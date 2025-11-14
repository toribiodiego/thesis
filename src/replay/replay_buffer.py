"""
Uniform Experience Replay Buffer for DQN.

Implements a circular buffer storing (s, a, r, s', done) transitions.
Features:
- Ring buffer with wrap-around
- Episode boundary tracking to prevent cross-episode samples
- uint8 storage for memory efficiency
- Configurable capacity (default ~1M transitions)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import torch


class ReplayBuffer:
    """
    Circular replay buffer with episode boundary tracking.

    Stores transitions (state, action, reward, next_state, done) in a ring buffer.
    Prevents sampling across episode boundaries by tracking episode markers.

    Args:
        capacity: Maximum number of transitions to store (default 1_000_000)
        obs_shape: Shape of a single observation (H, W) or (C, H, W)
        dtype: Numpy dtype for observations (default uint8 for memory efficiency)
        normalize: Whether to normalize observations to [0,1] on sample (default True)
                   If True: uint8 [0,255] → float32 [0,1]
                   If False: uint8 [0,255] → float32 [0,255]
        min_size: Minimum number of transitions before sampling is allowed (default 50_000)
                  Used for warm-up phase to ensure sufficient exploration before training
        device: Target device for sampled tensors ('cpu', 'cuda', etc.). If None, returns NumPy arrays
        pin_memory: If True, use pinned memory for faster host-to-device transfer (only for GPU)

    Memory layout:
        - observations: (capacity, *obs_shape) array in uint8
        - actions: (capacity,) array in int64
        - rewards: (capacity,) array in float32
        - dones: (capacity,) array in bool
        - episode_starts: (capacity,) array in bool marking episode boundaries
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        dtype: np.dtype = np.uint8,
        normalize: bool = True,
        min_size: int = 50_000,
        device: Optional[Union[str, torch.device]] = None,
        pin_memory: bool = False
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.dtype = dtype
        self.normalize = normalize  # Whether to normalize to [0,1] on sample
        self.min_size = min_size  # Minimum buffer size before sampling is allowed
        self.device = torch.device(device) if device is not None else None
        self.pin_memory = pin_memory  # Use pinned memory for faster GPU transfer

        # Circular buffer index
        self.index = 0
        self.size = 0  # Current number of stored transitions

        # Storage arrays
        # Store observations (states) - we store both s_t and s_{t+1}
        self.observations = np.zeros(
            (capacity, *obs_shape),
            dtype=dtype
        )

        # Store actions, rewards, dones
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        # Episode boundary markers
        # episode_starts[i] = True means index i is the first transition of an episode
        self.episode_starts = np.zeros(capacity, dtype=bool)

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current observation, shape (*obs_shape,) in uint8 or float32
            action: Action taken (integer)
            reward: Reward received (float)
            next_state: Next observation, shape (*obs_shape,)
            done: Whether episode ended (bool)

        Notes:
            - If state/next_state are float32, they are converted to uint8
            - Episode boundaries are tracked via done flags
            - Buffer wraps around when full (circular buffer)
        """
        # Validate shapes
        assert state.shape == self.obs_shape, \
            f"State shape {state.shape} doesn't match expected {self.obs_shape}"
        assert next_state.shape == self.obs_shape, \
            f"Next state shape {next_state.shape} doesn't match expected {self.obs_shape}"

        # Convert to uint8 if needed
        if state.dtype != self.dtype:
            if state.dtype == np.float32 or state.dtype == np.float64:
                # Assume [0, 1] range, convert to [0, 255]
                state = (state * 255).astype(self.dtype)
            else:
                state = state.astype(self.dtype)

        # Store current state at index
        self.observations[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done

        # Mark episode start
        # First transition in buffer is always an episode start
        # After that, episode starts occur after done=True transitions
        if self.size == 0:
            self.episode_starts[self.index] = True
        else:
            # Check if previous transition was done
            prev_index = (self.index - 1) % self.capacity
            self.episode_starts[self.index] = self.dones[prev_index]

        # Advance index (circular)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _is_valid_index(self, idx: int) -> bool:
        """
        Check if an index is valid for sampling.

        An index is valid if:
        1. It's within the current buffer size
        2. It's not an episode start (we need previous states for frame stacking)
        3. For non-terminal transitions: next index exists and is in same episode
        4. For terminal transitions (done=True): always valid (next_state doesn't
           matter since TD target = r when done=True)

        Args:
            idx: Index to check

        Returns:
            True if index can be safely sampled
        """
        if idx >= self.size:
            return False

        # Check if this is an episode start
        # Episode starts can't be sampled because we need previous frames
        if self.episode_starts[idx]:
            return False

        # Terminal transitions are valid - next_state doesn't matter for TD target
        if self.dones[idx]:
            return True

        # For non-terminal transitions, check if next index exists and is in same episode
        next_idx = (idx + 1) % self.capacity
        if next_idx >= self.size:
            return False

        # If next index is an episode start, we've crossed an episode boundary
        if self.episode_starts[next_idx]:
            return False

        return True

    def _get_valid_indices(self) -> np.ndarray:
        """
        Get all valid indices for sampling.

        Returns:
            Array of valid indices that can be safely sampled
        """
        if self.size < self.capacity:
            # Buffer not full yet, check indices [0, size)
            valid = np.array([
                i for i in range(self.size)
                if self._is_valid_index(i)
            ], dtype=np.int64)
        else:
            # Buffer is full, check all indices
            valid = np.array([
                i for i in range(self.capacity)
                if self._is_valid_index(i)
            ], dtype=np.int64)

        return valid

    def sample(self, batch_size: int) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Sample a batch of transitions from the buffer.

        Samples without replacement from valid indices only (respects episode boundaries).
        Converts observations from uint8 to float32 and optionally normalizes to [0,1].
        If device is specified, returns PyTorch tensors on the target device.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing:
                - 'states': (batch_size, *obs_shape) float32 array/tensor
                - 'actions': (batch_size,) int64 array/tensor
                - 'rewards': (batch_size,) float32 array/tensor
                - 'next_states': (batch_size, *obs_shape) float32 array/tensor
                - 'dones': (batch_size,) bool array/tensor

            If self.device is None: returns NumPy arrays
            If self.device is set: returns PyTorch tensors on specified device

        Raises:
            ValueError: If batch_size > number of valid indices

        Notes:
            - Only samples from valid indices (no episode boundaries crossed)
            - Sampling is uniform without replacement within a batch
            - Observations converted from uint8 to float32 on sample
            - If self.normalize=True: values scaled to [0,1], else [0,255]
            - If pin_memory=True and device is GPU, uses pinned memory for faster transfer
        """
        # Get all valid indices
        valid_indices = self._get_valid_indices()

        # Check if we have enough valid samples
        if len(valid_indices) < batch_size:
            raise ValueError(
                f"Not enough valid samples in buffer. "
                f"Requested {batch_size}, but only {len(valid_indices)} valid samples available. "
                f"Buffer size: {self.size}"
            )

        # Sample without replacement
        sampled_indices = np.random.choice(
            valid_indices,
            size=batch_size,
            replace=False
        )

        # Gather transitions (still in uint8)
        states_uint8 = self.observations[sampled_indices]
        actions = self.actions[sampled_indices]
        rewards = self.rewards[sampled_indices]
        dones = self.dones[sampled_indices]

        # Get next states (index + 1)
        next_indices = (sampled_indices + 1) % self.capacity
        next_states_uint8 = self.observations[next_indices]

        # Convert observations to float32
        states = states_uint8.astype(np.float32)
        next_states = next_states_uint8.astype(np.float32)

        # Normalize to [0, 1] if configured
        if self.normalize:
            states = states / 255.0
            next_states = next_states / 255.0

        # If device specified, convert to PyTorch tensors and move to device
        if self.device is not None:
            # Convert to tensors (on CPU first, potentially with pinned memory)
            if self.pin_memory and self.device.type == 'cuda':
                # Use pinned memory for faster H2D transfer
                states_tensor = torch.from_numpy(states).pin_memory()
                next_states_tensor = torch.from_numpy(next_states).pin_memory()
                actions_tensor = torch.from_numpy(actions).pin_memory()
                rewards_tensor = torch.from_numpy(rewards).pin_memory()
                dones_tensor = torch.from_numpy(dones).pin_memory()
            else:
                # Regular tensors
                states_tensor = torch.from_numpy(states)
                next_states_tensor = torch.from_numpy(next_states)
                actions_tensor = torch.from_numpy(actions)
                rewards_tensor = torch.from_numpy(rewards)
                dones_tensor = torch.from_numpy(dones)

            # Move to target device
            states = states_tensor.to(self.device, non_blocking=True)
            next_states = next_states_tensor.to(self.device, non_blocking=True)
            actions = actions_tensor.to(self.device, non_blocking=True)
            rewards = rewards_tensor.to(self.device, non_blocking=True)
            dones = dones_tensor.to(self.device, non_blocking=True)

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }

    def can_sample(self, batch_size: int = None) -> bool:
        """
        Check if buffer has enough samples for training.

        Args:
            batch_size: Optional batch size to check. If None, only checks min_size.

        Returns:
            True if buffer has at least min_size transitions and (if batch_size provided)
            has enough valid indices to sample a batch.

        Notes:
            - Used by training loop to determine when to start optimization
            - Enforces warm-up period before training begins
            - If batch_size provided, also checks valid indices count
        """
        # Check if we've reached minimum size
        if self.size < self.min_size:
            return False

        # If batch_size specified, check we have enough valid samples
        if batch_size is not None:
            valid_indices = self._get_valid_indices()
            return len(valid_indices) >= batch_size

        return True

    def __len__(self) -> int:
        """
        Return current number of transitions stored.

        Returns:
            Number of transitions in buffer
        """
        return self.size
