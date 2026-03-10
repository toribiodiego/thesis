"""
Uniform Experience Replay Buffer for DQN.

Implements a circular buffer storing (s, a, r, s', done) transitions.
Features:
- Ring buffer with wrap-around
- Episode boundary tracking to prevent cross-episode samples
- uint8 storage for memory efficiency
- Configurable capacity (default ~1M transitions)
- Sequence sampling for SPR (K consecutive transitions)
- N-step return computation (optional, for Rainbow DQN)
"""

from collections import deque
from typing import Dict, Optional, Tuple, Union

import numpy as np
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
        n_step: Number of steps for multi-step returns (default 1).
                When n > 1, transitions are accumulated in a pending deque and
                stored with discounted n-step returns R^(n) = sum gamma^k r_k.
                Episode boundaries flush pending transitions with truncated returns.
        gamma: Discount factor for n-step return computation (default 0.99).
               Only used when n_step > 1.

    Memory layout:
        - observations: (capacity, *obs_shape) array in uint8
        - actions: (capacity,) array in int64
        - rewards: (capacity,) array in float32
        - dones: (capacity,) array in bool
        - episode_starts: (capacity,) array in bool marking episode boundaries
        - next_observations: (capacity, *obs_shape) array in uint8 (only when n_step > 1)
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        dtype: np.dtype = np.uint8,
        normalize: bool = True,
        min_size: int = 50_000,
        device: Optional[Union[str, torch.device]] = None,
        pin_memory: bool = False,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        self.capacity = capacity
        self.obs_shape = obs_shape
        self.dtype = dtype
        self.normalize = normalize  # Whether to normalize to [0,1] on sample
        self.min_size = min_size  # Minimum buffer size before sampling is allowed
        self.device = torch.device(device) if device is not None else None
        self.pin_memory = pin_memory  # Use pinned memory for faster GPU transfer
        self.n_step = n_step
        self.gamma = gamma

        # Circular buffer index
        self.index = 0
        self.size = 0  # Current number of stored transitions

        # Storage arrays
        # Store observations (states) - we store both s_t and s_{t+1}
        self.observations = np.zeros((capacity, *obs_shape), dtype=dtype)

        # Store actions, rewards, dones
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=bool)

        # Episode boundary markers
        # episode_starts[i] = True means index i is the first transition of an episode
        self.episode_starts = np.zeros(capacity, dtype=bool)

        # N-step return support
        if n_step > 1:
            # Pending deque accumulates raw transitions before computing
            # n-step returns and committing to the main buffer
            self._pending: deque = deque()
            # Explicit next_state storage (consecutive observation trick
            # breaks with n-step gaps between stored transitions)
            self.next_observations = np.zeros(
                (capacity, *obs_shape), dtype=dtype
            )
        else:
            self._pending = None
            self.next_observations = None

    def _convert_obs(self, obs: np.ndarray) -> np.ndarray:
        """Convert an observation to the storage dtype."""
        if obs.dtype == np.float32 or obs.dtype == np.float64:
            return (obs * 255).astype(self.dtype)
        return obs.astype(self.dtype)

    def _store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Write a single transition to the ring buffer.

        Low-level storage method. State must already be in storage dtype.
        For n_step > 1, next_state is stored in next_observations.
        """
        self.observations[self.index] = state
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.dones[self.index] = done

        if self.next_observations is not None:
            self.next_observations[self.index] = next_state

        # Mark episode start
        if self.size == 0:
            self.episode_starts[self.index] = True
        else:
            prev_index = (self.index - 1) % self.capacity
            self.episode_starts[self.index] = self.dones[prev_index]

        # Advance index (circular)
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def append(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        """
        Add a transition to the buffer.

        When n_step == 1, stores the transition directly. When n_step > 1,
        accumulates transitions in a pending deque and stores n-step
        transitions with discounted returns. Episode boundaries flush
        pending transitions with truncated returns.

        Args:
            state: Current observation, shape (*obs_shape,) in uint8 or float32
            action: Action taken (integer)
            reward: Reward received (float)
            next_state: Next observation, shape (*obs_shape,)
            done: Whether episode ended (bool)
        """
        # Validate shapes
        assert (
            state.shape == self.obs_shape
        ), f"State shape {state.shape} doesn't match expected {self.obs_shape}"
        assert (
            next_state.shape == self.obs_shape
        ), f"Next state shape {next_state.shape} doesn't match expected {self.obs_shape}"

        # Convert to storage dtype
        if state.dtype != self.dtype:
            state = self._convert_obs(state)

        if self.n_step == 1:
            self._store(state, action, reward, next_state, done)
            return

        # N-step: convert next_state and push to pending deque
        if next_state.dtype != self.dtype:
            next_state = self._convert_obs(next_state)

        self._pending.append((state, action, reward, next_state, done))

        # When deque has n transitions, commit the oldest as an n-step transition
        if len(self._pending) == self.n_step:
            self._commit_n_step()

        # Episode boundary: flush all remaining pending with truncated returns
        if done:
            self._flush_pending()

    def _compute_n_step_return(self, transitions) -> float:
        """Compute discounted return R^(n) = sum_{k=0}^{n-1} gamma^k * r_k."""
        R = 0.0
        for k, (_, _, r, _, _) in enumerate(transitions):
            R += (self.gamma ** k) * r
        return R

    def _commit_n_step(self) -> None:
        """Store the oldest pending transition as an n-step transition."""
        transitions = list(self._pending)
        s_0, a_0, _, _, _ = transitions[0]
        _, _, _, s_n, done_n = transitions[-1]
        R = self._compute_n_step_return(transitions)
        self._store(s_0, a_0, R, s_n, done_n)
        self._pending.popleft()

    def _flush_pending(self) -> None:
        """Flush remaining pending transitions with truncated n-step returns."""
        while self._pending:
            transitions = list(self._pending)
            s_0, a_0, _, _, _ = transitions[0]
            _, _, _, s_n, done_n = transitions[-1]
            R = self._compute_n_step_return(transitions)
            self._store(s_0, a_0, R, s_n, done_n)
            self._pending.popleft()

    def _is_valid_index(self, idx: int) -> bool:
        """
        Check if an index is valid for sampling.

        For n_step == 1:
            Valid if within buffer, not an episode start, and (terminal OR
            next index is in the same episode).
        For n_step > 1:
            Valid if within buffer (next_state stored explicitly).

        Args:
            idx: Index to check

        Returns:
            True if index can be safely sampled
        """
        if idx >= self.size:
            return False

        # n-step: next_state stored explicitly, no boundary checks needed
        if self.next_observations is not None:
            return True

        # n=1: original boundary-checking logic
        if self.episode_starts[idx]:
            return False

        if self.dones[idx]:
            return True

        next_idx = (idx + 1) % self.capacity
        if next_idx >= self.size:
            return False

        if self.episode_starts[next_idx]:
            return False

        return True

    def _get_valid_indices(self) -> np.ndarray:
        """
        Get all valid indices for sampling.

        Uses vectorized numpy operations instead of per-index Python loop.
        For n_step > 1 (explicit next_observations), all stored indices are
        valid since next_state is stored alongside each transition.

        Returns:
            Array of valid indices that can be safely sampled
        """
        n = self.size if self.size < self.capacity else self.capacity
        indices = np.arange(n, dtype=np.int64)

        # n-step: next_state stored explicitly, all indices valid
        if self.next_observations is not None:
            if self.size < self.capacity:
                return indices[indices < self.size]
            return indices

        # n=1: original episode-boundary logic
        valid_mask = np.ones(n, dtype=bool)

        # Exclude episode starts
        valid_mask &= ~self.episode_starts[:n]

        # Exclude indices beyond buffer size
        if self.size < self.capacity:
            valid_mask &= indices < self.size

        # For non-terminal transitions, check next index validity
        next_indices = (indices + 1) % self.capacity
        is_terminal = self.dones[:n]

        next_out_of_bounds = next_indices >= self.size if self.size < self.capacity else np.zeros(n, dtype=bool)
        next_is_episode_start = self.episode_starts[next_indices]

        # Invalidate non-terminal transitions with bad next indices
        valid_mask &= is_terminal | (~next_out_of_bounds & ~next_is_episode_start)

        return indices[valid_mask]

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
            valid_indices, size=batch_size, replace=False
        )

        # Gather transitions (still in uint8)
        states_uint8 = self.observations[sampled_indices]
        actions = self.actions[sampled_indices]
        rewards = self.rewards[sampled_indices]
        dones = self.dones[sampled_indices]

        # Get next states
        if self.next_observations is not None:
            # n-step: next_state stored explicitly
            next_states_uint8 = self.next_observations[sampled_indices]
        else:
            # n=1: next state is the consecutive observation
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
            if self.pin_memory and self.device.type == "cuda":
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
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
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

    def _get_valid_sequence_starts(self, seq_len: int) -> np.ndarray:
        """
        Get indices that can start a valid sequence of length seq_len.

        A starting index is valid if the next seq_len indices (inclusive)
        all contain stored data and do not cross the write pointer in a
        wrapped buffer. Episode boundaries are NOT filtered here -- the
        caller uses the returned done flags for masking.

        Args:
            seq_len: Number of transitions in the sequence. The sequence
                uses seq_len + 1 consecutive observation slots (the
                starting state plus seq_len next-states).

        Returns:
            Array of valid starting indices.
        """
        if self.size < seq_len + 1:
            return np.array([], dtype=np.int64)

        if self.size < self.capacity:
            # Buffer not full: indices 0..size-1 have data.
            # Need start + seq_len < size (to access obs[start+seq_len]).
            max_start = self.size - seq_len - 1
            if max_start < 0:
                return np.array([], dtype=np.int64)
            return np.arange(max_start + 1, dtype=np.int64)
        else:
            # Buffer full (wrapped). All capacity slots have data, but
            # sequences must not span the write pointer (self.index),
            # where old data was just overwritten.
            # The write pointer is at self.index. The freshest datum is
            # at (self.index - 1) % capacity, the oldest at self.index.
            #
            # A sequence start..start+seq_len (mod capacity) is invalid
            # if any of those indices equals self.index (the oldest
            # slot about to be overwritten is stale relative to its
            # neighbours).
            #
            # Equivalently, exclude starts in the range
            # [index - seq_len, index] (mod capacity).
            all_indices = np.arange(self.capacity, dtype=np.int64)
            # Distance from each index to the write pointer, going forward
            dist = (self.index - all_indices) % self.capacity
            valid_mask = dist > seq_len
            return all_indices[valid_mask]

    def sample_sequences(
        self, batch_size: int, seq_len: int
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Sample a batch of consecutive transition sequences.

        Each sequence contains seq_len transitions (and seq_len + 1
        observations). Episode boundaries are encoded in the returned
        done flags -- the caller is responsible for masking predictions
        that cross boundaries.

        Args:
            batch_size: Number of sequences to sample.
            seq_len: Length of each sequence (number of transitions).
                For SPR with K prediction steps, use seq_len=K.

        Returns:
            Dictionary containing:
                - 'states': (batch_size, seq_len+1, *obs_shape) float32
                    Observations s_t through s_{t+seq_len}.
                - 'actions': (batch_size, seq_len) int64
                    Actions a_t through a_{t+seq_len-1}.
                - 'rewards': (batch_size, seq_len) float32
                    Rewards r_t through r_{t+seq_len-1}.
                - 'dones': (batch_size, seq_len) bool
                    Done flags for each transition.

        Raises:
            ValueError: If not enough valid starting indices.
        """
        valid_starts = self._get_valid_sequence_starts(seq_len)

        if len(valid_starts) < batch_size:
            raise ValueError(
                f"Not enough valid sequence starts. "
                f"Requested {batch_size}, available {len(valid_starts)}. "
                f"Buffer size: {self.size}, seq_len: {seq_len}"
            )

        # Sample starting indices without replacement
        starts = np.random.choice(valid_starts, size=batch_size, replace=False)

        # Build index arrays for gathering: (batch_size, seq_len+1) for obs,
        # (batch_size, seq_len) for actions/rewards/dones
        offsets_obs = np.arange(seq_len + 1)  # 0..seq_len
        offsets_trans = np.arange(seq_len)  # 0..seq_len-1

        obs_indices = (starts[:, None] + offsets_obs[None, :]) % self.capacity
        trans_indices = (starts[:, None] + offsets_trans[None, :]) % self.capacity

        # Gather data
        states = self.observations[obs_indices].astype(np.float32)
        actions = self.actions[trans_indices]
        rewards = self.rewards[trans_indices]
        dones = self.dones[trans_indices]

        if self.normalize:
            states = states / 255.0

        # Convert to tensors if device is set
        if self.device is not None:
            if self.pin_memory and self.device.type == "cuda":
                states = torch.from_numpy(states).pin_memory()
                actions = torch.from_numpy(actions).pin_memory()
                rewards = torch.from_numpy(rewards).pin_memory()
                dones = torch.from_numpy(dones.copy()).pin_memory()
            else:
                states = torch.from_numpy(states)
                actions = torch.from_numpy(actions)
                rewards = torch.from_numpy(rewards)
                dones = torch.from_numpy(dones.copy())

            states = states.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            rewards = rewards.to(self.device, non_blocking=True)
            dones = dones.to(self.device, non_blocking=True)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
        }

    def __len__(self) -> int:
        """
        Return current number of transitions stored.

        Returns:
            Number of transitions in buffer
        """
        return self.size
