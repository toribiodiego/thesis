"""
Prioritized Experience Replay Buffer (Schaul et al. 2016).

Extends ReplayBuffer with sum-tree based proportional prioritization
and importance-sampling (IS) weight correction. Used by Rainbow DQN
where priorities are per-sample KL divergence rather than TD error.

Sampling probability: P(i) = p_i^alpha / sum_k p_k^alpha
IS weights:           w_i  = (N * P(i))^{-beta} / max_j(w_j)

New transitions receive max priority so they are sampled at least
once. After the first training step, update_priorities() is called
with per-sample losses to set accurate priorities.

sample_sequences() uses uniform sampling (inherited from ReplayBuffer)
for SPR, which needs consecutive observations independent of the
priority-weighted TD batch.
"""

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from .replay_buffer import ReplayBuffer
from .sum_tree import SumTree


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Replay buffer with proportional priority sampling and IS weights.

    Inherits ring-buffer storage, episode boundary tracking, n-step
    return computation, and sequence sampling from ReplayBuffer.
    Adds a sum-tree for O(log N) priority-based sampling.

    Args:
        capacity: Maximum number of transitions to store.
        obs_shape: Observation shape (C, H, W).
        dtype: Storage dtype for observations.
        normalize: Normalize observations to [0,1] on sample.
        min_size: Minimum transitions before sampling is allowed.
        device: Target device for sampled tensors.
        pin_memory: Use pinned memory for GPU transfer.
        n_step: Number of steps for multi-step returns.
        gamma: Discount factor for n-step returns.
        alpha: Priority exponent. 0 = uniform, 1 = full prioritization.
            Rainbow uses 0.5.
        beta_start: Initial IS correction exponent (anneals to beta_end).
        beta_end: Final IS correction exponent (typically 1.0).
        beta_frames: Number of frames over which beta anneals linearly.
        epsilon: Small constant added to raw priorities to prevent
            zero probability. Default 1e-6.

    Additional sample() output keys (beyond ReplayBuffer):
        - 'indices': (batch_size,) int64 array for update_priorities()
        - 'weights': (batch_size,) float32 IS weights
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
        alpha: float = 0.5,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_frames: int = 100_000,
        epsilon: float = 1e-6,
    ):
        super().__init__(
            capacity=capacity,
            obs_shape=obs_shape,
            dtype=dtype,
            normalize=normalize,
            min_size=min_size,
            device=device,
            pin_memory=pin_memory,
            n_step=n_step,
            gamma=gamma,
        )
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.tree = SumTree(capacity)

    def _store(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition and assign max priority."""
        # Set priority before parent advances self.index
        self.tree.update(self.index, self.tree.max_priority)
        super()._store(state, action, reward, next_state, done)

    def compute_beta(self, frame: int) -> float:
        """Linearly anneal beta from beta_start to beta_end."""
        progress = min(1.0, frame / self.beta_frames)
        return self.beta_start + progress * (self.beta_end - self.beta_start)

    def sample(
        self,
        batch_size: int,
        beta: float = None,
        frame: int = None,
    ) -> Dict[str, Union[np.ndarray, torch.Tensor]]:
        """
        Sample a batch proportional to priorities with IS weights.

        Args:
            batch_size: Number of transitions to sample.
            beta: IS correction exponent. If None, computed from frame
                or defaults to beta_start.
            frame: Current training frame for beta annealing. Ignored
                if beta is provided explicitly.

        Returns:
            Dictionary with standard keys (states, actions, rewards,
            next_states, dones) plus:
                - 'indices': (batch_size,) int64 -- for update_priorities
                - 'weights': (batch_size,) float32 -- IS weights
        """
        if beta is None:
            beta = self.compute_beta(frame) if frame is not None else self.beta_start

        # Stratified priority sampling via sum-tree
        indices = self.tree.batch_sample(batch_size)

        # Compute IS weights: w_i = (N * P(i))^{-beta} / max(w)
        total = self.tree.total
        N = self.size

        priorities = np.array([self.tree.get(i) for i in indices])
        probs = priorities / total

        weights = (N * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        # ---- Gather transitions (same logic as ReplayBuffer.sample) ----
        states_uint8 = self.observations[indices]
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        dones = self.dones[indices]

        if self.next_observations is not None:
            next_states_uint8 = self.next_observations[indices]
        else:
            next_indices = (indices + 1) % self.capacity
            next_states_uint8 = self.observations[next_indices]

        states = states_uint8.astype(np.float32)
        next_states = next_states_uint8.astype(np.float32)

        if self.normalize:
            states = states / 255.0
            next_states = next_states / 255.0

        # Convert to tensors if device is set
        if self.device is not None:
            if self.pin_memory and self.device.type == "cuda":
                states = torch.from_numpy(states).pin_memory()
                next_states = torch.from_numpy(next_states).pin_memory()
                actions = torch.from_numpy(actions).pin_memory()
                rewards = torch.from_numpy(rewards).pin_memory()
                dones = torch.from_numpy(dones).pin_memory()
                weights = torch.from_numpy(weights).pin_memory()
            else:
                states = torch.from_numpy(states)
                next_states = torch.from_numpy(next_states)
                actions = torch.from_numpy(actions)
                rewards = torch.from_numpy(rewards)
                dones = torch.from_numpy(dones)
                weights = torch.from_numpy(weights)

            states = states.to(self.device, non_blocking=True)
            next_states = next_states.to(self.device, non_blocking=True)
            actions = actions.to(self.device, non_blocking=True)
            rewards = rewards.to(self.device, non_blocking=True)
            dones = dones.to(self.device, non_blocking=True)
            weights = weights.to(self.device, non_blocking=True)

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
            "indices": indices,
            "weights": weights,
        }

    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        """
        Update priorities after loss computation.

        Args:
            indices: Buffer indices returned by sample().
            priorities: Raw priority values (e.g. per-sample KL
                divergence). Epsilon is added and alpha is applied
                internally: stored = (|p| + epsilon)^alpha.
        """
        for idx, p in zip(indices, priorities):
            stored = (abs(float(p)) + self.epsilon) ** self.alpha
            self.tree.update(int(idx), stored)

    def can_sample(self, batch_size: int = None) -> bool:
        """Check if buffer has enough data and nonzero priority mass."""
        if self.size < self.min_size:
            return False
        if self.tree.total <= 0:
            return False
        if batch_size is not None:
            return self.size >= batch_size
        return True
