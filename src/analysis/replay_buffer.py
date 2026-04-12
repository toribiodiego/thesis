"""Load replay buffer transitions from training checkpoints.

Reads replay_buffer_<step>.npz files produced by Dopamine's
bundle_and_checkpoint, filters cross-episode boundaries using
the terminal flag, and returns valid consecutive transition pairs
for downstream analysis (inverse dynamics probing, transition
model evaluation).
"""

import os
from dataclasses import dataclass

import numpy as np


@dataclass
class ReplayData:
    """Raw replay buffer contents.

    Attributes:
        observations: (N, 84, 84) uint8 single grayscale frames.
        actions: (N,) int32 actions taken.
        rewards: (N,) float32 rewards (clipped).
        terminals: (N,) bool, True at episode boundaries.
        add_count: Number of valid entries in the buffer.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    add_count: int


@dataclass
class TransitionData:
    """Valid consecutive transition pairs filtered by terminal flag.

    Each entry i represents a within-episode transition where
    terminal[source_index] == 0, guaranteeing obs and obs_next
    are from the same episode.

    Attributes:
        obs: (M, 84, 84) uint8 frames at time t.
        obs_next: (M, 84, 84) uint8 frames at time t+1.
        actions: (M,) int32 actions taken at time t.
        rewards: (M,) float32 rewards received at time t.
    """

    obs: np.ndarray
    obs_next: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray


def load_replay_buffer(run_dir: str, step: int) -> ReplayData:
    """Load raw replay buffer data from a checkpoint.

    Args:
        run_dir: Path to the run directory.
        step: Checkpoint step number.

    Returns:
        ReplayData with observations, actions, rewards, terminals.

    Raises:
        FileNotFoundError: If the replay buffer file is missing.
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    npz_path = os.path.join(ckpt_dir, f"replay_buffer_{step}.npz")

    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Replay buffer not found: {npz_path}")

    data = np.load(npz_path)

    # Arrays have shape (N, 1, ...) where dim 1 is the env index;
    # squeeze it since we always have a single training env.
    observations = data["observation"].squeeze(1)  # (N, 84, 84)
    actions = data["action"].squeeze(1)  # (N,)
    rewards = data["reward"].squeeze(1)  # (N,)
    terminals = data["terminal"].squeeze(1).astype(bool)  # (N,)

    # Read add_count to know how many entries are valid
    add_count_path = os.path.join(ckpt_dir, f"add_count_ckpt.{step}.gz")
    if os.path.isfile(add_count_path):
        import gzip

        with gzip.open(add_count_path, "rb") as f:
            add_count = int(np.load(f))
    else:
        add_count = len(observations)

    # Truncate to valid entries (buffer may be larger than add_count
    # if capacity > steps collected)
    if add_count < len(observations):
        observations = observations[:add_count]
        actions = actions[:add_count]
        rewards = rewards[:add_count]
        terminals = terminals[:add_count]

    return ReplayData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        add_count=add_count,
    )


def get_valid_transitions(replay: ReplayData) -> TransitionData:
    """Extract consecutive frame pairs that do not cross episode boundaries.

    For each timestep t where terminal[t] == False, the pair
    (obs[t], obs[t+1]) is a valid within-episode transition.
    Timesteps where terminal[t] == True are excluded because
    obs[t+1] belongs to a new episode.

    The last timestep is always excluded since there is no obs[t+1].

    Args:
        replay: Raw replay buffer data from load_replay_buffer.

    Returns:
        TransitionData with matched obs/obs_next/actions/rewards.
    """
    # Valid indices: not terminal and not the last entry
    valid = ~replay.terminals[:-1]

    return TransitionData(
        obs=replay.observations[:-1][valid],
        obs_next=replay.observations[1:][valid],
        actions=replay.actions[:-1][valid],
        rewards=replay.rewards[:-1][valid],
    )
