"""Extract multi-step frame sequences from replay buffer data.

Slides a window of K+1 steps over the replay buffer, keeping only
windows where no terminal flag falls within the first K positions
(all K+1 frames belong to the same episode). Used by M16 transition
model evaluation for autoregressive prediction over K steps.
"""

from dataclasses import dataclass

import numpy as np

from src.analysis.replay_buffer import ReplayData


@dataclass
class SequenceData:
    """Multi-step frame sequences from the replay buffer.

    Attributes:
        obs: (M, K+1, 84, 84) uint8 single grayscale frames.
            obs[:, 0] is the starting frame, obs[:, k] is k steps later.
        actions: (M, K) int32 actions taken at each step.
            actions[:, 0] is the action from obs[:,0] to obs[:,1], etc.
    """

    obs: np.ndarray
    actions: np.ndarray


def get_multi_step_sequences(
    replay: ReplayData, K: int = 5
) -> SequenceData:
    """Extract K+1-length sequences with no episode boundary crossings.

    For each starting index i, the window [i, i+K] (inclusive) is
    valid if terminals[i] through terminals[i+K-1] are all False.
    The terminal at position i+K is allowed to be True (the last
    frame can be a terminal state -- the sequence ends there).

    Args:
        replay: ReplayData with observations, actions, terminals.
        K: Number of forward steps (default 5). Sequences have
            K+1 frames and K actions.

    Returns:
        SequenceData with obs (M, K+1, 84, 84) and actions (M, K).
    """
    n = len(replay.observations)
    if n < K + 1:
        return SequenceData(
            obs=np.empty((0, K + 1, 84, 84), dtype=np.uint8),
            actions=np.empty((0, K), dtype=np.int32),
        )

    # Build validity mask: terminal[i:i+K] must all be False
    # (no episode boundary in the first K positions of the window)
    terms = replay.terminals
    valid = np.ones(n - K, dtype=bool)
    for offset in range(K):
        valid &= ~terms[offset : n - K + offset]

    indices = np.where(valid)[0]

    if len(indices) == 0:
        return SequenceData(
            obs=np.empty((0, K + 1, 84, 84), dtype=np.uint8),
            actions=np.empty((0, K), dtype=np.int32),
        )

    # Gather sequences using index offsets
    obs_seqs = np.stack(
        [replay.observations[indices + k] for k in range(K + 1)], axis=1
    )
    act_seqs = np.stack(
        [replay.actions[indices + k] for k in range(K)], axis=1
    )

    return SequenceData(obs=obs_seqs, actions=act_seqs)
