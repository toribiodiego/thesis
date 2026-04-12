"""Compute ground-truth discounted returns from replay buffer data.

Backward-iterates G_t = r_t + gamma * G_{t+1} within each complete
episode identified by terminal flags. States from the incomplete
final episode (if the buffer doesn't end on a terminal) are
excluded (NaN) since their true return is unknown.
"""

import numpy as np

from src.analysis.replay_buffer import ReplayData


def compute_returns(replay: ReplayData, gamma: float) -> np.ndarray:
    """Compute discounted returns for each timestep in the replay buffer.

    For each complete episode (ending with terminal=True), computes
    G_t = r_t + gamma * G_{t+1} by backward iteration from the
    terminal state (where G_T = r_T). States in the incomplete
    trailing episode (after the last terminal) get NaN.

    Args:
        replay: ReplayData with rewards and terminals arrays.
        gamma: Discount factor (e.g., 0.997 for BBF, 0.99 for SPR/DER).

    Returns:
        (N,) float32 array of discounted returns. NaN for timesteps
        in the incomplete final episode.
    """
    n = len(replay.rewards)
    returns = np.full(n, np.nan, dtype=np.float32)

    # Find episode boundaries (indices where terminal is True)
    terminal_indices = np.where(replay.terminals)[0]

    if len(terminal_indices) == 0:
        # No complete episodes -- all NaN
        return returns

    # Process each complete episode by backward iteration
    ep_start = 0
    for ep_end in terminal_indices:
        # Episode spans [ep_start, ep_end] inclusive
        # At the terminal state: G_T = r_T
        returns[ep_end] = replay.rewards[ep_end]

        # Backward iterate: G_t = r_t + gamma * G_{t+1}
        for t in range(ep_end - 1, ep_start - 1, -1):
            returns[t] = replay.rewards[t] + gamma * returns[t + 1]

        ep_start = ep_end + 1

    # States after the last terminal remain NaN (incomplete episode)
    return returns
