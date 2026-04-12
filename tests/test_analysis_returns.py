"""Tests for discounted return computation.

Uses synthetic replay data with hand-computed expected returns.
No checkpoint or GPU needed.
"""

import numpy as np
import pytest

from src.analysis.replay_buffer import ReplayData
from src.analysis.returns import compute_returns


def _make_replay(rewards, terminals):
    """Create minimal ReplayData from reward and terminal arrays."""
    n = len(rewards)
    return ReplayData(
        observations=np.zeros((n, 84, 84), dtype=np.uint8),
        actions=np.zeros(n, dtype=np.int32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=bool),
        add_count=n,
    )


class TestComputeReturns:

    def test_single_episode_gamma_1(self):
        """gamma=1: return is sum of future rewards."""
        replay = _make_replay(
            rewards=[1, 2, 3],
            terminals=[False, False, True],
        )
        G = compute_returns(replay, gamma=1.0)
        # G[2] = 3, G[1] = 2 + 3 = 5, G[0] = 1 + 5 = 6
        np.testing.assert_allclose(G, [6.0, 5.0, 3.0])

    def test_single_episode_gamma_half(self):
        """gamma=0.5: G_t = r_t + 0.5 * G_{t+1}."""
        replay = _make_replay(
            rewards=[1, 2, 3],
            terminals=[False, False, True],
        )
        G = compute_returns(replay, gamma=0.5)
        # G[2] = 3, G[1] = 2 + 0.5*3 = 3.5, G[0] = 1 + 0.5*3.5 = 2.75
        np.testing.assert_allclose(G, [2.75, 3.5, 3.0])

    def test_two_episodes(self):
        """Returns reset at episode boundaries."""
        replay = _make_replay(
            rewards=[1, 1, 1, 10, 10],
            terminals=[False, False, True, False, True],
        )
        G = compute_returns(replay, gamma=1.0)
        # Episode 1: [0,1,2], G = [3, 2, 1]
        # Episode 2: [3,4], G = [20, 10]
        np.testing.assert_allclose(G, [3.0, 2.0, 1.0, 20.0, 10.0])

    def test_incomplete_final_episode_is_nan(self):
        """States after last terminal get NaN."""
        replay = _make_replay(
            rewards=[1, 1, 1, 5, 5],
            terminals=[False, False, True, False, False],
        )
        G = compute_returns(replay, gamma=1.0)
        # Episode 1: [0,1,2], G = [3, 2, 1]
        # Trailing [3,4]: NaN (incomplete)
        np.testing.assert_allclose(G[:3], [3.0, 2.0, 1.0])
        assert np.isnan(G[3])
        assert np.isnan(G[4])

    def test_no_complete_episodes(self):
        """All NaN if no terminal flags."""
        replay = _make_replay(
            rewards=[1, 2, 3],
            terminals=[False, False, False],
        )
        G = compute_returns(replay, gamma=0.99)
        assert np.all(np.isnan(G))

    def test_immediate_terminal(self):
        """Single-step episode."""
        replay = _make_replay(
            rewards=[5],
            terminals=[True],
        )
        G = compute_returns(replay, gamma=0.99)
        np.testing.assert_allclose(G, [5.0])

    def test_all_terminals(self):
        """Every step is a terminal (each step is its own episode)."""
        replay = _make_replay(
            rewards=[1, 2, 3],
            terminals=[True, True, True],
        )
        G = compute_returns(replay, gamma=0.99)
        np.testing.assert_allclose(G, [1.0, 2.0, 3.0])

    def test_zero_rewards(self):
        replay = _make_replay(
            rewards=[0, 0, 0],
            terminals=[False, False, True],
        )
        G = compute_returns(replay, gamma=0.99)
        np.testing.assert_allclose(G, [0.0, 0.0, 0.0])

    def test_typical_gamma(self):
        """gamma=0.99 with a realistic sequence."""
        replay = _make_replay(
            rewards=[0, 0, 1, 0, 0],
            terminals=[False, False, False, False, True],
        )
        G = compute_returns(replay, gamma=0.99)
        # G[4]=0, G[3]=0+0.99*0=0, G[2]=1+0.99*0=1,
        # G[1]=0+0.99*1=0.99, G[0]=0+0.99*0.99=0.9801
        expected = [0.99**2, 0.99, 1.0, 0.0, 0.0]
        np.testing.assert_allclose(G, expected, atol=1e-6)
