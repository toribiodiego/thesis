"""Tests for the replay buffer transition loader.

CPU-only: loads npz files and validates shapes, terminal filtering,
and transition pair consistency against verification checkpoints.
"""

import os

import numpy as np
import pytest

from src.analysis.replay_buffer import (
    ReplayData,
    TransitionData,
    get_valid_transitions,
    load_replay_buffer,
)

RUNS_DIR = os.path.join("experiments", "dqn_atari", "runs")
BBF_RUN = os.path.join(RUNS_DIR, "bbf_crazy_climber_seed13")
SPR_RUN = os.path.join(RUNS_DIR, "spr_crazy_climber_seed13")
DER_RUN = os.path.join(RUNS_DIR, "der_crazy_climber_seed13")
STEP = 10000


def _has_run(run_dir):
    return os.path.isdir(run_dir)


# ---------------------------------------------------------------------------
# Unit tests for get_valid_transitions (synthetic data, no files)
# ---------------------------------------------------------------------------


class TestGetValidTransitions:

    def test_filters_terminal_transitions(self):
        replay = ReplayData(
            observations=np.arange(50).reshape(5, 10).astype(np.uint8),
            actions=np.array([0, 1, 2, 3, 4], dtype=np.int32),
            rewards=np.array([0.0, 1.0, 0.0, 0.0, 1.0], dtype=np.float32),
            terminals=np.array([False, True, False, False, True]),
            add_count=5,
        )
        trans = get_valid_transitions(replay)

        assert isinstance(trans, TransitionData)
        # Indices 0, 2, 3 are valid (not terminal, not last)
        # Index 1 is terminal, index 4 is last
        assert len(trans.obs) == 3
        assert len(trans.obs_next) == 3
        assert len(trans.actions) == 3
        assert len(trans.rewards) == 3

    def test_obs_next_is_consecutive(self):
        obs = np.arange(40).reshape(4, 10).astype(np.uint8)
        replay = ReplayData(
            observations=obs,
            actions=np.array([0, 1, 2, 3], dtype=np.int32),
            rewards=np.zeros(4, dtype=np.float32),
            terminals=np.array([False, False, False, False]),
            add_count=4,
        )
        trans = get_valid_transitions(replay)

        # 3 valid pairs: (0,1), (1,2), (2,3)
        assert len(trans.obs) == 3
        np.testing.assert_array_equal(trans.obs[0], obs[0])
        np.testing.assert_array_equal(trans.obs_next[0], obs[1])
        np.testing.assert_array_equal(trans.obs[2], obs[2])
        np.testing.assert_array_equal(trans.obs_next[2], obs[3])

    def test_all_terminal_yields_empty(self):
        replay = ReplayData(
            observations=np.zeros((3, 10), dtype=np.uint8),
            actions=np.zeros(3, dtype=np.int32),
            rewards=np.zeros(3, dtype=np.float32),
            terminals=np.array([True, True, True]),
            add_count=3,
        )
        trans = get_valid_transitions(replay)
        assert len(trans.obs) == 0

    def test_single_entry_yields_empty(self):
        replay = ReplayData(
            observations=np.zeros((1, 10), dtype=np.uint8),
            actions=np.zeros(1, dtype=np.int32),
            rewards=np.zeros(1, dtype=np.float32),
            terminals=np.array([False]),
            add_count=1,
        )
        trans = get_valid_transitions(replay)
        assert len(trans.obs) == 0


# ---------------------------------------------------------------------------
# Integration tests with verification checkpoints
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_run(BBF_RUN), reason="BBF verification run not found")
class TestLoadBBFReplayBuffer:

    def test_loads_successfully(self):
        replay = load_replay_buffer(BBF_RUN, STEP)
        assert isinstance(replay, ReplayData)

    def test_shapes(self):
        replay = load_replay_buffer(BBF_RUN, STEP)
        assert replay.observations.shape == (10000, 84, 84)
        assert replay.observations.dtype == np.uint8
        assert replay.actions.shape == (10000,)
        assert replay.rewards.shape == (10000,)
        assert replay.terminals.shape == (10000,)

    def test_add_count(self):
        replay = load_replay_buffer(BBF_RUN, STEP)
        assert replay.add_count == 10000

    def test_terminal_count_matches_episodes(self):
        """BBF verification run has 16 episodes = 16 terminal events."""
        replay = load_replay_buffer(BBF_RUN, STEP)
        assert replay.terminals.sum() == 16

    def test_valid_transitions(self):
        replay = load_replay_buffer(BBF_RUN, STEP)
        trans = get_valid_transitions(replay)

        # 10000 steps - 16 terminal - 1 last = 9983 valid pairs
        expected = 10000 - 16 - 1
        assert len(trans.obs) == expected
        assert trans.obs.shape[1:] == (84, 84)
        assert trans.obs_next.shape == trans.obs.shape

    def test_actions_in_valid_range(self):
        replay = load_replay_buffer(BBF_RUN, STEP)
        # CrazyClimber has 9 actions (0-8)
        assert replay.actions.min() >= 0
        assert replay.actions.max() <= 8


@pytest.mark.skipif(not _has_run(SPR_RUN), reason="SPR verification run not found")
class TestLoadSPRReplayBuffer:

    def test_loads_and_filters(self):
        replay = load_replay_buffer(SPR_RUN, STEP)
        trans = get_valid_transitions(replay)

        assert replay.observations.shape == (10000, 84, 84)
        assert len(trans.obs) > 0
        assert len(trans.obs) < 10000


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestLoadReplayBufferErrors:

    def test_missing_run_dir(self):
        with pytest.raises(FileNotFoundError):
            load_replay_buffer("/nonexistent/path", 10000)

    def test_missing_step(self):
        if _has_run(BBF_RUN):
            with pytest.raises(FileNotFoundError):
                load_replay_buffer(BBF_RUN, 99999)
        else:
            pytest.skip("BBF verification run not found")
