"""Tests for the observation collection module.

Unit tests for helpers (no environment needed) and a small smoke
test for greedy collection using a real checkpoint + Atari env.
All CPU-only; the smoke test runs only 20 steps.
"""

import os

import numpy as np
import pytest

from src.analysis.observations import (
    CollectedData,
    _game_to_env_id,
)

# ---------------------------------------------------------------------------
# Pure unit tests (no env, no JAX)
# ---------------------------------------------------------------------------


class TestGameToEnvId:

    def test_camel_case(self):
        assert _game_to_env_id("CrazyClimber") == "ALE/CrazyClimber-v5"

    def test_snake_case(self):
        assert _game_to_env_id("crazy_climber") == "ALE/CrazyClimber-v5"

    def test_single_word(self):
        assert _game_to_env_id("Pong") == "ALE/Pong-v5"

    def test_single_word_snake(self):
        assert _game_to_env_id("pong") == "ALE/Pong-v5"


# ---------------------------------------------------------------------------
# Environment smoke test (needs ale-py, no JAX)
# ---------------------------------------------------------------------------

_has_ale = True
try:
    import ale_py  # noqa: F401
except ImportError:
    _has_ale = False


@pytest.mark.skipif(not _has_ale, reason="ale-py not installed")
class TestEnvironmentCreation:

    def test_make_atari_env_shape(self):
        from src.envs import make_atari_env

        env = make_atari_env(
            "ALE/CrazyClimber-v5",
            noop_max=0,
            clip_rewards=True,
            episode_life=False,
        )
        obs, _ = env.reset(seed=42)
        # FrameStack outputs CHW: (4, 84, 84)
        assert obs.shape == (4, 84, 84)
        assert obs.dtype == np.uint8
        env.close()


# ---------------------------------------------------------------------------
# Greedy collection smoke test (needs checkpoint + ale-py + JAX on CPU)
# ---------------------------------------------------------------------------

BBF_RUN = os.path.join("experiments", "dqn_atari", "runs", "bbf_crazy_climber_seed13")
SPR_RUN = os.path.join("experiments", "dqn_atari", "runs", "spr_crazy_climber_seed13")
STEP = 10000


def _can_run_collection():
    """Check if we have ale-py and a verification checkpoint."""
    if not _has_ale:
        return False
    return os.path.isdir(BBF_RUN)


@pytest.mark.skipif(not _can_run_collection(), reason="ale-py or checkpoint missing")
class TestCollectGreedy:
    """Smoke test: 20 steps of greedy collection on CPU."""

    def test_bbf_collect_small(self):
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.observations import collect_greedy

        ckpt = load_checkpoint(BBF_RUN, STEP)
        data = collect_greedy(
            ckpt, game="CrazyClimber", num_steps=20,
            epsilon=0.05, seed=42, noop_max=0,
        )

        assert isinstance(data, CollectedData)
        assert data.observations.shape == (20, 84, 84, 4)
        assert data.observations.dtype == np.uint8
        assert data.actions.shape == (20,)
        assert data.rewards.shape == (20,)
        assert data.terminals.shape == (20,)
        # All actions should be valid
        assert np.all(data.actions >= 0)
        assert np.all(data.actions < ckpt.num_actions)

    def test_spr_collect_small(self):
        if not os.path.isdir(SPR_RUN):
            pytest.skip("SPR verification run not found")

        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.observations import collect_greedy

        ckpt = load_checkpoint(SPR_RUN, STEP)
        data = collect_greedy(
            ckpt, game="CrazyClimber", num_steps=20,
            epsilon=0.05, seed=42, noop_max=0,
        )

        assert data.observations.shape == (20, 84, 84, 4)
        assert np.all(data.actions < ckpt.num_actions)
