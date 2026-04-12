"""Tests for representation and Q-value extraction, and transition model evaluation.

Smoke tests using verification checkpoints on CPU. Each test
processes a small number of observations to verify shapes and dtype.
"""

import os

import numpy as np
import pytest

RUNS_DIR = os.path.join("experiments", "dqn_atari", "runs")
BBF_RUN = os.path.join(RUNS_DIR, "bbf_crazy_climber_seed13")
SPR_RUN = os.path.join(RUNS_DIR, "spr_crazy_climber_seed13")
DER_RUN = os.path.join(RUNS_DIR, "der_crazy_climber_seed13")
STEP = 10000


def _has_run(run_dir):
    return os.path.isdir(run_dir)


def _get_obs(run_dir, n=8):
    """Load n observations from replay buffer, stack into HWC format."""
    from src.analysis.replay_buffer import load_replay_buffer

    replay = load_replay_buffer(run_dir, STEP)
    # Stack 4 consecutive non-terminal frames into HWC
    frames = replay.observations
    obs_list = []
    i = 0
    while len(obs_list) < n and i + 3 < len(frames):
        # Skip if any of the 4 frames crosses an episode boundary
        if not any(replay.terminals[i : i + 3]):
            stack = np.stack(frames[i : i + 4], axis=-1)  # (84, 84, 4)
            obs_list.append(stack)
        i += 1
    return np.array(obs_list, dtype=np.uint8)  # (n, 84, 84, 4)


# ---------------------------------------------------------------------------
# BBF (IMPALA encoder, hidden_dim=2048)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_run(BBF_RUN), reason="BBF verification run not found")
class TestExtractRepresentationsBBF:

    def test_output_shape(self):
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.representations import extract_representations

        ckpt = load_checkpoint(BBF_RUN, STEP)
        obs = _get_obs(BBF_RUN, n=4)

        reps = extract_representations(ckpt, obs, batch_size=4)

        assert reps.shape == (4, 2048)
        assert reps.dtype == np.float32

    def test_nonzero_output(self):
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.representations import extract_representations

        ckpt = load_checkpoint(BBF_RUN, STEP)
        obs = _get_obs(BBF_RUN, n=2)

        reps = extract_representations(ckpt, obs, batch_size=2)

        assert np.any(reps != 0), "Representations should not be all zeros"

    def test_target_returns_none(self):
        """Verification runs have no target params."""
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.representations import extract_representations_target

        ckpt = load_checkpoint(BBF_RUN, STEP)
        obs = _get_obs(BBF_RUN, n=2)

        result = extract_representations_target(ckpt, obs)
        assert result is None


# ---------------------------------------------------------------------------
# SPR (DQN encoder, hidden_dim=512)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_run(SPR_RUN), reason="SPR verification run not found")
class TestExtractRepresentationsSPR:

    def test_output_shape(self):
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.representations import extract_representations

        ckpt = load_checkpoint(SPR_RUN, STEP)
        obs = _get_obs(SPR_RUN, n=4)

        reps = extract_representations(ckpt, obs, batch_size=4)

        assert reps.shape == (4, 512)
        assert reps.dtype == np.float32


# ---------------------------------------------------------------------------
# Chunked batch processing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_run(BBF_RUN), reason="BBF verification run not found")
class TestBatchProcessing:

    def test_chunking_matches_single_batch(self):
        """Results should be identical whether processed in 1 or 2 chunks."""
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.representations import extract_representations

        ckpt = load_checkpoint(BBF_RUN, STEP)
        obs = _get_obs(BBF_RUN, n=6)

        reps_single = extract_representations(ckpt, obs, batch_size=6, seed=0)
        reps_chunked = extract_representations(ckpt, obs, batch_size=3, seed=0)

        np.testing.assert_allclose(reps_single, reps_chunked, rtol=1e-5)


# ---------------------------------------------------------------------------
# Q-value extraction
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_run(BBF_RUN), reason="BBF verification run not found")
class TestExtractQValuesBBF:

    def test_output_shape(self):
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.representations import extract_q_values

        ckpt = load_checkpoint(BBF_RUN, STEP)
        obs = _get_obs(BBF_RUN, n=4)

        q = extract_q_values(ckpt, obs, batch_size=4)

        # CrazyClimber has 9 actions
        assert q.shape == (4, 9)
        assert q.dtype == np.float32

    def test_finite_values(self):
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.representations import extract_q_values

        ckpt = load_checkpoint(BBF_RUN, STEP)
        obs = _get_obs(BBF_RUN, n=2)

        q = extract_q_values(ckpt, obs, batch_size=2)

        assert np.all(np.isfinite(q)), "Q-values should be finite"


@pytest.mark.skipif(not _has_run(SPR_RUN), reason="SPR verification run not found")
class TestExtractQValuesSPR:

    def test_output_shape(self):
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.representations import extract_q_values

        ckpt = load_checkpoint(SPR_RUN, STEP)
        obs = _get_obs(SPR_RUN, n=4)

        q = extract_q_values(ckpt, obs, batch_size=4)

        assert q.shape == (4, 9)
        assert q.dtype == np.float32


# ---------------------------------------------------------------------------
# Transition model evaluation
# ---------------------------------------------------------------------------


def _get_transition_pairs(run_dir, n=4):
    """Build stacked frame transition pairs from replay buffer.

    Returns (obs, actions, obs_next) where obs/obs_next are
    (n, 84, 84, 4) uint8 stacked frames and actions is (n,) int32.
    """
    from src.analysis.replay_buffer import load_replay_buffer

    replay = load_replay_buffer(run_dir, STEP)
    frames = replay.observations
    obs_list, obs_next_list, act_list = [], [], []
    i = 0
    while len(obs_list) < n and i + 4 < len(frames):
        # Need 5 consecutive non-terminal frames for a valid pair:
        # frames[i:i+4] -> obs_t, frames[i+1:i+5] -> obs_{t+1}
        if not any(replay.terminals[i : i + 4]):
            obs_t = np.stack(frames[i : i + 4], axis=-1)       # (84, 84, 4)
            obs_next = np.stack(frames[i + 1 : i + 5], axis=-1)  # (84, 84, 4)
            obs_list.append(obs_t)
            obs_next_list.append(obs_next)
            act_list.append(replay.actions[i + 3])  # action at time of last frame
        i += 1
    return (
        np.array(obs_list, dtype=np.uint8),
        np.array(act_list, dtype=np.int32),
        np.array(obs_next_list, dtype=np.uint8),
    )


@pytest.mark.skipif(not _has_run(BBF_RUN), reason="BBF verification run not found")
class TestEvaluateTransitionModelBBF:

    def test_output_shape(self):
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.representations import evaluate_transition_model

        ckpt = load_checkpoint(BBF_RUN, STEP)
        obs, actions, obs_next = _get_transition_pairs(BBF_RUN, n=4)

        sims = evaluate_transition_model(ckpt, obs, actions, obs_next, batch_size=4)

        assert sims.shape == (4,)
        assert sims.dtype == np.float32

    def test_values_in_range(self):
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.representations import evaluate_transition_model

        ckpt = load_checkpoint(BBF_RUN, STEP)
        obs, actions, obs_next = _get_transition_pairs(BBF_RUN, n=4)

        sims = evaluate_transition_model(ckpt, obs, actions, obs_next, batch_size=4)

        assert np.all(sims >= -1.0 - 1e-6)
        assert np.all(sims <= 1.0 + 1e-6)
        assert np.all(np.isfinite(sims))


@pytest.mark.skipif(not _has_run(SPR_RUN), reason="SPR verification run not found")
class TestEvaluateTransitionModelSPR:

    def test_output_shape(self):
        from src.analysis.checkpoint import load_checkpoint
        from src.analysis.representations import evaluate_transition_model

        ckpt = load_checkpoint(SPR_RUN, STEP)
        obs, actions, obs_next = _get_transition_pairs(SPR_RUN, n=4)

        sims = evaluate_transition_model(ckpt, obs, actions, obs_next, batch_size=4)

        assert sims.shape == (4,)
        assert sims.dtype == np.float32
