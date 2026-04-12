"""Tests for the analysis checkpoint loader.

All tests are CPU-only: file I/O, msgpack deserialization, gin string
parsing, and param shape verification. No forward passes or GPU needed.
"""

import os

import pytest

from src.analysis.checkpoint import (
    CheckpointData,
    _infer_num_actions,
    _parse_gin_network_params,
    load_checkpoint,
)

RUNS_DIR = os.path.join("experiments", "dqn_atari", "runs")

# Verification runs: BBF (IMPALA encoder), SPR and DER (DQN encoder)
BBF_RUN = os.path.join(RUNS_DIR, "bbf_crazy_climber_seed13")
BBFC_RUN = os.path.join(RUNS_DIR, "bbfc_crazy_climber_seed13")
SPR_RUN = os.path.join(RUNS_DIR, "spr_crazy_climber_seed13")
DER_RUN = os.path.join(RUNS_DIR, "der_crazy_climber_seed13")
STEP = 10000

# CrazyClimber has 9 discrete actions
CRAZY_CLIMBER_ACTIONS = 9


# -- Gin config parsing tests -----------------------------------------------


class TestGinParsing:
    """Test gin config string parsing without touching any files."""

    SAMPLE_GIN = (
        "BBFAgent.noisy = False\n"
        "BBFAgent.dueling = True\n"
        "BBFAgent.distributional = True\n"
        "BBFAgent.num_atoms = 51\n"
        "RainbowDQNNetwork.encoder_type = 'impala'\n"
        "RainbowDQNNetwork.hidden_dim = 2048\n"
        "RainbowDQNNetwork.renormalize = True\n"
        "RainbowDQNNetwork.width_scale = 4\n"
    )

    def test_parses_all_required_fields(self):
        params = _parse_gin_network_params(self.SAMPLE_GIN)
        assert params["encoder_type"] == "impala"
        assert params["hidden_dim"] == 2048
        assert params["width_scale"] == 4.0
        assert params["renormalize"] is True
        assert params["noisy"] is False
        assert params["dueling"] is True
        assert params["distributional"] is True
        assert params["num_atoms"] == 51

    def test_dqn_encoder_config(self):
        gin = (
            "BBFAgent.noisy = True\n"
            "BBFAgent.dueling = True\n"
            "BBFAgent.distributional = True\n"
            "BBFAgent.num_atoms = 51\n"
            "RainbowDQNNetwork.encoder_type = 'dqn'\n"
            "RainbowDQNNetwork.hidden_dim = 512\n"
            "RainbowDQNNetwork.renormalize = True\n"
            "RainbowDQNNetwork.width_scale = 1\n"
        )
        params = _parse_gin_network_params(gin)
        assert params["encoder_type"] == "dqn"
        assert params["hidden_dim"] == 512
        assert params["width_scale"] == 1.0
        assert params["noisy"] is True

    def test_missing_encoder_type_raises(self):
        gin = "RainbowDQNNetwork.hidden_dim = 512\n"
        with pytest.raises(ValueError, match="encoder_type"):
            _parse_gin_network_params(gin)

    def test_missing_hidden_dim_raises(self):
        gin = "RainbowDQNNetwork.encoder_type = 'dqn'\n"
        with pytest.raises(ValueError, match="hidden_dim"):
            _parse_gin_network_params(gin)

    def test_defaults_for_optional_fields(self):
        gin = (
            "RainbowDQNNetwork.encoder_type = 'dqn'\n"
            "RainbowDQNNetwork.hidden_dim = 512\n"
            "RainbowDQNNetwork.width_scale = 1\n"
        )
        params = _parse_gin_network_params(gin)
        assert params["noisy"] is False
        assert params["dueling"] is False
        assert params["distributional"] is True
        assert params["renormalize"] is False
        assert params["num_atoms"] == 51


# -- num_actions inference tests ---------------------------------------------


class TestNumActionsInference:
    """Test inferring num_actions from param tree shapes."""

    def test_correct_inference(self):
        import numpy as np

        params = {
            "head": {
                "advantage": {
                    "net": {"kernel": np.zeros((512, 9 * 51))}
                }
            }
        }
        assert _infer_num_actions(params, num_atoms=51) == 9

    def test_missing_key_raises(self):
        with pytest.raises(ValueError, match="Cannot infer"):
            _infer_num_actions({"head": {}}, num_atoms=51)

    def test_indivisible_raises(self):
        import numpy as np

        params = {
            "head": {
                "advantage": {
                    "net": {"kernel": np.zeros((512, 100))}
                }
            }
        }
        with pytest.raises(ValueError, match="not divisible"):
            _infer_num_actions(params, num_atoms=51)


# -- Integration tests using verification checkpoints -----------------------


def _has_run(run_dir):
    return os.path.isdir(run_dir)


@pytest.mark.skipif(
    not _has_run(BBF_RUN), reason="BBF verification run not found"
)
class TestLoadBBFCheckpoint:
    """Load BBF checkpoint (IMPALA encoder, noisy=False)."""

    def test_loads_successfully(self):
        ckpt = load_checkpoint(BBF_RUN, STEP)
        assert isinstance(ckpt, CheckpointData)

    def test_metadata(self):
        ckpt = load_checkpoint(BBF_RUN, STEP)
        assert ckpt.metadata["step"] == STEP
        assert "training_steps" in ckpt.metadata

    def test_architecture_params(self):
        ckpt = load_checkpoint(BBF_RUN, STEP)
        assert ckpt.encoder_type == "impala"
        assert ckpt.hidden_dim == 2048
        assert ckpt.num_atoms == 51
        assert ckpt.num_actions == CRAZY_CLIMBER_ACTIONS

    def test_online_params_structure(self):
        ckpt = load_checkpoint(BBF_RUN, STEP)
        params = ckpt.online_params
        assert "encoder" in params
        assert "head" in params
        assert "projection" in params
        assert "transition_model" in params
        assert "predictor" in params

    def test_target_params_absent(self):
        """Verification runs predate target param saving."""
        ckpt = load_checkpoint(BBF_RUN, STEP)
        assert ckpt.target_params is None

    def test_support_vector(self):
        ckpt = load_checkpoint(BBF_RUN, STEP)
        assert ckpt.support.shape == (51,)
        assert float(ckpt.support[0]) == pytest.approx(-10.0)
        assert float(ckpt.support[-1]) == pytest.approx(10.0)

    def test_network_def_matches_params(self):
        ckpt = load_checkpoint(BBF_RUN, STEP)
        net = ckpt.network_def
        assert net.encoder_type == "impala"
        assert net.hidden_dim == 2048
        assert net.num_actions == CRAZY_CLIMBER_ACTIONS
        assert net.num_atoms == 51
        assert net.noisy is False
        assert net.dueling is True


@pytest.mark.skipif(
    not _has_run(SPR_RUN), reason="SPR verification run not found"
)
class TestLoadSPRCheckpoint:
    """Load SPR checkpoint (DQN encoder, noisy=True)."""

    def test_loads_successfully(self):
        ckpt = load_checkpoint(SPR_RUN, STEP)
        assert isinstance(ckpt, CheckpointData)

    def test_architecture_params(self):
        ckpt = load_checkpoint(SPR_RUN, STEP)
        assert ckpt.encoder_type == "dqn"
        assert ckpt.hidden_dim == 512
        assert ckpt.num_actions == CRAZY_CLIMBER_ACTIONS

    def test_network_def_matches_params(self):
        ckpt = load_checkpoint(SPR_RUN, STEP)
        net = ckpt.network_def
        assert net.encoder_type == "dqn"
        assert net.hidden_dim == 512
        assert net.noisy is True
        assert net.dueling is True


@pytest.mark.skipif(
    not _has_run(DER_RUN), reason="DER verification run not found"
)
class TestLoadDERCheckpoint:
    """Load DER checkpoint (DQN encoder, noisy=True)."""

    def test_loads_successfully(self):
        ckpt = load_checkpoint(DER_RUN, STEP)
        assert isinstance(ckpt, CheckpointData)

    def test_architecture_params(self):
        ckpt = load_checkpoint(DER_RUN, STEP)
        assert ckpt.encoder_type == "dqn"
        assert ckpt.hidden_dim == 512
        assert ckpt.num_actions == CRAZY_CLIMBER_ACTIONS


# -- Error handling tests ----------------------------------------------------


class TestLoadCheckpointErrors:

    def test_missing_run_dir(self):
        with pytest.raises(FileNotFoundError):
            load_checkpoint("/nonexistent/path", 10000)

    def test_missing_step(self):
        if _has_run(BBF_RUN):
            with pytest.raises(FileNotFoundError):
                load_checkpoint(BBF_RUN, 99999)
        else:
            pytest.skip("BBF verification run not found")
