"""
Verify that all condition-game gin config combinations parse without
errors and resolve to the expected parameter values.

8 conditions x 6 games = 48 combinations. Each test loads:
  base config -> condition override -> game binding
and checks that key parameters match the per-condition spec.
"""

import os
import sys

import pytest

pytestmark = pytest.mark.jax

_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_dir))

pytest.importorskip("jax", reason="JAX not installed")
pytest.importorskip("dopamine", reason="dopamine-rl not installed")
flax = pytest.importorskip("flax", reason="flax not installed")
flax.config.update("flax_return_frozendict", True)
pytest.importorskip("optax", reason="optax not installed")
gin = pytest.importorskip("gin", reason="gin-config not installed")
pytest.importorskip("tensorflow", reason="tensorflow not installed")

from unittest.mock import MagicMock

if "baselines" not in sys.modules:
    sys.modules["baselines"] = MagicMock()
    sys.modules["baselines.common"] = MagicMock()
    sys.modules["baselines.common.atari_wrappers"] = MagicMock()

import bigger_better_faster.bbf.eval_run_experiment  # noqa: E402,F401
import bigger_better_faster.bbf.agents.metric_agent  # noqa: E402,F401

_CFG = os.path.join(
    os.path.dirname(__file__),
    "..", "src", "bigger_better_faster", "bbf", "configs",
)

# Map condition name -> (base gin, condition overlay)
CONDITIONS = {
    "BBF":     ("BBF.gin",    "conditions/BBF.gin"),
    "BBFc":    ("BBF.gin",    "conditions/BBFc.gin"),
    "SR-SPR":  ("SR_SPR.gin", "conditions/SR_SPR.gin"),
    "SR-SPRc": ("SR_SPR.gin", "conditions/SR_SPRc.gin"),
    "SPR":     ("SPR.gin",    "conditions/SPR.gin"),
    "SPRc":    ("SPR.gin",    "conditions/SPRc.gin"),
    "DER":     ("SPR.gin",    "conditions/DER.gin"),
    "DERc":    ("SPR.gin",    "conditions/DERc.gin"),
}

GAMES = {
    "boxing":        "Boxing",
    "crazy_climber": "CrazyClimber",
    "frostbite":     "Frostbite",
    "kangaroo":      "Kangaroo",
    "road_runner":   "RoadRunner",
    "up_n_down":     "UpNDown",
}

# Expected resolved values per condition.
EXPECTED = {
    "BBF":     {"spr_weight": 5, "jumps": 5, "log_every": 1, "data_augmentation": False, "reset_every": 20_000},
    "BBFc":    {"spr_weight": 0, "jumps": 0, "log_every": 1, "data_augmentation": False, "reset_every": 20_000},
    "SR-SPR":  {"spr_weight": 5, "jumps": 5, "log_every": 1, "data_augmentation": False, "reset_every": 20_000, "replay_ratio": 64},
    "SR-SPRc": {"spr_weight": 0, "jumps": 0, "log_every": 1, "data_augmentation": False, "reset_every": 20_000, "replay_ratio": 64},
    "SPR":     {"spr_weight": 5, "jumps": 5, "log_every": 1, "data_augmentation": False, "noisy": True},
    "SPRc":    {"spr_weight": 0, "jumps": 0, "log_every": 1, "data_augmentation": False, "noisy": True},
    "DER":     {"spr_weight": 5, "jumps": 5, "log_every": 1, "data_augmentation": False, "replay_ratio": 1, "update_horizon": 20, "target_update_period": 8000, "target_update_tau": 1.0, "JaxDQNAgent.min_replay_history": 1600},
    "DERc":    {"spr_weight": 0, "jumps": 0, "log_every": 1, "data_augmentation": False, "replay_ratio": 1, "update_horizon": 20, "target_update_period": 8000, "target_update_tau": 1.0, "JaxDQNAgent.min_replay_history": 1600},
}


def _load_config(condition_name, game_file):
    """Parse base + condition + game gin files and return None on success."""
    base, overlay = CONDITIONS[condition_name]
    gin.clear_config()
    gin.parse_config_files_and_bindings(
        [
            os.path.join(_CFG, base),
            os.path.join(_CFG, overlay),
            os.path.join(_CFG, "games", game_file),
        ],
        [],
    )


@pytest.fixture(autouse=True)
def _clean_gin():
    gin.clear_config()
    yield
    gin.clear_config()


class TestGinConfigCombinations:
    """All 48 condition-game combinations parse without errors."""

    @pytest.mark.parametrize("condition", CONDITIONS.keys())
    @pytest.mark.parametrize("game_file,game_name", GAMES.items())
    def test_parses(self, condition, game_file, game_name):
        _load_config(condition, f"{game_file}.gin")
        resolved = gin.query_parameter("DataEfficientAtariRunner.game_name")
        assert resolved == game_name


class TestConditionParameterValues:
    """Resolved values match per-condition specs."""

    @pytest.mark.parametrize("condition", EXPECTED.keys())
    def test_expected_values(self, condition):
        _load_config(condition, "boxing.gin")
        for param, expected in EXPECTED[condition].items():
            if "." in param:
                query = param
            else:
                query = f"BBFAgent.{param}"
            resolved = gin.query_parameter(query)
            assert resolved == expected, (
                f"{condition}: {param} expected {expected}, got {resolved}"
            )
