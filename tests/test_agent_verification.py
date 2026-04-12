"""Agent behavioral verification tests.

Verify key properties of BBF-family agents: SPR loss activation,
C51 validity, reset behavior, gradient flow, and shape compatibility.

Fast tests (no @pytest.mark.slow) check agent init-time properties
and run locally. Slow tests require training and JIT compilation;
they are written here but verified on GPU in a separate task.

Requires the full JAX ecosystem; marked with @pytest.mark.jax.
"""

import os
import sys

import numpy as np
import pytest

pytestmark = pytest.mark.jax

_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_dir))

jax = pytest.importorskip("jax", reason="JAX not installed")
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

import bigger_better_faster.bbf.eval_run_experiment  # noqa: F401, E402
from bigger_better_faster.bbf.agents.metric_agent import MetricBBFAgent  # noqa: E402

_CONFIGS_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "src", "bigger_better_faster", "bbf", "configs",
)

# Maps each condition to its base gin file.
_CONDITION_BASE = {
    "BBF": "BBF.gin", "BBFc": "BBF.gin",
    "DER": "SPR.gin", "DERc": "SPR.gin",
    "SPR": "SPR.gin", "SPRc": "SPR.gin",
    "SR_SPR": "SR_SPR.gin", "SR_SPRc": "SR_SPR.gin",
}


@pytest.fixture(autouse=True)
def _clean_gin():
    """Clear gin state before and after each test."""
    gin.clear_config()
    yield
    gin.clear_config()


def _create_agent(tmp_path, condition="BBF", num_actions=4,
                  extra_bindings=None):
    """Create a MetricBBFAgent with the given condition config."""
    base_gin = os.path.join(_CONFIGS_DIR, _CONDITION_BASE[condition])
    condition_gin = os.path.join(_CONFIGS_DIR, "conditions", f"{condition}.gin")

    bindings = [
        "Runner.training_steps = 100",
        "JaxDQNAgent.min_replay_history = 64",
        "BBFAgent.log_every = 1",
    ]
    if extra_bindings:
        bindings.extend(extra_bindings)

    gin.parse_config_files_and_bindings(
        [base_gin, condition_gin],
        bindings=bindings,
    )
    agent = MetricBBFAgent(
        num_actions=num_actions,
        seed=42,
        summary_writer=str(tmp_path),
    )
    agent.eval_mode = False
    return agent


def _fill_replay_and_train(agent, num_transitions=80, num_actions=4):
    """Add synthetic transitions and run one training step.

    Fills the replay buffer past min_replay_history, then calls
    step() which triggers _train_step and populates _last_metrics.
    """
    obs_shape = agent.observation_shape  # (84, 84)

    # Initial observation -- trailing dim 1 matches dopamine's (84, 84, 1)
    obs = np.random.randint(0, 255, (1, *obs_shape, 1), dtype=np.uint8)
    agent.reset_all(obs)

    for i in range(num_transitions):
        agent.step()

        obs = np.random.randint(0, 255, (1, *obs_shape, 1), dtype=np.uint8)
        action = np.array([np.random.randint(0, num_actions)])
        reward = np.array([np.random.choice([-1.0, 0.0, 1.0])])
        terminal = np.array([i == num_transitions - 1])
        episode_end = terminal.copy()

        agent.log_transition(obs, action, reward, terminal, episode_end)

    # One more step to trigger training with the full buffer
    agent.step()


# =========================================================================
# Slow tests (require training / JIT compilation)
# =========================================================================


@pytest.mark.slow
def test_bbf_spr_loss_nonzero(tmp_path):
    """BBF (spr_weight=5, jumps=5) produces nonzero SPR loss after training."""
    agent = _create_agent(tmp_path, condition="BBF")

    assert agent.spr_weight > 0, "BBF should have spr_weight > 0"
    assert agent._jumps > 0, "BBF should have jumps > 0"

    _fill_replay_and_train(agent)

    metrics = agent._last_metrics
    assert "SPRLoss" in metrics, "SPRLoss should be in _last_metrics"
    assert float(metrics["SPRLoss"]) > 0, (
        f"BBF SPR loss should be nonzero, got {metrics['SPRLoss']}"
    )


@pytest.mark.slow
def test_bbfc_spr_loss_zero(tmp_path):
    """BBFc (spr_weight=0, jumps=0) produces zero SPR loss after training."""
    agent = _create_agent(tmp_path, condition="BBFc")

    assert agent.spr_weight == 0, "BBFc should have spr_weight == 0"
    assert agent._jumps == 0, "BBFc should have jumps == 0"

    _fill_replay_and_train(agent)

    metrics = agent._last_metrics
    assert "SPRLoss" in metrics, "SPRLoss should be in _last_metrics"
    assert float(metrics["SPRLoss"]) == 0, (
        f"BBFc SPR loss should be zero, got {metrics['SPRLoss']}"
    )
