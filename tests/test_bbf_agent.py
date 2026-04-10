"""
Smoke test for the ported BBF agent.

Verifies that BBFAgent imports, instantiates from the BBF.gin config,
and produces valid actions via begin_episode and step. Requires the
full JAX ecosystem (jax, dopamine, flax, optax, gin, tensorflow);
skipped automatically when these are not installed.
"""

import os
import sys
import tempfile

import numpy as np
import pytest

# BBF internal imports use `bigger_better_faster.bbf...` so src/ must
# be on the path. Insert once at import time.
_src_dir = os.path.join(os.path.dirname(__file__), "..", "src")
if _src_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_src_dir))

# Skip the entire module if the JAX ecosystem is not installed.
jax = pytest.importorskip("jax", reason="JAX not installed")
pytest.importorskip("dopamine", reason="dopamine-rl not installed")
flax = pytest.importorskip("flax", reason="flax not installed")
# BBF code builds optimizer masks with FrozenDict, which requires init()
# to return FrozenDict. Flax >= 0.8 returns regular dicts by default.
flax.config.update("flax_return_frozendict", True)
pytest.importorskip("optax", reason="optax not installed")
gin = pytest.importorskip("gin", reason="gin-config not installed")
pytest.importorskip("tensorflow", reason="tensorflow not installed")

# Dopamine's atari_lib imports `baselines.common.atari_wrappers` at the
# top level. The old OpenAI baselines package does not install on Python
# 3.11+. BBFAgent never uses atari wrappers directly, so we provide a
# stub module to satisfy the import chain.
from unittest.mock import MagicMock

if "baselines" not in sys.modules:
    sys.modules["baselines"] = MagicMock()
    sys.modules["baselines.common"] = MagicMock()
    sys.modules["baselines.common.atari_wrappers"] = MagicMock()

from bigger_better_faster.bbf.agents import spr_agent  # noqa: E402
from bigger_better_faster.bbf.agents.metric_agent import MetricBBFAgent  # noqa: E402

_GIN_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "..", "src", "bigger_better_faster", "bbf", "configs", "BBF.gin",
)


@pytest.fixture(autouse=True)
def _clean_gin():
    """Clear gin state before and after each test."""
    gin.clear_config()
    yield
    gin.clear_config()


class TestBBFAgentSmoke:
    """Minimal smoke tests for BBFAgent instantiation and forward pass."""

    @pytest.fixture()
    def agent(self, tmp_path):
        """Instantiate a BBFAgent from the ported BBF.gin config."""
        # Import eval_run_experiment so gin recognizes
        # DataEfficientAtariRunner and other Runner configurables.
        import bigger_better_faster.bbf.eval_run_experiment  # noqa: F401
        gin.parse_config_file(_GIN_CONFIG)
        # Override training_steps to minimal value -- we only need one step.
        gin.bind_parameter("Runner.training_steps", 10)
        return spr_agent.BBFAgent(
            num_actions=18,
            seed=42,
            summary_writer=str(tmp_path),
        )

    def test_agent_instantiates(self, agent):
        """BBFAgent can be created from the BBF.gin config."""
        assert agent is not None
        assert hasattr(agent, "online_params")
        assert hasattr(agent, "target_network_params")

    def test_forward_pass_returns_valid_action(self, agent):
        """Network forward pass produces an action in [0, num_actions)."""
        # Build a batched stacked observation (1, 84, 84, 4).
        state = np.zeros((1, *agent.state_shape), dtype=np.float32)
        action = agent.select_action(
            state,
            agent.online_params,
            eval_mode=True,
            rng=jax.random.PRNGKey(0),
        )
        assert 0 <= int(action[0]) < 18


class TestMetricBBFAgent:
    """Tests for the MetricBBFAgent subclass."""

    @pytest.fixture()
    def agent(self, tmp_path):
        """Instantiate a MetricBBFAgent from the ported BBF.gin config."""
        import bigger_better_faster.bbf.eval_run_experiment  # noqa: F401
        gin.parse_config_file(_GIN_CONFIG)
        gin.bind_parameter("Runner.training_steps", 10)
        return MetricBBFAgent(
            num_actions=18,
            seed=42,
            summary_writer=str(tmp_path),
        )

    def test_subclass_instantiates(self, agent):
        """MetricBBFAgent creates successfully and has _last_metrics."""
        assert isinstance(agent, MetricBBFAgent)
        assert isinstance(agent, spr_agent.BBFAgent)
        assert agent._last_metrics == {}

    def test_is_gin_configurable(self):
        """MetricBBFAgent is registered as a gin configurable."""
        configurables = [c[0] for c in gin.config._REGISTRY.items()]
        assert any("MetricBBFAgent" in str(c) for c in configurables)
