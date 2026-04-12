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
# Fast tests (init-time properties, no training needed)
# =========================================================================


def _forward_pass(agent):
    """Run a single forward pass through the network and return output."""
    obs = np.zeros((1, *agent.observation_shape, 1), dtype=np.uint8)
    agent.reset_all(obs)
    state = agent.state.astype(agent.dtype)
    rng = jax.random.PRNGKey(0)
    actions = jax.numpy.zeros((5,))
    output = agent.network_def.apply(
        agent.online_params,
        x=state,
        actions=actions,
        do_rollout=False,
        eval_mode=True,
        key=rng,
        support=agent._support,
    )
    return output


def test_impala_encoder_spatial_output(tmp_path):
    """BBF IMPALA encoder produces 11x11x128 spatial output from 84x84 input."""
    agent = _create_agent(tmp_path, condition="BBF")
    output = _forward_pass(agent)
    latent = np.asarray(output.latent)
    # spatial_latent shape: (height, width, channels) without batch
    assert latent.shape[-3:] == (11, 11, 128), (
        f"Expected IMPALA spatial output (11, 11, 128), got {latent.shape}"
    )


def test_c51_probabilities_valid(tmp_path):
    """C51 probabilities from the distributional head sum to 1 per action."""
    agent = _create_agent(tmp_path, condition="BBF")
    output = _forward_pass(agent)
    probs = np.asarray(output.probabilities)
    # probabilities shape: (num_actions, num_atoms)
    assert probs.shape[-1] == 51, (
        f"Expected 51 atoms, got {probs.shape[-1]}"
    )
    per_action_sums = probs.sum(axis=-1)
    np.testing.assert_allclose(per_action_sums, 1.0, atol=1e-5, err_msg=(
        f"C51 probabilities should sum to 1, got sums {per_action_sums}"
    ))
    assert (probs >= 0).all(), "Probabilities should be non-negative"


def test_optimizer_mask_keys(tmp_path):
    """Optimizer masks partition all param keys into encoder and head groups."""
    agent = _create_agent(tmp_path, condition="BBF")

    param_keys = set(agent.online_params["params"].keys())
    encoder_true = {k for k, v in agent.encoder_mask["params"].items() if v}
    head_true = {k for k, v in agent.head_mask["params"].items() if v}

    # Encoder group: encoder + transition_model
    assert encoder_true == {"encoder", "transition_model"}, (
        f"Encoder mask should select encoder and transition_model, got {encoder_true}"
    )
    # Head group: projection + predictor + head
    assert head_true == {"projection", "predictor", "head"}, (
        f"Head mask should select projection, predictor, head, got {head_true}"
    )
    # Masks are exhaustive and non-overlapping
    assert encoder_true | head_true == param_keys, (
        f"Masks should cover all param keys. Missing: {param_keys - (encoder_true | head_true)}"
    )
    assert encoder_true & head_true == set(), "Masks should not overlap"


def test_gradient_norm_keys(tmp_path):
    """Online params have the expected top-level keys for per-layer gradient norms."""
    agent = _create_agent(tmp_path, condition="BBF")

    param_keys = set(agent.online_params["params"].keys())
    expected = {"encoder", "transition_model", "projection", "predictor", "head"}
    assert param_keys == expected, (
        f"Expected param keys {expected}, got {param_keys}"
    )


def test_spr_nature_cnn_spatial_output(tmp_path):
    """SPR Nature CNN encoder produces 11x11x64 spatial output from 84x84 input."""
    agent = _create_agent(tmp_path, condition="SPR")
    output = _forward_pass(agent)
    latent = np.asarray(output.latent)
    assert latent.shape[-3:] == (11, 11, 64), (
        f"Expected Nature CNN spatial output (11, 11, 64), got {latent.shape}"
    )


def test_spr_no_resets(tmp_path):
    """SPR has no periodic resets (reset_every defaults to -1)."""
    agent = _create_agent(tmp_path, condition="SPR")
    assert agent.reset_every == -1, (
        f"SPR reset_every should be -1, got {agent.reset_every}"
    )


def test_spr_fixed_hyperparameters(tmp_path):
    """SPR uses fixed gamma=0.99 and n_step=10 with no scheduling."""
    agent = _create_agent(tmp_path, condition="SPR")
    # Schedulers should return constants regardless of cycle step
    assert float(agent.gamma_scheduler(0)) == 0.99
    assert float(agent.gamma_scheduler(5000)) == 0.99
    assert int(agent.update_horizon_scheduler(0)) == 10
    assert int(agent.update_horizon_scheduler(5000)) == 10


def test_spr_noisy_nets(tmp_path):
    """SPR uses noisy nets for exploration with epsilon forced to zero."""
    agent = _create_agent(tmp_path, condition="SPR")
    assert agent._noisy is True, "SPR should have noisy=True"
    assert agent.epsilon_train == 0, (
        f"SPR epsilon_train should be 0, got {agent.epsilon_train}"
    )


def test_sr_spr_shrink_perturb_factors(tmp_path):
    """SR-SPR uses shrink_factor=0.8 and perturb_factor=0.2."""
    agent = _create_agent(tmp_path, condition="SR_SPR")
    assert agent.shrink_factor == 0.8, (
        f"SR-SPR shrink_factor should be 0.8, got {agent.shrink_factor}"
    )
    assert agent.perturb_factor == 0.2, (
        f"SR-SPR perturb_factor should be 0.2, got {agent.perturb_factor}"
    )


def test_sr_spr_target_action_selection(tmp_path):
    """SR-SPR selects actions using the target network during training."""
    agent = _create_agent(tmp_path, condition="SR_SPR")
    assert agent.target_action_selection is True, (
        "SR-SPR should have target_action_selection=True"
    )


def test_der_config_structure(tmp_path):
    """DER differs from DERc by exactly spr_weight and jumps."""
    agent_der = _create_agent(tmp_path, condition="DER")
    gin.clear_config()
    agent_derc = _create_agent(tmp_path, condition="DERc")

    # DER has SPR system active
    assert agent_der.spr_weight == 5
    assert agent_der._jumps == 5
    # DERc has SPR system off
    assert agent_derc.spr_weight == 0
    assert agent_derc._jumps == 0
    # Both share the same base parameters
    assert agent_der.target_update_period == agent_derc.target_update_period == 8000
    assert agent_der.learning_rate == agent_derc.learning_rate


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


@pytest.mark.slow
def test_reset_fires_at_interval(tmp_path):
    """BBF reset triggers after training_steps exceeds next_reset.

    Uses a small reset_every=5 and reset_offset=0 so the first reset
    fires within a short training run. Verifies MetricBBFAgent._reset_log
    captures the event.
    """
    agent = _create_agent(
        tmp_path, condition="BBF",
        extra_bindings=[
            "BBFAgent.reset_every = 5",
            "BBFAgent.reset_offset = 0",
        ],
    )

    assert agent.reset_every == 5, "reset_every should be overridden to 5"
    assert agent.next_reset == 5, "next_reset should be reset_every + offset = 5"

    _fill_replay_and_train(agent)

    assert len(agent._reset_log) >= 1, (
        f"Expected at least one reset, got {len(agent._reset_log)}"
    )
    first_reset = agent._reset_log[0]
    assert "training_step" in first_reset
    assert "cumulative_resets" in first_reset
    assert first_reset["cumulative_resets"] == 1


@pytest.mark.slow
def test_loss_shape_compatibility(tmp_path):
    """Uniform replay with spr_weight>0 does not crash from shape mismatch.

    Before the fix, loss_weights had shape (batch, subseq_len) from
    state.shape[0:2] which could not broadcast against the (batch,)
    loss vector. The fix uses state.shape[0:1]. This test forces
    uniform replay on BBF to exercise the fixed code path.
    """
    agent = _create_agent(
        tmp_path, condition="BBF",
        extra_bindings=[
            'BBFAgent.replay_scheme = "uniform"',
        ],
    )

    assert agent.spr_weight > 0, "BBF should have spr_weight > 0"
    assert agent._replay_scheme == "uniform"

    # If the shape fix is broken, this will raise ValueError
    _fill_replay_and_train(agent)

    metrics = agent._last_metrics
    assert "DQNLoss" in metrics, "DQNLoss should be in _last_metrics"
    assert np.isfinite(float(metrics["DQNLoss"])), (
        f"DQNLoss should be finite, got {metrics['DQNLoss']}"
    )


@pytest.mark.slow
def test_spr_target_hard_copy(tmp_path):
    """SPR target network is an exact copy of online, giving zero divergence.

    SPR uses target_update_period=1 and target_update_tau=1.0, so
    the target update is target = 1.0*online + 0.0*target = online
    every step. TargetDivergence should be 0.0 after training.
    """
    agent = _create_agent(tmp_path, condition="SPR")

    _fill_replay_and_train(agent)

    metrics = agent._last_metrics
    assert "TargetDivergence" in metrics
    assert float(metrics["TargetDivergence"]) == 0.0, (
        f"SPR TargetDivergence should be 0.0, got {metrics['TargetDivergence']}"
    )


@pytest.mark.slow
def test_sr_spr_ema_target_divergence(tmp_path):
    """SR-SPR EMA target update produces nonzero TargetDivergence.

    SR-SPR uses tau=0.005 with period=1, so the target network lags
    behind online via slow EMA. After training, the param difference
    should be nonzero.
    """
    agent = _create_agent(tmp_path, condition="SR_SPR")

    _fill_replay_and_train(agent)

    metrics = agent._last_metrics
    assert "TargetDivergence" in metrics
    assert float(metrics["TargetDivergence"]) > 0, (
        "SR-SPR TargetDivergence should be nonzero (EMA lag), "
        f"got {metrics['TargetDivergence']}"
    )


@pytest.mark.slow
def test_der_target_copy_sawtooth(tmp_path):
    """DER hard-copies target every 8000 steps, producing nonzero divergence between copies.

    With target_update_period=8000 and tau=1.0, the target is a stale
    snapshot. After a few training steps (well under 8000), online
    params have diverged, so TargetDivergence should be nonzero.
    """
    agent = _create_agent(
        tmp_path, condition="DER",
        extra_bindings=[
            # Override replay_ratio so update_period=1 (otherwise
            # DER's replay_ratio=1 gives update_period=32, and
            # training never fires within 80 transitions).
            "BBFAgent.replay_ratio = 64",
        ],
    )

    assert agent.target_update_period == 8000

    _fill_replay_and_train(agent)

    metrics = agent._last_metrics
    assert "TargetDivergence" in metrics
    assert float(metrics["TargetDivergence"]) > 0, (
        "DER TargetDivergence should be nonzero between hard copies, "
        f"got {metrics['TargetDivergence']}"
    )
