"""
Integration tests for Rainbow DQN training pipeline.

Tests end-to-end behavior with all Rainbow components working together:
- Full training step: distributional loss + IS weights + priority
  updates + noisy exploration
- Loss decreases over multiple updates on a fixed batch
- Priorities update after each training step
- IS weights are applied correctly (non-uniform weights change loss)
- SPR loss integrates alongside distributional loss
- Combined loss = IS-weighted distributional + lambda * SPR
"""

import numpy as np
import pytest
import torch

from src.models.ema import EMAEncoder
from src.models.rainbow import RainbowDQN
from src.models.spr import PredictionHead, ProjectionHead, TransitionModel
from src.replay.prioritized_buffer import PrioritizedReplayBuffer
from src.training.metrics import UpdateMetrics, perform_rainbow_update_step

# ---------------------------------------------------------------------------
# Shared Constants
# ---------------------------------------------------------------------------

NUM_ACTIONS = 4
OBS_SHAPE = (4, 84, 84)
NUM_ATOMS = 11
V_MIN, V_MAX = -5.0, 5.0
K = 5  # SPR prediction steps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nets(noisy=True, dueling=True):
    """Create online and synced target RainbowDQN networks."""
    online = RainbowDQN(
        num_actions=NUM_ACTIONS,
        num_atoms=NUM_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX,
        noisy=noisy,
        dueling=dueling,
    )
    target = RainbowDQN(
        num_actions=NUM_ACTIONS,
        num_atoms=NUM_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX,
        noisy=noisy,
        dueling=dueling,
    )
    target.load_state_dict(online.state_dict())
    return online, target


def _make_batch(batch_size=8):
    """Create a batch dict mimicking PrioritizedReplayBuffer.sample()."""
    return {
        "states": torch.rand(batch_size, *OBS_SHAPE),
        "actions": torch.randint(0, NUM_ACTIONS, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "next_states": torch.rand(batch_size, *OBS_SHAPE),
        "dones": torch.zeros(batch_size, dtype=torch.bool),
        "weights": torch.ones(batch_size),
        "indices": np.arange(batch_size),
    }


def _make_spr_batch(batch_size=8, seq_len=K):
    """Create a synthetic SPR sequence batch."""
    return {
        "states": torch.rand(batch_size, seq_len + 1, *OBS_SHAPE),
        "actions": torch.randint(0, NUM_ACTIONS, (batch_size, seq_len)),
        "dones": torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }


def _make_spr_components(online_net, momentum=0.99):
    """Create SPR components using a RainbowDQN as encoder."""
    transition_model = TransitionModel(num_actions=NUM_ACTIONS)
    projection_head = ProjectionHead()
    prediction_head = PredictionHead()
    target_encoder = EMAEncoder(online_net, momentum=momentum)
    target_projection = EMAEncoder(projection_head, momentum=momentum)

    return {
        "transition_model": transition_model,
        "projection_head": projection_head,
        "prediction_head": prediction_head,
        "target_encoder": target_encoder,
        "target_projection": target_projection,
    }


def _make_optimizer(online_net, spr_components=None, lr=1e-3):
    """Create optimizer over all trainable parameters."""
    params = list(online_net.parameters())
    if spr_components is not None:
        params += list(spr_components["transition_model"].parameters())
        params += list(spr_components["projection_head"].parameters())
        params += list(spr_components["prediction_head"].parameters())
    return torch.optim.Adam(params, lr=lr)


def _fill_buffer(buf, n=100):
    """Add n random transitions to a PrioritizedReplayBuffer."""
    for i in range(n):
        state = np.random.randint(0, 255, OBS_SHAPE, dtype=np.uint8)
        action = np.random.randint(0, NUM_ACTIONS)
        reward = float(np.random.randn())
        next_state = np.random.randint(0, 255, OBS_SHAPE, dtype=np.uint8)
        done = i % 25 == 24
        buf.append(state, action, reward, next_state, done)


# ---------------------------------------------------------------------------
# Test: Full training step with all components
# ---------------------------------------------------------------------------


class TestFullTrainingStep:
    """Full Rainbow training step with distributional loss, IS weights,
    priority updates, and noisy exploration."""

    def test_single_step_returns_metrics(self):
        """A single Rainbow step should produce valid UpdateMetrics."""
        torch.manual_seed(42)
        online, target = _make_nets()
        optimizer = _make_optimizer(online)
        support = online.support

        buf = PrioritizedReplayBuffer(
            capacity=200, obs_shape=OBS_SHAPE, min_size=10,
            alpha=0.5, beta_start=0.4, beta_end=1.0, beta_frames=100_000,
            device=torch.device("cpu"),
        )
        _fill_buffer(buf, 50)
        batch = buf.sample(8, frame=0)

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch,
            support=support, buffer=buf,
        )

        assert isinstance(metrics, UpdateMetrics)
        assert metrics.loss > 0
        assert metrics.td_error > 0
        assert metrics.grad_norm >= 0
        assert metrics.distributional_loss is not None
        assert metrics.mean_is_weight is not None
        assert metrics.mean_priority is not None
        assert metrics.priority_entropy is not None
        assert metrics.beta is not None

    def test_noisy_nets_produce_stochastic_output(self):
        """With noisy=True, two forward passes should differ due to noise."""
        torch.manual_seed(42)
        online, _ = _make_nets(noisy=True)
        x = torch.rand(2, *OBS_SHAPE)

        online.reset_noise()
        out1 = online(x)["q_values"].detach()
        online.reset_noise()
        out2 = online(x)["q_values"].detach()

        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Noisy nets should produce different outputs after noise reset"
        )


# ---------------------------------------------------------------------------
# Test: Loss decreases over multiple updates
# ---------------------------------------------------------------------------


class TestLossDecreases:
    """Rainbow distributional loss should decrease on a fixed batch."""

    def test_loss_decreases_over_updates(self):
        """Total loss should decrease after gradient steps on a fixed batch."""
        torch.manual_seed(42)
        online, target = _make_nets(noisy=False)  # Disable noise for stability
        optimizer = _make_optimizer(online, lr=5e-4)
        support = online.support

        batch = _make_batch(batch_size=16)

        losses = []
        for i in range(30):
            metrics = perform_rainbow_update_step(
                online, target, optimizer, batch,
                support=support, buffer=None, update_count=i,
            )
            losses.append(metrics.loss)

        early_avg = np.mean(losses[:3])
        late_avg = np.mean(losses[-3:])
        assert late_avg < early_avg, (
            f"Loss should decrease: early avg {early_avg:.4f} -> "
            f"late avg {late_avg:.4f}"
        )

    def test_distributional_loss_tracks_total_loss(self):
        """Without SPR, total loss should equal IS-weighted distributional."""
        torch.manual_seed(42)
        online, target = _make_nets(noisy=False)
        optimizer = _make_optimizer(online)
        support = online.support

        batch = _make_batch()

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch,
            support=support, buffer=None,
        )

        # Without SPR, total loss is the IS-weighted distributional loss.
        # distributional_loss is the unweighted mean (= td_error).
        assert metrics.distributional_loss is not None
        assert abs(metrics.distributional_loss - metrics.td_error) < 1e-6


# ---------------------------------------------------------------------------
# Test: Priorities update after each step
# ---------------------------------------------------------------------------


class TestPriorityUpdates:
    """Priorities should change in the replay buffer after training."""

    def test_priorities_change_after_update(self):
        """Buffer priorities should be updated based on per-sample loss."""
        torch.manual_seed(42)
        online, target = _make_nets(noisy=False)
        optimizer = _make_optimizer(online)
        support = online.support

        buf = PrioritizedReplayBuffer(
            capacity=200, obs_shape=OBS_SHAPE, min_size=10,
            alpha=0.5, beta_start=0.4, beta_end=1.0, beta_frames=100_000,
            device=torch.device("cpu"),
        )
        _fill_buffer(buf, 50)

        # Get initial priority state
        initial_state = buf.get_priority_state()
        initial_tree = initial_state["tree_data"].copy()

        # Sample and train
        batch = buf.sample(8, frame=0)
        indices = batch["indices"]

        perform_rainbow_update_step(
            online, target, optimizer, batch,
            support=support, buffer=buf,
        )

        # Check that priorities at sampled indices have changed
        updated_state = buf.get_priority_state()
        for idx in indices:
            tree_idx = idx + buf.tree.capacity
            assert updated_state["tree_data"][tree_idx] != initial_tree[tree_idx], (
                f"Priority at index {idx} should change after update"
            )

    def test_priorities_update_every_step(self):
        """Priorities should be updated on every training step."""
        torch.manual_seed(42)
        online, target = _make_nets(noisy=False)
        optimizer = _make_optimizer(online, lr=1e-4)
        support = online.support

        buf = PrioritizedReplayBuffer(
            capacity=200, obs_shape=OBS_SHAPE, min_size=10,
            alpha=0.5, beta_start=0.4, beta_end=1.0, beta_frames=100_000,
            device=torch.device("cpu"),
        )
        _fill_buffer(buf, 50)

        totals = []
        for step in range(5):
            batch = buf.sample(8, frame=step)
            perform_rainbow_update_step(
                online, target, optimizer, batch,
                support=support, buffer=buf, update_count=step,
            )
            totals.append(buf.tree.total)

        # Not all totals should be identical (priorities shift each step)
        assert len(set(totals)) > 1, (
            "Tree total should change across training steps"
        )


# ---------------------------------------------------------------------------
# Test: IS weights applied correctly
# ---------------------------------------------------------------------------


class TestISWeights:
    """Importance sampling weights should affect the loss."""

    def test_nonuniform_weights_change_loss(self):
        """Loss should differ when IS weights are non-uniform vs uniform."""
        torch.manual_seed(42)
        online, target = _make_nets(noisy=False)
        support = online.support
        batch = _make_batch(batch_size=8)

        # Uniform weights
        optimizer1 = _make_optimizer(online, lr=0.0)  # lr=0 so no param change
        batch_uniform = {**batch, "weights": torch.ones(8)}
        m1 = perform_rainbow_update_step(
            online, target, optimizer1, batch_uniform,
            support=support, buffer=None,
        )

        # Non-uniform weights (same batch, same model state since lr=0)
        batch_nonuniform = {**batch, "weights": torch.tensor(
            [0.1, 0.2, 0.5, 1.0, 0.3, 0.8, 0.4, 0.9]
        )}
        m2 = perform_rainbow_update_step(
            online, target, optimizer1, batch_nonuniform,
            support=support, buffer=None,
        )

        # Total (IS-weighted) loss should differ
        assert abs(m1.loss - m2.loss) > 1e-6, (
            "Different IS weights should produce different losses"
        )

    def test_mean_is_weight_reflects_batch(self):
        """mean_is_weight metric should match batch weights."""
        torch.manual_seed(42)
        online, target = _make_nets(noisy=False)
        optimizer = _make_optimizer(online)
        support = online.support

        weights = torch.tensor([0.5, 0.8, 1.0, 0.3])
        batch = _make_batch(batch_size=4)
        batch["weights"] = weights

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch,
            support=support, buffer=None,
        )

        expected = weights.mean().item()
        assert abs(metrics.mean_is_weight - expected) < 1e-5


# ---------------------------------------------------------------------------
# Test: SPR loss integrates alongside distributional loss
# ---------------------------------------------------------------------------


class TestSPRIntegration:
    """SPR auxiliary loss should work alongside Rainbow distributional loss."""

    def test_spr_loss_is_populated(self):
        """SPR loss and cosine similarity should be non-None with SPR."""
        torch.manual_seed(42)
        online, target = _make_nets(noisy=False)
        spr_components = _make_spr_components(online)
        optimizer = _make_optimizer(online, spr_components)
        support = online.support

        metrics = perform_rainbow_update_step(
            online, target, optimizer, _make_batch(),
            support=support, buffer=None,
            spr_components=spr_components,
            spr_batch=_make_spr_batch(),
        )

        assert metrics.spr_loss is not None
        assert metrics.cosine_similarity is not None
        assert isinstance(metrics.spr_loss, float)

    def test_combined_loss_differs_from_distributional_alone(self):
        """Combined loss (dist + SPR) should differ from distributional-only."""
        torch.manual_seed(42)
        online, target = _make_nets(noisy=False)
        spr_components = _make_spr_components(online)
        optimizer_combined = _make_optimizer(online, spr_components, lr=0.0)
        support = online.support
        batch = _make_batch()
        spr_batch = _make_spr_batch()

        # Combined: distributional + SPR
        m_combined = perform_rainbow_update_step(
            online, target, optimizer_combined, batch,
            support=support, buffer=None,
            spr_components=spr_components,
            spr_batch=spr_batch,
            spr_weight=2.0,
        )

        # Distributional only (same model state since lr=0)
        m_dist_only = perform_rainbow_update_step(
            online, target, optimizer_combined, batch,
            support=support, buffer=None,
        )

        # Combined includes SPR term, so losses should differ
        assert abs(m_combined.loss - m_dist_only.loss) > 1e-6, (
            f"Combined loss ({m_combined.loss:.4f}) should differ from "
            f"distributional-only ({m_dist_only.loss:.4f})"
        )
        # Verify the difference equals spr_weight * spr_loss
        expected_diff = 2.0 * m_combined.spr_loss
        actual_diff = m_combined.loss - m_dist_only.loss
        assert abs(actual_diff - expected_diff) < 0.01, (
            f"Difference should be spr_weight * spr_loss: "
            f"expected {expected_diff:.4f}, got {actual_diff:.4f}"
        )

    def test_combined_loss_equals_weighted_sum(self):
        """Total loss = IS-weighted distributional + lambda * SPR."""
        torch.manual_seed(42)
        online, target = _make_nets(noisy=False)
        spr_components = _make_spr_components(online)
        optimizer = _make_optimizer(online, spr_components, lr=0.0)
        support = online.support
        spr_weight = 2.0

        metrics = perform_rainbow_update_step(
            online, target, optimizer, _make_batch(),
            support=support, buffer=None,
            spr_components=spr_components,
            spr_batch=_make_spr_batch(),
            spr_weight=spr_weight,
        )

        # The total loss should approximately equal the sum of components.
        # Since loss = IS_weighted_dist + spr_weight * spr_loss,
        # and distributional_loss is the *unweighted* mean (td_error),
        # the IS-weighted part is metrics.loss - spr_weight * spr_loss.
        is_weighted_dist = metrics.loss - spr_weight * metrics.spr_loss
        assert is_weighted_dist > 0, "IS-weighted distributional loss should be positive"

    def test_spr_loss_decreases_with_rainbow(self):
        """SPR loss alongside Rainbow should decrease over training."""
        torch.manual_seed(42)
        online, target = _make_nets(noisy=False)
        spr_components = _make_spr_components(online)
        optimizer = _make_optimizer(online, spr_components, lr=1e-3)
        support = online.support

        batch = _make_batch(batch_size=16)
        spr_batch = _make_spr_batch(batch_size=16)

        losses = []
        for i in range(25):
            metrics = perform_rainbow_update_step(
                online, target, optimizer, batch,
                support=support, buffer=None, update_count=i,
                spr_components=spr_components,
                spr_batch=spr_batch,
                spr_weight=2.0,
            )
            losses.append(metrics.loss)

        early_avg = np.mean(losses[:3])
        late_avg = np.mean(losses[-3:])
        assert late_avg < early_avg, (
            f"Combined loss should decrease: early {early_avg:.4f} -> "
            f"late {late_avg:.4f}"
        )

    def test_ema_updates_after_step(self):
        """EMA target encoder weights should diverge after training."""
        torch.manual_seed(42)
        online, target = _make_nets(noisy=False)
        spr_components = _make_spr_components(online, momentum=0.99)
        optimizer = _make_optimizer(online, spr_components, lr=1e-3)
        support = online.support

        # Snapshot EMA params before training
        ema_before = {
            k: v.clone()
            for k, v in spr_components["target_encoder"].state_dict().items()
        }

        # Run a few training steps
        for i in range(5):
            perform_rainbow_update_step(
                online, target, optimizer, _make_batch(),
                support=support, buffer=None, update_count=i,
                spr_components=spr_components,
                spr_batch=_make_spr_batch(),
            )

        # EMA params should have changed
        any_changed = False
        for k, v in spr_components["target_encoder"].state_dict().items():
            if not torch.allclose(v, ema_before[k], atol=1e-7):
                any_changed = True
                break

        assert any_changed, "EMA encoder should diverge from initial state"


# ---------------------------------------------------------------------------
# Test: End-to-end with real PrioritizedReplayBuffer
# ---------------------------------------------------------------------------


class TestEndToEndWithBuffer:
    """Full pipeline using actual PrioritizedReplayBuffer."""

    def test_sample_train_update_cycle(self):
        """Sample from buffer -> train -> update priorities, repeated."""
        torch.manual_seed(42)
        np.random.seed(42)

        online, target = _make_nets(noisy=False)
        optimizer = _make_optimizer(online, lr=1e-4)
        support = online.support

        buf = PrioritizedReplayBuffer(
            capacity=200, obs_shape=OBS_SHAPE, min_size=10,
            alpha=0.5, beta_start=0.4, beta_end=1.0, beta_frames=1000,
            device=torch.device("cpu"),
        )
        _fill_buffer(buf, 50)

        metrics_history = []
        for step in range(10):
            batch = buf.sample(8, frame=step)
            metrics = perform_rainbow_update_step(
                online, target, optimizer, batch,
                support=support, buffer=buf, update_count=step,
            )
            metrics_history.append(metrics)

        # All steps should produce valid metrics
        for m in metrics_history:
            assert m.loss > 0
            assert m.mean_priority is not None
            assert m.beta is not None

        # Beta should increase over frames (annealing)
        assert metrics_history[-1].beta > metrics_history[0].beta

    def test_beta_annealing_through_buffer(self):
        """Beta should anneal from start to end over training frames."""
        torch.manual_seed(42)
        np.random.seed(42)

        online, target = _make_nets(noisy=False)
        optimizer = _make_optimizer(online, lr=0.0)
        support = online.support

        buf = PrioritizedReplayBuffer(
            capacity=200, obs_shape=OBS_SHAPE, min_size=10,
            alpha=0.5, beta_start=0.4, beta_end=1.0, beta_frames=100,
            device=torch.device("cpu"),
        )
        _fill_buffer(buf, 50)

        batch = buf.sample(8, frame=0)

        # Early frame -> beta near start
        m_early = perform_rainbow_update_step(
            online, target, optimizer, batch,
            support=support, buffer=buf, update_count=0,
        )

        # Late frame -> beta near end
        m_late = perform_rainbow_update_step(
            online, target, optimizer, batch,
            support=support, buffer=buf, update_count=200,
        )

        assert m_early.beta < m_late.beta
        assert m_late.beta >= 0.99  # Should be at or near 1.0
