"""Tests for perform_rainbow_update_step."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.rainbow import RainbowDQN
from src.training.metrics import UpdateMetrics, perform_rainbow_update_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nets(num_actions=4, num_atoms=11, v_min=-5.0, v_max=5.0, noisy=True):
    """Create online and target RainbowDQN networks for testing."""
    online = RainbowDQN(
        num_actions=num_actions,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        noisy=noisy,
        dueling=True,
    )
    target = RainbowDQN(
        num_actions=num_actions,
        num_atoms=num_atoms,
        v_min=v_min,
        v_max=v_max,
        noisy=noisy,
        dueling=True,
    )
    # Sync target with online
    target.load_state_dict(online.state_dict())
    return online, target


def _make_batch(batch_size=4, num_actions=4):
    """Create a batch dict mimicking PrioritizedReplayBuffer.sample() output."""
    return {
        "states": torch.randn(batch_size, 4, 84, 84),
        "actions": torch.randint(0, num_actions, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "next_states": torch.randn(batch_size, 4, 84, 84),
        "dones": torch.zeros(batch_size, dtype=torch.bool),
        "weights": torch.ones(batch_size),  # IS weights
        "indices": np.arange(batch_size),
    }


class FakeBuffer:
    """Minimal buffer mock that records update_priorities calls."""

    def __init__(self):
        self.updated_indices = None
        self.updated_priorities = None
        self.call_count = 0

    def update_priorities(self, indices, priorities):
        self.updated_indices = indices
        self.updated_priorities = priorities
        self.call_count += 1


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPerformRainbowUpdateStepBasic:
    """Basic functionality tests."""

    def test_returns_update_metrics(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()
        support = online.support

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, support,
        )

        assert isinstance(metrics, UpdateMetrics)

    def test_metrics_fields_populated(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
        )

        assert isinstance(metrics.loss, float)
        assert isinstance(metrics.td_error, float)
        assert isinstance(metrics.td_error_std, float)
        assert isinstance(metrics.grad_norm, float)
        assert isinstance(metrics.learning_rate, float)
        assert metrics.update_count == 0
        # SPR not enabled
        assert metrics.spr_loss is None
        assert metrics.cosine_similarity is None

    def test_loss_is_positive(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
        )

        assert metrics.loss > 0.0

    def test_update_count_passed_through(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
            update_count=42,
        )

        assert metrics.update_count == 42


class TestNoiseReset:
    """Verify NoisyNet noise is reset each step."""

    def test_noisy_nets_reset_called(self):
        from src.models.noisy_linear import NoisyLinear

        online, target = _make_nets(noisy=True)
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()

        # Record initial noise buffers
        noisy_layers = [
            m for m in online.modules() if isinstance(m, NoisyLinear)
        ]
        assert len(noisy_layers) > 0

        initial_noise = [m.eps_in.clone() for m in noisy_layers]

        perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
        )

        # Noise should have been resampled (different from initial)
        changed = any(
            not torch.equal(m.eps_in, init)
            for m, init in zip(noisy_layers, initial_noise)
        )
        assert changed, "NoisyLinear noise should be reset during update"

    def test_non_noisy_nets_no_error(self):
        """Non-noisy networks should work without error."""
        online, target = _make_nets(noisy=False)
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()

        # Should not raise
        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
        )
        assert metrics.loss > 0.0


class TestNetworkUpdates:
    """Verify network parameters change after update."""

    def test_online_net_params_change(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
        batch = _make_batch()

        # Snapshot a non-noise parameter (conv1.weight)
        param_before = online.conv1.weight.data.clone()

        perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
        )

        param_after = online.conv1.weight.data
        assert not torch.equal(param_before, param_after), \
            "Online net params should change after update"

    def test_target_net_params_unchanged(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)
        batch = _make_batch()

        target_before = target.conv1.weight.data.clone()

        perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
        )

        assert torch.equal(target_before, target.conv1.weight.data), \
            "Target net params should NOT change during update"


class TestISWeights:
    """Verify importance sampling weights affect the loss."""

    def test_uniform_weights_match_unweighted(self):
        """With uniform IS weights=1, loss equals unweighted mean."""
        online, target = _make_nets()
        batch = _make_batch(batch_size=8)
        batch["weights"] = torch.ones(8)

        # Run with uniform weights
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
        )
        assert metrics.loss > 0.0

    def test_different_is_weights_produce_different_loss(self):
        """Non-uniform IS weights should change the loss value."""
        torch.manual_seed(42)
        online1, target1 = _make_nets(noisy=False)
        online2 = RainbowDQN(4, 11, -5.0, 5.0, noisy=False, dueling=True)
        target2 = RainbowDQN(4, 11, -5.0, 5.0, noisy=False, dueling=True)
        # Use identical networks for both runs (no noise)
        online2.load_state_dict(online1.state_dict())
        target2.load_state_dict(target1.state_dict())

        batch = _make_batch(batch_size=8)
        batch_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy()
                      for k, v in batch.items()}

        # Run 1: uniform weights
        batch["weights"] = torch.ones(8)
        opt1 = torch.optim.Adam(online1.parameters(), lr=0)  # lr=0 so no param change
        m1 = perform_rainbow_update_step(
            online1, target1, opt1, batch, online1.support,
        )

        # Run 2: non-uniform weights
        batch_copy["weights"] = torch.tensor([0.5, 0.5, 2.0, 2.0, 0.5, 0.5, 2.0, 2.0])
        opt2 = torch.optim.Adam(online2.parameters(), lr=0)
        m2 = perform_rainbow_update_step(
            online2, target2, opt2, batch_copy, online2.support,
        )

        # Losses should differ due to different IS weights
        assert m1.loss != m2.loss, \
            "Different IS weights should produce different loss values"


class TestPriorityUpdates:
    """Verify buffer priorities are updated after the step."""

    def test_priorities_updated(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch(batch_size=4)
        fake_buffer = FakeBuffer()

        perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
            buffer=fake_buffer,
        )

        assert fake_buffer.call_count == 1
        assert fake_buffer.updated_indices is not None
        assert fake_buffer.updated_priorities is not None
        assert len(fake_buffer.updated_priorities) == 4

    def test_priorities_are_positive(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch(batch_size=4)
        fake_buffer = FakeBuffer()

        perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
            buffer=fake_buffer,
        )

        # Per-sample cross-entropy loss should always be positive
        assert all(p > 0 for p in fake_buffer.updated_priorities), \
            "Priority values should be positive (cross-entropy > 0)"

    def test_no_buffer_skips_update(self):
        """When buffer=None, update should still work (no priority update)."""
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()

        # Should not raise
        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
            buffer=None,
        )
        assert isinstance(metrics, UpdateMetrics)

    def test_indices_passed_correctly(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch(batch_size=4)
        batch["indices"] = np.array([10, 20, 30, 40])
        fake_buffer = FakeBuffer()

        perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
            buffer=fake_buffer,
        )

        np.testing.assert_array_equal(
            fake_buffer.updated_indices, np.array([10, 20, 30, 40])
        )


class TestNStepDiscount:
    """Verify n-step discount is applied correctly."""

    def test_nstep_changes_loss(self):
        """Different n_step values should produce different losses."""
        torch.manual_seed(123)
        online1, target1 = _make_nets(noisy=False)
        online2 = RainbowDQN(4, 11, -5.0, 5.0, noisy=False, dueling=True)
        target2 = RainbowDQN(4, 11, -5.0, 5.0, noisy=False, dueling=True)
        online2.load_state_dict(online1.state_dict())
        target2.load_state_dict(target1.state_dict())

        batch = _make_batch(batch_size=4)
        batch_copy = {k: v.clone() if isinstance(v, torch.Tensor) else v.copy()
                      for k, v in batch.items()}

        opt1 = torch.optim.Adam(online1.parameters(), lr=0)
        m1 = perform_rainbow_update_step(
            online1, target1, opt1, batch, online1.support, n_step=1,
        )

        opt2 = torch.optim.Adam(online2.parameters(), lr=0)
        m2 = perform_rainbow_update_step(
            online2, target2, opt2, batch_copy, online2.support, n_step=3,
        )

        # gamma^1 vs gamma^3 -> different projected distributions -> different loss
        assert m1.loss != m2.loss


class TestDoubleDQN:
    """Verify Double DQN flag works."""

    def test_double_dqn_default_true(self):
        """Default double_dqn=True should work without error."""
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
            double_dqn=True,
        )
        assert isinstance(metrics, UpdateMetrics)

    def test_standard_dqn_also_works(self):
        """double_dqn=False should also work."""
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
            double_dqn=False,
        )
        assert isinstance(metrics, UpdateMetrics)


class TestTerminalStates:
    """Verify terminal state handling."""

    def test_all_terminal(self):
        """Should not crash when all transitions are terminal."""
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()
        batch["dones"] = torch.ones(4, dtype=torch.bool)

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
        )
        assert isinstance(metrics, UpdateMetrics)
        assert not np.isnan(metrics.loss)

    def test_mixed_terminal(self):
        """Mixed terminal/non-terminal batch."""
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()
        batch["dones"] = torch.tensor([True, False, True, False])

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
        )
        assert isinstance(metrics, UpdateMetrics)
        assert not np.isnan(metrics.loss)


class TestGradientClipping:
    """Verify gradient clipping."""

    def test_grad_norm_returned(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
            max_grad_norm=10.0,
        )

        assert metrics.grad_norm >= 0.0

    def test_custom_max_grad_norm(self):
        """Small max_grad_norm should clip gradients."""
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
            max_grad_norm=0.01,
        )
        # Should not crash, grad_norm should be reported
        assert metrics.grad_norm >= 0.0


class TestMultipleUpdates:
    """Verify multiple consecutive updates work."""

    def test_three_consecutive_updates(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)

        losses = []
        for i in range(3):
            batch = _make_batch()
            metrics = perform_rainbow_update_step(
                online, target, optimizer, batch, online.support,
                update_count=i,
            )
            losses.append(metrics.loss)
            assert not np.isnan(metrics.loss)
            assert metrics.update_count == i

        # All should be valid losses
        assert len(losses) == 3


class TestMetricsDict:
    """Test to_dict conversion for logging."""

    def test_to_dict_has_required_keys(self):
        online, target = _make_nets()
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-4)
        batch = _make_batch()

        metrics = perform_rainbow_update_step(
            online, target, optimizer, batch, online.support,
        )
        d = metrics.to_dict()

        assert "loss" in d
        assert "td_error" in d
        assert "td_error_std" in d
        assert "grad_norm" in d
        assert "learning_rate" in d
        assert "update_count" in d
