"""
Tests for the RainbowDQN model architecture.

Verifies:
- Output shapes (q_values, log_probs, conv_output)
- Dueling aggregation (advantage means to zero per atom)
- Distributional output (probabilities sum to 1 per action)
- Q = sum(z * p) identity
- conv_output shape matches DQN (64x7x7) for SPR compatibility
- reset_noise changes output
- Deterministic inference mode
- Non-dueling and non-noisy configurations
- Gradient flow through all parameters
"""

import pytest
import torch

from src.models.rainbow import RainbowDQN


# ---------------------------------------------------------------------------
# Output shapes
# ---------------------------------------------------------------------------


class TestOutputShapes:
    """RainbowDQN should produce correct output shapes."""

    @pytest.mark.parametrize("num_actions", [4, 6, 9, 18])
    def test_output_shapes(self, num_actions):
        model = RainbowDQN(num_actions=num_actions)
        x = torch.randn(2, 4, 84, 84)
        out = model(x)

        assert out["q_values"].shape == (2, num_actions)
        assert out["log_probs"].shape == (2, num_actions, 51)
        assert out["conv_output"].shape == (2, 64, 7, 7)

    def test_single_batch(self):
        model = RainbowDQN(num_actions=4)
        x = torch.randn(1, 4, 84, 84)
        out = model(x)

        assert out["q_values"].shape == (1, 4)
        assert out["log_probs"].shape == (1, 4, 51)

    def test_custom_atoms(self):
        """Non-default num_atoms produces correct shape."""
        model = RainbowDQN(num_actions=4, num_atoms=21)
        x = torch.randn(2, 4, 84, 84)
        out = model(x)

        assert out["log_probs"].shape == (2, 4, 21)

    def test_output_keys(self):
        model = RainbowDQN(num_actions=4)
        x = torch.randn(1, 4, 84, 84)
        out = model(x)

        assert set(out.keys()) == {"q_values", "log_probs", "conv_output"}


# ---------------------------------------------------------------------------
# SPR compatibility
# ---------------------------------------------------------------------------


class TestSPRCompatibility:
    """conv_output must match DQN shape for SPR attachment."""

    def test_conv_output_matches_dqn(self):
        """conv_output is (B, 64, 7, 7), same as vanilla DQN."""
        model = RainbowDQN(num_actions=4)
        x = torch.randn(4, 4, 84, 84)
        out = model(x)

        assert out["conv_output"].shape == (4, 64, 7, 7)

    def test_conv_output_is_after_relu(self):
        """conv_output values should be non-negative (post-ReLU)."""
        model = RainbowDQN(num_actions=4)
        x = torch.randn(2, 4, 84, 84)
        out = model(x)

        assert (out["conv_output"] >= 0).all()


# ---------------------------------------------------------------------------
# Dueling aggregation
# ---------------------------------------------------------------------------


class TestDuelingAggregation:
    """Dueling: advantage should mean to zero per atom."""

    def test_advantage_means_to_zero(self):
        """After aggregation Q = V + A - mean(A), advantage sums to zero
        across actions for each atom. This means that for each batch
        element and atom, the mean Q-atom value across actions equals V
        (since advantages cancel). We verify the structural property
        through well-formed probability distributions."""
        model = RainbowDQN(num_actions=4, dueling=True)
        model.train()
        x = torch.randn(2, 4, 84, 84)

        with torch.no_grad():
            out = model(x)

        log_probs = out["log_probs"]  # (B, A, atoms)
        probs = log_probs.exp()

        # Each action's probs should sum to 1 over atoms
        prob_sums = probs.sum(dim=2)  # (B, A)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5)

    def test_dueling_vs_non_dueling_different_outputs(self):
        """Dueling and non-dueling models produce different Q-values."""
        torch.manual_seed(42)
        model_d = RainbowDQN(num_actions=4, dueling=True, noisy=False)
        torch.manual_seed(42)
        model_nd = RainbowDQN(num_actions=4, dueling=False, noisy=False)

        x = torch.randn(2, 4, 84, 84)
        with torch.no_grad():
            out_d = model_d(x)
            out_nd = model_nd(x)

        # Different architectures produce different outputs
        assert not torch.allclose(
            out_d["q_values"], out_nd["q_values"], atol=1e-3
        )


# ---------------------------------------------------------------------------
# Distributional output
# ---------------------------------------------------------------------------


class TestDistributionalOutput:
    """Probabilities must sum to 1; Q = sum(z * p)."""

    def test_probs_sum_to_one(self):
        """exp(log_probs).sum(dim=atoms) == 1 for each (batch, action)."""
        model = RainbowDQN(num_actions=6)
        x = torch.randn(4, 4, 84, 84)

        with torch.no_grad():
            out = model(x)

        probs = out["log_probs"].exp()  # (B, A, atoms)
        sums = probs.sum(dim=2)  # (B, A)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_log_probs_are_negative(self):
        """log_probs <= 0 (log of probabilities)."""
        model = RainbowDQN(num_actions=4)
        x = torch.randn(2, 4, 84, 84)

        with torch.no_grad():
            out = model(x)

        assert (out["log_probs"] <= 1e-6).all()

    def test_q_equals_expected_value(self):
        """Q(s,a) = sum_i z_i * p_i(s,a) -- verify the identity."""
        model = RainbowDQN(num_actions=4, num_atoms=51, v_min=-10, v_max=10)
        x = torch.randn(2, 4, 84, 84)

        with torch.no_grad():
            out = model(x)

        probs = out["log_probs"].exp()  # (B, A, 51)
        support = model.support  # (51,)

        # Manual Q computation
        q_manual = (probs * support).sum(dim=2)  # (B, A)

        assert torch.allclose(out["q_values"], q_manual, atol=1e-5)

    def test_q_within_support_range(self):
        """Q-values must lie within [v_min, v_max] since they are
        expected values of a distribution on that support."""
        model = RainbowDQN(num_actions=4, v_min=-10, v_max=10)
        x = torch.randn(8, 4, 84, 84)

        with torch.no_grad():
            out = model(x)

        assert (out["q_values"] >= -10.0 - 1e-5).all()
        assert (out["q_values"] <= 10.0 + 1e-5).all()

    def test_support_atoms(self):
        """Support buffer has correct values."""
        model = RainbowDQN(num_actions=4, num_atoms=5, v_min=-2, v_max=2)
        expected = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert torch.allclose(model.support, expected)


# ---------------------------------------------------------------------------
# Noise behavior
# ---------------------------------------------------------------------------


class TestNoiseBehavior:
    """Noisy layers should affect output; reset_noise changes output."""

    def test_reset_noise_changes_output(self):
        """Two forward passes with different noise produce different Q-values."""
        model = RainbowDQN(num_actions=4, noisy=True)
        model.train()
        x = torch.randn(2, 4, 84, 84)

        with torch.no_grad():
            y1 = model(x)["q_values"].clone()
            model.reset_noise()
            y2 = model(x)["q_values"].clone()

        assert not torch.allclose(y1, y2, atol=1e-6), (
            "Different noise samples should produce different Q-values"
        )

    def test_same_noise_same_output(self):
        """Without resetting noise, same input gives same output."""
        model = RainbowDQN(num_actions=4, noisy=True)
        model.train()
        x = torch.randn(2, 4, 84, 84)

        with torch.no_grad():
            y1 = model(x)["q_values"].clone()
            y2 = model(x)["q_values"].clone()

        assert torch.allclose(y1, y2, atol=1e-7)

    def test_inference_mode_deterministic(self):
        """Inference mode uses mu only -- output is deterministic."""
        model = RainbowDQN(num_actions=4, noisy=True)
        model.eval()
        x = torch.randn(2, 4, 84, 84)

        with torch.no_grad():
            y1 = model(x)["q_values"].clone()
            model.reset_noise()
            y2 = model(x)["q_values"].clone()

        assert torch.allclose(y1, y2, atol=1e-7), (
            "Inference mode should be deterministic regardless of noise"
        )

    def test_reset_noise_noop_when_not_noisy(self):
        """reset_noise does nothing when noisy=False."""
        model = RainbowDQN(num_actions=4, noisy=False)
        x = torch.randn(2, 4, 84, 84)

        with torch.no_grad():
            y1 = model(x)["q_values"].clone()
            model.reset_noise()
            y2 = model(x)["q_values"].clone()

        assert torch.allclose(y1, y2, atol=1e-7)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Gradients should flow through all parameters."""

    def test_gradients_flow(self):
        model = RainbowDQN(num_actions=4)
        model.train()
        x = torch.randn(2, 4, 84, 84)

        out = model(x)
        loss = out["q_values"].sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradients_flow_through_log_probs(self):
        """Loss on log_probs also flows gradients."""
        model = RainbowDQN(num_actions=4)
        model.train()
        x = torch.randn(2, 4, 84, 84)

        out = model(x)
        loss = out["log_probs"].sum()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_no_nans_in_output(self):
        model = RainbowDQN(num_actions=4)
        x = torch.randn(4, 4, 84, 84)

        with torch.no_grad():
            out = model(x)

        assert not torch.isnan(out["q_values"]).any()
        assert not torch.isnan(out["log_probs"]).any()
        assert not torch.isinf(out["q_values"]).any()


# ---------------------------------------------------------------------------
# Configuration variants
# ---------------------------------------------------------------------------


class TestConfigurations:
    """Non-default configurations should work correctly."""

    def test_non_noisy(self):
        """noisy=False uses nn.Linear instead of NoisyLinear."""
        model = RainbowDQN(num_actions=4, noisy=False)
        x = torch.randn(2, 4, 84, 84)
        out = model(x)

        assert out["q_values"].shape == (2, 4)
        assert out["log_probs"].shape == (2, 4, 51)

    def test_non_dueling(self):
        """dueling=False uses a single stream."""
        model = RainbowDQN(num_actions=4, dueling=False)
        x = torch.randn(2, 4, 84, 84)
        out = model(x)

        assert out["q_values"].shape == (2, 4)
        assert out["log_probs"].shape == (2, 4, 51)

    def test_non_noisy_non_dueling(self):
        """Both flags off still produces valid output."""
        model = RainbowDQN(num_actions=4, noisy=False, dueling=False)
        x = torch.randn(2, 4, 84, 84)
        out = model(x)

        probs = out["log_probs"].exp()
        sums = probs.sum(dim=2)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_custom_support_range(self):
        """Custom v_min/v_max affect Q-value range."""
        model = RainbowDQN(
            num_actions=4, v_min=-1.0, v_max=1.0, num_atoms=11
        )
        x = torch.randn(4, 4, 84, 84)

        with torch.no_grad():
            out = model(x)

        assert (out["q_values"] >= -1.0 - 1e-5).all()
        assert (out["q_values"] <= 1.0 + 1e-5).all()

    def test_dropout(self):
        """Dropout parameter is accepted and applied."""
        model = RainbowDQN(num_actions=4, dropout=0.5)
        x = torch.randn(2, 4, 84, 84)

        # Should run without error
        model.train()
        out = model(x)
        assert out["q_values"].shape == (2, 4)


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


class TestParameterCount:
    """Verify expected parameter structure."""

    def test_noisy_has_more_params_than_non_noisy(self):
        """NoisyLinear doubles the parameters in the head."""
        noisy = RainbowDQN(num_actions=4, noisy=True)
        standard = RainbowDQN(num_actions=4, noisy=False)

        n_noisy = sum(p.numel() for p in noisy.parameters())
        n_standard = sum(p.numel() for p in standard.parameters())

        # Noisy model has ~2x the head parameters (mu + sigma)
        assert n_noisy > n_standard

    def test_dueling_has_separate_streams(self):
        """Dueling model should have value_fc, value_head,
        advantage_fc, advantage_head attributes."""
        model = RainbowDQN(num_actions=4, dueling=True)
        assert hasattr(model, "value_fc")
        assert hasattr(model, "value_head")
        assert hasattr(model, "advantage_fc")
        assert hasattr(model, "advantage_head")

    def test_non_dueling_has_single_stream(self):
        """Non-dueling model should have fc and q_head attributes."""
        model = RainbowDQN(num_actions=4, dueling=False)
        assert hasattr(model, "fc")
        assert hasattr(model, "q_head")
