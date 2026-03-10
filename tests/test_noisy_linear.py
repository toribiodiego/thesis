"""
Tests for the NoisyLinear layer.

Verifies:
- Output shape matches nn.Linear for various dimensions
- Noise changes output between forward passes (training mode)
- Eval mode is deterministic (mu only, no noise)
- Gradients flow through both mu and sigma parameters
- reset_noise produces different eps vectors
- Initialization matches paper specification
- Parameter count is correct (2x standard linear)
"""

import math

import pytest
import torch

from src.models.noisy_linear import NoisyLinear


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


class TestOutputShape:
    """NoisyLinear should produce the same output shape as nn.Linear."""

    @pytest.mark.parametrize(
        "in_f, out_f, batch",
        [(32, 16, 1), (512, 256, 4), (3136, 512, 8), (64, 64, 2)],
    )
    def test_output_shape(self, in_f, out_f, batch):
        layer = NoisyLinear(in_f, out_f)
        x = torch.randn(batch, in_f)
        y = layer(x)
        assert y.shape == (batch, out_f)

    def test_single_sample(self):
        """Works with batch size 1."""
        layer = NoisyLinear(128, 64)
        x = torch.randn(1, 128)
        y = layer(x)
        assert y.shape == (1, 64)


# ---------------------------------------------------------------------------
# Noise behavior
# ---------------------------------------------------------------------------


class TestNoiseBehavior:
    """Training mode adds noise; different noise samples produce different outputs."""

    def test_noise_changes_output_in_train_mode(self):
        """Two forward passes with different noise produce different outputs."""
        layer = NoisyLinear(64, 32)
        layer.train()
        x = torch.randn(4, 64)

        y1 = layer(x).detach().clone()
        layer.reset_noise()
        y2 = layer(x).detach().clone()

        assert not torch.allclose(y1, y2, atol=1e-6), (
            "Different noise samples should produce different outputs"
        )

    def test_same_noise_same_output(self):
        """Without resetting noise, same input gives same output."""
        layer = NoisyLinear(64, 32)
        layer.train()
        x = torch.randn(4, 64)

        y1 = layer(x).detach().clone()
        y2 = layer(x).detach().clone()

        assert torch.allclose(y1, y2, atol=1e-7), (
            "Same noise and input should produce identical output"
        )

    def test_noise_magnitude_nonzero(self):
        """The noise contribution should be non-negligible."""
        layer = NoisyLinear(64, 32)
        layer.train()
        x = torch.randn(8, 64)

        # Compute output with noise
        y_noisy = layer(x).detach()

        # Compute output without noise (mu only)
        layer.eval()
        y_mu = layer(x).detach()

        diff = (y_noisy - y_mu).abs().mean().item()
        assert diff > 1e-6, (
            f"Noise contribution too small: {diff}"
        )


# ---------------------------------------------------------------------------
# Eval mode
# ---------------------------------------------------------------------------


class TestEvalMode:
    """Eval mode should be deterministic (mu only, no noise)."""

    def test_eval_is_deterministic(self):
        """Multiple forward passes in eval mode produce identical output."""
        layer = NoisyLinear(64, 32)
        layer.eval()
        x = torch.randn(4, 64)

        with torch.no_grad():
            y1 = layer(x).clone()
            layer.reset_noise()  # should not affect eval output
            y2 = layer(x).clone()

        assert torch.allclose(y1, y2, atol=1e-7), (
            "Eval mode should ignore noise and produce deterministic output"
        )

    def test_eval_uses_mu_only(self):
        """Eval output matches manual mu-only computation."""
        layer = NoisyLinear(16, 8)
        layer.eval()
        x = torch.randn(2, 16)

        with torch.no_grad():
            y = layer(x)
            y_manual = x @ layer.weight_mu.t() + layer.bias_mu

        assert torch.allclose(y, y_manual, atol=1e-6)

    def test_train_eval_toggle(self):
        """Switching between train and eval modes works correctly."""
        layer = NoisyLinear(32, 16)
        x = torch.randn(2, 32)

        # Train mode: noisy
        layer.train()
        y_train = layer(x).detach().clone()

        # Eval mode: deterministic
        layer.eval()
        with torch.no_grad():
            y_eval1 = layer(x).clone()
            y_eval2 = layer(x).clone()

        # Eval outputs should match each other
        assert torch.allclose(y_eval1, y_eval2, atol=1e-7)

        # Train output is likely different from eval (due to noise)
        # (not guaranteed but extremely likely with non-zero sigma)
        assert not torch.allclose(y_train, y_eval1, atol=1e-6)


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


class TestGradientFlow:
    """Gradients should flow through both mu and sigma parameters."""

    def test_gradients_flow_through_mu(self):
        """Mu parameters receive gradients during backprop."""
        layer = NoisyLinear(32, 16)
        layer.train()
        x = torch.randn(4, 32)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.weight_mu.grad is not None
        assert layer.bias_mu.grad is not None
        assert layer.weight_mu.grad.abs().sum() > 0
        assert layer.bias_mu.grad.abs().sum() > 0

    def test_gradients_flow_through_sigma(self):
        """Sigma parameters receive gradients during backprop."""
        layer = NoisyLinear(32, 16)
        layer.train()
        x = torch.randn(4, 32)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert layer.weight_sigma.grad is not None
        assert layer.bias_sigma.grad is not None
        assert layer.weight_sigma.grad.abs().sum() > 0
        assert layer.bias_sigma.grad.abs().sum() > 0

    def test_no_gradient_on_eps_buffers(self):
        """Eps buffers should not accumulate gradients."""
        layer = NoisyLinear(32, 16)
        layer.train()
        x = torch.randn(4, 32)

        y = layer(x)
        loss = y.sum()
        loss.backward()

        assert not layer.eps_in.requires_grad
        assert not layer.eps_out.requires_grad

    def test_optimizer_updates_all_params(self):
        """A gradient step modifies both mu and sigma."""
        layer = NoisyLinear(32, 16)
        layer.train()
        opt = torch.optim.SGD(layer.parameters(), lr=0.1)

        mu_w_before = layer.weight_mu.data.clone()
        sigma_w_before = layer.weight_sigma.data.clone()

        x = torch.randn(4, 32)
        loss = layer(x).sum()
        loss.backward()
        opt.step()

        assert not torch.allclose(layer.weight_mu.data, mu_w_before)
        assert not torch.allclose(layer.weight_sigma.data, sigma_w_before)


# ---------------------------------------------------------------------------
# reset_noise
# ---------------------------------------------------------------------------


class TestResetNoise:
    """reset_noise should resample the eps buffers."""

    def test_reset_changes_eps(self):
        """Eps vectors change after reset_noise."""
        layer = NoisyLinear(64, 32)

        eps_in_before = layer.eps_in.clone()
        eps_out_before = layer.eps_out.clone()

        layer.reset_noise()

        # Extremely unlikely to resample identical noise
        assert not torch.allclose(layer.eps_in, eps_in_before, atol=1e-8)
        assert not torch.allclose(layer.eps_out, eps_out_before, atol=1e-8)

    def test_eps_shapes(self):
        """Eps buffers have correct shapes (factorised: p and q)."""
        layer = NoisyLinear(128, 64)
        assert layer.eps_in.shape == (128,)
        assert layer.eps_out.shape == (64,)

    def test_noise_transform_properties(self):
        """f(x) = sgn(x) * sqrt(|x|) preserves sign and compresses magnitude."""
        x = torch.tensor([-4.0, -1.0, 0.0, 1.0, 4.0])
        f_x = NoisyLinear._f(x)
        expected = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert torch.allclose(f_x, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


class TestInitialization:
    """Parameters should be initialized per paper specification."""

    def test_sigma_init_value(self):
        """Sigma initialized to sigma_0 / sqrt(in_features)."""
        in_f = 256
        sigma_0 = 0.5
        layer = NoisyLinear(in_f, 64, sigma_0=sigma_0)

        expected_sigma = sigma_0 / math.sqrt(in_f)
        assert torch.allclose(
            layer.weight_sigma.data,
            torch.full_like(layer.weight_sigma.data, expected_sigma),
            atol=1e-7,
        )
        assert torch.allclose(
            layer.bias_sigma.data,
            torch.full_like(layer.bias_sigma.data, expected_sigma),
            atol=1e-7,
        )

    def test_mu_init_range(self):
        """Mu initialized uniformly in [-1/sqrt(p), 1/sqrt(p)]."""
        in_f = 100
        layer = NoisyLinear(in_f, 64)
        bound = 1.0 / math.sqrt(in_f)

        assert layer.weight_mu.data.min() >= -bound - 1e-6
        assert layer.weight_mu.data.max() <= bound + 1e-6
        assert layer.bias_mu.data.min() >= -bound - 1e-6
        assert layer.bias_mu.data.max() <= bound + 1e-6

    def test_parameter_count(self):
        """NoisyLinear has 2x the parameters of nn.Linear (mu + sigma)."""
        in_f, out_f = 128, 64
        noisy = NoisyLinear(in_f, out_f)
        linear = torch.nn.Linear(in_f, out_f)

        noisy_params = sum(p.numel() for p in noisy.parameters())
        linear_params = sum(p.numel() for p in linear.parameters())

        assert noisy_params == 2 * linear_params

    def test_custom_sigma_0(self):
        """Non-default sigma_0 is respected."""
        layer = NoisyLinear(64, 32, sigma_0=1.0)
        expected = 1.0 / math.sqrt(64)
        assert torch.allclose(
            layer.weight_sigma.data,
            torch.full_like(layer.weight_sigma.data, expected),
            atol=1e-7,
        )


# ---------------------------------------------------------------------------
# Device handling
# ---------------------------------------------------------------------------


class TestDeviceHandling:
    """Eps buffers should follow the module to the correct device."""

    def test_buffers_on_cpu(self):
        """Buffers start on CPU."""
        layer = NoisyLinear(32, 16)
        assert layer.eps_in.device.type == "cpu"
        assert layer.eps_out.device.type == "cpu"

    def test_reset_noise_uses_buffer_device(self):
        """reset_noise generates noise on the same device as buffers."""
        layer = NoisyLinear(32, 16)
        # After construction, buffers are on CPU
        layer.reset_noise()
        assert layer.eps_in.device.type == "cpu"
        assert layer.eps_out.device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_cuda_transfer(self):
        """Moving to CUDA transfers buffers and noise works correctly."""
        layer = NoisyLinear(32, 16).cuda()
        assert layer.eps_in.device.type == "cuda"
        assert layer.eps_out.device.type == "cuda"

        # Forward pass should work on CUDA
        x = torch.randn(2, 32, device="cuda")
        y = layer(x)
        assert y.device.type == "cuda"
        assert y.shape == (2, 16)

        # reset_noise should work on CUDA
        layer.reset_noise()
        assert layer.eps_in.device.type == "cuda"

    def test_state_dict_includes_buffers(self):
        """Eps buffers appear in state_dict for checkpointing."""
        layer = NoisyLinear(32, 16)
        sd = layer.state_dict()
        assert "eps_in" in sd
        assert "eps_out" in sd
        assert "weight_mu" in sd
        assert "weight_sigma" in sd
