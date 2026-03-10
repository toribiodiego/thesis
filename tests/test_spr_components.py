"""
Tests for SPR core components.

Verifies each component independently:
- TransitionModel: output shape, action conditioning, gradient flow
- ProjectionHead: output shape, flatten+ReLU behavior, gradient flow
- PredictionHead: output shape, linearity, gradient flow
- EMAEncoder: momentum update, direct copy, frozen gradients, buffers
- compute_spr_loss: cosine similarity, K-step aggregation, masking
"""

import pytest
import torch

from src.models.dqn import DQN
from src.models.ema import EMAEncoder
from src.models.spr import PredictionHead, ProjectionHead, TransitionModel
from src.training.spr_loss import compute_spr_loss


# ---------------------------------------------------------------------------
# TransitionModel
# ---------------------------------------------------------------------------


class TestTransitionModel:
    """Tests for the action-conditioned transition model."""

    def test_output_shape(self):
        """Output preserves spatial dims (B, 64, 7, 7)."""
        for num_actions in [4, 6, 18]:
            model = TransitionModel(num_actions=num_actions)
            conv_out = torch.randn(4, 64, 7, 7)
            action = torch.randint(0, num_actions, (4,))
            out = model(conv_out, action)
            assert out.shape == (4, 64, 7, 7)

    def test_action_conditioning(self):
        """Different actions produce different outputs for the same state."""
        model = TransitionModel(num_actions=4)
        model.train(False)
        conv_out = torch.randn(1, 64, 7, 7)

        with torch.no_grad():
            out_a0 = model(conv_out, torch.tensor([0]))
            out_a1 = model(conv_out, torch.tensor([1]))

        assert not torch.allclose(out_a0, out_a1, atol=1e-6), (
            "Different actions should produce different outputs"
        )

    def test_same_action_deterministic(self):
        """Same state and action produce identical output in eval mode."""
        model = TransitionModel(num_actions=4)
        model.train(False)
        conv_out = torch.randn(1, 64, 7, 7)
        action = torch.tensor([2])

        with torch.no_grad():
            out1 = model(conv_out, action)
            out2 = model(conv_out, action)

        assert torch.allclose(out1, out2)

    def test_gradient_flow(self):
        """Gradients flow through the transition model."""
        model = TransitionModel(num_actions=4)
        conv_out = torch.randn(2, 64, 7, 7, requires_grad=True)
        action = torch.randint(0, 4, (2,))

        out = model(conv_out, action)
        out.mean().backward()

        assert conv_out.grad is not None
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"

    def test_no_nans(self):
        """Forward pass produces no NaNs or Infs."""
        model = TransitionModel(num_actions=6)
        conv_out = torch.rand(4, 64, 7, 7)
        action = torch.randint(0, 6, (4,))
        out = model(conv_out, action)

        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_iterative_application(self):
        """Model can be applied iteratively (K steps) without errors."""
        model = TransitionModel(num_actions=4)
        z = torch.randn(2, 64, 7, 7)

        for _ in range(5):
            action = torch.randint(0, 4, (2,))
            z = model(z, action)

        assert z.shape == (2, 64, 7, 7)
        assert not torch.isnan(z).any()

    def test_batch_norm_present(self):
        """Transition model has BatchNorm after first conv."""
        model = TransitionModel(num_actions=4)
        bn_found = any(
            isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()
        )
        assert bn_found


# ---------------------------------------------------------------------------
# ProjectionHead
# ---------------------------------------------------------------------------


class TestProjectionHead:
    """Tests for the SPR projection head."""

    def test_output_shape(self):
        """Flattens (B, 64, 7, 7) to (B, 3136) then projects to (B, 512)."""
        proj = ProjectionHead(input_dim=3136, output_dim=512)
        conv_out = torch.randn(4, 64, 7, 7)
        out = proj(conv_out)
        assert out.shape == (4, 512)

    def test_custom_dims(self):
        """Supports custom input and output dimensions."""
        proj = ProjectionHead(input_dim=3136, output_dim=256)
        out = proj(torch.randn(2, 64, 7, 7))
        assert out.shape == (2, 256)

    def test_relu_activation(self):
        """Output is non-negative (ReLU applied)."""
        proj = ProjectionHead()
        # Use large enough batch for statistical reliability
        out = proj(torch.randn(32, 64, 7, 7))
        assert (out >= 0).all(), "ReLU should produce non-negative outputs"

    def test_gradient_flow(self):
        """Gradients flow back through the projection head."""
        proj = ProjectionHead()
        conv_out = torch.randn(2, 64, 7, 7, requires_grad=True)
        out = proj(conv_out)
        out.mean().backward()

        assert conv_out.grad is not None
        assert proj.fc.weight.grad is not None

    def test_no_nans(self):
        """Forward pass produces no NaNs or Infs."""
        proj = ProjectionHead()
        out = proj(torch.rand(4, 64, 7, 7))
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# PredictionHead
# ---------------------------------------------------------------------------


class TestPredictionHead:
    """Tests for the SPR prediction head."""

    def test_output_shape(self):
        """Input and output have the same shape (B, dim)."""
        pred = PredictionHead(dim=512)
        x = torch.randn(4, 512)
        out = pred(x)
        assert out.shape == (4, 512)

    def test_custom_dim(self):
        """Supports custom dimension."""
        pred = PredictionHead(dim=256)
        out = pred(torch.randn(2, 256))
        assert out.shape == (2, 256)

    def test_no_activation(self):
        """Prediction head has no nonlinear activation (pure affine)."""
        pred = PredictionHead(dim=4)
        # Affine: f(ax + (1-a)y) = a*f(x) + (1-a)*f(y)
        x = torch.randn(1, 4)
        y = torch.randn(1, 4)
        a = 0.3
        with torch.no_grad():
            out_combo = pred(a * x + (1 - a) * y)
            out_parts = a * pred(x) + (1 - a) * pred(y)
        assert torch.allclose(out_combo, out_parts, atol=1e-5)

    def test_gradient_flow(self):
        """Gradients flow back through the prediction head."""
        pred = PredictionHead(dim=512)
        x = torch.randn(2, 512, requires_grad=True)
        out = pred(x)
        out.mean().backward()

        assert x.grad is not None
        assert pred.fc.weight.grad is not None

    def test_online_path_gradient_flow(self):
        """Gradients flow through full online path: proj -> pred."""
        proj = ProjectionHead()
        pred = PredictionHead(dim=512)
        conv_out = torch.randn(2, 64, 7, 7, requires_grad=True)

        projected = proj(conv_out)
        predicted = pred(projected)
        predicted.mean().backward()

        assert conv_out.grad is not None
        assert proj.fc.weight.grad is not None
        assert pred.fc.weight.grad is not None


# ---------------------------------------------------------------------------
# EMAEncoder
# ---------------------------------------------------------------------------


class TestEMAEncoder:
    """Tests for the EMA target encoder."""

    def test_initial_copy(self):
        """EMA model starts as exact copy of online model."""
        online = ProjectionHead(input_dim=3136, output_dim=512)
        ema = EMAEncoder(online, momentum=0.99)

        for op, ep in zip(online.parameters(), ema.model.parameters()):
            assert torch.allclose(op, ep)

    def test_frozen_gradients(self):
        """EMA model parameters have requires_grad=False."""
        online = ProjectionHead()
        ema = EMAEncoder(online, momentum=0.99)

        for p in ema.model.parameters():
            assert not p.requires_grad

    def test_ema_update_formula(self):
        """EMA update follows theta_m <- tau*theta_m + (1-tau)*theta_o."""
        online = ProjectionHead(input_dim=16, output_dim=8)
        ema = EMAEncoder(online, momentum=0.9)

        # Perturb online weights
        with torch.no_grad():
            for p in online.parameters():
                p.add_(torch.randn_like(p))

        # Save pre-update EMA weights
        pre = [p.data.clone() for p in ema.model.parameters()]
        ema.update(online)

        for pre_p, ema_p, online_p in zip(
            pre, ema.model.parameters(), online.parameters()
        ):
            expected = 0.9 * pre_p + 0.1 * online_p.data
            assert torch.allclose(ema_p.data, expected, atol=1e-6)

    def test_tau_zero_is_direct_copy(self):
        """momentum=0 produces a direct copy of online params."""
        online = ProjectionHead(input_dim=16, output_dim=8)
        ema = EMAEncoder(online, momentum=0.0)

        with torch.no_grad():
            for p in online.parameters():
                p.add_(torch.randn_like(p) * 0.5)

        ema.update(online)

        for ep, op in zip(ema.model.parameters(), online.parameters()):
            assert torch.allclose(ep.data, op.data, atol=1e-7)

    def test_tau_one_ignores_online(self):
        """momentum=1 keeps EMA unchanged."""
        online = ProjectionHead(input_dim=16, output_dim=8)
        ema = EMAEncoder(online, momentum=1.0)
        pre = [p.data.clone() for p in ema.model.parameters()]

        with torch.no_grad():
            for p in online.parameters():
                p.add_(torch.randn_like(p) * 10.0)

        ema.update(online)

        for pre_p, ema_p in zip(pre, ema.model.parameters()):
            assert torch.allclose(ema_p.data, pre_p, atol=1e-7)

    def test_buffer_copy(self):
        """EMA copies buffers directly (e.g., BatchNorm running stats)."""
        trans = TransitionModel(num_actions=4)
        trans.train()
        # Run forward to populate BN stats
        _ = trans(torch.randn(8, 64, 7, 7), torch.randint(0, 4, (8,)))

        ema = EMAEncoder(trans, momentum=0.99)

        # Change online BN stats with another forward pass
        _ = trans(torch.randn(8, 64, 7, 7), torch.randint(0, 4, (8,)))
        ema.update(trans)

        for eb, ob in zip(ema.model.buffers(), trans.buffers()):
            assert torch.allclose(eb.data, ob.data)

    def test_forward_delegation(self):
        """EMA forward delegates to the internal model."""
        proj = ProjectionHead(input_dim=3136, output_dim=512)
        ema = EMAEncoder(proj, momentum=0.99)
        conv_out = torch.randn(2, 64, 7, 7)

        with torch.no_grad():
            out = ema(conv_out)
        assert out.shape == (2, 512)

    def test_independence_from_dqn_target_net(self):
        """EMA encoder is independent from DQN hard-update target net."""
        online = DQN(num_actions=4)
        ema = EMAEncoder(online, momentum=0.99)

        # Simulate DQN target net hard update
        from src.training.target_network import init_target_network

        target = init_target_network(online, num_actions=4)

        # Perturb online, update EMA only
        with torch.no_grad():
            for p in online.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        ema.update(online)

        # EMA should differ from target (target hasn't been updated)
        ema_params = list(ema.model.parameters())
        target_params = list(target.parameters())
        diffs = [
            not torch.allclose(e, t) for e, t in zip(ema_params, target_params)
        ]
        assert any(diffs), "EMA should differ from stale DQN target net"

    def test_momentum_validation(self):
        """Invalid momentum values raise ValueError."""
        online = ProjectionHead(input_dim=16, output_dim=8)
        with pytest.raises(ValueError):
            EMAEncoder(online, momentum=-0.1)
        with pytest.raises(ValueError):
            EMAEncoder(online, momentum=1.5)


# ---------------------------------------------------------------------------
# compute_spr_loss
# ---------------------------------------------------------------------------


class TestSPRLoss:
    """Tests for the SPR auxiliary loss computation."""

    def test_return_keys(self):
        """Loss function returns expected keys."""
        K, B, dim = 5, 4, 64
        result = compute_spr_loss(
            torch.randn(K, B, dim),
            torch.randn(K, B, dim),
            torch.zeros(K, B, dtype=torch.bool),
        )
        assert "loss" in result
        assert "per_step_loss" in result
        assert "num_valid" in result
        assert "cosine_similarity" in result

    def test_identical_vectors(self):
        """Identical predictions and targets give cos_sim=1, loss=-1."""
        K, B, dim = 3, 2, 32
        v = torch.randn(K, B, dim)
        dones = torch.zeros(K, B, dtype=torch.bool)
        result = compute_spr_loss(v.clone(), v.clone(), dones)

        assert abs(result["cosine_similarity"].item() - 1.0) < 1e-5
        assert abs(result["loss"].item() - (-1.0)) < 1e-5

    def test_opposite_vectors(self):
        """Opposite predictions and targets give cos_sim=-1, loss=1."""
        K, B, dim = 3, 2, 32
        v = torch.randn(K, B, dim)
        dones = torch.zeros(K, B, dtype=torch.bool)
        result = compute_spr_loss(v.clone(), -v.clone(), dones)

        assert abs(result["cosine_similarity"].item() - (-1.0)) < 1e-5
        assert abs(result["loss"].item() - 1.0) < 1e-5

    def test_orthogonal_vectors(self):
        """Orthogonal vectors give cos_sim near 0, loss near 0."""
        # Construct orthogonal pair
        a = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
        b = torch.tensor([[[0.0, 1.0]]])
        dones = torch.zeros(1, 1, dtype=torch.bool)
        result = compute_spr_loss(a, b, dones)

        assert abs(result["cosine_similarity"].item()) < 1e-5
        assert abs(result["loss"].item()) < 1e-5

    def test_gradient_flow(self):
        """Gradients flow through predictions but not targets."""
        K, B, dim = 5, 4, 64
        preds = torch.randn(K, B, dim, requires_grad=True)
        targets = torch.randn(K, B, dim)  # no requires_grad
        dones = torch.zeros(K, B, dtype=torch.bool)

        result = compute_spr_loss(preds, targets, dones)
        result["loss"].backward()

        assert preds.grad is not None
        assert not torch.isnan(preds.grad).any()

    def test_num_valid_no_dones(self):
        """All steps valid when no episode boundaries."""
        K, B, dim = 5, 4, 64
        dones = torch.zeros(K, B, dtype=torch.bool)
        result = compute_spr_loss(
            torch.randn(K, B, dim), torch.randn(K, B, dim), dones
        )
        assert result["num_valid"].item() == K * B

    def test_masking_done_at_step_zero(self):
        """Done at step 0 masks all predictions."""
        K, B, dim = 5, 2, 32
        dones = torch.zeros(K, B, dtype=torch.bool)
        dones[0, :] = True  # all samples done at step 0

        v = torch.randn(K, B, dim)
        result = compute_spr_loss(v, v, dones)

        # All masked, loss should be 0 (clamped num_valid=1)
        assert result["loss"].item() == 0.0

    def test_masking_done_mid_sequence(self):
        """Done mid-sequence masks that step and all later steps."""
        K, B = 5, 1
        dim = 32
        dones = torch.zeros(K, B, dtype=torch.bool)
        dones[2, 0] = True  # done at step 2

        result = compute_spr_loss(
            torch.randn(K, B, dim), torch.randn(K, B, dim), dones
        )
        # cumprod([1, 1, 0, 1, 1]) = [1, 1, 0, 0, 0] -> 2 valid
        assert result["num_valid"].item() == 2.0

    def test_masking_per_sample(self):
        """Different samples can have different done patterns."""
        K, B, dim = 5, 4, 32
        dones = torch.zeros(K, B, dtype=torch.bool)
        dones[1, 0] = True  # sample 0: 1 valid step
        dones[3, 1] = True  # sample 1: 3 valid steps
        # samples 2, 3: all 5 valid
        # total = 1 + 3 + 5 + 5 = 14
        result = compute_spr_loss(
            torch.randn(K, B, dim), torch.randn(K, B, dim), dones
        )
        assert result["num_valid"].item() == 14.0

    def test_per_step_loss_shape(self):
        """Per-step loss has shape (K,)."""
        K, B, dim = 5, 4, 64
        dones = torch.zeros(K, B, dtype=torch.bool)
        result = compute_spr_loss(
            torch.randn(K, B, dim), torch.randn(K, B, dim), dones
        )
        assert result["per_step_loss"].shape == (K,)

    def test_loss_decreases_with_better_predictions(self):
        """Better-aligned predictions produce lower loss."""
        K, B, dim = 5, 4, 64
        target = torch.randn(K, B, dim)
        dones = torch.zeros(K, B, dtype=torch.bool)

        bad = torch.randn(K, B, dim)
        good = target + 0.01 * torch.randn(K, B, dim)

        r_bad = compute_spr_loss(bad, target.clone(), dones)
        r_good = compute_spr_loss(good, target.clone(), dones)

        assert r_good["loss"].item() < r_bad["loss"].item()

    def test_k_step_aggregation(self):
        """Loss aggregates over K steps correctly."""
        B, dim = 2, 32
        dones = torch.zeros(1, B, dtype=torch.bool)

        # Single step
        v = torch.randn(1, B, dim)
        r1 = compute_spr_loss(v.clone(), v.clone(), dones)

        # With 3 identical-vector steps, average should still be -1.0
        v3 = v.expand(3, -1, -1).clone()
        dones3 = torch.zeros(3, B, dtype=torch.bool)
        r3 = compute_spr_loss(v3.clone(), v3.clone(), dones3)

        assert abs(r1["loss"].item() - r3["loss"].item()) < 1e-5, (
            "Average loss should be same regardless of K for identical vectors"
        )
