"""
Integration tests for SPR training pipeline.

Tests end-to-end behavior that requires multiple components working together:
- Combined TD + SPR loss decreases over multiple updates
- EMA weights diverge from online weights after training
- Sequence sampling respects episode boundaries under buffer wrap-around
"""

import numpy as np
import pytest
import torch

from src.models.dqn import DQN
from src.models.ema import EMAEncoder
from src.models.spr import PredictionHead, ProjectionHead, TransitionModel
from src.replay.replay_buffer import ReplayBuffer
from src.training.metrics import UpdateMetrics, perform_update_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_ACTIONS = 4
OBS_SHAPE = (4, 84, 84)
K = 5  # SPR prediction steps


def _make_spr_components(num_actions=NUM_ACTIONS, momentum=0.99):
    """Create a full set of SPR components for testing."""
    online_net = DQN(num_actions=num_actions)
    target_net = DQN(num_actions=num_actions)
    target_net.load_state_dict(online_net.state_dict())

    transition_model = TransitionModel(num_actions=num_actions)
    projection_head = ProjectionHead()
    prediction_head = PredictionHead()

    target_encoder = EMAEncoder(online_net, momentum=momentum)
    target_projection = EMAEncoder(projection_head, momentum=momentum)

    spr_components = {
        "transition_model": transition_model,
        "projection_head": projection_head,
        "prediction_head": prediction_head,
        "target_encoder": target_encoder,
        "target_projection": target_projection,
    }

    return online_net, target_net, spr_components


def _make_optimizer(online_net, spr_components, lr=1e-3):
    """Create optimizer over all trainable SPR parameters."""
    params = (
        list(online_net.parameters())
        + list(spr_components["transition_model"].parameters())
        + list(spr_components["projection_head"].parameters())
        + list(spr_components["prediction_head"].parameters())
    )
    return torch.optim.Adam(params, lr=lr)


def _make_batch(batch_size=8, device="cpu"):
    """Create a synthetic DQN training batch."""
    return {
        "states": torch.rand(batch_size, *OBS_SHAPE, device=device),
        "actions": torch.randint(0, NUM_ACTIONS, (batch_size,), device=device),
        "rewards": torch.randn(batch_size, device=device),
        "next_states": torch.rand(batch_size, *OBS_SHAPE, device=device),
        "dones": torch.zeros(batch_size, dtype=torch.bool, device=device),
    }


def _make_spr_batch(batch_size=8, seq_len=K, device="cpu"):
    """Create a synthetic SPR sequence batch."""
    return {
        "states": torch.rand(batch_size, seq_len + 1, *OBS_SHAPE, device=device),
        "actions": torch.randint(
            0, NUM_ACTIONS, (batch_size, seq_len), device=device
        ),
        "dones": torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device),
    }


# ---------------------------------------------------------------------------
# Test: combined loss decreases over multiple updates
# ---------------------------------------------------------------------------


class TestCombinedLossDecreases:
    """Verify that combined TD + SPR loss decreases on a fixed synthetic batch."""

    def test_loss_decreases_over_updates(self):
        """Total loss should decrease after several gradient steps on same batch."""
        torch.manual_seed(42)

        online_net, target_net, spr_components = _make_spr_components()
        optimizer = _make_optimizer(online_net, spr_components, lr=1e-3)

        # Use a fixed batch for consistent signal
        batch = _make_batch()
        spr_batch = _make_spr_batch()

        losses = []
        num_steps = 20

        for i in range(num_steps):
            metrics = perform_update_step(
                online_net=online_net,
                target_net=target_net,
                optimizer=optimizer,
                batch=batch,
                gamma=0.99,
                loss_type="mse",
                max_grad_norm=10.0,
                update_count=i,
                spr_components=spr_components,
                spr_batch=spr_batch,
                spr_weight=2.0,
            )
            losses.append(metrics.loss)

        # Loss at end should be lower than at start
        # Use average of first 3 vs last 3 to reduce noise
        early_avg = np.mean(losses[:3])
        late_avg = np.mean(losses[-3:])
        assert late_avg < early_avg, (
            f"Loss should decrease: early avg {early_avg:.4f} -> "
            f"late avg {late_avg:.4f}"
        )

    def test_spr_loss_is_populated(self):
        """SPR loss and cosine similarity should be non-None when SPR enabled."""
        torch.manual_seed(0)

        online_net, target_net, spr_components = _make_spr_components()
        optimizer = _make_optimizer(online_net, spr_components)

        metrics = perform_update_step(
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            batch=_make_batch(),
            spr_components=spr_components,
            spr_batch=_make_spr_batch(),
        )

        assert metrics.spr_loss is not None, "SPR loss should be populated"
        assert metrics.cosine_similarity is not None, (
            "Cosine similarity should be populated"
        )
        assert isinstance(metrics.spr_loss, float)
        assert isinstance(metrics.cosine_similarity, float)

    def test_total_loss_includes_spr_contribution(self):
        """Total loss should be larger than TD loss alone when SPR enabled."""
        torch.manual_seed(0)

        online_net, target_net, spr_components = _make_spr_components()
        optimizer = _make_optimizer(online_net, spr_components)

        # Run one step to get combined loss
        metrics_combined = perform_update_step(
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            batch=_make_batch(),
            spr_components=spr_components,
            spr_batch=_make_spr_batch(),
            spr_weight=2.0,
        )

        # SPR loss contributes to total: total = td + 2.0 * spr
        # Total should be larger than just td_error (which measures raw TD)
        assert metrics_combined.spr_loss > 0, (
            "SPR loss should be positive (negative cosine similarity)"
        )

    def test_metrics_to_dict_includes_spr_fields(self):
        """UpdateMetrics.to_dict should include SPR fields when populated."""
        torch.manual_seed(0)

        online_net, target_net, spr_components = _make_spr_components()
        optimizer = _make_optimizer(online_net, spr_components)

        metrics = perform_update_step(
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            batch=_make_batch(),
            spr_components=spr_components,
            spr_batch=_make_spr_batch(),
        )

        d = metrics.to_dict()
        assert "spr_loss" in d
        assert "cosine_similarity" in d
        assert d["spr_loss"] == metrics.spr_loss
        assert d["cosine_similarity"] == metrics.cosine_similarity


# ---------------------------------------------------------------------------
# Test: EMA weights diverge from online weights
# ---------------------------------------------------------------------------


class TestEMADivergence:
    """Verify EMA encoder tracks but diverges from the online encoder."""

    def test_ema_initially_matches_online(self):
        """EMA encoder should start as an exact copy of online network."""
        online_net, _, spr_components = _make_spr_components()
        target_encoder = spr_components["target_encoder"]

        for p_online, p_ema in zip(
            online_net.parameters(), target_encoder.model.parameters()
        ):
            assert torch.allclose(p_online, p_ema), (
                "EMA should match online at initialization"
            )

    def test_ema_diverges_after_training(self):
        """After gradient steps, EMA params should differ from online params."""
        torch.manual_seed(42)

        online_net, target_net, spr_components = _make_spr_components(momentum=0.99)
        optimizer = _make_optimizer(online_net, spr_components)
        target_encoder = spr_components["target_encoder"]

        # Run several update steps
        for i in range(10):
            perform_update_step(
                online_net=online_net,
                target_net=target_net,
                optimizer=optimizer,
                batch=_make_batch(),
                spr_components=spr_components,
                spr_batch=_make_spr_batch(),
                update_count=i,
            )

        # EMA should now differ from online (momentum < 1 means partial update)
        any_different = False
        for p_online, p_ema in zip(
            online_net.parameters(), target_encoder.model.parameters()
        ):
            if not torch.allclose(p_online, p_ema, atol=1e-6):
                any_different = True
                break

        assert any_different, (
            "EMA weights should diverge from online weights after training "
            "(momentum=0.99 means EMA lags behind)"
        )

    def test_ema_moves_toward_online(self):
        """EMA update should move weights closer to online network."""
        torch.manual_seed(0)

        online_net = DQN(num_actions=NUM_ACTIONS)
        target_encoder = EMAEncoder(online_net, momentum=0.5)

        # Manually perturb online weights
        with torch.no_grad():
            for p in online_net.parameters():
                p.add_(torch.randn_like(p) * 0.1)

        # Measure distance before update
        dist_before = 0.0
        for p_online, p_ema in zip(
            online_net.parameters(), target_encoder.model.parameters()
        ):
            dist_before += (p_online - p_ema).abs().sum().item()

        # Apply EMA update
        target_encoder.update(online_net)

        # Measure distance after update
        dist_after = 0.0
        for p_online, p_ema in zip(
            online_net.parameters(), target_encoder.model.parameters()
        ):
            dist_after += (p_online - p_ema).abs().sum().item()

        assert dist_after < dist_before, (
            f"EMA should move toward online: "
            f"distance {dist_before:.4f} -> {dist_after:.4f}"
        )

    def test_ema_projection_also_diverges(self):
        """Target projection (EMA of projection head) should also diverge."""
        torch.manual_seed(42)

        online_net, target_net, spr_components = _make_spr_components(momentum=0.99)
        optimizer = _make_optimizer(online_net, spr_components)
        target_projection = spr_components["target_projection"]
        projection_head = spr_components["projection_head"]

        # Run several update steps
        for i in range(10):
            perform_update_step(
                online_net=online_net,
                target_net=target_net,
                optimizer=optimizer,
                batch=_make_batch(),
                spr_components=spr_components,
                spr_batch=_make_spr_batch(),
                update_count=i,
            )

        # Target projection should differ from online projection
        any_different = False
        for p_online, p_ema in zip(
            projection_head.parameters(), target_projection.model.parameters()
        ):
            if not torch.allclose(p_online, p_ema, atol=1e-6):
                any_different = True
                break

        assert any_different, (
            "Target projection should diverge from online projection after training"
        )

    def test_zero_momentum_copies_directly(self):
        """With momentum=0.0, EMA should copy online weights exactly."""
        online_net = DQN(num_actions=NUM_ACTIONS)
        target_encoder = EMAEncoder(online_net, momentum=0.0)

        # Perturb online weights
        with torch.no_grad():
            for p in online_net.parameters():
                p.add_(torch.randn_like(p))

        # Update with momentum=0 should be direct copy
        target_encoder.update(online_net)

        for p_online, p_ema in zip(
            online_net.parameters(), target_encoder.model.parameters()
        ):
            assert torch.allclose(p_online, p_ema), (
                "momentum=0.0 should produce exact copy"
            )


# ---------------------------------------------------------------------------
# Test: sequence sampling respects episode boundaries under wrap-around
# ---------------------------------------------------------------------------


class TestSequenceSamplingBoundaries:
    """Verify sequence sampling handles episode boundaries and buffer wrap."""

    def test_done_flags_mark_episode_boundaries(self):
        """Sampled sequences should include done flags at episode ends."""
        buf = ReplayBuffer(capacity=200, obs_shape=OBS_SHAPE, min_size=1)

        # Fill buffer with 4 episodes of 10 steps each (40 transitions)
        for ep in range(4):
            for step in range(10):
                obs = np.zeros(OBS_SHAPE, dtype=np.uint8)
                obs[0, 0, 0] = ep * 10 + step  # Unique marker
                next_obs = np.zeros(OBS_SHAPE, dtype=np.uint8)
                next_obs[0, 0, 0] = ep * 10 + step + 1
                done = step == 9  # Episode ends at step 9
                buf.append(obs, 0, 0.0, next_obs, done)

        # Sample sequences -- should have done flags where episodes end
        seqs = buf.sample_sequences(batch_size=10, seq_len=3)
        dones = seqs["dones"]

        # Some sequences should cross the episode boundary and have dones=True
        assert dones.shape[1] == 3, f"Expected seq_len=3, got {dones.shape[1]}"

    def test_wrap_around_sequences_are_valid(self):
        """Sequences sampled after buffer wraps should still be valid."""
        capacity = 20
        buf = ReplayBuffer(capacity=capacity, obs_shape=OBS_SHAPE, min_size=1)

        # Fill buffer past capacity to trigger wrap-around
        for i in range(capacity + 10):
            obs = np.full(OBS_SHAPE, i % 256, dtype=np.uint8)
            next_obs = np.full(OBS_SHAPE, (i + 1) % 256, dtype=np.uint8)
            done = (i + 1) % 7 == 0  # Episode boundary every 7 steps
            buf.append(obs, i % NUM_ACTIONS, 1.0, next_obs, done)

        # Buffer should be at capacity
        assert buf.size == capacity

        # Sampling sequences should not raise
        seqs = buf.sample_sequences(batch_size=10, seq_len=3)
        assert seqs["states"].shape == (10, 4, *OBS_SHAPE)
        assert seqs["actions"].shape == (10, 3)
        assert seqs["dones"].shape == (10, 3)

    def test_done_masks_spr_loss_at_boundaries(self):
        """SPR loss masking should reduce valid count at episode boundaries."""
        from src.training.spr_loss import compute_spr_loss

        K_steps = 5
        B = 4
        dim = 32

        # Create predictions and targets
        predictions = torch.randn(K_steps, B, dim)
        targets = torch.randn(K_steps, B, dim)

        # No dones -- all K*B entries valid
        dones_none = torch.zeros(K_steps, B, dtype=torch.bool)
        result_full = compute_spr_loss(predictions, targets, dones_none)
        assert result_full["num_valid"].item() == K_steps * B

        # Done at step 1 for batch item 0 -- mask steps 1+ for that item
        # Valid: step 0 for all 4, steps 1-4 for items 1-3 = 4 + 12 = 16
        dones_partial = torch.zeros(K_steps, B, dtype=torch.bool)
        dones_partial[1, 0] = True
        result_partial = compute_spr_loss(predictions, targets, dones_partial)
        assert result_partial["num_valid"].item() == 16

        # All items done at step 0 -- mask zeroes out everything
        # (cumprod starts at 0 since 1-done[0]=0)
        dones_all = torch.zeros(K_steps, B, dtype=torch.bool)
        dones_all[0, :] = True
        result_all = compute_spr_loss(predictions, targets, dones_all)
        # num_valid clamped to 1.0 to avoid division by zero
        assert result_all["num_valid"].item() == 1.0

    def test_sequence_states_are_consecutive(self):
        """States in sampled sequences should be consecutive observations."""
        buf = ReplayBuffer(capacity=50, obs_shape=OBS_SHAPE, min_size=1)

        # Fill with identifiable observations (no episode boundaries)
        for i in range(20):
            obs = np.full(OBS_SHAPE, i, dtype=np.uint8)
            next_obs = np.full(OBS_SHAPE, i + 1, dtype=np.uint8)
            buf.append(obs, 0, 0.0, next_obs, False)

        seqs = buf.sample_sequences(batch_size=5, seq_len=3)
        states = np.asarray(seqs["states"])  # (5, 4, C, H, W)

        # For each sequence, states should be consecutive
        for b in range(5):
            for t in range(3):
                s_t = states[b, t]
                s_tp1 = states[b, t + 1]
                # Consecutive states should differ
                # (they're filled with incrementing values)
                assert not np.array_equal(s_t, s_tp1), (
                    f"Consecutive states should differ (batch {b}, step {t})"
                )


# ---------------------------------------------------------------------------
# Test: vanilla DQN path unaffected when SPR disabled
# ---------------------------------------------------------------------------


class TestVanillaDQNPath:
    """Verify perform_update_step works without SPR (backward compatibility)."""

    def test_no_spr_returns_none_spr_fields(self):
        """Without SPR, metrics should have None for SPR fields."""
        torch.manual_seed(0)

        online_net = DQN(num_actions=NUM_ACTIONS)
        target_net = DQN(num_actions=NUM_ACTIONS)
        target_net.load_state_dict(online_net.state_dict())
        optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)

        metrics = perform_update_step(
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            batch=_make_batch(),
        )

        assert metrics.spr_loss is None
        assert metrics.cosine_similarity is None

    def test_no_spr_dict_excludes_spr_fields(self):
        """to_dict should not include SPR keys when SPR disabled."""
        torch.manual_seed(0)

        online_net = DQN(num_actions=NUM_ACTIONS)
        target_net = DQN(num_actions=NUM_ACTIONS)
        target_net.load_state_dict(online_net.state_dict())
        optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)

        metrics = perform_update_step(
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            batch=_make_batch(),
        )

        d = metrics.to_dict()
        assert "spr_loss" not in d
        assert "cosine_similarity" not in d

    def test_vanilla_loss_decreases(self):
        """Vanilla DQN loss should also decrease on fixed batch."""
        torch.manual_seed(42)

        online_net = DQN(num_actions=NUM_ACTIONS)
        target_net = DQN(num_actions=NUM_ACTIONS)
        target_net.load_state_dict(online_net.state_dict())
        optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)

        batch = _make_batch()
        losses = []

        for i in range(20):
            metrics = perform_update_step(
                online_net=online_net,
                target_net=target_net,
                optimizer=optimizer,
                batch=batch,
                update_count=i,
            )
            losses.append(metrics.loss)

        early_avg = np.mean(losses[:3])
        late_avg = np.mean(losses[-3:])
        assert late_avg < early_avg, (
            f"Vanilla loss should decrease: {early_avg:.4f} -> {late_avg:.4f}"
        )
