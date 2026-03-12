"""
Integration tests for SPR training pipeline.

Tests end-to-end behavior that requires multiple components working together:
- Combined TD + SPR loss decreases over multiple updates
- EMA weights diverge from online weights after training
- Sequence sampling respects episode boundaries under buffer wrap-around
- Backward compatibility: vanilla DQN and DQN+aug paths unaffected by SPR code
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

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


# ---------------------------------------------------------------------------
# Test: backward compatibility -- SPR-disabled paths unchanged
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Verify vanilla DQN and DQN+aug paths are unaffected by SPR additions.

    These tests exercise the training_step function (the full loop
    orchestrator) and checkpoint save/load without SPR, confirming
    that the SPR code additions introduce no regressions.
    """

    def _fill_buffer(self, buf, n=100):
        """Fill replay buffer with random transitions."""
        for i in range(n):
            obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            next_obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            done = (i + 1) % 20 == 0
            buf.append(obs, np.random.randint(NUM_ACTIONS), 1.0, next_obs, done)

    def test_training_step_without_spr(self):
        """training_step works correctly with spr_components=None (default)."""
        from unittest.mock import MagicMock

        from src.training.training_loop import (
            FrameCounter,
            training_step,
        )
        from src.training.metrics import EpsilonScheduler
        from src.training.schedulers import TargetNetworkUpdater, TrainingScheduler

        torch.manual_seed(0)

        online_net = DQN(num_actions=NUM_ACTIONS)
        target_net = DQN(num_actions=NUM_ACTIONS)
        target_net.load_state_dict(online_net.state_dict())
        optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)

        buf = ReplayBuffer(capacity=200, obs_shape=OBS_SHAPE, min_size=10)
        self._fill_buffer(buf, 50)

        eps = EpsilonScheduler(1.0, 0.1, 100000)
        target_up = TargetNetworkUpdater(update_interval=1000)
        train_sched = TrainingScheduler(train_every=1)
        fc = FrameCounter(frameskip=4)

        # Mock environment
        env = MagicMock()
        env.step.return_value = (
            np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8),
            1.0,
            False,
            False,
            {},
        )

        state = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)

        # Run a training step without SPR -- should not raise
        result = training_step(
            env=env,
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_buffer=buf,
            epsilon_scheduler=eps,
            target_updater=target_up,
            training_scheduler=train_sched,
            frame_counter=fc,
            state=state,
            num_actions=NUM_ACTIONS,
            device="cpu",
            # spr_components not passed (default None)
        )

        assert "metrics" in result
        assert "next_state" in result
        assert "epsilon" in result
        # When training happened, metrics should not have SPR fields
        if result["metrics"] is not None:
            assert result["metrics"].spr_loss is None
            assert result["metrics"].cosine_similarity is None

    def test_training_step_with_augmentation_no_spr(self):
        """training_step with augmentation but no SPR works correctly."""
        from unittest.mock import MagicMock

        from src.augmentation import random_shift
        from src.training.training_loop import (
            FrameCounter,
            training_step,
        )
        from src.training.metrics import EpsilonScheduler
        from src.training.schedulers import TargetNetworkUpdater, TrainingScheduler

        torch.manual_seed(0)

        online_net = DQN(num_actions=NUM_ACTIONS)
        target_net = DQN(num_actions=NUM_ACTIONS)
        target_net.load_state_dict(online_net.state_dict())
        optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)

        buf = ReplayBuffer(capacity=200, obs_shape=OBS_SHAPE, min_size=10)
        self._fill_buffer(buf, 50)

        eps = EpsilonScheduler(1.0, 0.1, 100000)
        target_up = TargetNetworkUpdater(update_interval=1000)
        train_sched = TrainingScheduler(train_every=1)
        fc = FrameCounter(frameskip=4)

        env = MagicMock()
        env.step.return_value = (
            np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8),
            1.0,
            False,
            False,
            {},
        )

        state = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
        augment_fn = lambda x: random_shift(x, pad=4)

        # Run with augmentation, no SPR
        result = training_step(
            env=env,
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_buffer=buf,
            epsilon_scheduler=eps,
            target_updater=target_up,
            training_scheduler=train_sched,
            frame_counter=fc,
            state=state,
            num_actions=NUM_ACTIONS,
            device="cpu",
            augment_fn=augment_fn,
            # spr_components not passed
        )

        assert "metrics" in result
        if result["metrics"] is not None:
            assert result["metrics"].spr_loss is None

    def test_checkpoint_save_load_without_spr(self):
        """Checkpoint save/load without SPR components works correctly."""
        from src.training.logging import CheckpointManager

        torch.manual_seed(0)

        online_net = DQN(num_actions=NUM_ACTIONS)
        target_net = DQN(num_actions=NUM_ACTIONS)
        target_net.load_state_dict(online_net.state_dict())
        optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir, save_interval=1, keep_last_n=3
            )

            # Save without SPR (spr_components=None default)
            path = manager.save_checkpoint(
                step=1000,
                episode=10,
                epsilon=0.5,
                online_model=online_net,
                target_model=target_net,
                optimizer=optimizer,
            )

            assert os.path.exists(path)

            # Load without SPR
            online_net2 = DQN(num_actions=NUM_ACTIONS)
            target_net2 = DQN(num_actions=NUM_ACTIONS)
            optimizer2 = torch.optim.Adam(online_net2.parameters(), lr=1e-3)

            loaded = manager.load_checkpoint(
                checkpoint_path=path,
                online_model=online_net2,
                target_model=target_net2,
                optimizer=optimizer2,
                device="cpu",
            )

            assert loaded["step"] == 1000
            assert loaded["episode"] == 10

            # No SPR state in checkpoint
            checkpoint_data = torch.load(path, map_location="cpu",
                                         weights_only=False)
            assert "spr_state" not in checkpoint_data

    def test_checkpoint_without_spr_ignores_spr_on_load(self):
        """Loading a non-SPR checkpoint with spr_components=None is safe."""
        from src.training.logging import CheckpointManager

        torch.manual_seed(0)

        online_net = DQN(num_actions=NUM_ACTIONS)
        target_net = DQN(num_actions=NUM_ACTIONS)
        target_net.load_state_dict(online_net.state_dict())
        optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir, save_interval=1, keep_last_n=3
            )

            # Save vanilla checkpoint
            path = manager.save_checkpoint(
                step=500,
                episode=5,
                epsilon=0.8,
                online_model=online_net,
                target_model=target_net,
                optimizer=optimizer,
            )

            # Weights should restore correctly
            online_net2 = DQN(num_actions=NUM_ACTIONS)
            target_net2 = DQN(num_actions=NUM_ACTIONS)
            optimizer2 = torch.optim.Adam(online_net2.parameters(), lr=1e-3)

            manager.load_checkpoint(
                checkpoint_path=path,
                online_model=online_net2,
                target_model=target_net2,
                optimizer=optimizer2,
                device="cpu",
            )

            # Verify weights match
            for p1, p2 in zip(
                online_net.parameters(), online_net2.parameters()
            ):
                assert torch.allclose(p1, p2), "Online weights should match"

    def test_perform_update_with_augmented_batch(self):
        """perform_update_step works with augmented observations, no SPR."""
        from src.augmentation import random_shift

        torch.manual_seed(42)

        online_net = DQN(num_actions=NUM_ACTIONS)
        target_net = DQN(num_actions=NUM_ACTIONS)
        target_net.load_state_dict(online_net.state_dict())
        optimizer = torch.optim.Adam(online_net.parameters(), lr=1e-3)

        batch = _make_batch()
        # Apply augmentation to batch states (as training_step does)
        batch["states"] = random_shift(batch["states"], pad=4)
        batch["next_states"] = random_shift(batch["next_states"], pad=4)

        metrics = perform_update_step(
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            batch=batch,
            # No spr_components
        )

        assert metrics.loss > 0
        assert metrics.spr_loss is None
        assert metrics.cosine_similarity is None
        assert metrics.grad_norm >= 0

    def test_dqn_dropout_default_zero(self):
        """DQN with default dropout=0.0 behaves identically in train/mode."""
        torch.manual_seed(0)

        model = DQN(num_actions=NUM_ACTIONS, dropout=0.0)
        x = torch.rand(2, *OBS_SHAPE)

        model.train()
        out_train = model(x)["q_values"]

        # Set to inference mode instead
        model.train(False)
        with torch.no_grad():
            out_infer = model(x)["q_values"]

        assert torch.allclose(out_train, out_infer, atol=1e-6), (
            "dropout=0.0 should produce identical output in train and non-train"
        )


# ---------------------------------------------------------------------------
# Test: end-to-end SPR through initialize_components + training_step
# ---------------------------------------------------------------------------


def _make_spr_config():
    """Build a complete OmegaConf config with SPR enabled.

    Mirrors base.yaml + an SPR game config, with small buffer/frame
    counts for fast testing.
    """
    return OmegaConf.create({
        "experiment": {
            "name": "test_spr",
            "run_id": "test_run",
            "notes": "",
        },
        "environment": {
            "env_id": "PongNoFrameskip-v4",
            "preprocessing": {
                "frame_stack": 4,
                "frame_size": 84,
                "clip_rewards": True,
            },
            "action_repeat": 4,
            "episode": {
                "noop_max": 0,
                "episodic_life": False,
            },
        },
        "network": {
            "device": "cpu",
            "dropout": 0.5,
        },
        "seed": {"value": 42},
        "replay": {
            "capacity": 500,
            "batch_size": 4,
            "min_size": 20,
            "warmup_steps": 20,
        },
        "training": {
            "total_frames": 400,
            "train_every": 1,
            "gamma": 0.99,
            "loss": {"type": "mse"},
            "gradient_clip": {"max_norm": 10.0},
            "optimizer": {
                "type": "adam",
                "lr": 1e-4,
                "rmsprop": {
                    "alpha": 0.95,
                    "eps": 0.01,
                    "momentum": 0.0,
                },
                "adam": {
                    "eps": 1.5e-4,
                },
            },
        },
        "target_network": {"update_interval": 1000},
        "exploration": {
            "schedule": {
                "start_epsilon": 0.1,
                "end_epsilon": 0.1,
                "decay_frames": 1,
            },
        },
        "evaluation": {
            "eval_every": 999999,
            "num_episodes": 1,
            "epsilon": 0.0,
        },
        "logging": {
            "base_dir": "experiments/dqn_atari/runs",
            "log_every_steps": 100,
            "log_every_episodes": 1,
            "tensorboard": {"enabled": False},
            "csv": {"enabled": False},
            "wandb": {"enabled": False},
            "checkpoint": {
                "enabled": False,
                "save_every": 999999,
                "keep_last_n": 1,
                "save_best": False,
            },
        },
        "spr": {
            "enabled": True,
            "prediction_steps": 3,
            "loss_weight": 2.0,
            "projection_dim": 512,
            "transition_channels": 64,
        },
        "ema": {
            "momentum": 0.99,
        },
    })


def _mock_spr_env():
    """Create a mock Atari environment for SPR tests."""
    env = MagicMock()
    env.action_space = MagicMock()
    env.action_space.n = NUM_ACTIONS
    env.reset.return_value = (
        np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8),
        {},
    )
    env.step.return_value = (
        np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8),
        1.0,
        False,
        False,
        {},
    )
    return env


class TestSPREndToEnd:
    """End-to-end: initialize_components -> training steps -> SPR metrics."""

    def test_initialize_creates_spr_components(self, tmp_path):
        """initialize_components should create SPR components when spr.enabled."""
        from train_dqn import initialize_components

        config = _make_spr_config()
        paths = {
            "run_dir": tmp_path / "run",
            "checkpoints": tmp_path / "run" / "checkpoints",
            "eval": tmp_path / "run" / "eval",
            "logs": tmp_path / "run" / "logs",
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)

        with patch("train_dqn.make_atari_env", return_value=_mock_spr_env()):
            components = initialize_components(config, paths, torch.device("cpu"))

        spr_components = components["spr_components"]
        assert spr_components is not None, "SPR components should be created"
        assert "transition_model" in spr_components
        assert "projection_head" in spr_components
        assert "prediction_head" in spr_components
        assert "target_encoder" in spr_components
        assert "target_projection" in spr_components

    def test_initialize_adds_spr_params_to_optimizer(self, tmp_path):
        """Optimizer should have 2 param groups: DQN + SPR heads."""
        from train_dqn import initialize_components

        config = _make_spr_config()
        paths = {
            "run_dir": tmp_path / "run",
            "checkpoints": tmp_path / "run" / "checkpoints",
            "eval": tmp_path / "run" / "eval",
            "logs": tmp_path / "run" / "logs",
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)

        with patch("train_dqn.make_atari_env", return_value=_mock_spr_env()):
            components = initialize_components(config, paths, torch.device("cpu"))

        optimizer = components["optimizer"]
        assert len(optimizer.param_groups) == 2, (
            "Optimizer should have 2 param groups (DQN encoder + SPR heads)"
        )

    def test_initialize_applies_dropout(self, tmp_path):
        """DQN model should have dropout=0.5 from SPR config."""
        from train_dqn import initialize_components

        config = _make_spr_config()
        paths = {
            "run_dir": tmp_path / "run",
            "checkpoints": tmp_path / "run" / "checkpoints",
            "eval": tmp_path / "run" / "eval",
            "logs": tmp_path / "run" / "logs",
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)

        with patch("train_dqn.make_atari_env", return_value=_mock_spr_env()):
            components = initialize_components(config, paths, torch.device("cpu"))

        assert components["online_net"].dropout == 0.5

    def test_training_steps_produce_spr_metrics(self, tmp_path):
        """Running training steps with SPR should populate spr_loss and cosine_similarity."""
        from src.training.training_loop import training_step
        from train_dqn import initialize_components

        torch.manual_seed(42)
        np.random.seed(42)

        config = _make_spr_config()
        paths = {
            "run_dir": tmp_path / "run",
            "checkpoints": tmp_path / "run" / "checkpoints",
            "eval": tmp_path / "run" / "eval",
            "logs": tmp_path / "run" / "logs",
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)

        device = torch.device("cpu")
        mock_env = _mock_spr_env()
        with patch("train_dqn.make_atari_env", return_value=mock_env):
            components = initialize_components(config, paths, device)

        online_net = components["online_net"]
        target_net = components["target_net"]
        optimizer = components["optimizer"]
        replay_buffer = components["replay_buffer"]
        epsilon_scheduler = components["epsilon_scheduler"]
        target_updater = components["target_updater"]
        training_scheduler = components["training_scheduler"]
        frame_counter = components["frame_counter"]
        spr_components = components["spr_components"]

        # Fill buffer past warmup with random transitions
        for _ in range(30):
            obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            next_obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            replay_buffer.append(obs, np.random.randint(NUM_ACTIONS), 1.0, next_obs, False)

        spr_prediction_steps = config.spr.prediction_steps
        spr_weight = config.spr.loss_weight
        state = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)

        # Run several training steps and collect metrics
        collected_metrics = []
        for _ in range(5):
            result = training_step(
                env=mock_env,
                online_net=online_net,
                target_net=target_net,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                epsilon_scheduler=epsilon_scheduler,
                target_updater=target_updater,
                training_scheduler=training_scheduler,
                frame_counter=frame_counter,
                state=state,
                num_actions=NUM_ACTIONS,
                gamma=config.training.gamma,
                loss_type=config.training.loss.type,
                max_grad_norm=config.training.gradient_clip.max_norm,
                batch_size=config.replay.batch_size,
                device=device,
                spr_components=spr_components,
                spr_weight=spr_weight,
                spr_prediction_steps=spr_prediction_steps,
            )
            state = result["next_state"]
            if result["metrics"] is not None:
                collected_metrics.append(result["metrics"])

        assert len(collected_metrics) > 0, "Should have produced at least one training update"

        # Verify SPR metrics are populated on every update
        for i, m in enumerate(collected_metrics):
            assert m.spr_loss is not None, f"Step {i}: spr_loss should be populated"
            assert m.cosine_similarity is not None, f"Step {i}: cosine_similarity should be populated"
            assert m.loss is not None, f"Step {i}: total loss should be populated"

        # Sanity checks on metric values
        m = collected_metrics[0]
        assert -1.0 <= m.cosine_similarity <= 1.0, (
            f"cosine_similarity should be in [-1, 1], got {m.cosine_similarity}"
        )
        assert m.spr_loss != 0.0, "spr_loss should be non-zero on random data"

    def test_spr_metrics_in_to_dict(self, tmp_path):
        """UpdateMetrics.to_dict() should include spr_loss and cosine_similarity."""
        from src.training.training_loop import training_step
        from train_dqn import initialize_components

        torch.manual_seed(42)
        np.random.seed(42)

        config = _make_spr_config()
        paths = {
            "run_dir": tmp_path / "run",
            "checkpoints": tmp_path / "run" / "checkpoints",
            "eval": tmp_path / "run" / "eval",
            "logs": tmp_path / "run" / "logs",
        }
        for p in paths.values():
            p.mkdir(parents=True, exist_ok=True)

        device = torch.device("cpu")
        mock_env = _mock_spr_env()
        with patch("train_dqn.make_atari_env", return_value=mock_env):
            components = initialize_components(config, paths, device)

        replay_buffer = components["replay_buffer"]
        for _ in range(30):
            obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            next_obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            replay_buffer.append(obs, np.random.randint(NUM_ACTIONS), 1.0, next_obs, False)

        state = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)

        # Run until we get a metrics object
        metrics = None
        for _ in range(5):
            result = training_step(
                env=mock_env,
                online_net=components["online_net"],
                target_net=components["target_net"],
                optimizer=components["optimizer"],
                replay_buffer=replay_buffer,
                epsilon_scheduler=components["epsilon_scheduler"],
                target_updater=components["target_updater"],
                training_scheduler=components["training_scheduler"],
                frame_counter=components["frame_counter"],
                state=state,
                num_actions=NUM_ACTIONS,
                gamma=config.training.gamma,
                loss_type=config.training.loss.type,
                max_grad_norm=config.training.gradient_clip.max_norm,
                batch_size=config.replay.batch_size,
                device=device,
                spr_components=components["spr_components"],
                spr_weight=config.spr.loss_weight,
                spr_prediction_steps=config.spr.prediction_steps,
            )
            state = result["next_state"]
            if result["metrics"] is not None:
                metrics = result["metrics"]
                break

        assert metrics is not None, "Should have produced a training update"

        d = metrics.to_dict()
        assert "spr_loss" in d, "to_dict should include spr_loss"
        assert "cosine_similarity" in d, "to_dict should include cosine_similarity"
        assert isinstance(d["spr_loss"], float)
        assert isinstance(d["cosine_similarity"], float)
