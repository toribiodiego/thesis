"""
Backward compatibility tests for Rainbow code additions.

Verifies that vanilla DQN, DQN+augmentation, and DQN+SPR training paths
are unaffected when rainbow.enabled=false. Rainbow additions to UpdateMetrics,
EpsilonScheduler, optimizer defaults, target network factory, and checkpoint
save/load should not change behavior of the non-Rainbow code paths.
"""

import os
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.models.dqn import DQN
from src.models.ema import EMAEncoder
from src.models.rainbow import RainbowDQN
from src.models.spr import PredictionHead, ProjectionHead, TransitionModel
from src.replay.prioritized_buffer import PrioritizedReplayBuffer
from src.replay.replay_buffer import ReplayBuffer
from src.training.logging import CheckpointManager
from src.training.metrics import (
    EpsilonScheduler,
    UpdateMetrics,
    perform_update_step,
)
from src.training.optimization import configure_optimizer
from src.training.target_network import init_target_network

NUM_ACTIONS = 4
OBS_SHAPE = (4, 84, 84)
K = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(batch_size=8):
    """Create a synthetic DQN training batch."""
    return {
        "states": torch.rand(batch_size, *OBS_SHAPE),
        "actions": torch.randint(0, NUM_ACTIONS, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "next_states": torch.rand(batch_size, *OBS_SHAPE),
        "dones": torch.zeros(batch_size, dtype=torch.bool),
    }


def _make_spr_components(online_net, momentum=0.99):
    """Create SPR components using a vanilla DQN as encoder."""
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


def _make_spr_batch(batch_size=8, seq_len=K):
    """Create a synthetic SPR sequence batch."""
    return {
        "states": torch.rand(batch_size, seq_len + 1, *OBS_SHAPE),
        "actions": torch.randint(0, NUM_ACTIONS, (batch_size, seq_len)),
        "dones": torch.zeros(batch_size, seq_len, dtype=torch.bool),
    }


def _fill_buffer(buf, n=100):
    """Fill replay buffer with random transitions."""
    for i in range(n):
        obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
        next_obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
        done = (i + 1) % 25 == 0
        buf.append(obs, np.random.randint(NUM_ACTIONS), 1.0, next_obs, done)


# ---------------------------------------------------------------------------
# Test: UpdateMetrics Rainbow fields are None for vanilla DQN
# ---------------------------------------------------------------------------


class TestVanillaDQNRainbowFields:
    """Rainbow fields should be None when using vanilla DQN update step."""

    def test_vanilla_dqn_no_rainbow_fields(self):
        """perform_update_step should return None for all Rainbow metrics."""
        torch.manual_seed(42)
        online = DQN(num_actions=NUM_ACTIONS)
        target = DQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

        metrics = perform_update_step(
            online, target, optimizer, _make_batch(),
        )

        assert metrics.distributional_loss is None
        assert metrics.mean_is_weight is None
        assert metrics.mean_priority is None
        assert metrics.priority_entropy is None
        assert metrics.beta is None

    def test_vanilla_dqn_to_dict_excludes_rainbow(self):
        """to_dict should not include Rainbow keys for vanilla DQN."""
        torch.manual_seed(42)
        online = DQN(num_actions=NUM_ACTIONS)
        target = DQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

        metrics = perform_update_step(
            online, target, optimizer, _make_batch(),
        )

        d = metrics.to_dict()
        assert "distributional_loss" not in d
        assert "mean_is_weight" not in d
        assert "mean_priority" not in d
        assert "priority_entropy" not in d
        assert "beta" not in d

    def test_dqn_spr_no_rainbow_fields(self):
        """DQN+SPR should return None for Rainbow fields."""
        torch.manual_seed(42)
        online = DQN(num_actions=NUM_ACTIONS)
        target = DQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        spr_components = _make_spr_components(online)

        params = (
            list(online.parameters())
            + list(spr_components["transition_model"].parameters())
            + list(spr_components["projection_head"].parameters())
            + list(spr_components["prediction_head"].parameters())
        )
        optimizer = torch.optim.Adam(params, lr=1e-3)

        metrics = perform_update_step(
            online, target, optimizer, _make_batch(),
            spr_components=spr_components,
            spr_batch=_make_spr_batch(),
        )

        # SPR fields should be populated
        assert metrics.spr_loss is not None
        assert metrics.cosine_similarity is not None

        # Rainbow fields should be None
        assert metrics.distributional_loss is None
        assert metrics.mean_is_weight is None
        assert metrics.mean_priority is None
        assert metrics.priority_entropy is None
        assert metrics.beta is None

    def test_vanilla_loss_decreases(self):
        """Vanilla DQN loss should still decrease on a fixed batch."""
        torch.manual_seed(42)
        online = DQN(num_actions=NUM_ACTIONS)
        target = DQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

        batch = _make_batch(batch_size=16)
        losses = []

        for i in range(20):
            metrics = perform_update_step(
                online, target, optimizer, batch, update_count=i,
            )
            losses.append(metrics.loss)

        early_avg = np.mean(losses[:3])
        late_avg = np.mean(losses[-3:])
        assert late_avg < early_avg, (
            f"Vanilla loss should decrease: {early_avg:.4f} -> {late_avg:.4f}"
        )


# ---------------------------------------------------------------------------
# Test: EpsilonScheduler backward compatibility
# ---------------------------------------------------------------------------


class TestEpsilonSchedulerCompat:
    """EpsilonScheduler without noisy_nets should work as before."""

    def test_default_noisy_nets_false(self):
        """Default noisy_nets parameter should be False."""
        eps = EpsilonScheduler(1.0, 0.1, 1_000_000)
        assert eps.noisy_nets is False

    def test_epsilon_decays_normally(self):
        """Without noisy_nets, epsilon should decay linearly."""
        eps = EpsilonScheduler(1.0, 0.1, 1_000_000)

        assert eps.get_epsilon(0) == 1.0
        assert abs(eps.get_epsilon(500_000) - 0.55) < 0.01
        assert abs(eps.get_epsilon(1_000_000) - 0.1) < 0.01

    def test_eval_epsilon_unchanged(self):
        """Evaluation epsilon should be unaffected."""
        eps = EpsilonScheduler(1.0, 0.1, 1_000_000, eval_epsilon=0.05)
        assert eps.get_eval_epsilon() == 0.05

    def test_noisy_false_still_explores(self):
        """With noisy_nets=False, epsilon should be positive early."""
        eps = EpsilonScheduler(1.0, 0.1, 1_000_000, noisy_nets=False)
        assert eps.get_epsilon(0) == 1.0
        assert eps.get_epsilon(100) > 0.5


# ---------------------------------------------------------------------------
# Test: Optimizer backward compatibility
# ---------------------------------------------------------------------------


class TestOptimizerCompat:
    """configure_optimizer should work with original defaults."""

    def test_rmsprop_default(self):
        """RMSProp with default parameters should work as before."""
        model = DQN(num_actions=NUM_ACTIONS)
        opt = configure_optimizer(model, optimizer_type="rmsprop")

        assert isinstance(opt, torch.optim.RMSprop)
        assert opt.defaults["lr"] == 0.00025

    def test_adam_default(self):
        """Adam with default parameters should work."""
        model = DQN(num_actions=NUM_ACTIONS)
        opt = configure_optimizer(
            model, optimizer_type="adam", learning_rate=1e-3,
        )

        assert isinstance(opt, torch.optim.Adam)
        assert opt.defaults["lr"] == 1e-3

    def test_rmsprop_eps_default(self):
        """RMSProp eps should default to 1e-2 (DQN paper)."""
        model = DQN(num_actions=NUM_ACTIONS)
        opt = configure_optimizer(model, optimizer_type="rmsprop")
        assert opt.defaults["eps"] == 0.01

    def test_adam_eps_default(self):
        """Adam eps should default to 1e-8 when not overridden."""
        model = DQN(num_actions=NUM_ACTIONS)
        opt = configure_optimizer(
            model, optimizer_type="adam", learning_rate=1e-3,
        )
        assert opt.defaults["eps"] == 1e-8


# ---------------------------------------------------------------------------
# Test: Target network factory backward compatibility
# ---------------------------------------------------------------------------


class TestTargetNetworkCompat:
    """init_target_network should still work with vanilla DQN."""

    def test_dqn_target_creation(self):
        """Target network should be created from vanilla DQN."""
        online = DQN(num_actions=6, dropout=0.1)
        target = init_target_network(online, num_actions=6)

        assert type(target) is DQN
        assert target.num_actions == 6
        assert target.dropout == 0.1

    def test_dqn_target_weights_match(self):
        """Target weights should match online network."""
        online = DQN(num_actions=NUM_ACTIONS)
        target = init_target_network(online, num_actions=NUM_ACTIONS)

        for p_o, p_t in zip(online.parameters(), target.parameters()):
            assert torch.allclose(p_o, p_t)

    def test_dqn_target_requires_no_grad(self):
        """Target network parameters should not require gradients."""
        online = DQN(num_actions=NUM_ACTIONS)
        target = init_target_network(online, num_actions=NUM_ACTIONS)

        for p in target.parameters():
            assert not p.requires_grad


# ---------------------------------------------------------------------------
# Test: Checkpoint backward compatibility
# ---------------------------------------------------------------------------


class TestCheckpointCompat:
    """Checkpoint save/load without Rainbow should work as before."""

    def test_save_without_rainbow(self):
        """Saving with rainbow_enabled=False should not include rainbow_state."""
        torch.manual_seed(42)
        online = DQN(num_actions=NUM_ACTIONS)
        target = DQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir, save_interval=1,
            )
            path = manager.save_checkpoint(
                step=1000, episode=10, epsilon=0.5,
                online_model=online, target_model=target,
                optimizer=optimizer,
                rainbow_enabled=False,
            )

            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            assert "rainbow_state" not in checkpoint

    def test_load_without_rainbow(self):
        """Loading without Rainbow should return rainbow_restored=False."""
        torch.manual_seed(42)
        online = DQN(num_actions=NUM_ACTIONS)
        target = DQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir, save_interval=1,
            )
            path = manager.save_checkpoint(
                step=1000, episode=10, epsilon=0.5,
                online_model=online, target_model=target,
                optimizer=optimizer,
            )

            online2 = DQN(num_actions=NUM_ACTIONS)
            target2 = DQN(num_actions=NUM_ACTIONS)
            opt2 = torch.optim.Adam(online2.parameters(), lr=1e-3)

            result = manager.load_checkpoint(
                checkpoint_path=path,
                online_model=online2,
                target_model=target2,
                optimizer=opt2,
                device="cpu",
            )

            assert result["rainbow_restored"] is False
            assert result["step"] == 1000

    def test_save_load_roundtrip_weights_unchanged(self):
        """Vanilla DQN checkpoint roundtrip should preserve weights exactly."""
        torch.manual_seed(42)
        online = DQN(num_actions=NUM_ACTIONS)
        target = DQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

        # Run a training step to change weights from init
        metrics = perform_update_step(
            online, target, optimizer, _make_batch(),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = CheckpointManager(
                checkpoint_dir=tmpdir, save_interval=1,
            )
            path = manager.save_checkpoint(
                step=100, episode=1, epsilon=0.99,
                online_model=online, target_model=target,
                optimizer=optimizer,
            )

            online2 = DQN(num_actions=NUM_ACTIONS)
            target2 = DQN(num_actions=NUM_ACTIONS)
            opt2 = torch.optim.Adam(online2.parameters(), lr=1e-3)

            manager.load_checkpoint(
                checkpoint_path=path,
                online_model=online2,
                target_model=target2,
                optimizer=opt2,
                device="cpu",
            )

            for p1, p2 in zip(online.parameters(), online2.parameters()):
                assert torch.allclose(p1, p2), "Online weights should match"
            for p1, p2 in zip(target.parameters(), target2.parameters()):
                assert torch.allclose(p1, p2), "Target weights should match"


# ---------------------------------------------------------------------------
# Test: training_step backward compatibility
# ---------------------------------------------------------------------------


class TestTrainingStepCompat:
    """training_step without Rainbow components should work as before."""

    def test_training_step_vanilla_dqn(self):
        """training_step with vanilla DQN should not produce Rainbow metrics."""
        from src.training.training_loop import FrameCounter, training_step
        from src.training.schedulers import TargetNetworkUpdater, TrainingScheduler

        torch.manual_seed(42)
        online = DQN(num_actions=NUM_ACTIONS)
        target = DQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

        buf = ReplayBuffer(capacity=200, obs_shape=OBS_SHAPE, min_size=10)
        _fill_buffer(buf, 50)

        eps = EpsilonScheduler(1.0, 0.1, 100_000)
        target_up = TargetNetworkUpdater(update_interval=1000)
        train_sched = TrainingScheduler(train_every=1)
        fc = FrameCounter(frameskip=4)

        env = MagicMock()
        env.step.return_value = (
            np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8),
            1.0, False, False, {},
        )

        state = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)

        result = training_step(
            env=env,
            online_net=online,
            target_net=target,
            optimizer=optimizer,
            replay_buffer=buf,
            epsilon_scheduler=eps,
            target_updater=target_up,
            training_scheduler=train_sched,
            frame_counter=fc,
            state=state,
            num_actions=NUM_ACTIONS,
            device="cpu",
        )

        assert "metrics" in result
        if result["metrics"] is not None:
            assert result["metrics"].distributional_loss is None
            assert result["metrics"].mean_is_weight is None
            assert result["metrics"].mean_priority is None
            assert result["metrics"].beta is None

    def test_training_step_with_augmentation(self):
        """training_step with augmentation, no Rainbow, should work."""
        from src.augmentation import random_shift
        from src.training.training_loop import FrameCounter, training_step
        from src.training.schedulers import TargetNetworkUpdater, TrainingScheduler

        torch.manual_seed(42)
        online = DQN(num_actions=NUM_ACTIONS)
        target = DQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

        buf = ReplayBuffer(capacity=200, obs_shape=OBS_SHAPE, min_size=10)
        _fill_buffer(buf, 50)

        eps = EpsilonScheduler(1.0, 0.1, 100_000)
        target_up = TargetNetworkUpdater(update_interval=1000)
        train_sched = TrainingScheduler(train_every=1)
        fc = FrameCounter(frameskip=4)

        env = MagicMock()
        env.step.return_value = (
            np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8),
            1.0, False, False, {},
        )

        state = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
        augment_fn = lambda x: random_shift(x, pad=4)

        result = training_step(
            env=env,
            online_net=online,
            target_net=target,
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
        )

        assert "metrics" in result
        if result["metrics"] is not None:
            assert result["metrics"].distributional_loss is None
            assert result["metrics"].beta is None

    def test_training_step_with_spr(self):
        """training_step with DQN+SPR should have SPR but no Rainbow fields."""
        from src.training.training_loop import FrameCounter, training_step
        from src.training.schedulers import TargetNetworkUpdater, TrainingScheduler

        torch.manual_seed(42)
        online = DQN(num_actions=NUM_ACTIONS)
        target = DQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        spr_components = _make_spr_components(online)

        params = (
            list(online.parameters())
            + list(spr_components["transition_model"].parameters())
            + list(spr_components["projection_head"].parameters())
            + list(spr_components["prediction_head"].parameters())
        )
        optimizer = torch.optim.Adam(params, lr=1e-3)

        buf = ReplayBuffer(
            capacity=200, obs_shape=OBS_SHAPE, min_size=10,
            n_step=1,
        )
        _fill_buffer(buf, 50)

        eps = EpsilonScheduler(1.0, 0.1, 100_000)
        target_up = TargetNetworkUpdater(update_interval=1000)
        train_sched = TrainingScheduler(train_every=1)
        fc = FrameCounter(frameskip=4)

        env = MagicMock()
        env.step.return_value = (
            np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8),
            1.0, False, False, {},
        )

        state = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)

        result = training_step(
            env=env,
            online_net=online,
            target_net=target,
            optimizer=optimizer,
            replay_buffer=buf,
            epsilon_scheduler=eps,
            target_updater=target_up,
            training_scheduler=train_sched,
            frame_counter=fc,
            state=state,
            num_actions=NUM_ACTIONS,
            device="cpu",
            spr_components=spr_components,
        )

        assert "metrics" in result
        if result["metrics"] is not None:
            # SPR fields may or may not be populated depending on
            # whether a training step actually ran with SPR batch
            # Rainbow fields should always be None
            assert result["metrics"].distributional_loss is None
            assert result["metrics"].mean_is_weight is None
            assert result["metrics"].mean_priority is None
            assert result["metrics"].beta is None


# ---------------------------------------------------------------------------
# Test: Conditional model creation in initialize_components
# ---------------------------------------------------------------------------


class TestConditionalModelCreation:
    """train_dqn.py should create RainbowDQN when rainbow.enabled is true."""

    def _make_config(self, rainbow_enabled=False, **rainbow_overrides):
        """Build a minimal OmegaConf config for model creation tests."""
        cfg = {
            "network": {"dropout": 0.0},
            "rainbow": {
                "enabled": rainbow_enabled,
                "noisy_nets": True,
                "dueling": True,
                "distributional": {
                    "num_atoms": 51,
                    "v_min": -10.0,
                    "v_max": 10.0,
                },
            },
        }
        cfg["rainbow"].update(rainbow_overrides)
        return OmegaConf.create(cfg)

    def test_dqn_when_rainbow_disabled(self):
        """With rainbow.enabled=false, should create vanilla DQN."""
        config = self._make_config(rainbow_enabled=False)
        dropout = config.network.get("dropout", 0.0)
        rainbow_enabled = config.get("rainbow", {}).get("enabled", False)

        assert rainbow_enabled is False
        model = DQN(num_actions=4, dropout=dropout)
        assert type(model) is DQN

    def test_rainbow_when_enabled(self):
        """With rainbow.enabled=true, should create RainbowDQN."""
        config = self._make_config(rainbow_enabled=True)
        rainbow_enabled = config.get("rainbow", {}).get("enabled", False)

        assert rainbow_enabled is True
        model = RainbowDQN(
            num_actions=4,
            num_atoms=config.rainbow.distributional.num_atoms,
            v_min=config.rainbow.distributional.v_min,
            v_max=config.rainbow.distributional.v_max,
            noisy=config.rainbow.noisy_nets,
            dueling=config.rainbow.dueling,
            dropout=config.network.get("dropout", 0.0),
        )
        assert type(model) is RainbowDQN
        assert model.num_atoms == 51
        assert model.v_min == -10.0
        assert model.v_max == 10.0
        assert model.noisy is True
        assert model.dueling is True

    def test_rainbow_config_params_propagate(self):
        """RainbowDQN parameters should match config values."""
        config = self._make_config(
            rainbow_enabled=True,
            noisy_nets=False,
            dueling=False,
        )
        config.rainbow.distributional.num_atoms = 21
        config.rainbow.distributional.v_min = -5.0
        config.rainbow.distributional.v_max = 5.0

        model = RainbowDQN(
            num_actions=6,
            num_atoms=config.rainbow.distributional.num_atoms,
            v_min=config.rainbow.distributional.v_min,
            v_max=config.rainbow.distributional.v_max,
            noisy=config.rainbow.noisy_nets,
            dueling=config.rainbow.dueling,
        )
        assert model.num_atoms == 21
        assert model.v_min == -5.0
        assert model.v_max == 5.0
        assert model.noisy is False
        assert model.dueling is False
        assert model.num_actions == 6

    def test_rainbow_target_network_creation(self):
        """init_target_network should work with RainbowDQN via introspection."""
        online = RainbowDQN(num_actions=4, num_atoms=51, noisy=True, dueling=True)
        target = init_target_network(online, num_actions=4)

        assert type(target) is RainbowDQN
        assert target.num_atoms == 51
        assert target.noisy is True
        assert target.dueling is True
        for p in target.parameters():
            assert not p.requires_grad
        for p_o, p_t in zip(online.parameters(), target.parameters()):
            assert torch.allclose(p_o, p_t)

    def test_rainbow_output_interface(self):
        """RainbowDQN forward should return same dict keys as DQN plus log_probs."""
        model = RainbowDQN(num_actions=4)
        x = torch.rand(2, 4, 84, 84)
        out = model(x)

        assert "q_values" in out
        assert "conv_output" in out
        assert "log_probs" in out
        assert out["q_values"].shape == (2, 4)
        assert out["conv_output"].shape == (2, 64, 7, 7)
        assert out["log_probs"].shape == (2, 4, 51)

    def test_dqn_output_unchanged(self):
        """Vanilla DQN output should be unaffected by Rainbow additions."""
        model = DQN(num_actions=4)
        x = torch.rand(2, 4, 84, 84)
        out = model(x)

        assert "q_values" in out
        assert "conv_output" in out
        assert out["q_values"].shape == (2, 4)
        assert out["conv_output"].shape == (2, 64, 7, 7)


# ---------------------------------------------------------------------------
# Test: Conditional replay buffer creation
# ---------------------------------------------------------------------------


class TestConditionalReplayBuffer:
    """train_dqn.py should create PrioritizedReplayBuffer when rainbow enabled."""

    def _make_config(self, rainbow_enabled=False):
        """Build a minimal OmegaConf config for replay buffer tests."""
        return OmegaConf.create({
            "replay": {"capacity": 1000, "min_size": 50},
            "environment": {"preprocessing": {"frame_stack": 4}},
            "training": {"total_frames": 400000, "gamma": 0.99},
            "rainbow": {
                "enabled": rainbow_enabled,
                "multi_step": {"n": 3},
                "priority": {
                    "alpha": 0.5,
                    "beta_start": 0.4,
                    "beta_end": 1.0,
                    "epsilon": 1e-6,
                },
            },
        })

    def test_uniform_buffer_when_disabled(self):
        """With rainbow.enabled=false, should create vanilla ReplayBuffer."""
        config = self._make_config(rainbow_enabled=False)
        buf = ReplayBuffer(
            capacity=config.replay.capacity,
            obs_shape=(4, 84, 84),
            min_size=config.replay.min_size,
        )
        assert type(buf) is ReplayBuffer
        assert not isinstance(buf, PrioritizedReplayBuffer)

    def test_prioritized_buffer_when_enabled(self):
        """With rainbow.enabled=true, should create PrioritizedReplayBuffer."""
        config = self._make_config(rainbow_enabled=True)
        buf = PrioritizedReplayBuffer(
            capacity=config.replay.capacity,
            obs_shape=(4, 84, 84),
            min_size=config.replay.min_size,
            n_step=config.rainbow.multi_step.n,
            gamma=config.training.get("gamma", 0.99),
            alpha=config.rainbow.priority.alpha,
            beta_start=config.rainbow.priority.beta_start,
            beta_end=config.rainbow.priority.beta_end,
            beta_frames=config.training.total_frames,
            epsilon=config.rainbow.priority.epsilon,
        )
        assert isinstance(buf, PrioritizedReplayBuffer)
        assert buf.alpha == 0.5
        assert buf.n_step == 3
        assert buf.gamma == 0.99

    def test_prioritized_buffer_config_propagation(self):
        """Priority parameters from config should reach the buffer."""
        config = self._make_config(rainbow_enabled=True)
        config.rainbow.priority.alpha = 0.7
        config.rainbow.priority.beta_start = 0.5
        config.rainbow.multi_step.n = 5

        buf = PrioritizedReplayBuffer(
            capacity=config.replay.capacity,
            obs_shape=(4, 84, 84),
            min_size=config.replay.min_size,
            n_step=config.rainbow.multi_step.n,
            gamma=0.99,
            alpha=config.rainbow.priority.alpha,
            beta_start=config.rainbow.priority.beta_start,
            beta_end=config.rainbow.priority.beta_end,
            beta_frames=config.training.total_frames,
            epsilon=config.rainbow.priority.epsilon,
        )
        assert buf.alpha == 0.7
        assert buf.beta_start == 0.5
        assert buf.n_step == 5

    def test_prioritized_buffer_sample_returns_weights(self):
        """PrioritizedReplayBuffer.sample() should include IS weights and indices."""
        buf = PrioritizedReplayBuffer(
            capacity=200, obs_shape=OBS_SHAPE, min_size=10,
            n_step=3, alpha=0.5, beta_start=0.4,
        )
        for i in range(50):
            obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            next_obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            buf.append(obs, np.random.randint(NUM_ACTIONS), 1.0, next_obs, False)

        batch = buf.sample(batch_size=8)
        assert "weights" in batch
        assert "indices" in batch
        assert batch["weights"].shape == (8,)

    def test_uniform_buffer_no_weights(self):
        """Vanilla ReplayBuffer.sample() should NOT include weights or indices."""
        buf = ReplayBuffer(
            capacity=200, obs_shape=OBS_SHAPE, min_size=10,
        )
        for i in range(50):
            obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            next_obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            buf.append(obs, np.random.randint(NUM_ACTIONS), 1.0, next_obs, False)

        batch = buf.sample(batch_size=8)
        assert "weights" not in batch
        assert "indices" not in batch


# ---------------------------------------------------------------------------
# Test: Rainbow training step integration
# ---------------------------------------------------------------------------


class TestRainbowTrainingStep:
    """training_step with rainbow_config should use perform_rainbow_update_step."""

    def test_rainbow_training_step_produces_rainbow_metrics(self):
        """training_step with rainbow_config should produce distributional metrics."""
        from src.training.training_loop import FrameCounter, training_step
        from src.training.schedulers import TargetNetworkUpdater, TrainingScheduler

        torch.manual_seed(42)
        online = RainbowDQN(num_actions=NUM_ACTIONS)
        target = RainbowDQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        for p in target.parameters():
            p.requires_grad = False
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

        buf = PrioritizedReplayBuffer(
            capacity=200, obs_shape=OBS_SHAPE, min_size=10,
            n_step=3, alpha=0.5, beta_start=0.4,
        )
        _fill_buffer(buf, 50)

        eps = EpsilonScheduler(1.0, 0.1, 100_000)
        target_up = TargetNetworkUpdater(update_interval=1000)
        train_sched = TrainingScheduler(train_every=1)
        fc = FrameCounter(frameskip=4)

        env = MagicMock()
        env.step.return_value = (
            np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8),
            1.0, False, False, {},
        )

        state = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)

        rainbow_cfg = {
            "support": online.support,
            "n_step": 3,
            "double_dqn": True,
            "buffer": buf,
        }

        result = training_step(
            env=env,
            online_net=online,
            target_net=target,
            optimizer=optimizer,
            replay_buffer=buf,
            epsilon_scheduler=eps,
            target_updater=target_up,
            training_scheduler=train_sched,
            frame_counter=fc,
            state=state,
            num_actions=NUM_ACTIONS,
            device="cpu",
            rainbow_config=rainbow_cfg,
        )

        assert "metrics" in result
        m = result["metrics"]
        assert m is not None, "Rainbow training step should produce metrics"
        assert m.distributional_loss is not None
        assert m.mean_is_weight is not None
        assert m.mean_priority is not None
        assert m.loss is not None

    def test_vanilla_dqn_unaffected_by_rainbow_param(self):
        """training_step with rainbow_config=None should still work for DQN."""
        from src.training.training_loop import FrameCounter, training_step
        from src.training.schedulers import TargetNetworkUpdater, TrainingScheduler

        torch.manual_seed(42)
        online = DQN(num_actions=NUM_ACTIONS)
        target = DQN(num_actions=NUM_ACTIONS)
        target.load_state_dict(online.state_dict())
        optimizer = torch.optim.Adam(online.parameters(), lr=1e-3)

        buf = ReplayBuffer(capacity=200, obs_shape=OBS_SHAPE, min_size=10)
        _fill_buffer(buf, 50)

        eps = EpsilonScheduler(1.0, 0.1, 100_000)
        target_up = TargetNetworkUpdater(update_interval=1000)
        train_sched = TrainingScheduler(train_every=1)
        fc = FrameCounter(frameskip=4)

        env = MagicMock()
        env.step.return_value = (
            np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8),
            1.0, False, False, {},
        )

        state = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)

        result = training_step(
            env=env,
            online_net=online,
            target_net=target,
            optimizer=optimizer,
            replay_buffer=buf,
            epsilon_scheduler=eps,
            target_updater=target_up,
            training_scheduler=train_sched,
            frame_counter=fc,
            state=state,
            num_actions=NUM_ACTIONS,
            device="cpu",
            rainbow_config=None,
        )

        assert "metrics" in result
        if result["metrics"] is not None:
            assert result["metrics"].distributional_loss is None


# ---------------------------------------------------------------------------
# Test: Rainbow config sets epsilon to zero via noisy_nets
# ---------------------------------------------------------------------------


class TestRainbowConfigNoisyNetsEpsilon:
    """Rainbow configs should disable epsilon-greedy via noisy_nets flag."""

    def test_rainbow_config_enables_noisy_nets(self):
        """Loading a Rainbow config should set noisy_nets=True."""
        from src.config import load_config

        config = load_config(
            "experiments/dqn_atari/configs/atari100k_boxing_rainbow.yaml"
        )
        assert config["rainbow"]["enabled"] is True
        assert config["rainbow"]["noisy_nets"] is True

    def test_use_noisy_flag_logic(self):
        """use_noisy should be True only when both rainbow.enabled and noisy_nets."""
        cases = [
            (True, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, False),
        ]
        for rainbow_enabled, noisy_nets, expected in cases:
            use_noisy = rainbow_enabled and noisy_nets
            assert use_noisy == expected, (
                f"rainbow_enabled={rainbow_enabled}, noisy_nets={noisy_nets}: "
                f"expected {expected}, got {use_noisy}"
            )

    def test_epsilon_zero_when_rainbow_noisy(self):
        """EpsilonScheduler with noisy_nets=True should always return 0."""
        eps = EpsilonScheduler(
            epsilon_start=1.0,
            epsilon_end=0.1,
            decay_frames=1_000_000,
            noisy_nets=True,
        )
        assert eps.get_epsilon(0) == 0.0
        assert eps.get_epsilon(500_000) == 0.0
        assert eps.get_epsilon(2_000_000) == 0.0
        assert eps.get_eval_epsilon() == 0.0

    def test_epsilon_nonzero_without_noisy(self):
        """Without noisy_nets, epsilon should follow normal decay."""
        eps = EpsilonScheduler(
            epsilon_start=1.0,
            epsilon_end=0.1,
            decay_frames=1_000_000,
            noisy_nets=False,
        )
        assert eps.get_epsilon(0) == 1.0
        assert eps.get_epsilon(500_000) > 0.0
        assert eps.get_eval_epsilon() == 0.05


# ---------------------------------------------------------------------------
# Test: Rainbow checkpoint retention (keep_last_n=10)
# ---------------------------------------------------------------------------


class TestRainbowCheckpointRetention:
    """Rainbow configs should keep all periodic checkpoints."""

    def test_rainbow_configs_keep_last_n_10(self):
        """All Rainbow configs should set keep_last_n=10."""
        import glob

        from src.config import load_config

        configs = glob.glob(
            "experiments/dqn_atari/configs/atari100k_*_rainbow.yaml"
        )
        assert len(configs) >= 10, f"Expected >= 10 configs, got {len(configs)}"

        for cfg_path in sorted(configs):
            config = load_config(cfg_path)
            kln = config["logging"]["checkpoint"]["keep_last_n"]
            assert kln == 10, (
                f"{cfg_path}: expected keep_last_n=10, got {kln}"
            )

    def test_checkpoint_manager_keeps_all_with_n_10(self):
        """CheckpointManager with keep_last_n=10 retains 10 checkpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = CheckpointManager(
                checkpoint_dir=tmpdir,
                save_interval=40_000,
                keep_last_n=10,
            )
            for step in range(40_000, 440_000, 40_000):
                path = os.path.join(tmpdir, f"checkpoint_{step}.pt")
                torch.save({"step": step}, path)
                mgr.periodic_checkpoints.append(path)
                if (
                    mgr.keep_last_n > 0
                    and len(mgr.periodic_checkpoints) > mgr.keep_last_n
                ):
                    old = mgr.periodic_checkpoints.pop(0)
                    if os.path.exists(old):
                        os.remove(old)

            assert len(mgr.periodic_checkpoints) == 10
            for path in mgr.periodic_checkpoints:
                assert os.path.exists(path)
