"""
End-to-end integration tests for Rainbow training through train_dqn.py.

Tests that initialize_components() correctly wires RainbowDQN and
PrioritizedReplayBuffer when rainbow.enabled=true, and that running
training steps through the full pipeline produces all expected
Rainbow-specific metrics.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from src.models.rainbow import RainbowDQN
from src.replay.prioritized_buffer import PrioritizedReplayBuffer

NUM_ACTIONS = 4
OBS_SHAPE = (4, 84, 84)


def _make_rainbow_config():
    """Build a complete OmegaConf config with Rainbow enabled.

    Mirrors the structure of base.yaml + a Rainbow game config,
    with small buffer/frame counts for fast testing.
    """
    return OmegaConf.create({
        "experiment": {
            "name": "test_rainbow",
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
            "dropout": 0.0,
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
                "lr": 6.25e-5,
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
                "start_epsilon": 0.0,
                "end_epsilon": 0.0,
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
        "rainbow": {
            "enabled": True,
            "double_dqn": True,
            "dueling": True,
            "noisy_nets": True,
            "distributional": {
                "num_atoms": 11,
                "v_min": -10.0,
                "v_max": 10.0,
            },
            "multi_step": {"n": 3},
            "priority": {
                "alpha": 0.5,
                "beta_start": 0.4,
                "beta_end": 1.0,
                "epsilon": 1e-6,
            },
        },
    })


def _mock_env():
    """Create a mock Atari environment with correct interface."""
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


def _make_paths(tmp_path):
    """Create a paths dict matching setup_run_directory output."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    eval_dir = run_dir / "eval"
    eval_dir.mkdir()
    logs = run_dir / "logs"
    logs.mkdir()
    return {
        "run_dir": run_dir,
        "checkpoints": checkpoints,
        "eval": eval_dir,
        "logs": logs,
    }


class TestRainbowTrainingIntegration:
    """End-to-end: initialize_components -> training steps -> Rainbow metrics."""

    def test_initialize_returns_rainbow_model(self, tmp_path):
        """initialize_components should return RainbowDQN when rainbow enabled."""
        from train_dqn import initialize_components

        config = _make_rainbow_config()
        paths = _make_paths(tmp_path)

        with patch("train_dqn.make_atari_env", return_value=_mock_env()):
            components = initialize_components(config, paths, torch.device("cpu"))

        assert isinstance(components["online_net"], RainbowDQN)
        assert isinstance(components["target_net"], RainbowDQN)

    def test_initialize_returns_prioritized_buffer(self, tmp_path):
        """initialize_components should return PrioritizedReplayBuffer for Rainbow."""
        from train_dqn import initialize_components

        config = _make_rainbow_config()
        paths = _make_paths(tmp_path)

        with patch("train_dqn.make_atari_env", return_value=_mock_env()):
            components = initialize_components(config, paths, torch.device("cpu"))

        buf = components["replay_buffer"]
        assert isinstance(buf, PrioritizedReplayBuffer)
        assert buf.alpha == 0.5
        assert buf.n_step == 3

    def test_initialize_uses_adam_with_correct_eps(self, tmp_path):
        """Rainbow optimizer should be Adam with eps=1.5e-4, not RMSProp eps."""
        from train_dqn import initialize_components

        config = _make_rainbow_config()
        paths = _make_paths(tmp_path)

        with patch("train_dqn.make_atari_env", return_value=_mock_env()):
            components = initialize_components(config, paths, torch.device("cpu"))

        optimizer = components["optimizer"]
        assert isinstance(optimizer, torch.optim.Adam)
        # Adam eps should be 1.5e-4, not 0.01 (RMSProp default)
        adam_eps = optimizer.param_groups[0]["eps"]
        assert adam_eps == pytest.approx(1.5e-4), (
            f"Expected Adam eps=1.5e-4, got {adam_eps}"
        )

    def test_training_steps_produce_rainbow_metrics(self, tmp_path):
        """Running training steps should produce all 5 Rainbow-specific metrics."""
        from src.training.training_loop import training_step

        from train_dqn import initialize_components

        torch.manual_seed(42)
        np.random.seed(42)

        config = _make_rainbow_config()
        paths = _make_paths(tmp_path)
        device = torch.device("cpu")

        mock_env = _mock_env()
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

        # Fill buffer past min_size with random transitions
        for _ in range(30):
            obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            next_obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            replay_buffer.append(obs, np.random.randint(NUM_ACTIONS), 1.0, next_obs, False)

        # Build rainbow_config matching what run_training builds
        rainbow_config = {
            "support": online_net.support,
            "n_step": config.rainbow.multi_step.n,
            "double_dqn": config.rainbow.double_dqn,
            "buffer": replay_buffer,
        }

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
                max_grad_norm=config.training.gradient_clip.max_norm,
                batch_size=config.replay.batch_size,
                device=device,
                rainbow_config=rainbow_config,
            )
            state = result["next_state"]
            if result["metrics"] is not None:
                collected_metrics.append(result["metrics"])

        assert len(collected_metrics) > 0, "Should have produced at least one training update"

        # Verify all 5 Rainbow-specific metrics are present
        m = collected_metrics[0]
        assert m.distributional_loss is not None, "Missing distributional_loss"
        assert m.mean_is_weight is not None, "Missing mean_is_weight"
        assert m.mean_priority is not None, "Missing mean_priority"
        assert m.priority_entropy is not None, "Missing priority_entropy"
        assert m.beta is not None, "Missing beta"

        # Sanity checks on metric values
        assert m.distributional_loss > 0, "Distributional loss should be positive"
        assert m.mean_is_weight > 0, "IS weights should be positive"
        assert m.mean_priority > 0, "Mean priority should be positive"
        assert m.beta >= 0.4, "Beta should start at beta_start (0.4)"
        assert m.loss is not None, "Total loss should be present"

    def test_all_training_steps_have_consistent_metrics(self, tmp_path):
        """Every training step should produce all 5 Rainbow metrics consistently."""
        from src.training.training_loop import training_step

        from train_dqn import initialize_components

        torch.manual_seed(42)
        np.random.seed(42)

        config = _make_rainbow_config()
        paths = _make_paths(tmp_path)
        device = torch.device("cpu")

        mock_env = _mock_env()
        with patch("train_dqn.make_atari_env", return_value=mock_env):
            components = initialize_components(config, paths, device)

        online_net = components["online_net"]
        target_net = components["target_net"]
        replay_buffer = components["replay_buffer"]

        # Fill buffer
        for _ in range(30):
            obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            next_obs = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)
            replay_buffer.append(obs, np.random.randint(NUM_ACTIONS), 1.0, next_obs, False)

        rainbow_config = {
            "support": online_net.support,
            "n_step": config.rainbow.multi_step.n,
            "double_dqn": config.rainbow.double_dqn,
            "buffer": replay_buffer,
        }

        state = np.random.randint(0, 256, OBS_SHAPE, dtype=np.uint8)

        # Run 10 steps, check every metric on every update
        rainbow_fields = [
            "distributional_loss", "mean_is_weight",
            "mean_priority", "priority_entropy", "beta",
        ]

        for step_i in range(10):
            result = training_step(
                env=mock_env,
                online_net=online_net,
                target_net=target_net,
                optimizer=components["optimizer"],
                replay_buffer=replay_buffer,
                epsilon_scheduler=components["epsilon_scheduler"],
                target_updater=components["target_updater"],
                training_scheduler=components["training_scheduler"],
                frame_counter=components["frame_counter"],
                state=state,
                num_actions=NUM_ACTIONS,
                gamma=config.training.gamma,
                max_grad_norm=config.training.gradient_clip.max_norm,
                batch_size=config.replay.batch_size,
                device=device,
                rainbow_config=rainbow_config,
            )
            state = result["next_state"]

            if result["metrics"] is not None:
                m = result["metrics"]
                for field in rainbow_fields:
                    val = getattr(m, field)
                    assert val is not None, (
                        f"Step {step_i}: {field} is None"
                    )
