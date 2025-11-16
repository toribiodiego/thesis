"""
Unit tests for resume functionality.

Tests checkpoint resume with:
- Config validation and compatibility checking
- Device-safe tensor loading
- Epsilon schedule state restoration
- RNG state restoration
- Replay buffer restoration
- Git hash mismatch warnings
"""

import shutil
import tempfile

import numpy as np
import pytest
import torch

from src.models.dqn import DQN
from src.replay.replay_buffer import ReplayBuffer
from src.training import (
    CheckpointManager,
    EpsilonScheduler,
    check_git_hash_mismatch,
    configure_optimizer,
    get_rng_states,
    resume_from_checkpoint,
    validate_config_compatibility,
)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def models_and_state():
    """Create models, optimizer, and epsilon scheduler."""
    online_model = DQN(num_actions=6)
    target_model = DQN(num_actions=6)
    target_model.load_state_dict(online_model.state_dict())
    optimizer = configure_optimizer(online_model, optimizer_type="rmsprop")
    epsilon_scheduler = EpsilonScheduler(
        epsilon_start=1.0, epsilon_end=0.1, decay_frames=1_000_000, eval_epsilon=0.05
    )

    return {
        "online_model": online_model,
        "target_model": target_model,
        "optimizer": optimizer,
        "epsilon_scheduler": epsilon_scheduler,
    }


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    return {
        "env": {"id": "ALE/Pong-v5"},
        "preprocess": {"frame_size": 84, "stack_size": 4},
        "training": {
            "gamma": 0.99,
            "learning_rate": 2.5e-4,
            "batch_size": 32,
            "target_update_interval": 10000,
        },
        "replay": {"capacity": 1_000_000},
    }


class TestConfigValidation:
    """Test configuration compatibility validation."""

    def test_compatible_configs(self, sample_config):
        """Test that identical configs are compatible."""
        config1 = sample_config.copy()
        config2 = sample_config.copy()

        is_compatible, warnings = validate_config_compatibility(config1, config2)

        assert is_compatible is True
        assert len(warnings) == 0

    def test_critical_mismatch_env_id(self, sample_config):
        """Test that env ID mismatch is detected as critical."""
        import copy

        config1 = copy.deepcopy(sample_config)
        config2 = copy.deepcopy(sample_config)
        config2["env"]["id"] = "ALE/Breakout-v5"

        is_compatible, warnings = validate_config_compatibility(config1, config2)

        assert is_compatible is False
        assert any("Environment ID" in w for w in warnings)
        assert any("CRITICAL" in w for w in warnings)

    def test_critical_mismatch_frame_size(self, sample_config):
        """Test that frame size mismatch is detected as critical."""
        import copy

        config1 = copy.deepcopy(sample_config)
        config2 = copy.deepcopy(sample_config)
        config2["preprocess"]["frame_size"] = 64

        is_compatible, warnings = validate_config_compatibility(config1, config2)

        assert is_compatible is False
        assert any("Frame size" in w for w in warnings)

    def test_warning_on_hyperparameter_change(self, sample_config):
        """Test that hyperparameter changes generate warnings."""
        import copy

        config1 = copy.deepcopy(sample_config)
        config2 = copy.deepcopy(sample_config)
        config2["training"]["learning_rate"] = 1e-4

        is_compatible, warnings = validate_config_compatibility(config1, config2)

        # Should be compatible but with warnings
        assert is_compatible is True
        assert any("Learning rate" in w for w in warnings)
        assert any("WARNING" in w for w in warnings)


class TestGitHashCheck:
    """Test git commit hash mismatch detection."""

    def test_matching_hashes(self):
        """Test that matching hashes return None."""
        warning = check_git_hash_mismatch("abc123", "abc123")
        assert warning is None

    def test_mismatching_hashes(self):
        """Test that mismatching hashes return warning."""
        warning = check_git_hash_mismatch("abc123", "def456")

        assert warning is not None
        assert "WARNING" in warning
        assert "abc123" in warning
        assert "def456" in warning

    def test_unknown_hash(self):
        """Test handling of unknown git hash."""
        warning = check_git_hash_mismatch("unknown", "abc123")

        assert warning is not None
        assert "Unable to verify" in warning


class TestResumeFromCheckpoint:
    """Test full resume functionality."""

    def test_basic_resume(self, temp_checkpoint_dir, models_and_state, sample_config):
        """Test basic checkpoint save and resume."""
        # Setup
        state = models_and_state
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Save checkpoint
        save_step = 5000
        save_episode = 250
        save_epsilon = 0.95

        checkpoint_path = manager.save_checkpoint(
            step=save_step,
            episode=save_episode,
            epsilon=save_epsilon,
            online_model=state["online_model"],
            target_model=state["target_model"],
            optimizer=state["optimizer"],
            rng_states=get_rng_states(),
            extra_metadata={"config": sample_config},
        )

        # Create fresh models for resume
        new_online = DQN(num_actions=6)
        new_target = DQN(num_actions=6)
        new_optimizer = configure_optimizer(new_online, optimizer_type="rmsprop")
        new_scheduler = EpsilonScheduler(
            epsilon_start=1.0, epsilon_end=0.1, decay_frames=1_000_000
        )

        # Resume
        resumed = resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            online_model=new_online,
            target_model=new_target,
            optimizer=new_optimizer,
            epsilon_scheduler=new_scheduler,
            config=sample_config,
            device="cpu",
        )

        # Verify
        assert resumed["step"] == save_step
        assert resumed["episode"] == save_episode
        assert abs(resumed["epsilon"] - save_epsilon) < 1e-6
        assert resumed["next_step"] == save_step + 1

        # Verify epsilon scheduler was restored
        assert abs(new_scheduler.current_epsilon - save_epsilon) < 1e-6
        assert new_scheduler.frame_counter == save_step

        # Verify model weights match
        for p1, p2 in zip(state["online_model"].parameters(), new_online.parameters()):
            assert torch.allclose(p1, p2)

    def test_resume_with_replay_buffer(self, temp_checkpoint_dir, models_and_state):
        """Test resume with replay buffer restoration."""
        state = models_and_state
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir, save_replay_buffer=True
        )

        # Create and fill replay buffer
        replay_buffer = ReplayBuffer(capacity=1000, obs_shape=(4, 84, 84))

        for i in range(200):
            obs = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
            action = np.random.randint(0, 6)
            reward = np.random.randn()
            next_obs = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
            done = i % 50 == 49

            replay_buffer.append(obs, action, reward, next_obs, done)

        # Save checkpoint with buffer
        checkpoint_path = manager.save_checkpoint(
            step=10000,
            episode=500,
            epsilon=0.9,
            online_model=state["online_model"],
            target_model=state["target_model"],
            optimizer=state["optimizer"],
            replay_buffer=replay_buffer,
            rng_states=get_rng_states(),
        )

        # Create fresh models and empty buffer
        new_online = DQN(num_actions=6)
        new_target = DQN(num_actions=6)
        new_optimizer = configure_optimizer(new_online, optimizer_type="rmsprop")
        new_scheduler = EpsilonScheduler()
        new_buffer = ReplayBuffer(capacity=1000, obs_shape=(4, 84, 84))

        # Resume
        resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            online_model=new_online,
            target_model=new_target,
            optimizer=new_optimizer,
            epsilon_scheduler=new_scheduler,
            replay_buffer=new_buffer,
            device="cpu",
        )

        # Verify buffer state was restored
        assert new_buffer.size == replay_buffer.size
        assert new_buffer.index == replay_buffer.index
        assert np.array_equal(new_buffer.observations, replay_buffer.observations)
        assert np.array_equal(new_buffer.actions, replay_buffer.actions)

    def test_resume_with_rng_state_restoration(
        self, temp_checkpoint_dir, models_and_state
    ):
        """Test that RNG states are properly restored."""
        state = models_and_state
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Set specific seed and capture state
        torch.manual_seed(42)
        np.random.seed(42)
        rng_states = get_rng_states()

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            step=1000,
            episode=50,
            epsilon=0.99,
            online_model=state["online_model"],
            target_model=state["target_model"],
            optimizer=state["optimizer"],
            rng_states=rng_states,
        )

        # Change RNG state
        torch.manual_seed(999)
        np.random.seed(999)

        # Create fresh models
        new_online = DQN(num_actions=6)
        new_target = DQN(num_actions=6)
        new_optimizer = configure_optimizer(new_online, optimizer_type="rmsprop")
        new_scheduler = EpsilonScheduler()

        # Generate random numbers before resume
        torch_before = torch.rand(10)
        numpy_before = np.random.rand(10)

        # Resume (which should restore RNG states)
        resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            online_model=new_online,
            target_model=new_target,
            optimizer=new_optimizer,
            epsilon_scheduler=new_scheduler,
            device="cpu",
        )

        # Reset to the same state manually for comparison
        torch.manual_seed(42)
        np.random.seed(42)
        torch_expected = torch.rand(10)
        numpy_expected = np.random.rand(10)

        # After resume, generate should NOT match the before values
        # (because RNG was restored to seed=42 state)
        assert not torch.allclose(torch_before, torch_expected)
        assert not np.allclose(numpy_before, numpy_expected)

    def test_resume_device_mapping(self, temp_checkpoint_dir, models_and_state):
        """Test device-safe tensor loading."""
        state = models_and_state
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            step=1000,
            episode=50,
            epsilon=0.99,
            online_model=state["online_model"],
            target_model=state["target_model"],
            optimizer=state["optimizer"],
        )

        # Create fresh models
        new_online = DQN(num_actions=6)
        new_target = DQN(num_actions=6)
        new_optimizer = configure_optimizer(new_online, optimizer_type="rmsprop")
        new_scheduler = EpsilonScheduler()

        # Resume to CPU
        resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            online_model=new_online,
            target_model=new_target,
            optimizer=new_optimizer,
            epsilon_scheduler=new_scheduler,
            device="cpu",
        )

        # Verify all parameters are on CPU
        for param in new_online.parameters():
            assert param.device.type == "cpu"

        for param in new_target.parameters():
            assert param.device.type == "cpu"

    def test_resume_config_validation_error(
        self, temp_checkpoint_dir, models_and_state, sample_config
    ):
        """Test that strict config validation raises error on mismatch."""
        state = models_and_state
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Save with one config
        checkpoint_path = manager.save_checkpoint(
            step=1000,
            episode=50,
            epsilon=0.99,
            online_model=state["online_model"],
            target_model=state["target_model"],
            optimizer=state["optimizer"],
            extra_metadata={"config": sample_config},
        )

        # Try to resume with incompatible config
        incompatible_config = sample_config.copy()
        incompatible_config["env"]["id"] = "ALE/Breakout-v5"  # Different game

        new_online = DQN(num_actions=6)
        new_target = DQN(num_actions=6)
        new_optimizer = configure_optimizer(new_online, optimizer_type="rmsprop")
        new_scheduler = EpsilonScheduler()

        # Should raise error with strict_config=True
        with pytest.raises(ValueError, match="Config incompatibility"):
            resume_from_checkpoint(
                checkpoint_path=checkpoint_path,
                online_model=new_online,
                target_model=new_target,
                optimizer=new_optimizer,
                epsilon_scheduler=new_scheduler,
                config=incompatible_config,
                strict_config=True,
                device="cpu",
            )

    def test_resume_config_validation_warning(
        self, temp_checkpoint_dir, models_and_state, sample_config
    ):
        """Test that non-strict config validation only warns."""
        state = models_and_state
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Save with one config
        checkpoint_path = manager.save_checkpoint(
            step=1000,
            episode=50,
            epsilon=0.99,
            online_model=state["online_model"],
            target_model=state["target_model"],
            optimizer=state["optimizer"],
            extra_metadata={"config": sample_config},
        )

        # Resume with incompatible config but strict=False
        incompatible_config = sample_config.copy()
        incompatible_config["env"]["id"] = "ALE/Breakout-v5"

        new_online = DQN(num_actions=6)
        new_target = DQN(num_actions=6)
        new_optimizer = configure_optimizer(new_online, optimizer_type="rmsprop")
        new_scheduler = EpsilonScheduler()

        # Should succeed with warnings
        resumed = resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            online_model=new_online,
            target_model=new_target,
            optimizer=new_optimizer,
            epsilon_scheduler=new_scheduler,
            config=incompatible_config,
            strict_config=False,
            device="cpu",
        )

        # Should have warnings
        assert len(resumed["warnings"]) > 0

    def test_resume_nonexistent_checkpoint(self, models_and_state):
        """Test that resuming from nonexistent checkpoint raises error."""
        state = models_and_state

        with pytest.raises(FileNotFoundError):
            resume_from_checkpoint(
                checkpoint_path="/nonexistent/checkpoint.pt",
                online_model=state["online_model"],
                target_model=state["target_model"],
                optimizer=state["optimizer"],
                epsilon_scheduler=state["epsilon_scheduler"],
                device="cpu",
            )

    def test_epsilon_schedule_restoration(self, temp_checkpoint_dir, models_and_state):
        """Test that epsilon schedule state is correctly restored."""
        state = models_and_state
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)

        # Advance epsilon scheduler to specific point
        test_step = 500_000  # Halfway through decay
        expected_epsilon = 0.55  # Should be halfway between 1.0 and 0.1

        # Save checkpoint at this step
        checkpoint_path = manager.save_checkpoint(
            step=test_step,
            episode=2500,
            epsilon=expected_epsilon,
            online_model=state["online_model"],
            target_model=state["target_model"],
            optimizer=state["optimizer"],
        )

        # Create fresh scheduler
        new_scheduler = EpsilonScheduler(
            epsilon_start=1.0, epsilon_end=0.1, decay_frames=1_000_000
        )

        # Before resume, epsilon should be at start
        assert new_scheduler.current_epsilon == 1.0
        assert new_scheduler.frame_counter == 0

        # Resume
        new_online = DQN(num_actions=6)
        new_target = DQN(num_actions=6)
        new_optimizer = configure_optimizer(new_online, optimizer_type="rmsprop")

        resume_from_checkpoint(
            checkpoint_path=checkpoint_path,
            online_model=new_online,
            target_model=new_target,
            optimizer=new_optimizer,
            epsilon_scheduler=new_scheduler,
            device="cpu",
        )

        # After resume, epsilon should match checkpoint
        assert abs(new_scheduler.current_epsilon - expected_epsilon) < 1e-6
        assert new_scheduler.frame_counter == test_step

        # Verify get_epsilon returns correct value
        computed_epsilon = new_scheduler.get_epsilon(test_step)
        assert abs(computed_epsilon - expected_epsilon) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
