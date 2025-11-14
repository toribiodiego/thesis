"""
Unit tests for checkpoint save/load functionality.

Tests complete checkpoint structure including:
- Online and target network weights
- Optimizer state
- Training counters (step, episode)
- Epsilon values
- Replay buffer state
- RNG states
- Metadata (schema, timestamp, commit hash)
- Atomic writes
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import shutil
from pathlib import Path

from src.models.dqn import DQN
from src.replay.replay_buffer import ReplayBuffer
from src.training import (
    CheckpointManager,
    get_rng_states,
    set_rng_states,
    verify_checkpoint_integrity,
    configure_optimizer
)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def models():
    """Create online and target models."""
    online_model = DQN(num_actions=6)
    target_model = DQN(num_actions=6)
    target_model.load_state_dict(online_model.state_dict())
    return online_model, target_model


@pytest.fixture
def optimizer(models):
    """Create optimizer."""
    online_model, _ = models
    return configure_optimizer(online_model, optimizer_type="rmsprop")


@pytest.fixture
def replay_buffer():
    """Create small replay buffer."""
    return ReplayBuffer(
        capacity=1000,
        obs_shape=(4, 84, 84),
        min_size=100
    )


@pytest.fixture
def filled_replay_buffer(replay_buffer):
    """Create replay buffer with some data."""
    # Add some dummy transitions
    for i in range(200):
        state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        action = np.random.randint(0, 6)
        reward = np.random.randn()
        next_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
        done = i % 50 == 49  # Episode ends every 50 steps

        replay_buffer.append(state, action, reward, next_state, done)

    return replay_buffer


class TestCheckpointManager:
    """Test CheckpointManager functionality."""

    def test_initialization(self, temp_checkpoint_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            save_interval=1_000_000,
            keep_last_n=3,
            save_best=True
        )

        assert manager.checkpoint_dir == temp_checkpoint_dir
        assert manager.save_interval == 1_000_000
        assert manager.keep_last_n == 3
        assert manager.save_best_enabled is True
        assert os.path.exists(temp_checkpoint_dir)

    def test_should_save(self, temp_checkpoint_dir):
        """Test should_save logic."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            save_interval=1_000_000
        )

        assert not manager.should_save(0)
        assert not manager.should_save(500_000)
        assert manager.should_save(1_000_000)
        assert manager.should_save(2_000_000)
        assert not manager.should_save(1_500_000)

    def test_save_checkpoint_basic(self, temp_checkpoint_dir, models, optimizer):
        """Test basic checkpoint saving."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        online_model, target_model = models

        # Save checkpoint
        checkpoint_path = manager.save_checkpoint(
            step=1000,
            episode=50,
            epsilon=0.95,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer
        )

        # Verify file exists
        assert os.path.exists(checkpoint_path)
        assert checkpoint_path.endswith('checkpoint_1000.pt')

        # Verify checkpoint can be loaded
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Check schema and metadata
        assert checkpoint['schema_version'] == CheckpointManager.SCHEMA_VERSION
        assert 'timestamp' in checkpoint
        assert 'commit_hash' in checkpoint

        # Check training state
        assert checkpoint['step'] == 1000
        assert checkpoint['episode'] == 50
        assert checkpoint['epsilon'] == 0.95

        # Check model and optimizer states
        assert 'online_model_state_dict' in checkpoint
        assert 'target_model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint

    def test_save_checkpoint_with_rng_states(
        self, temp_checkpoint_dir, models, optimizer
    ):
        """Test checkpoint saving with RNG states."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        online_model, target_model = models

        # Capture RNG states
        rng_states = get_rng_states()

        # Save checkpoint with RNG states
        checkpoint_path = manager.save_checkpoint(
            step=2000,
            episode=100,
            epsilon=0.9,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
            rng_states=rng_states
        )

        # Load and verify
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert 'rng_states' in checkpoint
        assert 'python_random' in checkpoint['rng_states']
        assert 'numpy_random' in checkpoint['rng_states']
        assert 'torch_cpu' in checkpoint['rng_states']

    def test_save_checkpoint_with_replay_buffer(
        self, temp_checkpoint_dir, models, optimizer, filled_replay_buffer
    ):
        """Test checkpoint saving with replay buffer state."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            save_replay_buffer=False  # Save only index/size
        )
        online_model, target_model = models

        checkpoint_path = manager.save_checkpoint(
            step=3000,
            episode=150,
            epsilon=0.85,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
            replay_buffer=filled_replay_buffer
        )

        # Load and verify
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        assert 'replay_buffer_state' in checkpoint

        buffer_state = checkpoint['replay_buffer_state']
        assert buffer_state['index'] == filled_replay_buffer.index
        assert buffer_state['size'] == filled_replay_buffer.size
        assert buffer_state['capacity'] == filled_replay_buffer.capacity
        assert buffer_state['obs_shape'] == filled_replay_buffer.obs_shape

        # Should not have full data (save_replay_buffer=False)
        assert 'data' not in buffer_state

    def test_save_checkpoint_with_full_replay_buffer(
        self, temp_checkpoint_dir, models, optimizer, filled_replay_buffer
    ):
        """Test checkpoint saving with full replay buffer data."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            save_replay_buffer=True  # Save full buffer
        )
        online_model, target_model = models

        checkpoint_path = manager.save_checkpoint(
            step=4000,
            episode=200,
            epsilon=0.8,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
            replay_buffer=filled_replay_buffer
        )

        # Load and verify
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        buffer_state = checkpoint['replay_buffer_state']

        # Should have full data
        assert 'data' in buffer_state
        assert 'observations' in buffer_state['data']
        assert 'actions' in buffer_state['data']
        assert 'rewards' in buffer_state['data']
        assert 'dones' in buffer_state['data']
        assert 'episode_starts' in buffer_state['data']

    def test_atomic_write(self, temp_checkpoint_dir, models, optimizer):
        """Test that checkpoint uses atomic write (no .tmp files left)."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        online_model, target_model = models

        checkpoint_path = manager.save_checkpoint(
            step=5000,
            episode=250,
            epsilon=0.75,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer
        )

        # Check no temporary files remain
        temp_files = list(Path(temp_checkpoint_dir).glob('*.tmp'))
        assert len(temp_files) == 0

        # Final checkpoint should exist
        assert os.path.exists(checkpoint_path)

    def test_load_checkpoint(self, temp_checkpoint_dir, models, optimizer):
        """Test checkpoint loading and state restoration."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        online_model, target_model = models

        # Save checkpoint
        original_step = 6000
        original_episode = 300
        original_epsilon = 0.7

        checkpoint_path = manager.save_checkpoint(
            step=original_step,
            episode=original_episode,
            epsilon=original_epsilon,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
            rng_states=get_rng_states()
        )

        # Create new models and optimizer
        new_online = DQN(num_actions=6)
        new_target = DQN(num_actions=6)
        new_optimizer = configure_optimizer(new_online, optimizer_type="rmsprop")

        # Load checkpoint
        loaded_state = manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            online_model=new_online,
            target_model=new_target,
            optimizer=new_optimizer
        )

        # Verify loaded state
        assert loaded_state['step'] == original_step
        assert loaded_state['episode'] == original_episode
        assert loaded_state['epsilon'] == original_epsilon
        assert 'rng_states' in loaded_state
        assert 'commit_hash' in loaded_state
        assert 'timestamp' in loaded_state

        # Verify model weights match
        for p1, p2 in zip(online_model.parameters(), new_online.parameters()):
            assert torch.allclose(p1, p2)

        for p1, p2 in zip(target_model.parameters(), new_target.parameters()):
            assert torch.allclose(p1, p2)

    def test_load_checkpoint_with_replay_buffer(
        self, temp_checkpoint_dir, models, optimizer, filled_replay_buffer
    ):
        """Test loading checkpoint with replay buffer restoration."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            save_replay_buffer=True
        )
        online_model, target_model = models

        # Save with replay buffer
        checkpoint_path = manager.save_checkpoint(
            step=7000,
            episode=350,
            epsilon=0.65,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
            replay_buffer=filled_replay_buffer
        )

        # Create new models and empty buffer
        new_online = DQN(num_actions=6)
        new_target = DQN(num_actions=6)
        new_optimizer = configure_optimizer(new_online, optimizer_type="rmsprop")
        new_buffer = ReplayBuffer(capacity=1000, obs_shape=(4, 84, 84))

        # Load checkpoint with buffer restoration
        loaded_state = manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            online_model=new_online,
            target_model=new_target,
            optimizer=new_optimizer,
            replay_buffer=new_buffer
        )

        # Verify buffer state restored
        assert new_buffer.index == filled_replay_buffer.index
        assert new_buffer.size == filled_replay_buffer.size

        # Verify buffer data restored
        assert np.array_equal(new_buffer.observations, filled_replay_buffer.observations)
        assert np.array_equal(new_buffer.actions, filled_replay_buffer.actions)
        assert np.array_equal(new_buffer.rewards, filled_replay_buffer.rewards)
        assert np.array_equal(new_buffer.dones, filled_replay_buffer.dones)

    def test_keep_last_n_checkpoints(self, temp_checkpoint_dir, models, optimizer):
        """Test that old checkpoints are cleaned up."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            keep_last_n=3
        )
        online_model, target_model = models

        # Save 5 checkpoints
        for i in range(5):
            manager.save_checkpoint(
                step=(i + 1) * 1000,
                episode=i * 50,
                epsilon=1.0 - i * 0.1,
                online_model=online_model,
                target_model=target_model,
                optimizer=optimizer
            )

        # Should only have 3 most recent checkpoints
        checkpoints = list(Path(temp_checkpoint_dir).glob('checkpoint_*.pt'))
        assert len(checkpoints) == 3

        # Verify it's the last 3
        assert Path(os.path.join(temp_checkpoint_dir, 'checkpoint_3000.pt')).exists()
        assert Path(os.path.join(temp_checkpoint_dir, 'checkpoint_4000.pt')).exists()
        assert Path(os.path.join(temp_checkpoint_dir, 'checkpoint_5000.pt')).exists()
        assert not Path(os.path.join(temp_checkpoint_dir, 'checkpoint_1000.pt')).exists()
        assert not Path(os.path.join(temp_checkpoint_dir, 'checkpoint_2000.pt')).exists()

    def test_save_best_model(self, temp_checkpoint_dir, models, optimizer):
        """Test best model tracking and saving."""
        manager = CheckpointManager(
            checkpoint_dir=temp_checkpoint_dir,
            save_best=True
        )
        online_model, target_model = models

        # Save first best model
        saved = manager.save_best(
            step=1000,
            episode=50,
            epsilon=0.95,
            eval_return=10.0,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer
        )

        assert saved is True
        assert manager.best_eval_return == 10.0
        assert os.path.exists(os.path.join(temp_checkpoint_dir, 'best_model.pt'))

        # Try to save worse model
        saved = manager.save_best(
            step=2000,
            episode=100,
            epsilon=0.9,
            eval_return=5.0,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer
        )

        assert saved is False
        assert manager.best_eval_return == 10.0

        # Save new best model
        saved = manager.save_best(
            step=3000,
            episode=150,
            epsilon=0.85,
            eval_return=15.0,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer
        )

        assert saved is True
        assert manager.best_eval_return == 15.0

        # Verify best model checkpoint has eval_return
        checkpoint = torch.load(
            os.path.join(temp_checkpoint_dir, 'best_model.pt'),
            map_location='cpu'
        )
        assert checkpoint['eval_return'] == 15.0


class TestRNGStates:
    """Test RNG state capture and restoration."""

    def test_get_rng_states(self):
        """Test capturing RNG states."""
        rng_states = get_rng_states()

        assert 'python_random' in rng_states
        assert 'numpy_random' in rng_states
        assert 'torch_cpu' in rng_states

        # CUDA states only if available
        if torch.cuda.is_available():
            assert 'torch_cuda' in rng_states

    def test_set_rng_states_reproducibility(self):
        """Test that setting RNG states gives reproducible results."""
        # Set initial seed
        torch.manual_seed(42)
        np.random.seed(42)

        # Capture state
        rng_states = get_rng_states()

        # Generate some random numbers
        torch_rand_1 = torch.rand(10)
        numpy_rand_1 = np.random.rand(10)

        # Restore state
        set_rng_states(rng_states)

        # Generate again - should be identical
        torch_rand_2 = torch.rand(10)
        numpy_rand_2 = np.random.rand(10)

        assert torch.allclose(torch_rand_1, torch_rand_2)
        assert np.allclose(numpy_rand_1, numpy_rand_2)


class TestCheckpointIntegrity:
    """Test checkpoint integrity verification."""

    def test_verify_valid_checkpoint(self, temp_checkpoint_dir, models, optimizer):
        """Test verification of valid checkpoint."""
        manager = CheckpointManager(checkpoint_dir=temp_checkpoint_dir)
        online_model, target_model = models

        checkpoint_path = manager.save_checkpoint(
            step=1000,
            episode=50,
            epsilon=0.95,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer
        )

        # Should verify successfully
        assert verify_checkpoint_integrity(checkpoint_path) is True

    def test_verify_invalid_checkpoint(self, temp_checkpoint_dir):
        """Test verification of corrupted checkpoint."""
        # Create invalid checkpoint file
        invalid_path = os.path.join(temp_checkpoint_dir, 'invalid.pt')
        with open(invalid_path, 'w') as f:
            f.write("not a valid checkpoint")

        # Should fail verification
        assert verify_checkpoint_integrity(invalid_path) is False

    def test_verify_incomplete_checkpoint(self, temp_checkpoint_dir):
        """Test verification of incomplete checkpoint (missing fields)."""
        incomplete_path = os.path.join(temp_checkpoint_dir, 'incomplete.pt')

        # Save checkpoint with missing required fields
        incomplete_data = {
            'step': 1000,
            'epsilon': 0.95,
            # Missing: episode, model states, optimizer state, etc.
        }

        torch.save(incomplete_data, incomplete_path)

        # Should fail verification
        assert verify_checkpoint_integrity(incomplete_path) is False


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
