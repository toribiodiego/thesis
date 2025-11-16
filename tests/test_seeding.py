"""
Unit tests for deterministic seeding utilities.

Tests that set_seed() properly seeds:
- Python random module
- NumPy random
- PyTorch CPU random
- PyTorch CUDA random (if available)
- Environment seeding
- Deterministic flags
- Multiprocessing worker isolation
"""

import random
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src.utils import seed_env, set_seed


class TestSetSeed:
    """Test set_seed() function for reproducibility."""

    def test_python_random_seeding(self):
        """Test that Python random module is properly seeded."""
        # Seed with value 42
        set_seed(42)
        values1 = [random.random() for _ in range(10)]

        # Seed again with same value
        set_seed(42)
        values2 = [random.random() for _ in range(10)]

        # Should generate identical sequence
        assert values1 == values2

    def test_numpy_random_seeding(self):
        """Test that NumPy random is properly seeded."""
        # Seed with value 42
        set_seed(42)
        values1 = np.random.rand(10)

        # Seed again with same value
        set_seed(42)
        values2 = np.random.rand(10)

        # Should generate identical sequence
        np.testing.assert_array_equal(values1, values2)

    def test_torch_cpu_seeding(self):
        """Test that PyTorch CPU random is properly seeded."""
        # Seed with value 42
        set_seed(42)
        values1 = torch.rand(10)

        # Seed again with same value
        set_seed(42)
        values2 = torch.rand(10)

        # Should generate identical sequence
        assert torch.allclose(values1, values2)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_torch_cuda_seeding(self):
        """Test that PyTorch CUDA random is properly seeded."""
        # Seed with value 42
        set_seed(42)
        values1 = torch.rand(10, device="cuda")

        # Seed again with same value
        set_seed(42)
        values2 = torch.rand(10, device="cuda")

        # Should generate identical sequence
        assert torch.allclose(values1, values2)

    def test_different_seeds_produce_different_values(self):
        """Test that different seeds produce different random values."""
        # Seed with value 42
        set_seed(42)
        values1 = torch.rand(10)

        # Seed with different value
        set_seed(123)
        values2 = torch.rand(10)

        # Should be different
        assert not torch.allclose(values1, values2)

    def test_deterministic_flag_sets_cudnn(self):
        """Test that deterministic=True sets cudnn flags."""
        # Reset flags first
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # Call with deterministic=True
        set_seed(42, deterministic=True)

        # Flags should be set
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_non_deterministic_leaves_cudnn_unchanged(self):
        """Test that deterministic=False doesn't modify cudnn flags."""
        # Set flags to known state
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # Call with deterministic=False (default)
        set_seed(42, deterministic=False)

        # Flags should remain unchanged
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True

    def test_env_seeding(self):
        """Test that environment is seeded when provided."""
        # Create mock environment
        mock_env = Mock()
        mock_env.reset = Mock(return_value=("obs", {"info": "dict"}))

        # Call set_seed with env
        result = set_seed(42, env=mock_env)

        # Verify env.reset was called with seed
        mock_env.reset.assert_called_once_with(seed=42)

        # Verify return value is from env.reset
        assert result == ("obs", {"info": "dict"})

    def test_without_env_returns_none(self):
        """Test that set_seed returns None when no env provided."""
        result = set_seed(42)
        assert result is None

    def test_multiprocessing_worker_isolation(self):
        """Test that different worker seeds produce isolated sequences."""
        # Simulate worker 0
        set_seed(42 + 0)
        worker0_values = torch.rand(5)

        # Simulate worker 1
        set_seed(42 + 1)
        worker1_values = torch.rand(5)

        # Should produce different values
        assert not torch.allclose(worker0_values, worker1_values)

        # Verify worker 0 can be reproduced
        set_seed(42 + 0)
        worker0_repro = torch.rand(5)
        assert torch.allclose(worker0_values, worker0_repro)


class TestSeedEnv:
    """Test seed_env() convenience function."""

    def test_seed_env_calls_reset_with_seed(self):
        """Test that seed_env calls env.reset with correct seed."""
        mock_env = Mock()
        mock_env.reset = Mock(return_value=("obs", {"info": "dict"}))

        result = seed_env(mock_env, 42)

        mock_env.reset.assert_called_once_with(seed=42)
        assert result == ("obs", {"info": "dict"})

    def test_seed_env_multiple_episodes(self):
        """Test seeding environment across multiple episodes."""
        mock_env = Mock()
        mock_env.reset = Mock(
            side_effect=[(f"obs_{i}", {"episode": i}) for i in range(5)]
        )

        base_seed = 100
        results = []
        for episode in range(5):
            obs, info = seed_env(mock_env, base_seed + episode)
            results.append((obs, info))

        # Verify correct seeds were used
        [((100 + i,),) for i in range(5)]
        actual_calls = [call[1] for call in mock_env.reset.call_args_list]

        # Extract just the keyword arguments (seed=...)
        actual_seeds = [call["seed"] for call in actual_calls]
        expected_seeds = [100 + i for i in range(5)]

        assert actual_seeds == expected_seeds


class TestEndToEndSeeding:
    """Test end-to-end seeding scenarios."""

    def test_full_training_initialization(self):
        """Test complete seeding setup for training start."""
        # Simulate training initialization
        base_seed = 42
        set_seed(base_seed, deterministic=True)

        # Generate some random values
        py_val = random.random()
        np_val = np.random.rand()
        torch_val = torch.rand(1).item()

        # Reset and verify reproducibility
        set_seed(base_seed, deterministic=True)

        py_val2 = random.random()
        np_val2 = np.random.rand()
        torch_val2 = torch.rand(1).item()

        assert py_val == py_val2
        assert np_val == np_val2
        assert torch_val == torch_val2

    def test_episode_specific_seeding(self):
        """Test seeding pattern for episode-specific randomness."""
        base_seed = 42
        episodes = []

        # Simulate 3 episodes with episode-specific seeds
        for episode in range(3):
            # Seed for this episode
            set_seed(base_seed + episode)

            # Generate episode data
            episode_data = {
                "episode": episode,
                "random_val": random.random(),
                "np_val": np.random.rand(),
                "torch_val": torch.rand(1).item(),
            }
            episodes.append(episode_data)

        # Reproduce episode 1
        set_seed(base_seed + 1)
        repro_random = random.random()
        repro_np = np.random.rand()
        repro_torch = torch.rand(1).item()

        # Should match episode 1 exactly
        assert repro_random == episodes[1]["random_val"]
        assert repro_np == episodes[1]["np_val"]
        assert repro_torch == episodes[1]["torch_val"]

    def test_resume_from_checkpoint_seeding(self):
        """Test that seeding works correctly when resuming from checkpoint."""
        # Initial run - save RNG states after seeding
        set_seed(42, deterministic=True)

        # Generate some values
        [random.random() for _ in range(5)]

        # Capture current RNG state (simulating checkpoint save)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()

        # Generate more values (simulating continued training)
        post_values = [random.random() for _ in range(5)]

        # Simulate resume - restore RNG states
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)

        # Generate values after resume
        resumed_values = [random.random() for _ in range(5)]

        # Should match post_values exactly
        assert resumed_values == post_values

    def test_seed_propagation_metadata(self):
        """Test that seed is properly recorded in metadata."""
        import json
        import shutil
        import tempfile

        from src.utils import save_run_metadata

        # Create temp directory
        temp_dir = tempfile.mkdtemp()

        try:
            # Save metadata with seed
            save_run_metadata(
                output_dir=temp_dir, config={"agent": {"gamma": 0.99}}, seed=42
            )

            # Load and verify
            meta_path = f"{temp_dir}/meta.json"
            with open(meta_path, "r") as f:
                metadata = json.load(f)

            assert "seed" in metadata
            assert metadata["seed"] == 42

        finally:
            # Clean up
            shutil.rmtree(temp_dir)


class TestDeterministicBehavior:
    """Test deterministic behavior with flags enabled."""

    def test_cudnn_deterministic_reproducibility(self):
        """Test that deterministic mode produces consistent results."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # First run with deterministic mode
        set_seed(42, deterministic=True)
        x = torch.randn(10, 10, device="cuda")
        y = torch.randn(10, 10, device="cuda")
        result1 = torch.matmul(x, y)

        # Second run with same seed
        set_seed(42, deterministic=True)
        x = torch.randn(10, 10, device="cuda")
        y = torch.randn(10, 10, device="cuda")
        result2 = torch.matmul(x, y)

        # Should be identical
        assert torch.allclose(result1, result2, rtol=1e-6, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
