"""Tests for CKA similarity analysis.

Tests the linear CKA computation and the observation collection
helper. Uses synthetic data only -- no checkpoints or GPU.
"""

import numpy as np
import pytest

# Import the CKA function directly from the script module
import importlib.util
import os
import sys

_script_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "analysis", "run_cka.py",
)
_spec = importlib.util.spec_from_file_location("run_cka", _script_path)
_mod = importlib.util.module_from_spec(_spec)
# Prevent the script from running main() on import
sys.modules["run_cka"] = _mod
_spec.loader.exec_module(_mod)

linear_cka = _mod.linear_cka
_get_observations = _mod._get_observations


class TestLinearCKA:

    def test_identical_representations(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-6)

    def test_independent_representations(self):
        # Independent random features should have low CKA
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10).astype(np.float32)
        Y = rng.randn(200, 10).astype(np.float32)
        cka = linear_cka(X, Y)
        assert cka < 0.1

    def test_symmetry(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        Y = rng.randn(100, 8).astype(np.float32)
        assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-6)

    def test_range_zero_to_one(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 10).astype(np.float32)
        Y = rng.randn(100, 10).astype(np.float32)
        cka = linear_cka(X, Y)
        assert 0.0 <= cka <= 1.0


class TestGetObservationsReplay:
    """Regression test for the _get_observations replay source bug
    (fixed in commit 4ad194d). The function previously accessed
    ckpt._run_dir which does not exist on CheckpointData."""

    def test_replay_source_uses_run_dir_arg(self, tmp_path):
        """Verify _get_observations with source='replay' uses the
        explicit run_dir/step args, not checkpoint attributes."""
        # Create a minimal replay buffer npz
        n = 100
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()

        np.savez(
            ckpt_dir / "replay_buffer_1000.npz",
            observation=np.random.randint(0, 255, (n, 1, 84, 84), dtype=np.uint8),
            action=np.zeros((n, 1), dtype=np.int32),
            reward=np.zeros((n, 1), dtype=np.float32),
            terminal=np.zeros((n, 1), dtype=np.uint8),
        )

        # Write add_count
        import gzip
        with gzip.open(str(ckpt_dir / "add_count_ckpt.1000.gz"), "wb") as f:
            np.save(f, np.array(n))

        # Should not raise AttributeError on ckpt._run_dir
        obs = _get_observations(
            game=None, num_steps=100, ckpt=None, seed=0,
            source="replay", run_dir=str(tmp_path), step=1000,
        )
        # With 100 frames and no terminals, we get 97 valid stacks
        # (need indices 3..99, checking terms[i-3:i])
        assert obs.shape[0] == 97
        assert obs.shape[1:] == (84, 84, 4)
