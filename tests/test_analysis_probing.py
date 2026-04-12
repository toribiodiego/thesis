"""Tests for the linear probe training function.

Uses synthetic data only -- no checkpoints, no GPU, no Atari.
"""

import numpy as np
import pytest

from src.analysis.probing import (
    ProbeResult,
    _normalized_entropy,
    train_probe,
    train_probes_multi,
)


# ---------------------------------------------------------------------------
# Entropy helper
# ---------------------------------------------------------------------------


class TestNormalizedEntropy:

    def test_uniform_distribution(self):
        labels = np.array([0, 1, 2, 3] * 25)
        ent = _normalized_entropy(labels)
        assert ent == pytest.approx(1.0, abs=1e-6)

    def test_constant_labels(self):
        labels = np.array([5] * 100)
        assert _normalized_entropy(labels) == 0.0

    def test_skewed_distribution(self):
        labels = np.array([0] * 95 + [1] * 5)
        ent = _normalized_entropy(labels)
        assert 0 < ent < 1.0


# ---------------------------------------------------------------------------
# Single probe training
# ---------------------------------------------------------------------------


class TestTrainProbe:

    def test_separable_data(self):
        """Perfectly separable data should yield high F1."""
        rng = np.random.RandomState(42)
        n = 500
        d = 10
        X = rng.randn(n, d).astype(np.float32)
        # Labels determined by sign of first feature
        y = (X[:, 0] > 0).astype(np.int32)

        result = train_probe(X, y, variable_name="test_var")

        assert isinstance(result, ProbeResult)
        assert not result.skipped
        assert result.f1_test > 0.9
        assert result.variable == "test_var"
        assert result.n_classes == 2

    def test_random_data_low_f1(self):
        """Random labels on random features should yield low F1."""
        rng = np.random.RandomState(42)
        n = 500
        d = 10
        X = rng.randn(n, d).astype(np.float32)
        y = rng.randint(0, 10, size=n)

        result = train_probe(X, y)
        # 10-class random: macro F1 should be low
        assert result.f1_test < 0.5

    def test_constant_labels_skipped(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        y = np.zeros(100, dtype=np.int32)

        result = train_probe(X, y)
        assert result.skipped
        assert result.skip_reason == "constant value"

    def test_low_entropy_skipped(self):
        """Label distribution dominated by one class should be skipped."""
        rng = np.random.RandomState(42)
        X = rng.randn(1000, 5).astype(np.float32)
        # 99% class 0, 1% class 1 -- very low entropy
        y = np.zeros(1000, dtype=np.int32)
        y[:10] = 1

        result = train_probe(X, y, entropy_threshold=0.6)
        assert result.skipped
        assert "low entropy" in result.skip_reason

    def test_seed_deterministic(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float32)
        y = rng.randint(0, 3, size=200)

        r1 = train_probe(X, y, seed=13)
        r2 = train_probe(X, y, seed=13)
        assert r1.f1_test == r2.f1_test

    def test_different_seeds_differ(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float32)
        y = rng.randint(0, 3, size=200)

        r1 = train_probe(X, y, seed=13)
        r2 = train_probe(X, y, seed=99)
        # Different splits should (usually) give different F1
        # Not guaranteed, but very likely with these parameters
        assert r1.f1_test != r2.f1_test


# ---------------------------------------------------------------------------
# Multi-variable probing
# ---------------------------------------------------------------------------


class TestTrainProbesMulti:

    def test_returns_one_result_per_variable(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5).astype(np.float32)
        labels_dict = {
            "var_a": rng.randint(0, 3, size=200),
            "var_b": rng.randint(0, 5, size=200),
            "var_c": np.zeros(200, dtype=np.int32),  # constant, will be skipped
        }

        results = train_probes_multi(X, labels_dict)

        assert len(results) == 3
        names = [r.variable for r in results]
        assert names == ["var_a", "var_b", "var_c"]  # sorted
        assert not results[0].skipped
        assert not results[1].skipped
        assert results[2].skipped
