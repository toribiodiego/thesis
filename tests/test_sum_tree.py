"""
Tests for the SumTree data structure.

Verifies:
- Insert and retrieve values
- Update propagates to root correctly
- Sampling distribution matches priorities over many draws
- Edge cases: single element, full capacity, zero priority, all-equal
- Batch sampling produces valid indices
- Error handling for invalid inputs
"""

import numpy as np
import pytest

from src.replay.sum_tree import SumTree


# ---------------------------------------------------------------------------
# Insert and retrieve
# ---------------------------------------------------------------------------


class TestInsertRetrieve:
    """Updating a leaf stores the priority and it can be read back."""

    def test_set_and_get(self):
        tree = SumTree(4)
        tree.update(0, 1.0)
        tree.update(1, 2.0)
        tree.update(2, 3.0)
        tree.update(3, 4.0)

        assert tree.get(0) == 1.0
        assert tree.get(1) == 2.0
        assert tree.get(2) == 3.0
        assert tree.get(3) == 4.0

    def test_overwrite(self):
        """Updating the same leaf replaces its value."""
        tree = SumTree(4)
        tree.update(0, 5.0)
        assert tree.get(0) == 5.0

        tree.update(0, 2.0)
        assert tree.get(0) == 2.0

    def test_initial_values_are_zero(self):
        """All leaves start at zero priority."""
        tree = SumTree(8)
        for i in range(8):
            assert tree.get(i) == 0.0


# ---------------------------------------------------------------------------
# Propagation to root
# ---------------------------------------------------------------------------


class TestPropagation:
    """Internal nodes must equal the sum of their children."""

    def test_total_equals_sum_of_leaves(self):
        tree = SumTree(4)
        tree.update(0, 1.0)
        tree.update(1, 2.0)
        tree.update(2, 3.0)
        tree.update(3, 4.0)

        assert tree.total == pytest.approx(10.0)

    def test_total_after_update(self):
        """Updating a leaf changes the total correctly."""
        tree = SumTree(4)
        tree.update(0, 5.0)
        assert tree.total == pytest.approx(5.0)

        tree.update(1, 3.0)
        assert tree.total == pytest.approx(8.0)

        # Overwrite leaf 0
        tree.update(0, 1.0)
        assert tree.total == pytest.approx(4.0)

    def test_internal_nodes_are_child_sums(self):
        """Verify the tree structure from the docstring example."""
        tree = SumTree(4)
        tree.update(0, 3.0)
        tree.update(1, 3.0)
        tree.update(2, 1.0)
        tree.update(3, 3.0)

        # Internal nodes: tree[2] = leaf0 + leaf1, tree[3] = leaf2 + leaf3
        assert tree.tree[2] == pytest.approx(6.0)
        assert tree.tree[3] == pytest.approx(4.0)
        # Root: tree[1] = tree[2] + tree[3]
        assert tree.tree[1] == pytest.approx(10.0)

    def test_propagation_with_large_tree(self):
        """Total is correct for a larger tree."""
        tree = SumTree(64)
        values = np.random.rand(64) * 10
        for i, v in enumerate(values):
            tree.update(i, float(v))

        assert tree.total == pytest.approx(values.sum(), rel=1e-10)


# ---------------------------------------------------------------------------
# Sampling distribution
# ---------------------------------------------------------------------------


class TestSamplingDistribution:
    """Sampling frequency should match priority proportions."""

    def test_deterministic_sample_boundaries(self):
        """Known values map to expected leaves."""
        tree = SumTree(4)
        tree.update(0, 1.0)  # cumulative [0, 1)
        tree.update(1, 2.0)  # cumulative [1, 3)
        tree.update(2, 3.0)  # cumulative [3, 6)
        tree.update(3, 4.0)  # cumulative [6, 10)

        assert tree.sample(0.5) == 0
        assert tree.sample(1.5) == 1
        assert tree.sample(4.0) == 2
        assert tree.sample(7.0) == 3

    def test_boundary_value_goes_left(self):
        """When value equals left child sum, it goes left (<=)."""
        tree = SumTree(4)
        tree.update(0, 2.0)
        tree.update(1, 3.0)
        tree.update(2, 5.0)
        tree.update(3, 0.0)

        # value=2.0 equals left child of root's left subtree
        assert tree.sample(2.0) == 0

    def test_statistical_distribution(self):
        """Over many draws, sample frequencies match priorities."""
        tree = SumTree(4)
        priorities = [1.0, 2.0, 3.0, 4.0]
        for i, p in enumerate(priorities):
            tree.update(i, p)

        n_samples = 100_000
        counts = np.zeros(4)
        for _ in range(n_samples):
            value = np.random.uniform(0, tree.total)
            idx = tree.sample(value)
            counts[idx] += 1

        # Expected proportions
        total = sum(priorities)
        expected = np.array(priorities) / total
        observed = counts / n_samples

        # Allow 2% tolerance for statistical test
        np.testing.assert_allclose(observed, expected, atol=0.02)

    def test_batch_sample_returns_valid_indices(self):
        """batch_sample returns indices in [0, capacity)."""
        tree = SumTree(8)
        for i in range(8):
            tree.update(i, float(i + 1))

        indices = tree.batch_sample(32)
        assert indices.shape == (32,)
        assert np.all(indices >= 0)
        assert np.all(indices < 8)

    def test_batch_sample_stratified_coverage(self):
        """Stratified sampling covers all non-zero leaves."""
        tree = SumTree(4)
        tree.update(0, 1.0)
        tree.update(1, 1.0)
        tree.update(2, 1.0)
        tree.update(3, 1.0)

        # With 4 segments and equal priorities, stratified sampling
        # should hit each leaf exactly once
        np.random.seed(42)
        indices = tree.batch_sample(4)
        assert set(indices.tolist()) == {0, 1, 2, 3}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Corner cases that could cause bugs."""

    def test_single_element(self):
        """Tree with capacity=1 works correctly."""
        tree = SumTree(1)
        tree.update(0, 5.0)

        assert tree.total == pytest.approx(5.0)
        assert tree.get(0) == 5.0
        assert tree.sample(2.5) == 0

    def test_zero_priority(self):
        """Zero-priority leaves are never sampled when others are nonzero."""
        tree = SumTree(4)
        tree.update(0, 0.0)
        tree.update(1, 0.0)
        tree.update(2, 5.0)
        tree.update(3, 0.0)

        # All samples should go to leaf 2
        for _ in range(100):
            value = np.random.uniform(0, tree.total)
            assert tree.sample(value) == 2

    def test_all_equal_priorities(self):
        """Equal priorities produce uniform-like sampling."""
        tree = SumTree(8)
        for i in range(8):
            tree.update(i, 1.0)

        n_samples = 80_000
        counts = np.zeros(8)
        for _ in range(n_samples):
            value = np.random.uniform(0, tree.total)
            counts[tree.sample(value)] += 1

        # Each leaf should get ~12.5% of samples
        expected = n_samples / 8
        for c in counts:
            assert abs(c - expected) / expected < 0.05

    def test_full_capacity_overwrite(self):
        """Writing all leaves and then overwriting works."""
        tree = SumTree(4)
        for i in range(4):
            tree.update(i, float(i + 1))
        assert tree.total == pytest.approx(10.0)

        # Overwrite all with different values
        for i in range(4):
            tree.update(i, 1.0)
        assert tree.total == pytest.approx(4.0)

    def test_very_small_priorities(self):
        """Very small but nonzero priorities work without underflow."""
        tree = SumTree(4)
        tree.update(0, 1e-15)
        tree.update(1, 1e-15)
        tree.update(2, 1e-15)
        tree.update(3, 1e-15)

        assert tree.total > 0
        assert tree.total == pytest.approx(4e-15, rel=1e-10)

    def test_very_large_priorities(self):
        """Large priorities work without overflow."""
        tree = SumTree(4)
        tree.update(0, 1e15)
        tree.update(1, 1e15)

        assert tree.total == pytest.approx(2e15, rel=1e-10)


# ---------------------------------------------------------------------------
# Max priority tracking
# ---------------------------------------------------------------------------


class TestMaxPriority:
    """max_priority should track the highest leaf value seen."""

    def test_initial_max_priority(self):
        """Default max_priority is 1.0 (for new transitions)."""
        tree = SumTree(4)
        assert tree.max_priority == 1.0

    def test_max_tracks_updates(self):
        tree = SumTree(4)
        tree.update(0, 3.0)
        assert tree.max_priority == 3.0

        tree.update(1, 5.0)
        assert tree.max_priority == 5.0

        # Lowering a leaf does not lower max (max is monotonic for
        # the purpose of new-transition initialization)
        tree.update(1, 1.0)
        assert tree.max_priority == 5.0


# ---------------------------------------------------------------------------
# Count tracking
# ---------------------------------------------------------------------------


class TestCount:
    """count should track how many leaves have been written."""

    def test_initial_count(self):
        tree = SumTree(4)
        assert tree.count == 0

    def test_count_increments(self):
        tree = SumTree(4)
        tree.update(0, 1.0)
        assert tree.count == 1

        tree.update(1, 2.0)
        assert tree.count == 2

    def test_count_caps_at_capacity(self):
        tree = SumTree(4)
        for i in range(4):
            tree.update(i, 1.0)
        assert tree.count == 4

        # Further updates don't increase count beyond capacity
        tree.update(0, 2.0)
        assert tree.count == 4


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Invalid inputs should raise clear errors."""

    def test_negative_capacity(self):
        with pytest.raises(ValueError, match="positive"):
            SumTree(-1)

    def test_zero_capacity(self):
        with pytest.raises(ValueError, match="positive"):
            SumTree(0)

    def test_leaf_index_out_of_range(self):
        tree = SumTree(4)
        with pytest.raises(IndexError):
            tree.update(4, 1.0)
        with pytest.raises(IndexError):
            tree.update(-1, 1.0)
        with pytest.raises(IndexError):
            tree.get(4)

    def test_negative_priority(self):
        tree = SumTree(4)
        with pytest.raises(ValueError, match="non-negative"):
            tree.update(0, -1.0)
