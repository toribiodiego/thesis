"""
Tests for the PrioritizedReplayBuffer.

Verifies:
- New transitions receive max priority
- High-priority transitions sampled proportionally more often
- IS weights match hand-computed values
- update_priorities changes sampling distribution
- Beta annealing from beta_start to beta_end
- can_sample checks min_size and tree total
- sample_sequences uses uniform sampling (for SPR)
- Interface compatibility with ReplayBuffer
"""

import numpy as np
import pytest

from src.replay import PrioritizedReplayBuffer, ReplayBuffer


# Small observation shape for fast tests
OBS_SHAPE = (1, 2, 2)


def _make_obs(value: int) -> np.ndarray:
    """Create a small observation filled with a single value."""
    return np.full(OBS_SHAPE, value, dtype=np.uint8)


def _fill_buffer(buf, n, start_value=0):
    """Fill buffer with n non-terminal transitions."""
    for i in range(n):
        buf.append(
            _make_obs(start_value + i),
            i % 4,
            float(i),
            _make_obs(start_value + i + 1),
            False,
        )


# ---------------------------------------------------------------------------
# New item priority
# ---------------------------------------------------------------------------


class TestNewItemPriority:
    """New transitions should receive the current max priority."""

    def test_first_item_gets_default_priority(self):
        """First item gets SumTree default max_priority=1.0."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1, epsilon=0.0,
        )
        buf.append(_make_obs(1), 0, 1.0, _make_obs(2), False)

        assert buf.tree.get(0) == 1.0
        assert buf.tree.total == 1.0

    def test_subsequent_items_get_max_priority(self):
        """All new items receive the current max_priority."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1, epsilon=0.0,
        )
        _fill_buffer(buf, 4)

        for i in range(4):
            assert buf.tree.get(i) == 1.0
        assert buf.tree.total == 4.0

    def test_new_items_after_update_get_new_max(self):
        """After update_priorities raises max, new items get that max."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0,
        )
        _fill_buffer(buf, 2)

        # stored = (5.0 + 0)^1.0 = 5.0 -> new max_priority
        buf.update_priorities(np.array([0]), np.array([5.0]))
        assert buf.tree.max_priority == 5.0

        # Next append should get priority 5.0
        buf.append(_make_obs(10), 0, 1.0, _make_obs(11), False)
        assert buf.tree.get(2) == 5.0


# ---------------------------------------------------------------------------
# Priority-proportional sampling
# ---------------------------------------------------------------------------


class TestPrioritySampling:
    """High-priority items should be sampled proportionally more."""

    def test_high_priority_sampled_more_often(self):
        """Item with 10x priority sampled roughly 10x more often."""
        buf = PrioritizedReplayBuffer(
            capacity=16, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0, beta_start=1.0,
        )
        _fill_buffer(buf, 2)

        # stored priorities: item 0 = 1.0, item 1 = 10.0
        buf.update_priorities(np.array([0, 1]), np.array([1.0, 10.0]))

        np.random.seed(42)
        counts = np.zeros(2)
        for _ in range(5000):
            batch = buf.sample(1, beta=1.0)
            counts[batch["indices"][0]] += 1

        ratio = counts[1] / max(counts[0], 1)
        assert 5.0 < ratio < 20.0, f"Expected ~10x ratio, got {ratio:.1f}"

    def test_update_priorities_shifts_distribution(self):
        """After updating priorities, the favored item changes."""
        buf = PrioritizedReplayBuffer(
            capacity=16, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0, beta_start=1.0,
        )
        _fill_buffer(buf, 4)
        # All equal -> roughly uniform
        np.random.seed(42)
        counts_before = np.zeros(4)
        for _ in range(2000):
            batch = buf.sample(1, beta=1.0)
            counts_before[batch["indices"][0]] += 1
        for c in counts_before:
            assert c > 200, f"Uniform sampling should hit all items: {counts_before}"

        # Make item 3 dominant
        buf.update_priorities(
            np.array([0, 1, 2, 3]),
            np.array([1.0, 1.0, 1.0, 100.0]),
        )
        np.random.seed(42)
        counts_after = np.zeros(4)
        for _ in range(2000):
            batch = buf.sample(1, beta=1.0)
            counts_after[batch["indices"][0]] += 1

        assert counts_after[3] > counts_after[0] * 5, (
            f"Item 3 should dominate after update: {counts_after}"
        )


# ---------------------------------------------------------------------------
# IS weight computation
# ---------------------------------------------------------------------------


class TestISWeights:
    """Importance-sampling weights must match the formula."""

    def _make_buffer_with_known_priorities(self):
        """Create buffer with 4 items and stored priorities [1, 2, 3, 4].

        Uses alpha=0.5, epsilon=0. Raw priorities [1, 4, 9, 16] yield
        stored = |p|^0.5 = [1, 2, 3, 4].  Total = 10, N = 4.
        """
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            alpha=0.5, epsilon=0.0, beta_start=1.0,
        )
        _fill_buffer(buf, 4)
        buf.update_priorities(
            np.array([0, 1, 2, 3]),
            np.array([1.0, 4.0, 9.0, 16.0]),
        )
        return buf

    def test_weights_beta_1(self):
        """IS weights with beta=1.0 match hand-computed values.

        P(i) = stored_i / 10, w_i = (4 * P(i))^{-1} / max(w)
        Expected normalized: {0: 1.0, 1: 0.5, 2: 1/3, 3: 0.25}
        """
        buf = self._make_buffer_with_known_priorities()
        expected = {0: 1.0, 1: 0.5, 2: 1.0 / 3.0, 3: 0.25}

        np.random.seed(42)
        batch = buf.sample(4, beta=1.0)

        for i, idx in enumerate(batch["indices"]):
            assert abs(float(batch["weights"][i]) - expected[idx]) < 1e-5, (
                f"idx={idx}: expected {expected[idx]:.4f}, "
                f"got {float(batch['weights'][i]):.4f}"
            )

    def test_weights_beta_half(self):
        """IS weights with beta=0.5 match hand-computed values.

        w_i = (4 * P(i))^{-0.5} / max(w)
        Expected normalized: {0: 1.0, 1: 1/sqrt(2), 2: 1/sqrt(3), 3: 0.5}
        """
        buf = self._make_buffer_with_known_priorities()
        expected = {
            0: 1.0,
            1: 1.0 / np.sqrt(2),
            2: 1.0 / np.sqrt(3),
            3: 0.5,
        }

        np.random.seed(42)
        batch = buf.sample(4, beta=0.5)

        for i, idx in enumerate(batch["indices"]):
            assert abs(float(batch["weights"][i]) - expected[idx]) < 1e-4, (
                f"idx={idx}: expected {expected[idx]:.4f}, "
                f"got {float(batch['weights'][i]):.4f}"
            )

    def test_equal_priorities_give_uniform_weights(self):
        """When all priorities are equal, all IS weights should be 1.0."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0, beta_start=1.0,
        )
        _fill_buffer(buf, 4)

        np.random.seed(42)
        batch = buf.sample(4, beta=1.0)
        np.testing.assert_allclose(batch["weights"], 1.0, atol=1e-6)

    def test_max_weight_is_one(self):
        """Normalized IS weights always have max = 1.0."""
        buf = PrioritizedReplayBuffer(
            capacity=16, obs_shape=OBS_SHAPE, min_size=1,
            alpha=0.5, epsilon=1e-6,
        )
        _fill_buffer(buf, 8)
        np.random.seed(0)
        buf.update_priorities(
            np.arange(8), np.random.uniform(0.1, 10.0, size=8),
        )

        batch = buf.sample(8, beta=0.7)
        assert abs(float(batch["weights"].max()) - 1.0) < 1e-6

    def test_weights_dtype_float32(self):
        """IS weights should be float32."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
        )
        _fill_buffer(buf, 4)

        batch = buf.sample(2, beta=1.0)
        assert batch["weights"].dtype == np.float32

    def test_beta_zero_gives_uniform_weights(self):
        """beta=0 means no IS correction: all weights should be 1.0."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0,
        )
        _fill_buffer(buf, 4)
        buf.update_priorities(
            np.array([0, 1, 2, 3]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        )

        np.random.seed(42)
        batch = buf.sample(4, beta=0.0)
        # (N * P(i))^0 = 1 for all i -> normalized = 1
        np.testing.assert_allclose(batch["weights"], 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Beta annealing
# ---------------------------------------------------------------------------


class TestBetaAnnealing:
    """Beta should anneal linearly from beta_start to beta_end."""

    def test_beta_at_frame_zero(self):
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            beta_start=0.4, beta_end=1.0, beta_frames=100_000,
        )
        assert buf.compute_beta(0) == 0.4

    def test_beta_at_end(self):
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            beta_start=0.4, beta_end=1.0, beta_frames=100_000,
        )
        assert abs(buf.compute_beta(100_000) - 1.0) < 1e-10

    def test_beta_at_midpoint(self):
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            beta_start=0.4, beta_end=1.0, beta_frames=100_000,
        )
        assert abs(buf.compute_beta(50_000) - 0.7) < 1e-10

    def test_beta_clamped_after_end(self):
        """Beta should not exceed beta_end after beta_frames."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            beta_start=0.4, beta_end=1.0, beta_frames=100_000,
        )
        assert abs(buf.compute_beta(200_000) - 1.0) < 1e-10

    def test_sample_uses_frame_for_beta(self):
        """sample(frame=N) computes beta from the frame number."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0,
            beta_start=0.4, beta_end=1.0, beta_frames=100,
        )
        _fill_buffer(buf, 4)
        buf.update_priorities(
            np.array([0, 1, 2, 3]),
            np.array([1.0, 2.0, 3.0, 4.0]),
        )

        np.random.seed(42)
        batch_early = buf.sample(4, frame=0)
        np.random.seed(42)
        batch_late = buf.sample(4, frame=100)

        # Different betas produce different weights
        assert not np.allclose(
            batch_early["weights"], batch_late["weights"], atol=1e-5,
        )

    def test_sample_defaults_to_beta_start(self):
        """sample() uses beta_start when neither beta nor frame given."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0, beta_start=0.4,
        )
        _fill_buffer(buf, 4)
        buf.update_priorities(np.array([0, 1]), np.array([1.0, 10.0]))

        np.random.seed(42)
        batch_default = buf.sample(4)
        np.random.seed(42)
        batch_explicit = buf.sample(4, beta=0.4)

        np.testing.assert_allclose(
            batch_default["weights"], batch_explicit["weights"], atol=1e-6,
        )


# ---------------------------------------------------------------------------
# can_sample
# ---------------------------------------------------------------------------


class TestCanSample:
    """can_sample should check min_size and tree total."""

    def test_false_below_min_size(self):
        buf = PrioritizedReplayBuffer(
            capacity=16, obs_shape=OBS_SHAPE, min_size=5,
        )
        _fill_buffer(buf, 3)
        assert not buf.can_sample()

    def test_true_at_min_size(self):
        buf = PrioritizedReplayBuffer(
            capacity=16, obs_shape=OBS_SHAPE, min_size=5,
        )
        _fill_buffer(buf, 5)
        assert buf.can_sample()

    def test_false_with_zero_total(self):
        """Buffer with data but zeroed priorities cannot sample."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0,
        )
        _fill_buffer(buf, 4)
        for i in range(4):
            buf.tree.update(i, 0.0)
        assert not buf.can_sample()

    def test_respects_batch_size(self):
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
        )
        _fill_buffer(buf, 3)
        assert buf.can_sample(batch_size=3)
        assert not buf.can_sample(batch_size=5)


# ---------------------------------------------------------------------------
# Sample output format
# ---------------------------------------------------------------------------


class TestSampleOutput:
    """sample() output should have correct keys, shapes, and types."""

    def test_output_keys(self):
        buf = PrioritizedReplayBuffer(
            capacity=16, obs_shape=OBS_SHAPE, min_size=1,
        )
        _fill_buffer(buf, 8)

        batch = buf.sample(4, beta=1.0)
        expected_keys = {
            "states", "actions", "rewards", "next_states",
            "dones", "indices", "weights",
        }
        assert set(batch.keys()) == expected_keys

    def test_output_shapes(self):
        buf = PrioritizedReplayBuffer(
            capacity=16, obs_shape=OBS_SHAPE, min_size=1,
        )
        _fill_buffer(buf, 8)

        batch = buf.sample(4, beta=1.0)
        assert batch["states"].shape == (4, *OBS_SHAPE)
        assert batch["actions"].shape == (4,)
        assert batch["rewards"].shape == (4,)
        assert batch["next_states"].shape == (4, *OBS_SHAPE)
        assert batch["dones"].shape == (4,)
        assert batch["indices"].shape == (4,)
        assert batch["weights"].shape == (4,)

    def test_indices_within_buffer(self):
        """Sampled indices should be within [0, size)."""
        buf = PrioritizedReplayBuffer(
            capacity=16, obs_shape=OBS_SHAPE, min_size=1,
        )
        _fill_buffer(buf, 8)

        batch = buf.sample(4, beta=1.0)
        assert (batch["indices"] >= 0).all()
        assert (batch["indices"] < buf.size).all()

    def test_states_normalized(self):
        """With normalize=True, states are in [0, 1]."""
        buf = PrioritizedReplayBuffer(
            capacity=16, obs_shape=OBS_SHAPE, min_size=1,
            normalize=True,
        )
        _fill_buffer(buf, 8)

        batch = buf.sample(4, beta=1.0)
        assert batch["states"].max() <= 1.0
        assert batch["states"].min() >= 0.0


# ---------------------------------------------------------------------------
# Sequence sampling (uniform, for SPR)
# ---------------------------------------------------------------------------


class TestSequenceSampling:
    """sample_sequences should use uniform sampling (inherited)."""

    def test_sample_sequences_returns_correct_shapes(self):
        buf = PrioritizedReplayBuffer(
            capacity=32, obs_shape=OBS_SHAPE, min_size=1,
        )
        _fill_buffer(buf, 20)

        batch = buf.sample_sequences(4, seq_len=3)
        assert batch["states"].shape == (4, 4, *OBS_SHAPE)
        assert batch["actions"].shape == (4, 3)
        assert batch["rewards"].shape == (4, 3)
        assert batch["dones"].shape == (4, 3)

    def test_sample_sequences_no_priority_bias(self):
        """Sequence sampling ignores priorities (uniform for SPR)."""
        buf = PrioritizedReplayBuffer(
            capacity=32, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0,
        )
        _fill_buffer(buf, 20)

        # Give index 0 an extreme priority
        buf.update_priorities(
            np.arange(20),
            np.concatenate([np.array([1000.0]), np.ones(19)]),
        )

        # sample_sequences is inherited from ReplayBuffer (uniform)
        np.random.seed(42)
        start_counts = np.zeros(20)
        for _ in range(1000):
            batch = buf.sample_sequences(1, seq_len=1)
            # Recover start index from the first observation value
            obs_val = int(round(float(batch["states"][0, 0, 0, 0, 0]) * 255))
            if obs_val < 20:
                start_counts[obs_val] += 1

        # Uniform: each of ~19 valid starts gets ~52 samples.
        # Index 0 should NOT dominate despite its high priority.
        assert start_counts[0] < 150, (
            f"Index 0 sampled {int(start_counts[0])} times -- "
            "sequence sampling should be uniform"
        )

    def test_sample_sequences_has_no_weights_key(self):
        """Sequence sampling (inherited) does not add IS weights."""
        buf = PrioritizedReplayBuffer(
            capacity=32, obs_shape=OBS_SHAPE, min_size=1,
        )
        _fill_buffer(buf, 20)

        batch = buf.sample_sequences(4, seq_len=3)
        assert "weights" not in batch
        assert "indices" not in batch


# ---------------------------------------------------------------------------
# Interface compatibility with ReplayBuffer
# ---------------------------------------------------------------------------


class TestInterfaceCompatibility:
    """PrioritizedReplayBuffer should be a drop-in for ReplayBuffer."""

    def test_is_subclass(self):
        assert issubclass(PrioritizedReplayBuffer, ReplayBuffer)

    def test_isinstance_check(self):
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
        )
        assert isinstance(buf, ReplayBuffer)

    def test_append_same_interface(self):
        """append() takes the same arguments as ReplayBuffer."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
        )
        buf.append(_make_obs(1), 0, 1.0, _make_obs(2), False)
        assert buf.size == 1

    def test_sample_returns_superset_of_replay_keys(self):
        """sample() returns all ReplayBuffer keys plus indices and weights."""
        buf = PrioritizedReplayBuffer(
            capacity=16, obs_shape=OBS_SHAPE, min_size=1,
        )
        _fill_buffer(buf, 8)

        batch = buf.sample(4, beta=1.0)
        rb_keys = {"states", "actions", "rewards", "next_states", "dones"}
        assert rb_keys.issubset(set(batch.keys()))
        assert "indices" in batch
        assert "weights" in batch

    def test_len_works(self):
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
        )
        assert len(buf) == 0
        _fill_buffer(buf, 3)
        assert len(buf) == 3

    def test_n_step_support(self):
        """PrioritizedReplayBuffer works with n_step > 1."""
        buf = PrioritizedReplayBuffer(
            capacity=16, obs_shape=OBS_SHAPE, min_size=1,
            n_step=3, gamma=0.99,
        )
        for i in range(6):
            buf.append(_make_obs(i), i % 4, 1.0, _make_obs(i + 1), False)

        # 6 appends with n=3: 4 committed transitions
        assert buf.size == 4
        assert buf.tree.total > 0

        batch = buf.sample(2, beta=1.0)
        assert batch["states"].shape == (2, *OBS_SHAPE)
        assert "weights" in batch
        assert "indices" in batch


# ---------------------------------------------------------------------------
# Epsilon handling
# ---------------------------------------------------------------------------


class TestEpsilonHandling:
    """Epsilon prevents zero stored priorities."""

    def test_zero_raw_priority_gets_nonzero_stored(self):
        """With epsilon > 0, raw priority 0 still stores a positive value."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            alpha=0.5, epsilon=1e-6,
        )
        _fill_buffer(buf, 2)

        buf.update_priorities(np.array([0]), np.array([0.0]))
        stored = buf.tree.get(0)
        assert stored > 0, f"Stored priority should be > 0, got {stored}"

    def test_negative_raw_priority_uses_absolute(self):
        """update_priorities takes abs of raw priority."""
        buf = PrioritizedReplayBuffer(
            capacity=8, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0,
        )
        _fill_buffer(buf, 2)

        buf.update_priorities(np.array([0]), np.array([-5.0]))
        assert buf.tree.get(0) == 5.0


# ---------------------------------------------------------------------------
# Buffer wrapping
# ---------------------------------------------------------------------------


class TestBufferWrapping:
    """Priorities should be set correctly when the buffer wraps."""

    def test_overwritten_index_gets_new_max_priority(self):
        """When buffer wraps, overwritten index gets fresh max_priority."""
        buf = PrioritizedReplayBuffer(
            capacity=4, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0,
        )
        _fill_buffer(buf, 4)
        assert buf.size == 4

        # Update index 0 to high priority
        buf.update_priorities(np.array([0]), np.array([10.0]))
        assert buf.tree.get(0) == 10.0
        assert buf.tree.max_priority == 10.0

        # Overwrite index 0 by appending one more
        buf.append(_make_obs(99), 0, 1.0, _make_obs(100), False)

        # Index 0 now has max_priority (10.0) from the new write
        assert buf.tree.get(0) == 10.0
        assert buf.size == 4


class TestPriorityStats:
    """Test get_priority_stats for logging."""

    def test_empty_buffer(self):
        """Empty buffer returns zero stats."""
        buf = PrioritizedReplayBuffer(
            capacity=10, obs_shape=OBS_SHAPE, min_size=1,
        )
        stats = buf.get_priority_stats()
        assert stats["mean_priority"] == 0.0
        assert stats["priority_entropy"] == 0.0

    def test_uniform_priorities_max_entropy(self):
        """Uniform priorities should give maximum entropy."""
        buf = PrioritizedReplayBuffer(
            capacity=10, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0,
        )
        _fill_buffer(buf, 4)
        # Set all priorities equal
        for i in range(4):
            buf.update_priorities(np.array([i]), np.array([1.0]))

        stats = buf.get_priority_stats()
        assert stats["mean_priority"] == 1.0
        # Entropy of uniform distribution over 4 items = ln(4)
        import math
        expected_entropy = math.log(4)
        assert abs(stats["priority_entropy"] - expected_entropy) < 1e-6

    def test_skewed_priorities_lower_entropy(self):
        """Skewed priorities should have lower entropy than uniform."""
        buf = PrioritizedReplayBuffer(
            capacity=10, obs_shape=OBS_SHAPE, min_size=1,
            alpha=1.0, epsilon=0.0,
        )
        _fill_buffer(buf, 4)

        # Uniform baseline
        for i in range(4):
            buf.update_priorities(np.array([i]), np.array([1.0]))
        uniform_stats = buf.get_priority_stats()

        # Skewed: one very high priority
        buf.update_priorities(np.array([0]), np.array([100.0]))
        skewed_stats = buf.get_priority_stats()

        assert skewed_stats["priority_entropy"] < uniform_stats["priority_entropy"]
        assert skewed_stats["mean_priority"] > uniform_stats["mean_priority"]
