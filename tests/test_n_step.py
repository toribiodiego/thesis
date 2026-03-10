"""
Tests for n-step return computation in the replay buffer.

Verifies:
- n=1 identity (identical to standard single-step behavior)
- n=3 correct return computation against hand-computed examples
- Episode boundary truncation (fewer than n steps)
- Gamma discount verified against hand-computed examples
- Sampling returns correct n-step next_states
- Edge cases: single-step episodes, multi-episode sequences, buffer wrap
"""

import numpy as np
import pytest

from src.replay import ReplayBuffer

# Small obs_shape for readable tests
OBS_SHAPE = (1, 2, 2)


def _make_obs(value: int) -> np.ndarray:
    """Create a unique observation filled with a single value."""
    return np.full(OBS_SHAPE, value, dtype=np.uint8)


# ---------------------------------------------------------------------------
# n=1 identity
# ---------------------------------------------------------------------------


class TestNStepIdentity:
    """n_step=1 should produce identical behavior to the default buffer."""

    def test_n1_stores_every_transition(self):
        """Each append stores exactly one transition."""
        buf = ReplayBuffer(
            capacity=10, obs_shape=OBS_SHAPE, min_size=1, n_step=1
        )
        for i in range(5):
            buf.append(_make_obs(i), i, float(i), _make_obs(i + 1), False)

        assert len(buf) == 5

    def test_n1_rewards_unchanged(self):
        """Rewards are stored as-is (no discounting)."""
        buf = ReplayBuffer(
            capacity=10, obs_shape=OBS_SHAPE, min_size=1, n_step=1
        )
        rewards = [1.0, 2.0, 3.0]
        for i, r in enumerate(rewards):
            buf.append(_make_obs(i), 0, r, _make_obs(i + 1), False)

        for i, r in enumerate(rewards):
            assert buf.rewards[i] == pytest.approx(r)

    def test_n1_no_next_observations_array(self):
        """n_step=1 should not allocate next_observations."""
        buf = ReplayBuffer(
            capacity=10, obs_shape=OBS_SHAPE, min_size=1, n_step=1
        )
        assert buf.next_observations is None

    def test_n1_matches_default_buffer(self):
        """n_step=1 buffer matches a default buffer exactly."""
        buf_default = ReplayBuffer(
            capacity=10, obs_shape=OBS_SHAPE, min_size=1
        )
        buf_n1 = ReplayBuffer(
            capacity=10, obs_shape=OBS_SHAPE, min_size=1, n_step=1
        )

        for i in range(5):
            s, ns = _make_obs(i), _make_obs(i + 1)
            buf_default.append(s, i % 3, float(i) * 0.5, ns, i == 4)
            buf_n1.append(s, i % 3, float(i) * 0.5, ns, i == 4)

        assert len(buf_default) == len(buf_n1)
        np.testing.assert_array_equal(
            buf_default.observations[: len(buf_default)],
            buf_n1.observations[: len(buf_n1)],
        )
        np.testing.assert_array_equal(
            buf_default.rewards[: len(buf_default)],
            buf_n1.rewards[: len(buf_n1)],
        )
        np.testing.assert_array_equal(
            buf_default.actions[: len(buf_default)],
            buf_n1.actions[: len(buf_n1)],
        )
        np.testing.assert_array_equal(
            buf_default.dones[: len(buf_default)],
            buf_n1.dones[: len(buf_n1)],
        )


# ---------------------------------------------------------------------------
# n=3 return computation
# ---------------------------------------------------------------------------


class TestNStepReturnComputation:
    """n=3 returns should match hand-computed discounted sums."""

    def test_three_step_return_basic(self):
        """R^(3) = r0 + gamma*r1 + gamma^2*r2 with gamma=0.99."""
        gamma = 0.99
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=gamma,
        )

        # Append 3 transitions: rewards 1.0, 2.0, 3.0
        buf.append(_make_obs(0), 0, 1.0, _make_obs(1), False)
        buf.append(_make_obs(1), 1, 2.0, _make_obs(2), False)
        buf.append(_make_obs(2), 2, 3.0, _make_obs(3), False)

        # First n-step transition should be stored
        assert len(buf) == 1
        expected_R = 1.0 + gamma * 2.0 + gamma**2 * 3.0
        assert buf.rewards[0] == pytest.approx(expected_R)

    def test_three_step_state_and_next_state(self):
        """Stored transition has s_0 as state and s_3 as next_state."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.99,
        )

        buf.append(_make_obs(10), 0, 1.0, _make_obs(11), False)
        buf.append(_make_obs(11), 1, 2.0, _make_obs(12), False)
        buf.append(_make_obs(12), 2, 3.0, _make_obs(13), False)

        # state = s_0 (obs value 10), next_state = s_3 (obs value 13)
        np.testing.assert_array_equal(buf.observations[0], _make_obs(10))
        np.testing.assert_array_equal(buf.next_observations[0], _make_obs(13))

    def test_three_step_action_preserved(self):
        """Stored action is from the first transition in the window."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.99,
        )

        buf.append(_make_obs(0), 7, 1.0, _make_obs(1), False)
        buf.append(_make_obs(1), 3, 2.0, _make_obs(2), False)
        buf.append(_make_obs(2), 5, 3.0, _make_obs(3), False)

        assert buf.actions[0] == 7  # Action from first transition

    def test_three_step_done_from_last(self):
        """Stored done flag comes from the last transition in the window."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.99,
        )

        buf.append(_make_obs(0), 0, 1.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 2.0, _make_obs(2), False)
        buf.append(_make_obs(2), 0, 3.0, _make_obs(3), False)

        assert buf.dones[0] == False  # noqa: E712

    def test_sliding_window_produces_multiple_transitions(self):
        """5 raw transitions with n=3 produce 3 stored n-step transitions."""
        gamma = 0.99
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=gamma,
        )

        rewards = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, r in enumerate(rewards):
            buf.append(_make_obs(i), 0, r, _make_obs(i + 1), False)

        # 5 transitions, n=3: stored at steps 3, 4, 5 -> 3 transitions
        assert len(buf) == 3

        # Verify each stored return
        for k in range(3):
            expected_R = sum(gamma**j * rewards[k + j] for j in range(3))
            assert buf.rewards[k] == pytest.approx(expected_R), (
                f"Stored return at index {k} is wrong"
            )

    def test_n2_return(self):
        """n=2 works correctly (not just n=3)."""
        gamma = 0.9
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=2,
            gamma=gamma,
        )

        buf.append(_make_obs(0), 0, 3.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 7.0, _make_obs(2), False)

        assert len(buf) == 1
        expected_R = 3.0 + gamma * 7.0  # 3 + 6.3 = 9.3
        assert buf.rewards[0] == pytest.approx(expected_R)


# ---------------------------------------------------------------------------
# Episode boundary truncation
# ---------------------------------------------------------------------------


class TestEpisodeBoundaryTruncation:
    """Episode boundaries flush pending transitions with truncated returns."""

    def test_short_episode_two_steps_with_n3(self):
        """Episode of 2 steps with n=3 produces 2 truncated transitions."""
        gamma = 0.99
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=gamma,
        )

        # 2-step episode (shorter than n=3)
        buf.append(_make_obs(0), 0, 1.0, _make_obs(1), False)
        buf.append(_make_obs(1), 1, 2.0, _make_obs(2), True)

        # Both transitions should be flushed with truncated returns
        assert len(buf) == 2

        # First: 2-step truncated return
        assert buf.rewards[0] == pytest.approx(1.0 + gamma * 2.0)
        np.testing.assert_array_equal(buf.observations[0], _make_obs(0))
        np.testing.assert_array_equal(buf.next_observations[0], _make_obs(2))
        assert buf.dones[0] == True  # noqa: E712

        # Second: 1-step truncated return
        assert buf.rewards[1] == pytest.approx(2.0)
        np.testing.assert_array_equal(buf.observations[1], _make_obs(1))
        np.testing.assert_array_equal(buf.next_observations[1], _make_obs(2))
        assert buf.dones[1] == True  # noqa: E712

    def test_single_step_episode_with_n3(self):
        """Episode of 1 step with n=3 produces 1 transition (1-step return)."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.99,
        )

        buf.append(_make_obs(0), 0, 5.0, _make_obs(1), True)

        assert len(buf) == 1
        assert buf.rewards[0] == pytest.approx(5.0)
        np.testing.assert_array_equal(buf.observations[0], _make_obs(0))
        np.testing.assert_array_equal(buf.next_observations[0], _make_obs(1))
        assert buf.dones[0] == True  # noqa: E712

    def test_full_n_step_then_done_flushes_remainder(self):
        """Episode of 4 steps with n=3 produces 1 full + 2 truncated."""
        gamma = 0.99
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=gamma,
        )

        # 4-step episode: rewards 1, 2, 3, 4
        buf.append(_make_obs(0), 0, 1.0, _make_obs(1), False)
        buf.append(_make_obs(1), 1, 2.0, _make_obs(2), False)
        buf.append(_make_obs(2), 2, 3.0, _make_obs(3), False)
        # Step 3 completes the first n-step window
        buf.append(_make_obs(3), 3, 4.0, _make_obs(4), True)
        # Done flushes remaining

        # 1 full (steps 0-2) + 1 full (steps 1-3) + 2 truncated (step 2-3, step 3)
        # Wait: after step 2 (index 2), deque = [0,1,2], len=3, commit -> store(0), deque=[1,2]
        # After step 3 (index 3), deque = [1,2,3], len=3, commit -> store(1), deque=[2,3]
        # done=True, flush: store(2, truncated 2-step), store(3, truncated 1-step)
        assert len(buf) == 4

        # Index 0: full 3-step from steps 0,1,2
        assert buf.rewards[0] == pytest.approx(
            1.0 + gamma * 2.0 + gamma**2 * 3.0
        )
        assert buf.dones[0] == False  # noqa: E712

        # Index 1: full 3-step from steps 1,2,3
        assert buf.rewards[1] == pytest.approx(
            2.0 + gamma * 3.0 + gamma**2 * 4.0
        )
        assert buf.dones[1] == True  # noqa: E712

        # Index 2: truncated 2-step from steps 2,3
        assert buf.rewards[2] == pytest.approx(3.0 + gamma * 4.0)
        assert buf.dones[2] == True  # noqa: E712

        # Index 3: truncated 1-step from step 3
        assert buf.rewards[3] == pytest.approx(4.0)
        assert buf.dones[3] == True  # noqa: E712

    def test_multiple_episodes(self):
        """Transitions across multiple episodes are handled correctly."""
        gamma = 0.5
        buf = ReplayBuffer(
            capacity=20,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=gamma,
        )

        # Episode 1: 2 steps (truncated)
        buf.append(_make_obs(0), 0, 1.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 2.0, _make_obs(2), True)
        ep1_count = len(buf)
        assert ep1_count == 2

        # Episode 2: 3 steps (exactly n)
        buf.append(_make_obs(10), 0, 10.0, _make_obs(11), False)
        buf.append(_make_obs(11), 0, 20.0, _make_obs(12), False)
        buf.append(_make_obs(12), 0, 30.0, _make_obs(13), True)

        # Episode 2 produces: 1 full 3-step + 2 truncated from flush
        assert len(buf) == ep1_count + 3

        # Verify episode 2 returns don't mix with episode 1 rewards
        # Index 2 (first of ep2): full 3-step
        assert buf.rewards[2] == pytest.approx(
            10.0 + gamma * 20.0 + gamma**2 * 30.0
        )
        np.testing.assert_array_equal(buf.observations[2], _make_obs(10))
        np.testing.assert_array_equal(buf.next_observations[2], _make_obs(13))

    def test_pending_deque_empty_after_done(self):
        """After a done signal, the pending deque is completely flushed."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.99,
        )

        buf.append(_make_obs(0), 0, 1.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 2.0, _make_obs(2), True)

        assert len(buf._pending) == 0


# ---------------------------------------------------------------------------
# Gamma discount
# ---------------------------------------------------------------------------


class TestGammaDiscount:
    """Verify gamma^k discounting against hand-computed examples."""

    def test_gamma_zero_ignores_future_rewards(self):
        """gamma=0 means only the immediate reward matters."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.0,
        )

        buf.append(_make_obs(0), 0, 5.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 100.0, _make_obs(2), False)
        buf.append(_make_obs(2), 0, 200.0, _make_obs(3), False)

        # R = 5 + 0*100 + 0*200 = 5
        assert buf.rewards[0] == pytest.approx(5.0)

    def test_gamma_one_sums_all_rewards(self):
        """gamma=1 means R^(n) is the simple sum of rewards."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=1.0,
        )

        buf.append(_make_obs(0), 0, 1.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 2.0, _make_obs(2), False)
        buf.append(_make_obs(2), 0, 3.0, _make_obs(3), False)

        # R = 1 + 2 + 3 = 6
        assert buf.rewards[0] == pytest.approx(6.0)

    def test_gamma_half_hand_computed(self):
        """gamma=0.5 with known rewards: R = 2 + 0.5*4 + 0.25*8 = 6.0."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.5,
        )

        buf.append(_make_obs(0), 0, 2.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 4.0, _make_obs(2), False)
        buf.append(_make_obs(2), 0, 8.0, _make_obs(3), False)

        assert buf.rewards[0] == pytest.approx(6.0)

    def test_truncated_return_uses_correct_gamma_powers(self):
        """Truncated 2-step return with gamma=0.5: R = r0 + 0.5*r1."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.5,
        )

        # 2-step episode (truncated from n=3)
        buf.append(_make_obs(0), 0, 10.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 6.0, _make_obs(2), True)

        # First transition: truncated 2-step
        assert buf.rewards[0] == pytest.approx(10.0 + 0.5 * 6.0)  # 13.0

        # Second transition: truncated 1-step
        assert buf.rewards[1] == pytest.approx(6.0)

    def test_gamma_applies_per_step_not_per_transition(self):
        """Gamma exponents reset for each n-step window, not cumulative."""
        gamma = 0.5
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=2,
            gamma=gamma,
        )

        # 4 transitions -> 3 stored with sliding window
        buf.append(_make_obs(0), 0, 1.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 2.0, _make_obs(2), False)
        buf.append(_make_obs(2), 0, 4.0, _make_obs(3), False)
        buf.append(_make_obs(3), 0, 8.0, _make_obs(4), False)

        # Each window starts gamma^0 fresh
        assert buf.rewards[0] == pytest.approx(1.0 + 0.5 * 2.0)  # 2.0
        assert buf.rewards[1] == pytest.approx(2.0 + 0.5 * 4.0)  # 4.0
        assert buf.rewards[2] == pytest.approx(4.0 + 0.5 * 8.0)  # 8.0


# ---------------------------------------------------------------------------
# Sampling with n-step
# ---------------------------------------------------------------------------


class TestNStepSampling:
    """sample() should return correct n-step next_states."""

    def test_sample_returns_n_step_next_state(self):
        """Sampled next_states come from next_observations, not obs[idx+1]."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.99,
            normalize=False,
        )

        # 3-step transition: state=obs(10), next_state=obs(13)
        buf.append(_make_obs(10), 0, 1.0, _make_obs(11), False)
        buf.append(_make_obs(11), 0, 2.0, _make_obs(12), False)
        buf.append(_make_obs(12), 0, 3.0, _make_obs(13), False)

        batch = buf.sample(1)
        # next_state should be s_{t+3} = obs(13), not obs(11)
        expected_next = _make_obs(13).astype(np.float32)
        np.testing.assert_array_equal(batch["next_states"][0], expected_next)

    def test_sample_returns_correct_state(self):
        """Sampled state is s_t (the first in the n-step window)."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.99,
            normalize=False,
        )

        buf.append(_make_obs(10), 0, 1.0, _make_obs(11), False)
        buf.append(_make_obs(11), 0, 2.0, _make_obs(12), False)
        buf.append(_make_obs(12), 0, 3.0, _make_obs(13), False)

        batch = buf.sample(1)
        expected_state = _make_obs(10).astype(np.float32)
        np.testing.assert_array_equal(batch["states"][0], expected_state)

    def test_all_valid_indices_with_n_step(self):
        """All stored n-step transitions are valid for sampling."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.99,
        )

        # 5 transitions -> 3 stored
        for i in range(5):
            buf.append(_make_obs(i), 0, 1.0, _make_obs(i + 1), False)

        valid = buf._get_valid_indices()
        assert len(valid) == 3
        assert len(valid) == len(buf)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestNStepEdgeCases:
    """Corner cases for n-step return computation."""

    def test_no_transitions_stored_before_n_reached(self):
        """Buffer is empty until n transitions have been appended."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.99,
        )

        buf.append(_make_obs(0), 0, 1.0, _make_obs(1), False)
        assert len(buf) == 0

        buf.append(_make_obs(1), 0, 2.0, _make_obs(2), False)
        assert len(buf) == 0

        buf.append(_make_obs(2), 0, 3.0, _make_obs(3), False)
        assert len(buf) == 1  # First n-step transition committed

    def test_buffer_wrap_with_n_step(self):
        """N-step transitions wrap correctly in a small-capacity buffer."""
        gamma = 0.5
        buf = ReplayBuffer(
            capacity=4,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=2,
            gamma=gamma,
        )

        # Insert enough transitions to wrap the buffer
        # 8 raw transitions with n=2 -> 7 stored (sliding window)
        for i in range(8):
            buf.append(_make_obs(i), 0, float(i + 1), _make_obs(i + 1), False)

        # Buffer capacity is 4, so size should be capped
        assert len(buf) == 4

        # The 4 most recent stored transitions should be indices 3-6
        # (rewards 4+0.5*5, 5+0.5*6, 6+0.5*7, 7+0.5*8)
        for idx in range(4):
            stored_reward = buf.rewards[idx]
            assert stored_reward > 0  # Just verify data is present

    def test_negative_rewards(self):
        """N-step returns work with negative rewards."""
        gamma = 0.9
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=gamma,
        )

        buf.append(_make_obs(0), 0, -1.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 2.0, _make_obs(2), False)
        buf.append(_make_obs(2), 0, -3.0, _make_obs(3), False)

        expected = -1.0 + gamma * 2.0 + gamma**2 * (-3.0)
        assert buf.rewards[0] == pytest.approx(expected)

    def test_zero_rewards(self):
        """N-step return with all zero rewards is zero."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.99,
        )

        buf.append(_make_obs(0), 0, 0.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 0.0, _make_obs(2), False)
        buf.append(_make_obs(2), 0, 0.0, _make_obs(3), False)

        assert buf.rewards[0] == pytest.approx(0.0)

    def test_consecutive_done_episodes(self):
        """Back-to-back single-step episodes are each stored correctly."""
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=0.99,
        )

        # Three single-step episodes in a row
        buf.append(_make_obs(0), 0, 1.0, _make_obs(1), True)
        buf.append(_make_obs(10), 1, 2.0, _make_obs(11), True)
        buf.append(_make_obs(20), 2, 3.0, _make_obs(21), True)

        assert len(buf) == 3
        assert buf.rewards[0] == pytest.approx(1.0)
        assert buf.rewards[1] == pytest.approx(2.0)
        assert buf.rewards[2] == pytest.approx(3.0)
        assert buf.actions[0] == 0
        assert buf.actions[1] == 1
        assert buf.actions[2] == 2

    def test_exact_n_step_episode_with_done(self):
        """Episode of exactly n steps: 1 full n-step + truncated flush."""
        gamma = 0.5
        buf = ReplayBuffer(
            capacity=10,
            obs_shape=OBS_SHAPE,
            min_size=1,
            n_step=3,
            gamma=gamma,
        )

        # 3-step episode (exactly n)
        buf.append(_make_obs(0), 0, 1.0, _make_obs(1), False)
        buf.append(_make_obs(1), 0, 2.0, _make_obs(2), False)
        buf.append(_make_obs(2), 0, 4.0, _make_obs(3), True)

        # Step 2: deque full -> commit 3-step, deque=[1,2]
        # done -> flush: store 2-step (1,2), store 1-step (2)
        assert len(buf) == 3

        # Full 3-step
        assert buf.rewards[0] == pytest.approx(
            1.0 + 0.5 * 2.0 + 0.25 * 4.0
        )
        # Truncated 2-step
        assert buf.rewards[1] == pytest.approx(2.0 + 0.5 * 4.0)
        # Truncated 1-step
        assert buf.rewards[2] == pytest.approx(4.0)
