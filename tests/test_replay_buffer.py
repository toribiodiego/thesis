"""
Tests for Replay Buffer.

Verifies:
- Circular buffer mechanics with wrap-around
- Episode boundary tracking
- Valid index detection
- Proper storage and retrieval
- Shape and dtype correctness
"""

import numpy as np
import pytest
from src.replay import ReplayBuffer


def test_replay_buffer_init():
    """Test replay buffer initialization."""
    buffer = ReplayBuffer(capacity=1000, obs_shape=(4, 84, 84))

    assert buffer.capacity == 1000
    assert buffer.obs_shape == (4, 84, 84)
    assert buffer.size == 0
    assert buffer.index == 0
    assert len(buffer) == 0

    # Check storage arrays are allocated
    assert buffer.observations.shape == (1000, 4, 84, 84)
    assert buffer.actions.shape == (1000,)
    assert buffer.rewards.shape == (1000,)
    assert buffer.dones.shape == (1000,)
    assert buffer.episode_starts.shape == (1000,)


def test_replay_buffer_append_single():
    """Test appending a single transition."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
    next_state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
    action = 2
    reward = 1.0
    done = False

    buffer.append(state, action, reward, next_state, done)

    assert len(buffer) == 1
    assert buffer.index == 1
    assert buffer.size == 1

    # Check stored values
    assert np.array_equal(buffer.observations[0], state)
    assert buffer.actions[0] == action
    assert buffer.rewards[0] == reward
    assert buffer.dones[0] == done

    # First transition is always an episode start
    assert buffer.episode_starts[0] == True


def test_replay_buffer_append_multiple():
    """Test appending multiple transitions."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    # Add 10 transitions
    for i in range(10):
        state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
        next_state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
        buffer.append(state, i, float(i), next_state, False)

    assert len(buffer) == 10
    assert buffer.index == 10
    assert buffer.size == 10

    # Check actions stored correctly
    for i in range(10):
        assert buffer.actions[i] == i
        assert buffer.rewards[i] == float(i)


def test_replay_buffer_episode_boundary_tracking():
    """Test episode boundary markers."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)

    # Episode 1: 5 transitions
    for i in range(5):
        done = (i == 4)  # Last transition ends episode
        buffer.append(state, i, 1.0, state, done)

    # Episode 2: 3 transitions
    for i in range(3):
        done = (i == 2)
        buffer.append(state, i, 1.0, state, done)

    # Check episode starts
    assert buffer.episode_starts[0] == True   # First transition
    assert buffer.episode_starts[1] == False  # Episode 1 continues
    assert buffer.episode_starts[4] == False  # Last of episode 1
    assert buffer.episode_starts[5] == True   # Episode 2 starts (after done)
    assert buffer.episode_starts[6] == False  # Episode 2 continues


def test_replay_buffer_circular_wrap():
    """Test circular buffer wraps around correctly."""
    capacity = 10
    buffer = ReplayBuffer(capacity=capacity, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)

    # Fill buffer to capacity
    for i in range(capacity):
        buffer.append(state, i, float(i), state, False)

    assert len(buffer) == capacity
    assert buffer.size == capacity
    assert buffer.index == 0  # Wrapped around

    # Add one more - should overwrite index 0
    buffer.append(state, 999, 999.0, state, False)

    assert len(buffer) == capacity  # Size stays at capacity
    assert buffer.index == 1  # Advanced to 1
    assert buffer.actions[0] == 999  # Overwrote first element
    assert buffer.rewards[0] == 999.0


def test_replay_buffer_dtype_conversion():
    """Test automatic dtype conversion from float32 to uint8."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84), dtype=np.uint8)

    # Create float32 state in [0, 1] range
    state_float = np.random.rand(4, 84, 84).astype(np.float32)

    buffer.append(state_float, 0, 1.0, state_float, False)

    # Should be converted to uint8 [0, 255]
    stored = buffer.observations[0]
    assert stored.dtype == np.uint8
    assert stored.min() >= 0
    assert stored.max() <= 255

    # Check conversion is approximately correct (within 1 due to rounding)
    expected = (state_float * 255).astype(np.uint8)
    assert np.allclose(stored, expected, atol=1)


def test_replay_buffer_shape_validation():
    """Test shape validation raises errors."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    # Wrong shape should raise
    wrong_state = np.random.randint(0, 255, size=(84, 84), dtype=np.uint8)

    with pytest.raises(AssertionError):
        buffer.append(wrong_state, 0, 1.0, wrong_state, False)


def test_replay_buffer_valid_index_detection():
    """Test valid index detection for sampling."""
    buffer = ReplayBuffer(capacity=20, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)

    # Add episode 1: 5 transitions
    for i in range(5):
        done = (i == 4)
        buffer.append(state, i, 1.0, state, done)

    # Add episode 2: 3 transitions
    for i in range(3):
        done = (i == 2)
        buffer.append(state, i, 1.0, state, done)

    # Index 0 is episode start - not valid
    assert buffer._is_valid_index(0) == False

    # Index 1-4 should be valid (episode 1, not starts)
    for i in range(1, 5):
        assert buffer._is_valid_index(i) == True

    # Index 5 is episode start - not valid
    assert buffer._is_valid_index(5) == False

    # Index 6-7 should be valid (episode 2, not starts)
    for i in range(6, 8):
        assert buffer._is_valid_index(i) == True

    # Index 8+ don't exist yet - not valid
    assert buffer._is_valid_index(8) == False


def test_replay_buffer_get_valid_indices():
    """Test getting all valid indices."""
    buffer = ReplayBuffer(capacity=20, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)

    # Add episode 1: 5 transitions
    for i in range(5):
        done = (i == 4)
        buffer.append(state, i, 1.0, state, done)

    # Add episode 2: 5 transitions
    for i in range(5):
        done = (i == 4)
        buffer.append(state, i, 1.0, state, done)

    valid_indices = buffer._get_valid_indices()

    # Should have valid indices from both episodes
    # Episode 1: indices 1-4 (not 0, it's episode start)
    # Episode 2: indices 6-9 (not 5, it's episode start)
    expected = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=np.int64)

    assert len(valid_indices) == len(expected)
    assert np.array_equal(np.sort(valid_indices), expected)


def test_replay_buffer_wrap_around_boundary():
    """Test episode boundary tracking across wrap-around."""
    capacity = 10
    buffer = ReplayBuffer(capacity=capacity, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)

    # Fill buffer with one long episode
    for i in range(capacity):
        buffer.append(state, i, 1.0, state, False)

    assert buffer.size == capacity
    assert buffer.index == 0  # Wrapped

    # Now start a new episode by marking done
    buffer.append(state, 999, 1.0, state, True)

    assert buffer.index == 1
    assert buffer.dones[0] == True  # Overwrote with done=True

    # Next append should be marked as episode start
    buffer.append(state, 888, 1.0, state, False)

    assert buffer.index == 2
    assert buffer.episode_starts[1] == True  # New episode after done


def test_replay_buffer_empty_valid_indices():
    """Test valid indices when buffer is empty."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    valid = buffer._get_valid_indices()
    assert len(valid) == 0


def test_replay_buffer_single_transition_no_valid():
    """Test that single transition has no valid indices (it's episode start)."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
    buffer.append(state, 0, 1.0, state, False)

    # First transition is episode start, so not valid for sampling
    valid = buffer._get_valid_indices()
    assert len(valid) == 0


def test_replay_buffer_rewards_dtype():
    """Test rewards are stored as float32."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
    buffer.append(state, 0, 1.5, state, False)

    assert buffer.rewards.dtype == np.float32
    assert buffer.rewards[0] == 1.5


def test_replay_buffer_actions_dtype():
    """Test actions are stored as int64."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
    buffer.append(state, 5, 1.0, state, False)

    assert buffer.actions.dtype == np.int64
    assert buffer.actions[0] == 5


def test_replay_buffer_dones_dtype():
    """Test dones are stored as bool."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
    buffer.append(state, 0, 1.0, state, True)

    assert buffer.dones.dtype == bool
    assert buffer.dones[0] == True


if __name__ == "__main__":
    # Run tests manually
    print("Running replay buffer tests...")

    test_replay_buffer_init()
    print("✓ Initialization test passed")

    test_replay_buffer_append_single()
    print("✓ Single append test passed")

    test_replay_buffer_append_multiple()
    print("✓ Multiple append test passed")

    test_replay_buffer_episode_boundary_tracking()
    print("✓ Episode boundary tracking test passed")

    test_replay_buffer_circular_wrap()
    print("✓ Circular wrap test passed")

    test_replay_buffer_dtype_conversion()
    print("✓ Dtype conversion test passed")

    test_replay_buffer_shape_validation()
    print("✓ Shape validation test passed")

    test_replay_buffer_valid_index_detection()
    print("✓ Valid index detection test passed")

    test_replay_buffer_get_valid_indices()
    print("✓ Get valid indices test passed")

    test_replay_buffer_wrap_around_boundary()
    print("✓ Wrap-around boundary test passed")

    test_replay_buffer_empty_valid_indices()
    print("✓ Empty valid indices test passed")

    test_replay_buffer_single_transition_no_valid()
    print("✓ Single transition no valid test passed")

    test_replay_buffer_rewards_dtype()
    print("✓ Rewards dtype test passed")

    test_replay_buffer_actions_dtype()
    print("✓ Actions dtype test passed")

    test_replay_buffer_dones_dtype()
    print("✓ Dones dtype test passed")

    print("\nAll tests passed! ✓")
