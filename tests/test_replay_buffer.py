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


def test_replay_buffer_sample_basic():
    """Test basic sampling functionality."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    # Add two episodes with multiple transitions
    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)

    # Episode 1: 10 transitions
    for i in range(10):
        done = (i == 9)
        buffer.append(state, i, float(i), state, done)

    # Episode 2: 10 transitions
    for i in range(10):
        done = (i == 9)
        buffer.append(state, i + 10, float(i + 10), state, done)

    # Sample a batch
    batch_size = 5
    batch = buffer.sample(batch_size)

    # Check batch structure
    assert 'states' in batch
    assert 'actions' in batch
    assert 'rewards' in batch
    assert 'next_states' in batch
    assert 'dones' in batch

    # Check shapes
    assert batch['states'].shape == (batch_size, 4, 84, 84)
    assert batch['actions'].shape == (batch_size,)
    assert batch['rewards'].shape == (batch_size,)
    assert batch['next_states'].shape == (batch_size, 4, 84, 84)
    assert batch['dones'].shape == (batch_size,)

    # Check dtypes
    assert batch['states'].dtype == np.uint8
    assert batch['actions'].dtype == np.int64
    assert batch['rewards'].dtype == np.float32
    assert batch['next_states'].dtype == np.uint8
    assert batch['dones'].dtype == bool


def test_replay_buffer_sample_insufficient_samples():
    """Test sampling raises error when batch_size > valid samples."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)

    # Add only 3 transitions (only 2 will be valid - not counting episode start)
    for i in range(3):
        buffer.append(state, i, float(i), state, False)

    # Try to sample more than available
    with pytest.raises(ValueError, match="Not enough valid samples"):
        buffer.sample(batch_size=5)


def test_replay_buffer_sample_empty_buffer():
    """Test sampling from empty buffer raises error."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    with pytest.raises(ValueError, match="Not enough valid samples"):
        buffer.sample(batch_size=1)


def test_replay_buffer_sample_without_replacement():
    """Test sampling without replacement (no duplicates in batch)."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    # Add transitions with unique actions
    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)
    for i in range(20):
        done = (i == 9 or i == 19)  # Two episodes
        buffer.append(state, i, float(i), state, done)

    # Sample a batch
    batch_size = 10
    batch = buffer.sample(batch_size)

    # Check no duplicate actions (since we made them unique)
    actions = batch['actions']
    assert len(np.unique(actions)) == len(actions), "Found duplicate samples"


def test_replay_buffer_sample_respects_boundaries():
    """Test sampling doesn't cross episode boundaries."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    # Create distinct states for each transition
    states = [
        np.full((4, 84, 84), i, dtype=np.uint8)
        for i in range(20)
    ]

    # Episode 1: indices 0-9
    for i in range(10):
        done = (i == 9)
        buffer.append(states[i], i, float(i), states[i], done)

    # Episode 2: indices 10-19
    for i in range(10, 20):
        done = (i == 19)
        buffer.append(states[i], i, float(i), states[i], done)

    # Sample many batches
    for _ in range(10):
        batch = buffer.sample(batch_size=5)

        # For each sampled transition, verify next_state is consistent
        for j in range(len(batch['states'])):
            state = batch['states'][j]
            next_state = batch['next_states'][j]
            action = batch['actions'][j]

            # Find which index this is
            state_val = state[0, 0, 0]  # Unique identifier

            # next_state should be state_val + 1 (except at episode ends)
            next_val = next_state[0, 0, 0]

            # If not done, next state should be consecutive
            if not batch['dones'][j]:
                # For our setup, consecutive states differ by 1
                # But we can't guarantee this at episode boundaries
                # Just check that we didn't jump episodes
                assert abs(int(next_val) - int(state_val)) <= 1, \
                    "Crossed episode boundary in sampling"


def test_replay_buffer_sample_correct_next_states():
    """Test that next_states are correctly retrieved."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    # Create states with identifiable values
    states = []
    for i in range(15):
        state = np.full((4, 84, 84), i * 10, dtype=np.uint8)
        states.append(state)

    # Add episode (no done until the end)
    for i in range(14):
        buffer.append(states[i], i, float(i), states[i + 1], False)

    # Sample and verify
    batch = buffer.sample(batch_size=5)

    for i in range(len(batch['states'])):
        state_val = batch['states'][i][0, 0, 0]
        next_val = batch['next_states'][i][0, 0, 0]
        action = batch['actions'][i]

        # next_state should have value state_val + 10
        # (because we increment by 10 each time)
        expected_next = state_val + 10
        assert next_val == expected_next, \
            f"Next state mismatch: expected {expected_next}, got {next_val}"


def test_replay_buffer_sample_after_wrap():
    """Test sampling works correctly after buffer wraps around."""
    capacity = 10
    buffer = ReplayBuffer(capacity=capacity, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)

    # Fill buffer past capacity
    for i in range(20):
        done = (i % 10 == 9)  # Episode ends every 10 transitions
        buffer.append(state, i, float(i), state, done)

    # Should be able to sample
    batch = buffer.sample(batch_size=3)

    assert batch['states'].shape == (3, 4, 84, 84)
    assert batch['actions'].shape == (3,)


def test_replay_buffer_sample_deterministic_with_seed():
    """Test sampling is deterministic with fixed random seed."""
    buffer = ReplayBuffer(capacity=100, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)

    # Add transitions
    for i in range(20):
        done = (i == 19)
        buffer.append(state, i, float(i), state, done)

    # Sample with fixed seed
    np.random.seed(42)
    batch1 = buffer.sample(batch_size=5)

    # Sample again with same seed
    np.random.seed(42)
    batch2 = buffer.sample(batch_size=5)

    # Should get identical samples
    assert np.array_equal(batch1['actions'], batch2['actions'])
    assert np.array_equal(batch1['rewards'], batch2['rewards'])


def test_replay_buffer_sample_full_capacity():
    """Test sampling when buffer is at full capacity."""
    capacity = 50
    buffer = ReplayBuffer(capacity=capacity, obs_shape=(4, 84, 84))

    state = np.random.randint(0, 255, size=(4, 84, 84), dtype=np.uint8)

    # Fill to capacity with one long episode
    for i in range(capacity):
        buffer.append(state, i, float(i), state, False)

    # Should be able to sample near capacity
    # (minus 1 for episode start)
    batch = buffer.sample(batch_size=capacity - 1)

    assert len(batch['actions']) == capacity - 1


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

    test_replay_buffer_sample_basic()
    print("✓ Sample basic test passed")

    test_replay_buffer_sample_insufficient_samples()
    print("✓ Sample insufficient samples test passed")

    test_replay_buffer_sample_empty_buffer()
    print("✓ Sample empty buffer test passed")

    test_replay_buffer_sample_without_replacement()
    print("✓ Sample without replacement test passed")

    test_replay_buffer_sample_respects_boundaries()
    print("✓ Sample respects boundaries test passed")

    test_replay_buffer_sample_correct_next_states()
    print("✓ Sample correct next states test passed")

    test_replay_buffer_sample_after_wrap()
    print("✓ Sample after wrap test passed")

    test_replay_buffer_sample_deterministic_with_seed()
    print("✓ Sample deterministic with seed test passed")

    test_replay_buffer_sample_full_capacity()
    print("✓ Sample full capacity test passed")

    print("\nAll tests passed! ✓")
