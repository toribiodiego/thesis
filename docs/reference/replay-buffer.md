# Replay Buffer

---

**Prerequisites:**
- Completed [DQN Setup](dqn-setup.md) - Environment configured and dependencies installed
- Understand [Atari Wrappers](atari-env-wrapper.md) - Frame preprocessing and stacking
- Optional: [DQN Paper Section 3](../resear../research/papers/dqn-2013-notes.md) - Experience replay background

**Related Docs:**
- [DQN Training](dqn-training.md) - How sampled batches are used for Q-learning
- [Training Loop](training-loop-runtime.md) - Replay integration with optimization
- [Checkpointing](checkpointing.md) - Saving/loading replay buffer state

---

## Overview

The replay buffer implements uniform experience replay for DQN training, storing transitions in a circular buffer with episode boundary tracking. It provides memory-efficient storage (uint8), deferred float32 conversion, configurable normalization, device transfer to GPU, and warm-up enforcement.

**Key features:**
- Circular buffer with wrap-around (default 1M capacity)
- Episode boundary tracking to prevent cross-episode sampling
- Memory-efficient uint8 storage (4x reduction vs float32)
- Deferred conversion and optional normalization on sampling
- GPU acceleration with optional pinned memory
- Configurable warm-up threshold (default 50K transitions)

## Memory Layout

### Storage Arrays

```
ReplayBuffer (capacity=1,000,000)
├── observations:     (1M, 4, 84, 84) uint8     ~27 GB
├── actions:          (1M,) int64                ~8 MB
├── rewards:          (1M,) float32              ~4 MB
├── dones:            (1M,) bool                 ~1 MB
└── episode_starts:   (1M,) bool                 ~1 MB

Total memory: ~27 GB (vs ~108 GB if observations were float32)
```

### Circular Buffer Mechanics

```
Initial state (capacity=10, empty):
┌─────────────────────────────────────┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │
├─────────────────────────────────────┤
│   │   │   │   │   │   │   │   │   │   │
└─────────────────────────────────────┘
  ↑
 index=0, size=0

After adding 5 transitions:
┌─────────────────────────────────────┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │
├─────────────────────────────────────┤
│ T │ T │ T │ T │ T │   │   │   │   │   │
└─────────────────────────────────────┘
                  ↑
                index=5, size=5

After adding 12 total (wrapped around):
┌─────────────────────────────────────┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │
├─────────────────────────────────────┤
│ T │ T │ T │ T │ T │ T │ T │ T │ T │ T │
└─────────────────────────────────────┘
      ↑
    index=2, size=10 (full, wrapped, overwrote 0-1)
```

### Episode Boundary Tracking

```
Episode 1: transitions 0-4 (done at index 4)
Episode 2: transitions 5-9 (done at index 9)

episode_starts array:
┌─────────────────────────────────────┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │
├─────────────────────────────────────┤
│ T │ F │ F │ F │ F │ T │ F │ F │ F │ F │
└─────────────────────────────────────┘
  ↑                   ↑
  Episode 1 start     Episode 2 start

Valid for sampling: [1, 2, 3, 4, 6, 7, 8, 9]
NOT valid: [0, 5] (episode starts - need previous frames)
```

## Sampling Pseudocode

### High-Level Algorithm

```python
def sample(batch_size):
    # 1. Get valid indices (no episode starts, no boundary crossings)
    valid_indices = _get_valid_indices()

    # 2. Check sufficient samples available
    if len(valid_indices) < batch_size:
        raise ValueError("Not enough valid samples")

    # 3. Sample without replacement
    sampled_indices = np.random.choice(valid_indices, batch_size, replace=False)

    # 4. Gather data (still uint8)
    states_uint8 = observations[sampled_indices]
    actions = actions[sampled_indices]
    rewards = rewards[sampled_indices]
    dones = dones[sampled_indices]

    # 5. Get next states (handle wrap-around)
    next_indices = (sampled_indices + 1) % capacity
    next_states_uint8 = observations[next_indices]

    # 6. Convert to float32
    states = states_uint8.astype(float32)
    next_states = next_states_uint8.astype(float32)

    # 7. Normalize if configured
    if normalize:
        states /= 255.0
        next_states /= 255.0

    # 8. Device transfer if configured
    if device is not None:
        # Convert to PyTorch tensors
        if pin_memory and device == 'cuda':
            # Use pinned memory for faster H2D transfer
            tensors = [torch.from_numpy(arr).pin_memory() for arr in arrays]
        else:
            tensors = [torch.from_numpy(arr) for arr in arrays]

        # Move to device
        tensors = [t.to(device, non_blocking=True) for t in tensors]

    return {'states', 'actions', 'rewards', 'next_states', 'dones'}
```

### Valid Index Detection

```python
def _is_valid_index(idx):
    # Must be within buffer size
    if idx >= size:
        return False

    # Can't be episode start (need previous frames for stacking)
    if episode_starts[idx]:
        return False

    # Terminal transitions (done=True) are always valid
    # The next_state value doesn't matter because TD target = r when done=True
    # The (1-done) term in the TD target computation zeros out the Q(s',a') bootstrap
    if dones[idx]:
        return True

    # For non-terminal transitions: next index must exist and be in same episode
    next_idx = (idx + 1) % capacity
    if next_idx >= size:
        return False

    # Check for episode boundary crossing
    if episode_starts[next_idx]:
        return False

    return True
```

**Terminal Transition Handling:**

Terminal transitions (done=True) are valid for sampling because the TD target for terminal states is just the reward:

```
TD target = r + γ * (1 - done) * max_a' Q(s', a')

When done=True:
TD target = r + γ * 0 * max_a' Q(s', a')
         = r
```

The `(1-done)` term zeros out the next-state contribution, so the value returned for `next_states[i]` doesn't affect training. The buffer returns `observations[(idx+1) % capacity]` (which may be from the next episode), but this is safe because it's multiplied by zero in the loss computation.

**Why this matters:**
- Terminal transitions provide critical learning signal about episode-ending states
- Without sampling terminal transitions, the agent can't learn values for states that lead to termination
- The overlapping storage design (next_state = observations[idx+1]) is memory-efficient while still being correct

## Warm-Up Policy

### Purpose

Ensures sufficient exploration before training begins. Prevents training on sparse, poorly-distributed data.

### Configuration

```python
buffer = ReplayBuffer(
    capacity=1_000_000,
    min_size=50_000  # Default warm-up threshold
)

# Training loop
while step < max_steps:
    # Collect experience
    buffer.append(state, action, reward, next_state, done)

    # Check if ready for training
    if buffer.can_sample(batch_size=32):
        # Buffer has >= min_size transitions
        batch = buffer.sample(32)
        train(batch)
    else:
        # Still in warm-up phase
        continue
```

### can_sample() Logic

```python
def can_sample(batch_size=None):
    # Check minimum size threshold
    if size < min_size:
        return False

    # Optionally check valid indices count
    if batch_size is not None:
        valid_indices = _get_valid_indices()
        return len(valid_indices) >= batch_size

    return True
```

## Configuration Flags

### Core Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `capacity` | 1,000,000 | Maximum transitions to store |
| `obs_shape` | (4, 84, 84) | Shape of a single observation |
| `dtype` | np.uint8 | Storage dtype for observations |
| `normalize` | True | Normalize to [0,1] on sample |
| `min_size` | 50,000 | Warm-up threshold |
| `device` | None | Target device for samples |
| `pin_memory` | False | Use pinned memory for GPU |

### Configuration Examples

**Memory-efficient CPU training:**
```python
buffer = ReplayBuffer(
    capacity=1_000_000,
    normalize=True,    # [0,1] normalization
    min_size=50_000,
    device=None        # Returns NumPy arrays
)
```

**GPU training with fast transfer:**
```python
buffer = ReplayBuffer(
    capacity=1_000_000,
    normalize=True,
    min_size=50_000,
    device='cuda',      # Returns GPU tensors
    pin_memory=True     # Faster H2D transfer
)
```

**Custom warm-up:**
```python
buffer = ReplayBuffer(
    capacity=500_000,
    min_size=10_000,   # Smaller warm-up for faster iteration
    normalize=False     # Keep [0, 255] range
)
```

## Known Failure Modes

### 1. Episode Leakage

**Symptom:** Model learns to predict transitions across episode boundaries

**Cause:** Sampling indices that cross episode boundaries (e.g., sampling last transition of episode, getting first transition of next episode as "next_state")

**Detection:**
```python
# Check for episode leakage
valid_indices = buffer._get_valid_indices()
for idx in valid_indices:
    assert not buffer.episode_starts[idx], f"Episode start at {idx} is valid!"
    next_idx = (idx + 1) % buffer.capacity
    if buffer.episode_starts[next_idx]:
        print(f"WARNING: Potential leakage at {idx}")
```

**Prevention:**
- Buffer automatically excludes episode starts from valid indices
- `_is_valid_index()` checks for boundary crossings
- Tests verify no cross-episode sampling

**Fix:** Already prevented by design. If you see this, check for bugs in `episode_starts` tracking.

### 2. Dtype Mismatch

**Symptom:** `RuntimeError: expected Float but got Byte` or similar

**Cause:** Passing uint8 observations directly to PyTorch model

**Detection:**
```python
batch = buffer.sample(32)
print(f"States dtype: {batch['states'].dtype}")  # Should be float32
print(f"Actions dtype: {batch['actions'].dtype}")  # Should be int64
```

**Prevention:**
- Buffer automatically converts uint8 → float32 on sample
- Device transfer preserves dtypes

**Fix:**
```python
# If you get uint8, check:
assert buffer.normalize == True  # Should normalize
batch = buffer.sample(32)
assert batch['states'].dtype == np.float32  # Or torch.float32
```

### 3. Insufficient Valid Samples

**Symptom:** `ValueError: Not enough valid samples in buffer`

**Cause:** Requesting batch_size larger than valid indices count

**Detection:**
```python
valid_indices = buffer._get_valid_indices()
print(f"Valid samples: {len(valid_indices)}")
print(f"Requested batch: {batch_size}")
```

**Common scenarios:**
- Many short episodes (lots of episode starts)
- Buffer not full yet
- Batch size too large

**Fix:**
```python
# Check before sampling
if buffer.can_sample(batch_size=32):
    batch = buffer.sample(32)
else:
    # Use smaller batch or wait for more data
    pass
```

### 4. Wrap-Around Boundary Issues

**Symptom:** Sampling fails or produces inconsistent data after buffer wraps

**Cause:** Incorrect handling of modulo indexing when index wraps to 0

**Detection:**
```python
# After buffer wraps
print(f"Index: {buffer.index}")  # Should be < capacity
print(f"Size: {buffer.size}")    # Should be == capacity
batch = buffer.sample(10)  # Should succeed
```

**Prevention:**
- All indexing uses `(index + offset) % capacity`
- Tests verify wrap-around correctness

**Fix:** Already handled. If you see this, check for bugs in index calculations.

### 5. Memory Overflow

**Symptom:** `MemoryError` or system swap thrashing

**Cause:** Buffer capacity too large for available RAM

**Detection:**
```python
import numpy as np
capacity = 1_000_000
obs_shape = (4, 84, 84)
bytes_needed = capacity * np.prod(obs_shape) * 1  # uint8 = 1 byte
print(f"Memory needed: {bytes_needed / 1e9:.2f} GB")
```

**Fix:**
```python
# Reduce capacity
buffer = ReplayBuffer(capacity=500_000)  # Half capacity = half memory

# Or use smaller observations (if possible)
buffer = ReplayBuffer(obs_shape=(4, 64, 64))  # 64x64 instead of 84x84
```

### 6. Device Transfer Slowdown

**Symptom:** Slow training loop, GPU underutilized

**Cause:** Not using pinned memory for GPU transfer

**Detection:**
```python
import time
batch_size = 32

# Without pinned memory
buffer = ReplayBuffer(device='cuda', pin_memory=False)
start = time.time()
batch = buffer.sample(batch_size)
print(f"Without pinned: {time.time() - start:.4f}s")

# With pinned memory
buffer = ReplayBuffer(device='cuda', pin_memory=True)
start = time.time()
batch = buffer.sample(batch_size)
print(f"With pinned: {time.time() - start:.4f}s")  # Should be faster
```

**Fix:**
```python
buffer = ReplayBuffer(device='cuda', pin_memory=True)
```

## Validation Commands

### Running Tests

**All tests:**
```bash
pytest tests/test_replay_buffer.py -v
```

**Specific test categories:**
```bash
# Sampling tests
pytest tests/test_replay_buffer.py -k "sample" -v

# Boundary tests
pytest tests/test_replay_buffer.py -k "boundary" -v

# Device tests
pytest tests/test_replay_buffer.py -k "device" -v

# Comprehensive integration
pytest tests/test_replay_buffer.py::test_replay_buffer_comprehensive_integration -v
```

**Manual test runner:**
```bash
python tests/test_replay_buffer.py
```

### Inspecting Buffer State

**Check buffer contents:**
```python
from src.replay import ReplayBuffer
import numpy as np

buffer = ReplayBuffer(capacity=100, min_size=10)

# Add some transitions
state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
for i in range(50):
    done = (i % 10 == 9)
    buffer.append(state, i, float(i), state, done)

# Inspect state
print(f"Size: {buffer.size}/{buffer.capacity}")
print(f"Index: {buffer.index}")
print(f"Valid indices count: {len(buffer._get_valid_indices())}")
print(f"Can sample (batch=32): {buffer.can_sample(32)}")

# Check episode boundaries
episode_starts = np.where(buffer.episode_starts[:buffer.size])[0]
print(f"Episode starts at indices: {episode_starts}")
```

### Dumping Sample Batches

**Save batch to disk for inspection:**
```python
import numpy as np
import torch

buffer = ReplayBuffer(capacity=1000, min_size=10, device='cpu')

# ... fill buffer ...

# Sample batch
batch = buffer.sample(batch_size=5)

# Save as NumPy
if isinstance(batch['states'], np.ndarray):
    np.savez('batch_dump.npz',
             states=batch['states'],
             actions=batch['actions'],
             rewards=batch['rewards'],
             next_states=batch['next_states'],
             dones=batch['dones'])
    print("Saved to batch_dump.npz")
else:  # PyTorch tensors
    torch.save(batch, 'batch_dump.pt')
    print("Saved to batch_dump.pt")

# Load and inspect
loaded = np.load('batch_dump.npz')
print(f"States shape: {loaded['states'].shape}")
print(f"States dtype: {loaded['states'].dtype}")
print(f"States range: [{loaded['states'].min()}, {loaded['states'].max()}]")
print(f"Actions: {loaded['actions']}")
print(f"Rewards: {loaded['rewards']}")
```

**Visualize frame stack:**
```python
import matplotlib.pyplot as plt

batch = buffer.sample(1)
state = batch['states'][0]  # (4, 84, 84)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i in range(4):
    axes[i].imshow(state[i], cmap='gray')
    axes[i].set_title(f'Frame t-{3-i}')
    axes[i].axis('off')
plt.savefig('frame_stack.png')
print("Saved to frame_stack.png")
```

## Troubleshooting Guide

### Problem: Training is unstable early on

**Check:** Warm-up threshold
```python
print(f"Min size: {buffer.min_size}")
print(f"Current size: {buffer.size}")
print(f"Can sample: {buffer.can_sample()}")
```

**Solution:** Increase warm-up
```python
buffer = ReplayBuffer(min_size=100_000)  # Increase from 50K
```

### Problem: Out of memory

**Check:** Buffer memory usage
```python
capacity = buffer.capacity
obs_bytes = capacity * np.prod(buffer.obs_shape) * 1  # uint8
other_bytes = capacity * (8 + 4 + 1 + 1)  # actions, rewards, dones, starts
total_gb = (obs_bytes + other_bytes) / 1e9
print(f"Buffer uses ~{total_gb:.2f} GB")
```

**Solution:** Reduce capacity
```python
buffer = ReplayBuffer(capacity=500_000)
```

### Problem: Batch contains episode boundaries

**Check:** Valid indices
```python
valid = buffer._get_valid_indices()
for idx in valid:
    assert not buffer.episode_starts[idx]
    next_idx = (idx + 1) % buffer.capacity
    if buffer.episode_starts[next_idx]:
        print(f"ERROR: Index {idx} has next as episode start")
```

**Solution:** This should never happen. If it does, file a bug report.

### Problem: Slow GPU training

**Check:** Device transfer setup
```python
print(f"Device: {buffer.device}")
print(f"Pin memory: {buffer.pin_memory}")

# Time a sample
import time
start = time.time()
batch = buffer.sample(32)
print(f"Sample time: {(time.time() - start) * 1000:.2f}ms")
```

**Solution:** Enable pinned memory
```python
buffer = ReplayBuffer(device='cuda', pin_memory=True)
```

### Problem: Samples are not normalized

**Check:** Normalization setting
```python
print(f"Normalize: {buffer.normalize}")
batch = buffer.sample(1)
print(f"States range: [{batch['states'].min()}, {batch['states'].max()}]")
```

**Solution:** Enable normalization
```python
buffer = ReplayBuffer(normalize=True)  # Should give [0, 1]
```

## Design Decisions

### Why uint8 storage?

**Memory efficiency:** 4x reduction (1 byte vs 4 bytes for float32)
- 1M transitions × (4×84×84) = 28.2M values
- float32: 112.9 MB per array × 1M = ~113 GB
- uint8: 28.2 MB per array × 1M = ~28 GB

**Precision:** Atari frames are naturally [0, 255], no precision loss

**Performance:** Conversion to float32 is fast (~1ms for batch of 32)

### Why defer normalization to sampling?

**Flexibility:** Some algorithms prefer [0, 255] (e.g., for integer-based methods)

**Efficiency:** Normalization is cheap (single division), no benefit to pre-compute

**Compatibility:** Easier to switch between normalized/unnormalized

### Why pinned memory?

**Speed:** Host-to-device transfer is 2-3x faster with pinned memory

**Asynchronous:** Enables `non_blocking=True` for overlap with compute

**Cost:** Pinned memory is a limited resource, but worth it for training bottleneck

### Why track episode starts instead of episode IDs?

**Memory:** 1 bit per transition vs 32/64 bits for ID

**Performance:** Single boolean check vs integer comparison

**Sufficient:** Only need to know if index is start, not which episode

## Quick Reference

```python
# Create buffer
from src.replay import ReplayBuffer

buffer = ReplayBuffer(
    capacity=1_000_000,    # Max transitions
    obs_shape=(4, 84, 84), # Observation shape
    normalize=True,        # [0,1] normalization
    min_size=50_000,       # Warm-up threshold
    device='cuda',         # GPU tensors
    pin_memory=True        # Faster transfer
)

# Add transition
buffer.append(state, action, reward, next_state, done)

# Check ready
if buffer.can_sample(batch_size=32):
    # Sample batch
    batch = buffer.sample(32)
    # batch = {
    #     'states': (32, 4, 84, 84) float32/tensor,
    #     'actions': (32,) int64/tensor,
    #     'rewards': (32,) float32/tensor,
    #     'next_states': (32, 4, 84, 84) float32/tensor,
    #     'dones': (32,) bool/tensor
    # }

# Check size
print(f"Buffer: {len(buffer)}/{buffer.capacity}")

# Validate
pytest tests/test_replay_buffer.py -v
```
