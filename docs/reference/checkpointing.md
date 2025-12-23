# Checkpointing and Resume

**Objective:** Save and restore complete training state to enable resumption and reproducibility.

---

## Table of Contents

**Quick Links:** [Quick Start](#quick-start-90-seconds) | [CLI Usage](#usage) | [Resume Training](#resume-training) | [Determinism](#deterministic-seeding) | [Troubleshooting](#troubleshooting)

### Getting Started
1. [Quick Start (90 seconds)](#quick-start-90-seconds)
2. [Checkpoint Structure](#checkpoint-structure)
3. [File Format](#file-format)
4. [Atomic Writes](#atomic-writes)
5. [Usage (CLI)](#usage)

### Resuming Training
6. [Resume Training](#resume-training)
7. [RNG State Management](#rng-state-management)
8. [Checkpoint Validation](#checkpoint-validation)

### Deterministic Execution
9. [Deterministic Seeding](#deterministic-seeding)
10. [Checkpoint/Resume Procedures](#checkpointresume-procedures)

### Reference
11. [Metadata Files](#metadata-files-in-resumed-runs)
12. [Best Practices](#best-practices)
13. [Testing](#testing)
14. [Troubleshooting](#troubleshooting)
15. [Related Documentation](#related-documentation)

---

## Quick Start (90 seconds)

### Save a Checkpoint (Automatic)

Checkpoints are saved automatically during training:

```bash
# Training automatically saves checkpoints every 1M frames
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42
```

**Expected output:**
```
experiments/dqn_atari/runs/pong_42/checkpoints/
├── checkpoint_1000000.pt    # Every 1M frames
├── checkpoint_2000000.pt
└── best_model.pt            # Best eval score
```

### Resume from Checkpoint

```bash
# Basic resume (continues training)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --resume experiments/dqn_atari/runs/pong_42/checkpoints/checkpoint_1000000.pt

# Resume with strict config validation
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --resume experiments/dqn_atari/runs/pong_42/checkpoints/checkpoint_1000000.pt \
  --strict-resume
```

### Verify Determinism

```bash
# Run smoke test to verify save/resume determinism
pytest tests/test_save_resume_determinism.py -v -s

# Expected output:
# DONE PERFECT DETERMINISM - All metrics match exactly
```

**Need more details?** Jump to [Command-Line Interface](#command-line-interface) or [Deterministic Execution](#deterministic-execution).

---

## Checkpoint Structure

Checkpoints save the complete training state including:

- **Online and target Q-network weights** (state_dict for both models)
- **Optimizer state** (momentum buffers, learning rate, etc.)
- **Training counters:**
  - Environment step counter
  - Episode counter
  - Current epsilon value
- **Replay buffer state:**
  - Write index and size (always saved)
  - Full buffer content (optional, configurable via `save_replay_buffer=True`)
- **RNG states** for reproducibility:
  - Python `random` module state
  - NumPy random state
  - PyTorch CPU random state
  - PyTorch CUDA random state (if GPU available)
  - Environment RNG state (if supported)
- **Metadata:**
  - Schema version (for compatibility checking)
  - Timestamp (ISO 8601 format)
  - Git commit hash (for code version tracking)
  - Custom metadata (optional)

## File Format

Checkpoints are saved as PyTorch `.pt` files using `torch.save()`:

```
experiments/dqn_atari/checkpoints/
├── checkpoint_1000000.pt    # Periodic checkpoint at 1M steps
├── checkpoint_2000000.pt    # Periodic checkpoint at 2M steps
├── checkpoint_3000000.pt    # Periodic checkpoint at 3M steps
└── best_model.pt            # Best model by eval return
```

### Schema

```python
{
    # Metadata
    'schema_version': '1.0.0',
    'timestamp': '2025-01-15T10:30:45.123456',
    'commit_hash': 'a1b2c3d',

    # Training state
    'step': 1000000,
    'episode': 5000,
    'epsilon': 0.5,

    # Model weights
    'online_model_state_dict': {...},
    'target_model_state_dict': {...},

    # Optimizer
    'optimizer_state_dict': {...},

    # RNG states
    'rng_states': {
        'python_random': (...),
        'numpy_random': (...),
        'torch_cpu': tensor(...),
        'torch_cuda': [tensor(...), ...],  # If CUDA available
        'env': {...}  # If environment supports it
    },

    # Replay buffer (conditional)
    'replay_buffer_state': {
        'index': 250000,
        'size': 250000,
        'capacity': 1000000,
        'obs_shape': (4, 84, 84),
        # Optional full data (only if save_replay_buffer=True)
        'data': {
            'observations': ndarray(...),
            'actions': ndarray(...),
            'rewards': ndarray(...),
            'dones': ndarray(...),
            'episode_starts': ndarray(...)
        }
    },

    # Optional user metadata
    'metadata': {
        'config': {...},
        'notes': '...'
    }
}
```

## Atomic Writes

Checkpoints use atomic writes to prevent corruption if the process is killed during save:

1. Write to temporary file with `.pt.tmp` suffix
2. Use `os.replace()` to atomically rename temp file to final name
3. Clean up temp file on any errors

This ensures checkpoints are never partially written.

## Usage

### Saving Checkpoints

```python
from src.training import CheckpointManager, get_rng_states

# Initialize manager
manager = CheckpointManager(
    checkpoint_dir='runs/pong_123/checkpoints',
    save_interval=1_000_000,  # Save every 1M steps
    keep_last_n=3,            # Keep last 3 periodic checkpoints
    save_best=True,           # Track and save best model
    save_replay_buffer=False  # Don't save full buffer (saves space)
)

# During training loop
if manager.should_save(step):
    # Capture RNG states
    rng_states = get_rng_states(env)

    # Save periodic checkpoint
    checkpoint_path = manager.save_checkpoint(
        step=step,
        episode=episode_count,
        epsilon=epsilon,
        online_model=online_model,
        target_model=target_model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        rng_states=rng_states,
        extra_metadata={'config': config}
    )
    print(f"Saved checkpoint: {checkpoint_path}")

# After evaluation
if eval_return > best_return:
    saved = manager.save_best(
        step=step,
        episode=episode_count,
        epsilon=epsilon,
        eval_return=eval_return,
        online_model=online_model,
        target_model=target_model,
        optimizer=optimizer,
        rng_states=rng_states
    )
    if saved:
        print(f"New best model! Return: {eval_return}")
```

### Loading Checkpoints

```python
from src.training import CheckpointManager, set_rng_states

# Initialize manager and models
manager = CheckpointManager(checkpoint_dir='runs/pong_123/checkpoints')
online_model = DQN(num_actions=6)
target_model = DQN(num_actions=6)
optimizer = torch.optim.RMSprop(online_model.parameters(), lr=2.5e-4)
replay_buffer = ReplayBuffer(capacity=1_000_000)

# Load checkpoint
checkpoint_path = 'runs/pong_123/checkpoints/checkpoint_2000000.pt'
loaded_state = manager.load_checkpoint(
    checkpoint_path=checkpoint_path,
    online_model=online_model,
    target_model=target_model,
    optimizer=optimizer,
    replay_buffer=replay_buffer,  # Will restore index and size
    device='cuda'
)

# Resume training from loaded state
step = loaded_state['step']
episode = loaded_state['episode']
epsilon = loaded_state['epsilon']

# Restore RNG states for reproducibility
set_rng_states(loaded_state['rng_states'], env)

print(f"Resumed from step {step}, episode {episode}")
print(f"Commit: {loaded_state['commit_hash']}")
print(f"Saved at: {loaded_state['timestamp']}")
```

## RNG State Management

### Capturing RNG States

```python
from src.training import get_rng_states

# Capture all RNG states
rng_states = get_rng_states(env)

# Returns dict with:
# - python_random: Python random.getstate()
# - numpy_random: numpy.random.get_state()
# - torch_cpu: torch.get_rng_state()
# - torch_cuda: torch.cuda.get_rng_state_all() (if CUDA)
# - env: env.get_rng_state() (if env supports it)
```

### Restoring RNG States

```python
from src.training import set_rng_states

# Restore from checkpoint
set_rng_states(checkpoint['rng_states'], env)

# Verifies deterministic behavior after restore
```

## Checkpoint Validation

```python
from src.training import verify_checkpoint_integrity

# Verify checkpoint is valid before loading
is_valid = verify_checkpoint_integrity('path/to/checkpoint.pt')

if is_valid:
    print("Checkpoint is valid")
else:
    print("Checkpoint is corrupted or incomplete")
```

## Best Practices

### Storage Management

- **Don't save full replay buffer** by default (`save_replay_buffer=False`)
  - Saves significant disk space (1M transitions ~300GB uncompressed)
  - Index and size are sufficient to resume (buffer refills quickly)

- **Limit periodic checkpoints** (`keep_last_n=3`)
  - Prevents unbounded disk usage
  - Keep enough for safety but not too many

- **Always save best model** (`save_best=True`)
  - Track best performance even if training diverges
  - Useful for final model selection

### Deterministic Resumption

For fully reproducible resumes:

1. **Capture RNG states** when saving
2. **Restore RNG states** when loading
3. **Set deterministic flags** in PyTorch:
   ```python
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   ```
4. **Track git commit hash** to ensure code version matches

### Error Handling

```python
try:
    checkpoint_path = manager.save_checkpoint(...)
except Exception as e:
    print(f"Checkpoint save failed: {e}")
    # Continue training (atomic writes prevent corruption)
```

### Resume CLI Pattern

```python
if args.resume:
    # Load checkpoint
    loaded = manager.load_checkpoint(
        args.resume,
        online_model,
        target_model,
        optimizer,
        replay_buffer
    )

    # Restore state
    start_step = loaded['step']
    episode_count = loaded['episode']
    epsilon = loaded['epsilon']
    set_rng_states(loaded['rng_states'], env)

    print(f"Resumed from {args.resume}")
else:
    # Fresh start
    start_step = 0
    episode_count = 0
    epsilon = 1.0
```

## Troubleshooting

### Schema Version Mismatch

If loading shows warning about schema mismatch:
- Check `loaded_state['schema_version']`
- Current version: `1.0.0`
- Older checkpoints may be incompatible

### Missing RNG States

If RNG states are missing from old checkpoints:
- Set fresh seeds: `torch.manual_seed(seed)`
- Accept non-deterministic resume
- Re-save checkpoint with RNG states

### Replay Buffer Not Restored

If replay buffer data wasn't saved:
- Only index/size are saved by default
- Buffer refills during warm-up
- For exact resume, use `save_replay_buffer=True`

### Torch Load Errors

If getting `UnpicklingError` or `weights_only` warnings:
- Checkpoints use `weights_only=False` (required for numpy states)
- This is safe for checkpoints you created yourself
- Don't load untrusted checkpoints

## Testing

Run checkpoint tests:

```bash
pytest tests/test_checkpoint.py -v
```

Tests cover:
- Save/load round-trip
- RNG state capture/restore
- Replay buffer state
- Atomic writes
- Schema validation
- Best model tracking
- Checkpoint rotation (keep_last_n)

## Resume Training

### CLI Integration

Add resume arguments to your training script:

```python
from src.training import add_resume_args

parser = argparse.ArgumentParser()
add_resume_args(parser)
args = parser.parse_args()

# Usage:
# python train_dqn.py --resume checkpoints/checkpoint_1000000.pt
# python train_dqn.py --resume checkpoints/checkpoint_1000000.pt --strict-resume
```

### Full Resume Example

```python
from src.training import resume_from_checkpoint, CheckpointManager, EpsilonScheduler
from src.models.dqn import DQN
from src.replay import ReplayBuffer

# Initialize fresh objects
online_model = DQN(num_actions=6)
target_model = DQN(num_actions=6)
optimizer = torch.optim.RMSprop(online_model.parameters(), lr=2.5e-4)
epsilon_scheduler = EpsilonScheduler(
    epsilon_start=1.0,
    epsilon_end=0.1,
    decay_frames=1_000_000
)
replay_buffer = ReplayBuffer(capacity=1_000_000)

# Resume from checkpoint
resumed = resume_from_checkpoint(
    checkpoint_path=args.resume,
    online_model=online_model,
    target_model=target_model,
    optimizer=optimizer,
    epsilon_scheduler=epsilon_scheduler,
    replay_buffer=replay_buffer,
    env=env,
    config=config,
    device='cuda',
    strict_config=args.strict_resume
)

# Extract resumed state
start_step = resumed['next_step']  # Resume from next step
episode_count = resumed['episode']
current_epsilon = resumed['epsilon']

print(f"Resuming training from step {start_step}")

# Continue training loop
for step in range(start_step, total_steps):
    # Training continues normally...
    epsilon = epsilon_scheduler.get_epsilon(step)
    # ...
```

### Config Validation

Resume validates configuration compatibility automatically:

**Critical Parameters (must match):**
- Environment ID
- Frame size
- Frame stack size

**Important Parameters (warnings only):**
- Learning rate
- Discount factor (gamma)
- Batch size
- Target update interval
- Replay capacity

**Strict Mode:**
```bash
# Enforce strict config match (error on mismatch)
python train_dqn.py --resume checkpoint.pt --strict-resume
```

**Warnings Example:**
```
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  CRITICAL: Environment ID mismatch - checkpoint: ALE/Pong-v5, current: ALE/Breakout-v5
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

WARNING: Learning rate differs - checkpoint: 0.00025, current: 0.0001
```

### Git Hash Verification

Resume automatically checks for code changes:

```
WARNING: Git commit hash mismatch
  Checkpoint was saved at commit: abc1234
  Current commit: def5678
  Code changes may affect reproducibility
```

### State Restoration

What gets restored:
- DONE Model weights (online and target)
- DONE Optimizer state (momentum, learning rate, etc.)
- DONE Training counters (step, episode)
- DONE Epsilon value and scheduler state
- DONE Replay buffer (index, size, optionally full data)
- DONE RNG states (Python, NumPy, PyTorch, CUDA, environment)

### Epsilon Schedule Restoration

The epsilon scheduler state is fully restored:

```python
# Checkpoint saved at step=500,000 with epsilon=0.55
# After resume:
assert epsilon_scheduler.frame_counter == 500_000
assert epsilon_scheduler.current_epsilon == 0.55
assert epsilon_scheduler.get_epsilon(500_000) == 0.55

# Training continues with correct decay
epsilon_scheduler.get_epsilon(600_000)  # Returns 0.51
```

### Troubleshooting Resume

**Error: FileNotFoundError**
- Check checkpoint path is correct
- Use absolute paths or relative to working directory

**Error: Config incompatibility**
- Review warnings for critical mismatches
- Use `--strict-resume` to enforce compatibility
- Ensure checkpoint matches current architecture

**Warning: Epsilon mismatch**
- Minor differences (< 0.01) are normal due to floating point
- Large differences indicate scheduler misconfiguration

**Warning: Unable to verify git hash**
- Not in a git repository
- Checkpoint saved outside git repo
- Acceptable if not tracking code versions

**Replay buffer not restored**
- Check if checkpoint saved with `save_replay_buffer=True`
- Default only saves index/size (buffer refills during warm-up)

## Deterministic Seeding

### Centralized Seeding Function

The `set_seed()` function provides centralized seeding for all random number generators:

```python
from src.utils import set_seed

# Initial seeding (once at training start)
set_seed(seed=42, deterministic=True)

# Seed with environment reset
obs, info = set_seed(seed=42 + episode, env=env)

# For multiprocessing workers
set_seed(seed=base_seed + worker_id, deterministic=True)
```

### When to Call set_seed()

**1. Training Initialization**
```python
# Once at the very start of training
set_seed(args.seed, deterministic=args.deterministic)
```

**2. Every Environment Reset**
```python
# Use episode-specific seed for each reset
for episode in range(num_episodes):
    obs, info = env.reset(seed=base_seed + episode)
    # Or use convenience function:
    obs, info = seed_env(env, base_seed + episode)
```

**3. After Resume from Checkpoint**
```python
# RNG states are restored automatically by resume_from_checkpoint()
# But you can manually seed if needed:
set_seed(checkpoint['seed'])
```

**4. Multiprocessing Workers**
```python
def worker_init_fn(worker_id):
    """Initialize worker with unique seed."""
    worker_seed = base_seed + worker_id
    set_seed(worker_seed, deterministic=True)
```

### What Gets Seeded

`set_seed()` seeds all random number generators:
- DONE Python `random` module
- DONE NumPy `np.random`
- DONE PyTorch CPU (`torch.manual_seed`)
- DONE PyTorch CUDA (`torch.cuda.manual_seed_all`)
- DONE Environment (via `env.reset(seed=...)`)

### Deterministic Mode Configuration

Configure deterministic behavior for reproducibility vs performance trade-off:

```python
from src.utils import configure_determinism

# Basic determinism (recommended for reproducibility)
configure_determinism(enabled=True, strict=False)

# Strict determinism (for debugging non-determinism sources)
configure_determinism(enabled=True, strict=True, warn_only=True)

# Disable determinism (faster training)
configure_determinism(enabled=False)
```

#### Configuration Options

**1. Basic Deterministic Mode (`enabled=True`)**

Sets:
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

Effects:
- DONE Ensures reproducible cuDNN operations
- DONE Disables autotuning that may vary between runs
- TODO ~10-20% slower training speed
- TODO No benefit on CPU-only training

**2. Strict Deterministic Mode (`strict=True`)**

Sets:
- All basic mode settings
- `torch.use_deterministic_algorithms(True)`

Effects:
- DONE Enforces deterministic implementations for all operations
- DONE Helps debug sources of non-determinism
- TODO May raise errors for operations without deterministic implementations
- TODO Additional performance overhead
- TODO Not all PyTorch operations have deterministic versions

**3. Warn-Only Mode (`warn_only=True`)**

When combined with `strict=True`:
- Warns instead of raising errors on non-deterministic operations
- Useful for identifying non-deterministic code without breaking training
- Requires PyTorch >= 1.11

#### Configuration via Config File

Add to `base.yaml`:

```yaml
experiment:
  deterministic:
    enabled: false  # Set true for reproducibility
    strict: false   # Set true for debugging only
    warn_only: true # Warn instead of error in strict mode
```

Apply in training script:

```python
from src.utils import configure_determinism

# Load from config
det_config = config['experiment']['deterministic']
settings = configure_determinism(
    enabled=det_config['enabled'],
    strict=det_config['strict'],
    warn_only=det_config['warn_only']
)

print(f"Determinism configured: {settings}")
```

#### Performance Impact

**Benchmark Results (DQN Atari Pong, 1M frames):**

| Mode | Training Time | FPS | Reproducibility |
|------|--------------|-----|-----------------|
| Disabled (`enabled=False`) | 100% baseline | ~1000 FPS | FAIL Non-deterministic |
| Basic (`enabled=True, strict=False`) | ~110-120% | ~850 FPS | DONE Reproducible |
| Strict (`enabled=True, strict=True`) | ~120-130% | ~800 FPS | DONEDONE Fully deterministic |

**Recommendations:**

- **Production training:** `enabled=False` (fastest)
- **Reproducibility experiments:** `enabled=True, strict=False` (recommended)
- **Debugging non-determinism:** `enabled=True, strict=True, warn_only=True`
- **Paper reproduction:** `enabled=True, strict=False` (good balance)

#### Checking Current Status

```python
from src.utils import get_determinism_status

status = get_determinism_status()
print(f"cuDNN deterministic: {status['cudnn_deterministic']}")
print(f"cuDNN benchmark: {status['cudnn_benchmark']}")
print(f"Strict algorithms: {status['strict_algorithms']}")
```

#### Known Limitations

**Operations Without Deterministic Implementations:**

Some PyTorch operations don't have deterministic versions:
- `torch.nn.functional.interpolate` (some modes)
- Scatter/gather operations with CUDA
- Some CUDA atomic operations
- Non-deterministic sampling operations

**Workarounds:**
- Use `warn_only=True` to identify issues
- Replace non-deterministic operations with alternatives
- Accept minor non-determinism for these specific operations

**Hardware Differences:**

Even with determinism enabled:
- Different GPUs may produce slightly different results
- CPU vs GPU results may differ due to floating-point precision
- Multi-GPU training may have synchronization issues

**PyTorch Version Compatibility:**

- `torch.use_deterministic_algorithms()` requires PyTorch >= 1.8
- `warn_only` parameter requires PyTorch >= 1.11
- Older versions fall back gracefully with warnings

### Seed Recording in Metadata

Seeds are automatically recorded in run metadata:

```python
from src.utils import save_run_metadata

save_run_metadata(
    output_dir='runs/pong_123',
    config=config,
    seed=42  # Recorded in meta.json
)
```

The metadata file (`meta.json`) includes:
```json
{
  "seed": 42,
  "git": {
    "commit": "abc123",
    "branch": "main",
    "dirty": false
  },
  "config": {...}
}
```

### Episode-Specific Seeding Pattern

For deterministic episode sequences:

```python
base_seed = 42

# Episode 0: seed=42
obs, info = env.reset(seed=base_seed + 0)
# ... run episode ...

# Episode 1: seed=43
obs, info = env.reset(seed=base_seed + 1)
# ... run episode ...

# Episode 2: seed=44
obs, info = env.reset(seed=base_seed + 2)
# ... run episode ...
```

This ensures:
- Each episode has deterministic behavior
- Episodes can be reproduced individually
- Different episodes have different randomness

### Testing Determinism

Verify deterministic behavior:

```bash
# Run seeding tests
pytest tests/test_seeding.py -v
```

Tests verify:
- Python/NumPy/PyTorch seeding
- Environment seeding
- Deterministic flags
- Multiprocessing isolation
- Resume reproducibility

### Troubleshooting Determinism

**Non-deterministic results after seeding:**
1. Check that `deterministic=True` is set
2. Verify environment supports seeding
3. Some CUDA operations are inherently non-deterministic
4. Check for uninitialized random calls

**Different results on CPU vs GPU:**
- GPU operations may have numerical differences
- Use `torch.backends.cudnn.deterministic=True`
- Some operations don't have deterministic implementations

**Multiprocessing issues:**
- Ensure each worker calls `set_seed(base_seed + worker_id)`
- Workers inherit parent's seed if not explicitly set
- Use `worker_init_fn` for DataLoader workers

## Checkpoint/Resume Procedures

### Complete Checkpoint Save Procedure

**Step 1: Initialize CheckpointManager**

```python
from src.training import CheckpointManager, get_rng_states

manager = CheckpointManager(
    checkpoint_dir='runs/pong_123/checkpoints',
    save_interval=1_000_000,  # Every 1M steps
    keep_last_n=3,            # Keep last 3 checkpoints
    save_best=True,           # Track best eval score
    save_replay_buffer=False  # Don't save full buffer (saves space)
)
```

**Step 2: During Training Loop**

```python
# Check if it's time to save
if manager.should_save(current_step):
    # Capture current RNG states
    rng_states = get_rng_states(env)

    # Save periodic checkpoint
    checkpoint_path = manager.save_checkpoint(
        step=current_step,
        episode=episode_count,
        epsilon=current_epsilon,
        online_model=online_model,
        target_model=target_model,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        rng_states=rng_states,
        extra_metadata={'config': config}
    )

    print(f"Saved checkpoint: {checkpoint_path}")
```

**Step 3: Save Best Model After Evaluation**

```python
# After running evaluation
if eval_return > best_return:
    best_return = eval_return

    saved = manager.save_best(
        step=current_step,
        episode=episode_count,
        epsilon=current_epsilon,
        eval_return=eval_return,
        online_model=online_model,
        target_model=target_model,
        optimizer=optimizer,
        rng_states=rng_states
    )

    if saved:
        print(f"New best model! Return: {eval_return:.2f}")
```

### Complete Resume Procedure

**Step 1: Add CLI Arguments**

```python
import argparse
from src.training import add_resume_args

parser = argparse.ArgumentParser()
add_resume_args(parser)
args = parser.parse_args()

# Usage:
# python train.py --resume checkpoints/checkpoint_1000000.pt
# python train.py --resume checkpoints/checkpoint_1000000.pt --strict-resume
```

**Step 2: Initialize Fresh Objects**

```python
from src.training import resume_from_checkpoint
from src.models.dqn import DQN

# Create fresh objects (will be loaded from checkpoint)
online_model = DQN(num_actions=num_actions)
target_model = DQN(num_actions=num_actions)
optimizer = configure_optimizer(online_model, optimizer_type='rmsprop')
epsilon_scheduler = EpsilonScheduler(
    epsilon_start=1.0,
    epsilon_end=0.1,
    decay_frames=1_000_000
)
replay_buffer = ReplayBuffer(capacity=1_000_000)
```

**Step 3: Resume from Checkpoint**

```python
if args.resume:
    resumed = resume_from_checkpoint(
        checkpoint_path=args.resume,
        online_model=online_model,
        target_model=target_model,
        optimizer=optimizer,
        epsilon_scheduler=epsilon_scheduler,
        replay_buffer=replay_buffer,
        env=env,
        config=config,
        device=device,
        strict_config=args.strict_resume
    )

    # Extract resumed state
    start_step = resumed['next_step']  # Resume from next step
    episode_count = resumed['episode']
    current_epsilon = resumed['epsilon']

    print(f"Resumed from step {start_step}")
else:
    # Fresh start
    start_step = 0
    episode_count = 0
    current_epsilon = 1.0
```

**Step 4: Continue Training Loop**

```python
for step in range(start_step, total_steps):
    # Training continues normally
    epsilon = epsilon_scheduler.get_epsilon(step)
    # ... rest of training loop
```

### Saved Tensors Reference

Complete list of tensors and data saved in checkpoints:

**Model Weights:**
- `online_model_state_dict`: Online Q-network parameters
- `target_model_state_dict`: Target Q-network parameters

**Optimizer:**
- `optimizer_state_dict`: Momentum buffers, learning rate, step counts

**Training State:**
- `step`: Environment step counter (int)
- `episode`: Episode counter (int)
- `epsilon`: Current epsilon value (float)

**Replay Buffer:**
- `replay_buffer_state`:
  - `index`: Write index (int)
  - `size`: Current size (int)
  - `capacity`: Maximum capacity (int)
  - `obs_shape`: Observation shape (tuple)
  - `data`: Full buffer content (optional, only if `save_replay_buffer=True`)
    - `observations`: numpy array
    - `actions`: numpy array
    - `rewards`: numpy array
    - `dones`: numpy array
    - `episode_starts`: numpy array

**RNG States:**
- `rng_states`:
  - `python_random`: Python random.getstate()
  - `numpy_random`: numpy.random.get_state()
  - `torch_cpu`: torch.get_rng_state()
  - `torch_cuda`: torch.cuda.get_rng_state_all() (if CUDA)
  - `env`: env.get_rng_state() (if supported)

**Metadata:**
- `schema_version`: Checkpoint format version (string)
- `timestamp`: ISO 8601 timestamp (string)
- `commit_hash`: Git commit hash (string)
- `metadata`: Custom user metadata (dict, optional)

### Metadata Schema

Checkpoints use schema version `1.0.0` with the following structure:

```python
{
    # Required fields
    'schema_version': '1.0.0',
    'timestamp': '2025-01-15T10:30:45.123456',
    'commit_hash': 'a1b2c3d',
    'step': 1000000,
    'episode': 5000,
    'epsilon': 0.5,

    # Model states
    'online_model_state_dict': OrderedDict(...),
    'target_model_state_dict': OrderedDict(...),

    # Optimizer
    'optimizer_state_dict': {...},

    # RNG states
    'rng_states': {
        'python_random': (...),
        'numpy_random': (...),
        'torch_cpu': tensor(...),
        'torch_cuda': [tensor(...), ...],  # If CUDA
        'env': {...}  # If env supports it
    },

    # Replay buffer
    'replay_buffer_state': {
        'index': 250000,
        'size': 250000,
        'capacity': 1000000,
        'obs_shape': (4, 84, 84),
        # Optional 'data' dict if save_replay_buffer=True
    },

    # Optional metadata
    'metadata': {
        'config': {...},
        'notes': '...'
    }
}
```

### Resume CLI Usage

**Basic Resume:**
```bash
python train_dqn.py --resume checkpoints/checkpoint_1000000.pt
```

**Strict Resume (enforce config compatibility):**
```bash
python train_dqn.py --resume checkpoints/checkpoint_1000000.pt --strict-resume
```

**Resume without RNG restoration:**
```bash
python train_dqn.py --resume checkpoints/checkpoint_1000000.pt --no-restore-rng
```

**Resume from best model:**
```bash
python train_dqn.py --resume checkpoints/best_model.pt
```

### Verifying Deterministic Resumes

**Checklist for Deterministic Resumes:**

1. DONE **Enable deterministic mode:**
   ```python
   configure_determinism(enabled=True, strict=False)
   set_seed(seed, deterministic=True)
   ```

2. DONE **Save checkpoint with RNG states:**
   ```python
   rng_states = get_rng_states(env)
   checkpoint_path = manager.save_checkpoint(..., rng_states=rng_states)
   ```

3. DONE **Resume with RNG restoration:**
   ```python
   resumed = resume_from_checkpoint(..., env=env)  # RNG states auto-restored
   ```

4. DONE **Verify git commit hash matches:**
   - Check warning: "Git commit hash mismatch"
   - Ensure code version is identical

5. DONE **Check config compatibility:**
   - Critical params must match (env ID, frame size, stack size)
   - Important params should match (learning rate, gamma, batch size)

**Run Smoke Test:**

```bash
# Run determinism verification smoke test
pytest tests/test_save_resume_determinism.py -v -s

# Expected output:
# DONE PERFECT DETERMINISM - All metrics match exactly
# Epsilon Matches: 100.0%
# Reward Matches: 100.0%
# Action Matches: 100.0%
# Checksum Match: DONE PASS
```

The smoke test:
- Runs 5000 steps total
- Saves checkpoint at step 2500
- Resumes from checkpoint
- Verifies epsilon, rewards, and actions match exactly
- Reports detailed comparison with checksums

### Debugging Mismatched States

**Issue: Different epsilon values after resume**

Check:
1. Epsilon scheduler state restoration:
   ```python
   assert epsilon_scheduler.frame_counter == checkpoint['step']
   assert abs(epsilon_scheduler.current_epsilon - checkpoint['epsilon']) < 1e-6
   ```

2. Epsilon calculation is deterministic:
   ```python
   eps = epsilon_scheduler.get_epsilon(step)
   # Should return same value for same step
   ```

**Issue: Different actions selected after resume**

Check:
1. RNG states were restored:
   ```python
   assert 'rng_states' in checkpoint
   set_rng_states(checkpoint['rng_states'], env)
   ```

2. Action selection uses same epsilon:
   ```python
   # Both runs should use identical epsilon
   action = select_epsilon_greedy_action(model, state, epsilon, num_actions)
   ```

3. Model weights are identical:
   ```python
   for p1, p2 in zip(model1.parameters(), model2.parameters()):
       assert torch.allclose(p1, p2)
   ```

**Issue: Different rewards after resume**

Check:
1. Environment seeding:
   ```python
   obs, info = env.reset(seed=seed + episode)
   ```

2. Environment RNG state restored:
   ```python
   if hasattr(env, 'set_rng_state'):
       env.set_rng_state(checkpoint['rng_states']['env'])
   ```

3. Action sequence is identical:
   ```python
   # Different actions lead to different rewards
   # Verify actions match first
   ```

**Issue: Training diverges after resume**

Check:
1. Optimizer state restored:
   ```python
   optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
   # Momentum buffers should be restored
   ```

2. Target network weights match:
   ```python
   target_model.load_state_dict(checkpoint['target_model_state_dict'])
   ```

3. Replay buffer state restored:
   ```python
   assert replay_buffer.index == checkpoint['replay_buffer_state']['index']
   assert replay_buffer.size == checkpoint['replay_buffer_state']['size']
   ```

### Commands for Creating/Restoring Checkpoints

**Create Checkpoint Manually:**

```python
from src.training import CheckpointManager, get_rng_states
import torch

# Initialize manager
manager = CheckpointManager(checkpoint_dir='checkpoints')

# Save checkpoint
checkpoint_path = manager.save_checkpoint(
    step=10000,
    episode=500,
    epsilon=0.95,
    online_model=online_model,
    target_model=target_model,
    optimizer=optimizer,
    replay_buffer=replay_buffer,
    rng_states=get_rng_states(env),
    extra_metadata={'note': 'Manual checkpoint'}
)
```

**Load Checkpoint Manually:**

```python
# Load checkpoint
loaded = manager.load_checkpoint(
    checkpoint_path='checkpoints/checkpoint_10000.pt',
    online_model=online_model,
    target_model=target_model,
    optimizer=optimizer,
    replay_buffer=replay_buffer,
    device='cuda'
)

# Access state
step = loaded['step']
episode = loaded['episode']
epsilon = loaded['epsilon']
timestamp = loaded['timestamp']
```

**Verify Checkpoint Integrity:**

```python
from src.training import verify_checkpoint_integrity

is_valid = verify_checkpoint_integrity('checkpoints/checkpoint_10000.pt')

if is_valid:
    print("DONE Checkpoint is valid")
else:
    print("TODO Checkpoint is corrupted")
```

**Inspect Checkpoint Contents:**

```python
import torch

checkpoint = torch.load('checkpoints/checkpoint_10000.pt', weights_only=False)

print(f"Schema version: {checkpoint['schema_version']}")
print(f"Step: {checkpoint['step']}")
print(f"Epsilon: {checkpoint['epsilon']}")
print(f"Timestamp: {checkpoint['timestamp']}")
print(f"Commit: {checkpoint['commit_hash']}")

# Check replay buffer
if 'replay_buffer_state' in checkpoint:
    buf = checkpoint['replay_buffer_state']
    print(f"Buffer size: {buf['size']} / {buf['capacity']}")
    has_data = 'data' in buf
    print(f"Full buffer saved: {has_data}")
```

## Metadata Files in Resumed Runs

### Run Metadata (`meta.json`)

Every training run creates a `meta.json` file in the run directory:

```
experiments/dqn_atari/runs/pong_123/meta.json
```

**Schema:**

```json
{
  "seed": 42,
  "git": {
    "commit": "a1b2c3d4e5f6...",
    "branch": "main",
    "dirty": false
  },
  "config": {
    "experiment": {...},
    "env": {...},
    "agent": {...},
    "training": {...}
  },
  "ale_settings": {
    "frameskip": 4,
    "repeat_action_probability": 0.0,
    "max_noop_start": 30
  },
  "extra": {
    "created_at": "2025-01-15T10:30:00",
    "python_version": "3.10.13",
    "pytorch_version": "2.4.1+cu121"
  }
}
```

### Checkpoint Metadata

Each checkpoint file (`.pt`) contains embedded metadata:

**Checkpoint structure:**

```python
checkpoint = {
    # Metadata
    'schema_version': '1.0.0',
    'timestamp': '2025-01-15T10:30:45.123456',
    'commit_hash': 'a1b2c3d',

    # Training state
    'step': 1000000,
    'episode': 5000,
    'epsilon': 0.5,

    # Model and optimizer
    'online_model_state_dict': {...},
    'target_model_state_dict': {...},
    'optimizer_state_dict': {...},

    # RNG states (for determinism)
    'rng_states': {
        'python_random': (...),
        'numpy_random': (...),
        'torch_cpu': tensor(...),
        'torch_cuda': [...],  # If CUDA
        'env': {...}  # If supported
    },

    # Replay buffer
    'replay_buffer_state': {
        'index': 250000,
        'size': 250000,
        'capacity': 1000000,
        'obs_shape': (4, 84, 84)
    },

    # Optional user metadata
    'metadata': {
        'config': {...},
        'notes': 'Manual checkpoint'
    }
}
```

### Metadata Before vs After Resume

**Before Resume (Original Run):**

```json
// meta.json created at run start
{
  "seed": 42,
  "git": {
    "commit": "abc123",
    "branch": "main",
    "dirty": false
  },
  "config": {...}
}
```

Checkpoint at step 1M:
```python
{
    'timestamp': '2025-01-15T10:30:45',
    'commit_hash': 'abc123',
    'step': 1000000,
    'rng_states': {...}  # Captured at save
}
```

**After Resume:**

The resumed run continues using the same `meta.json` (no changes).
The checkpoint loaded contains the saved RNG states, which are restored:

```python
# Resume restores from checkpoint
loaded = torch.load('checkpoint_1000000.pt')

# RNG states restored
set_rng_states(loaded['rng_states'], env)

# Training continues from step 1,000,001
# All random operations now deterministic from this point
```

**Verification:**

To confirm RNG states were restored, check console output:

```
Restoring RNG states for reproducibility...
  DONE Python random state restored
  DONE NumPy random state restored
  DONE PyTorch random state restored
  DONE CUDA random state restored
  DONE Environment random state restored
```

### Finding Metadata in Resumed Runs

**Run metadata:**

```bash
# View run metadata
cat experiments/dqn_atari/runs/pong_123/meta.json | jq

# Check git commit
jq '.git.commit' experiments/dqn_atari/runs/pong_123/meta.json

# Check seed
jq '.seed' experiments/dqn_atari/runs/pong_123/meta.json
```

**Checkpoint metadata:**

```python
import torch

# Load checkpoint
checkpoint = torch.load('checkpoint_1000000.pt', weights_only=False)

# Check metadata
print(f"Schema: {checkpoint['schema_version']}")
print(f"Saved at: {checkpoint['timestamp']}")
print(f"Commit: {checkpoint['commit_hash']}")
print(f"Step: {checkpoint['step']}")

# Check RNG states present
print(f"RNG states saved: {'rng_states' in checkpoint}")
if 'rng_states' in checkpoint:
    print(f"  Python random: {'python_random' in checkpoint['rng_states']}")
    print(f"  NumPy random: {'numpy_random' in checkpoint['rng_states']}")
    print(f"  PyTorch CPU: {'torch_cpu' in checkpoint['rng_states']}")
    print(f"  CUDA: {'torch_cuda' in checkpoint['rng_states']}")
    print(f"  Environment: {'env' in checkpoint['rng_states']}")
```

**Expected fields after resume:**

| Field | Location | Purpose | Expected on Resume |
|-------|----------|---------|-------------------|
| `seed` | `meta.json` | Base random seed | Same as original |
| `git.commit` | `meta.json` | Code version | Same (warns if different) |
| `timestamp` | Checkpoint | Save time | From checkpoint |
| `step` | Checkpoint | Training step | Loaded and used |
| `epsilon` | Checkpoint | Exploration rate | Restored to scheduler |
| `rng_states` | Checkpoint | RNG states | Restored to all sources |
| `replay_buffer_state` | Checkpoint | Buffer state | Index/size restored |

### Troubleshooting Metadata Issues

**Issue: Git commit hash mismatch**

Resume shows:
```
WARNING: Git commit hash mismatch
  Checkpoint was saved at commit: abc123
  Current commit: def456
  Code changes may affect reproducibility
```

**Solution:**
```bash
# Check current commit
git rev-parse HEAD

# Check checkpoint commit
python -c "import torch; print(torch.load('checkpoint.pt', weights_only=False)['commit_hash'])"

# Return to checkpoint commit
git checkout abc123

# Or accept non-deterministic resume
# (training will continue but may not be bit-for-bit reproducible)
```

**Issue: Missing RNG states in checkpoint**

```python
checkpoint = torch.load('checkpoint.pt', weights_only=False)
assert 'rng_states' not in checkpoint  # Old checkpoint!
```

**Solution:**
- Old checkpoints don't have RNG states
- Resume will work but won't be deterministic
- Re-run with updated checkpoint code
- Or accept non-deterministic resume

**Issue: Metadata not found**

```bash
ls experiments/dqn_atari/runs/pong_123/meta.json
# No such file
```

**Solution:**
- `meta.json` created by `save_run_metadata()`
- Add to training script initialization:
  ```python
  from src.utils import save_run_metadata
  save_run_metadata(output_dir, config, seed, ale_settings)
  ```

## Related Documentation

- [Training Loop](training_loop_runtime.md) - Integration with training loop
- [Replay Buffer Design](replay_buffer.md) - Buffer structure and state
- [Metadata Persistence](../design/reproducibility.md) - Git tracking and config snapshots
