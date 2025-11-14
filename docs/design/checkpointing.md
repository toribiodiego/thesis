# Checkpointing and Resume

**Objective:** Save and restore complete training state to enable resumption and reproducibility.

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
- ✓ Model weights (online and target)
- ✓ Optimizer state (momentum, learning rate, etc.)
- ✓ Training counters (step, episode)
- ✓ Epsilon value and scheduler state
- ✓ Replay buffer (index, size, optionally full data)
- ✓ RNG states (Python, NumPy, PyTorch, CUDA, environment)

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
- ✓ Python `random` module
- ✓ NumPy `np.random`
- ✓ PyTorch CPU (`torch.manual_seed`)
- ✓ PyTorch CUDA (`torch.cuda.manual_seed_all`)
- ✓ Environment (via `env.reset(seed=...)`)

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
- ✓ Ensures reproducible cuDNN operations
- ✓ Disables autotuning that may vary between runs
- ✗ ~10-20% slower training speed
- ✗ No benefit on CPU-only training

**2. Strict Deterministic Mode (`strict=True`)**

Sets:
- All basic mode settings
- `torch.use_deterministic_algorithms(True)`

Effects:
- ✓ Enforces deterministic implementations for all operations
- ✓ Helps debug sources of non-determinism
- ✗ May raise errors for operations without deterministic implementations
- ✗ Additional performance overhead
- ✗ Not all PyTorch operations have deterministic versions

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
| Disabled (`enabled=False`) | 100% baseline | ~1000 FPS | ❌ Non-deterministic |
| Basic (`enabled=True, strict=False`) | ~110-120% | ~850 FPS | ✓ Reproducible |
| Strict (`enabled=True, strict=True`) | ~120-130% | ~800 FPS | ✓✓ Fully deterministic |

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

## Related Documentation

- [Training Loop](training_loop_runtime.md) - Integration with training loop
- [Replay Buffer Design](replay_buffer.md) - Buffer structure and state
- [Metadata Persistence](../design/reproducibility.md) - Git tracking and config snapshots
