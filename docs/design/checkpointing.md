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

## Related Documentation

- [Training Loop](training_loop_runtime.md) - Integration with training loop
- [Replay Buffer Design](replay_buffer.md) - Buffer structure and state
- [Metadata Persistence](../design/reproducibility.md) - Git tracking and config snapshots
