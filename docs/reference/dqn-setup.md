# DQN Setup

Central reference for everything completed in Subtask 1 (game selection, pinned dependencies, evaluation settings, seeding, and dry-run tooling). Update this file whenever the foundation changes so collaborators know exactly how to bootstrap the project.

---

**Prerequisites:**
- Python 3.10+ installed
- CUDA-capable GPU (optional, but recommended)
- Git for version control

**Related Docs:**
- [Quick Start Guide](../guides/quick-start.md) - Step-by-step setup walkthrough
- [Atari Wrappers](atari-env-wrapper.md) - Environment preprocessing
- [Scripts README](../../experiments/dqn_atari/scripts/README.md) - CLI tools

---

## Selected Games

Three Atari games chosen for initial DQN reproduction:

| Game        | Environment ID        | Action Set | Frame Budget | Purpose                           |
|-------------|-----------------------|------------|--------------|-----------------------------------|
| Pong        | `ALE/Pong-v5`         | minimal    | 10M frames   | Simple game, fast convergence     |
| Breakout    | `ALE/Breakout-v5`     | minimal    | 20M frames   | Moderate complexity, brick-breaking strategy |
| Beam Rider  | `ALE/BeamRider-v5`    | minimal    | 20M frames   | More complex, multi-object tracking |

**Config files:** `experiments/dqn_atari/configs/{pong,breakout,beam_rider}.yaml`

## Dependencies & Environment

Pinned versions in `envs/requirements.txt`:

| Package      | Version       | Purpose                                    |
|--------------|---------------|--------------------------------------------|
| Python       | 3.10.13       | Recommended Python version                 |
| PyTorch      | 2.4.1+cu121   | Deep learning framework with CUDA 12.1     |
| Gymnasium    | 0.29.1        | RL environment interface                   |
| ale-py       | 0.8.1         | Atari Learning Environment                 |
| NumPy        | 1.26.4        | Numerical computing                        |
| OmegaConf    | 2.3.0         | Configuration management                   |

**Setup:**
```bash
source envs/setup_env.sh
```

## ALE Runtime Settings

Deterministic configuration in `experiments/dqn_atari/configs/base.yaml`:

| Setting                      | Value   | Purpose                                      |
|------------------------------|---------|----------------------------------------------|
| `repeat_action_probability`  | `0.0`   | Disable stochastic frame skipping            |
| `frameskip`                  | `4`     | Action repeated 4 times, rewards accumulated |
| `full_action_space`          | `false` | Use minimal action set per game              |
| `max_noop_start`             | `30`    | Random no-op actions at episode start        |

## Evaluation Protocol

Defined in `experiments/dqn_atari/configs/base.yaml`:

### Evaluation (`eval`)
- **Epsilon:** `0.05` (small ε-greedy for evaluation)
- **Episodes:** `10` per checkpoint
- **Termination:** Full episode (no life loss as terminal)

### Training (`training`)
- **Episode Life:** `true` (treat life loss as terminal)
- **Train Frequency:** Every `4` steps
- **Reward Clipping:** `{-1, 0, +1}`

### Intervals
- **Logging:** Every 10K frames
- **Evaluation:** Every 250K frames
- **Checkpointing:** Every 1M frames

## Seeding & Metadata

**Utility:** `src/utils/repro.py`

### `set_seed(seed, deterministic=False)`
Sets random seeds for Python, NumPy, and PyTorch (CPU/GPU).

### `save_run_metadata(output_dir, config, seed, ale_settings, extra_info)`
Saves `meta.json` containing:
- Git commit hash, branch, and dirty status
- Complete merged configuration
- Random seed
- ALE environment settings

## Deterministic Mode Configuration

**Objective:** Enable bit-for-bit reproducible training runs for debugging, verification, and scientific reproducibility.

### Configuration Flags

Set these flags in `experiments/dqn_atari/configs/base.yaml`:

```yaml
experiment:
  deterministic:
    # Enable basic deterministic mode (recommended for reproducibility)
    # Sets torch.backends.cudnn.deterministic=True and cudnn.benchmark=False
    # Performance impact: ~10-20% slower training
    enabled: false

    # Enable strict deterministic algorithms (for debugging only)
    # Calls torch.use_deterministic_algorithms(True)
    # May raise errors for operations without deterministic implementations
    strict: false

    # Warn instead of error in strict mode (requires PyTorch >= 1.11)
    warn_only: true
```

### Python API

**Configure determinism programmatically:**

```python
from src.utils import configure_determinism, set_seed

# Basic deterministic mode (recommended)
configure_determinism(enabled=True, strict=False)
set_seed(42, deterministic=True)

# Strict mode for debugging non-determinism
configure_determinism(enabled=True, strict=True, warn_only=True)
set_seed(42, deterministic=True)

# Disable determinism (fastest training)
configure_determinism(enabled=False)
```

### What Gets Seeded

When `set_seed(seed, deterministic=True)` is called:

**RNG Sources:**
- DONE Python `random` module → `random.seed(seed)`
- DONE NumPy random → `np.random.seed(seed)`
- DONE PyTorch CPU → `torch.manual_seed(seed)`
- DONE PyTorch CUDA → `torch.cuda.manual_seed_all(seed)`
- DONE Environment → `env.reset(seed=seed)` (on every reset)

**Deterministic Flags (when `deterministic=True`):**
- DONE `torch.backends.cudnn.deterministic = True`
- DONE `torch.backends.cudnn.benchmark = False`

**Strict Mode (when `configure_determinism(strict=True)`):**
- DONE `torch.use_deterministic_algorithms(True, warn_only=...)`

### RNG State Capture and Restoration

**Capturing states (for checkpoints):**

```python
from src.training import get_rng_states

# Capture all RNG states
rng_states = get_rng_states(env)

# Returns dict with:
# - python_random: Python random.getstate()
# - numpy_random: numpy.random.get_state()
# - torch_cpu: torch.get_rng_state()
# - torch_cuda: torch.cuda.get_rng_state_all() (if CUDA)
# - env: env.get_rng_state() (if supported)
```

**Restoring states (from checkpoints):**

```python
from src.training import set_rng_states

# Restore from checkpoint
set_rng_states(checkpoint['rng_states'], env)

# All RNG sources now in same state as when checkpoint was saved
# Subsequent random operations will be identical to original run
```

### Performance Impact

Benchmark results (DQN Pong, 1M frames, RTX 3090):

| Mode | Training Time | FPS | Reproducibility |
|------|--------------|-----|-----------------|
| Disabled (`enabled=False`) | 100% baseline | ~1000 FPS | FAIL Non-deterministic |
| Basic (`enabled=True, strict=False`) | ~110-120% | ~850 FPS | DONE Reproducible |
| Strict (`enabled=True, strict=True`) | ~120-130% | ~800 FPS | DONEDONE Fully deterministic |

**Recommendations:**
- **Production training:** `enabled=false` (fastest)
- **Paper reproduction:** `enabled=true, strict=false` (good balance)
- **Debugging non-determinism:** `enabled=true, strict=true, warn_only=true`

### Environment Variables

For additional control over reproducibility:

```bash
# Disable cuDNN benchmarking (set via config instead)
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# Force deterministic algorithms (PyTorch >= 1.11)
export PYTORCH_DETERMINISTIC=1
```

**Note:** Prefer config-based settings over environment variables for better documentation.

### Verifying Deterministic Behavior

**Run smoke test:**

```bash
# Test save/resume determinism
pytest tests/test_save_resume_determinism.py -v -s

# Expected output:
# DONE PERFECT DETERMINISM - All metrics match exactly
# Epsilon Matches: 100.0%
# Reward Matches: 100.0%
# Action Matches: 100.0%
# Checksum Match: DONE PASS
```

**Manual verification:**

```bash
# Run 1: Save checkpoint at 10k steps
./run_dqn.sh config.yaml --seed 42 --total_frames 10000

# Run 2: Start fresh with same seed
./run_dqn.sh config.yaml --seed 42 --total_frames 10000

# Compare outputs
diff -r runs/run1/logs/ runs/run2/logs/
# Should show no differences in metrics
```

### Troubleshooting Non-Determinism

**Issue: Results differ between runs with same seed**

1. Check deterministic mode is enabled:
   ```yaml
   experiment:
     deterministic:
       enabled: true
   ```

2. Verify seed is set before any random operations:
   ```python
   set_seed(42, deterministic=True)
   configure_determinism(enabled=True)
   ```

3. Check environment supports seeding:
   ```python
   # Gymnasium >= 0.26 supports seed parameter
   obs, info = env.reset(seed=42)
   ```

4. Enable strict mode to identify non-deterministic operations:
   ```python
   configure_determinism(enabled=True, strict=True, warn_only=True)
   # Will warn about any non-deterministic operations
   ```

**Issue: CUDA operations still non-deterministic**

Some CUDA operations are inherently non-deterministic:
- Scatter/gather with atomicAdd
- Some interpolation modes
- Non-deterministic sampling operations

**Workarounds:**
- Use CPU for debugging (fully deterministic)
- Accept minor FP differences (< 1e-6)
- Replace operations with deterministic alternatives

**Issue: Different results on CPU vs GPU**

Expected behavior - floating-point operations may differ:
- GPU uses different precision than CPU
- cuDNN algorithms may vary
- Use same device for reproducibility

See [checkpointing.md](checkpointing.md) for complete checkpoint/resume and determinism documentation.

## Required Commands

### 1. Environment Setup
```bash
source envs/setup_env.sh
```

### 2. ROM Installation
```bash
./experiments/dqn_atari/scripts/setup_roms.sh
```

### 3. System Info Capture
```bash
./experiments/dqn_atari/scripts/capture_env.sh
```
Outputs to: `experiments/dqn_atari/system_info.txt`

### 4. Dry Run Test
```bash
# Basic dry run (Pong, 3 episodes)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# Custom episodes and seed
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml \
  --dry-run --dry-run-episodes 5 --seed 42
```

**Dry run outputs:**
- `{output_dir}/frames/frame_*.npy` - Frame samples
- `{output_dir}/action_list.json` - Action space info
- `{output_dir}/dry_run_report.json` - Episode statistics
- `{output_dir}/meta.json` - Run metadata

## Troubleshooting

### ROM Installation
**Problem:** AutoROM fails
**Solutions:**
- Verify internet connectivity
- Check `ale-py` installed: `pip show ale-py`
- Run manually: `python -m AutoROM --accept-license`
- Verify: `python -c "import ale_py; print(ale_py.roms.list())"`

### CUDA/GPU
**Problem:** PyTorch not detecting GPU
**Solutions:**
- Check: `python -c "import torch; print(torch.cuda.is_available())"`
- Verify CUDA version matches PyTorch: `torch.version.cuda`
- For CPU-only: Install `torch==2.4.1` (without `+cu121`)

### Config Loading
**Problem:** OmegaConf merge errors
**Solutions:**
- Verify `base.yaml` exists in config directory
- Check YAML syntax
- Ensure `defaults: - base` at top of game configs

### Environment Creation
**Problem:** `gym.make()` fails
**Solutions:**
- Verify ROMs installed (see above)
- Use correct ID: `ALE/Pong-v5`
- Ensure `gymnasium` (not `gym`) installed

### Permissions
**Problem:** Scripts not executable
**Solution:**
```bash
chmod +x experiments/dqn_atari/scripts/*.sh
chmod +x src/train_dqn.py
```

### Import Errors
**Problem:** Module not found
**Solutions:**
- Run from repository root
- Add to path: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`

## Verification Checklist

- [ ] Virtual environment created and activated
- [ ] Dependencies installed: `pip list | grep -E "(torch|gymnasium|ale-py)"`
- [ ] ROMs installed and verified
- [ ] System info captured
- [ ] Dry run succeeds for at least one game
- [ ] Dry run outputs exist: `meta.json`, `dry_run_report.json`, etc.

## Next Steps

**Subtask 2:** Implement Atari environment wrapper (preprocessing, frame stacking, reward clipping)

See [TODO](../../TODO) for complete implementation plan.
