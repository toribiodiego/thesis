# DQN Foundation Setup (Subtask 1)

Central reference for everything completed in Subtask 1 (game selection, pinned dependencies, evaluation settings, seeding, and dry-run tooling). Update this file whenever the foundation changes so collaborators know exactly how to bootstrap the project.

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

See `docs/roadmap.md` for complete implementation plan.
