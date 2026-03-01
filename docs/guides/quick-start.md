# Quick Start

Complete end-to-end workflow for running DQN on Atari games. This guide ties together environment setup, ROM download, dry-run validation, full training, and evaluation.

> **Note**: This is a step-by-step workflow guide. For authoritative setup reference (pinned versions, seeding, evaluation settings), see [DQN Setup](../reference/dqn-setup.md).

## Overview

**Time to first run:** ~10 minutes (setup) + 2 minutes (dry run) + hours/days (training)

**What you'll do:**
1. Set up Python environment and dependencies
2. Download Atari ROMs
3. Validate preprocessing with dry run
4. Start full training
5. Evaluate and plot results

**Prerequisites:**
- Python 3.8+ installed
- Git repository cloned
- ~2GB disk space for dependencies and ROMs
- (Optional) CUDA-capable GPU for faster training

<br><br>

## Step 1: Environment Setup

### Install Dependencies

```bash
# Navigate to repository root
cd /path/to/thesis

# Install required packages
pip install -r setup/requirements.txt
```

**What gets installed:**
- PyTorch (deep learning framework)
- Gymnasium (RL environment interface)
- ALE-py (Atari Learning Environment)
- AutoROM (ROM downloader)
- OmegaConf (configuration management)
- Additional utilities (numpy, opencv, etc.)

**Expected time:** 2-5 minutes depending on internet speed

**Troubleshooting:** See [dqn-setup.md](../reference/dqn-setup.md) for detailed setup instructions and common issues.

<br><br>

## Step 2: Download Atari ROMs

Atari 2600 ROMs are required to run ALE environments. We use AutoROM to download legally-redistributable ROMs.

```bash
# Run ROM setup script
./setup/setup_roms.sh
```

**What happens:**
- Downloads ~50 Atari 2600 ROMs
- Installs to location expected by ale-py
- Accepts license terms automatically

**Verification:**
```bash
# Check installed ROMs
python -c 'import ale_py; print(ale_py.roms.list())'
```

**Expected output:** List of ROM names including Pong, Breakout, BeamRider, etc.

**Expected time:** 1-2 minutes

**Troubleshooting:** If ROMs fail to install, see [scripts documentation](../experiments/dqn_atari/scripts/README.md#setup_romssh).

<br><br>

## Step 3: Capture Environment Info (Optional)

Capture system and package versions for reproducibility.

```bash
# Capture environment state
./setup/capture_env.sh
```

**Output:** `experiments/dqn_atari/system_info.txt`

**What's captured:**
- System details (OS, kernel, architecture)
- Python version and package versions
- Git commit hash and branch
- CUDA availability and GPU info

**When to run:**
- Before major training runs
- When reporting bugs
- For reproducibility documentation

<br><br>

## Step 4: Validate Setup with Dry Run

Run a short random-policy rollout to verify preprocessing pipeline and generate debug artifacts.

```bash
# Basic dry run with Pong (3 episodes)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run
```

**What happens:**
1. Creates environment with full wrapper chain
2. Runs 3 episodes with random actions
3. Saves preprocessed frame samples (PNGs)
4. Generates rollout log with shape verification
5. Creates metadata file with config and git hash

**Outputs:** `experiments/dqn_atari/runs/pong_0/`
- `frames/reset_*_frame_*.png` – Preprocessed frames (4 per stack, up to 5 stacks)
- `rollout_log.json` – Complete debug log
- `meta.json` – Git hash, config, seed
- `action_list.json` – Action space details

**Verification checklist:**
- [ ] Frames saved successfully (check `runs/pong_0/frames/`)
- [ ] Frames are 84×84 grayscale (inspect PNGs)
- [ ] Rollout log shows correct shapes: `(4, 84, 84)`
- [ ] No error messages during run

**Expected time:** ~30 seconds

**Advanced dry run options:**
```bash
# Custom seed and more episodes
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml \
  --dry-run --seed 42 --dry-run-episodes 5

# Different game
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/beam_rider.yaml --dry-run
```

**Troubleshooting:** See [atari-env-wrapper.md](../reference/atari-env-wrapper.md#troubleshooting) for common preprocessing issues.

<br><br>

## Step 5: Start Full Training

Launch DQN training with specified config and seed.

### Quick Test Run (Smoke Test)

```bash
# Short training run (100K frames, ~10 minutes)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 0 --training.total_frames 100000
```

**Use for:**
- Verifying training loop works end-to-end
- Testing code changes quickly
- Debugging before long runs

### Full Training Run

```bash
# Full Pong training (10M frames, ~12-24 hours on GPU)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --seed 123

# Breakout (20M frames, ~24-48 hours on GPU)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml --seed 123

# Beam Rider (20M frames, ~24-48 hours on GPU)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/beam_rider.yaml --seed 123
```

**Training outputs:** `experiments/dqn_atari/runs/{game}_{seed}/`
- `checkpoints/` – Model weights saved every 1M frames
- `logs/` – Training metrics (loss, TD error, grad norm, episode returns)
- `reference_states.pt` – Fixed batch for Q-value tracking
- `config.yaml` – Complete config used for the run

**Monitoring training:**
```bash
# Check logs (if using file logging)
tail -f experiments/dqn_atari/runs/pong_123/logs/training.log

# Check tensorboard (if enabled)
tensorboard --logdir experiments/dqn_atari/tensorboard
```

**Key metrics to watch:**
- **Loss:** Should decrease over time
- **TD error:** Should decrease and stabilize
- **Episode return:** Should increase (agent improving)
- **Gradient norm:** Should stabilize (not exploding)

**Expected time:**
- Pong: 12-24 hours (GPU), 3-7 days (CPU)
- Breakout: 24-48 hours (GPU), 7-14 days (CPU)
- Beam Rider: 24-48 hours (GPU), 7-14 days (CPU)

**Troubleshooting:** See [dqn-training.md](../reference/dqn-training.md#debugging-unstable-training) for debugging convergence issues.

<br><br>

## Step 6: Evaluation and Plotting

### Evaluate Trained Agent

```bash
# Evaluate checkpoint (Subtask 6, not yet implemented)
# Placeholder for future evaluation script
python src/eval_dqn.py \
  --checkpoint experiments/dqn_atari/runs/pong_123/checkpoints/step_10000000.pt \
  --config experiments/dqn_atari/configs/pong.yaml \
  --episodes 30 --seed 999
```

**What happens:**
1. Loads trained Q-network from checkpoint
2. Runs N evaluation episodes with ε=0.05 (or fully greedy)
3. Records episode returns, lengths, and Q-values
4. Saves evaluation report

**Outputs:**
- `eval_report.json` – Episode statistics
- `eval_video.mp4` – (Optional) Gameplay recording

### Plot Training Curves

```bash
# Plot training metrics (Subtask 6, not yet implemented)
# Placeholder for future plotting script
python scripts/plot_training.py \
  --run-dir experiments/dqn_atari/runs/pong_123 \
  --output plots/pong_training.png
```

**What gets plotted:**
1. Episode return vs. frames
2. Loss vs. update count
3. TD error vs. update count
4. Gradient norm vs. update count
5. Reference max-Q vs. frames

### Compare Multiple Runs

```bash
# Compare different seeds or hyperparameters
# Placeholder for future comparison script
python scripts/compare_runs.py \
  --runs experiments/dqn_atari/runs/pong_* \
  --output plots/pong_comparison.png
```

**Note:** Evaluation and plotting scripts are planned for Subtask 6 (Training Loop). For now, you can inspect logs and checkpoints manually.

<br><br>

## Common Workflows

### First-Time Setup (One-Time)

```bash
# Complete setup from scratch
pip install -r setup/requirements.txt
./setup/setup_roms.sh
./setup/capture_env.sh
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run
```

### Running New Experiment

```bash
# 1. Validate preprocessing
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# 2. Quick smoke test
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 0 --training.total_frames 100000

# 3. Full training run
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --seed 123
```

### Reproducing Subtasks 1-2

```bash
# Subtask 1: Environment setup
pip install -r setup/requirements.txt
./setup/setup_roms.sh
./setup/capture_env.sh

# Subtask 2: Wrapper validation
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run --seed 0

./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml --dry-run --seed 0

./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/beam_rider.yaml --dry-run --seed 0

# Inspect artifacts
ls experiments/dqn_atari/runs/*/frames/
cat experiments/dqn_atari/runs/*/rollout_log.json
```

### Debugging Preprocessing Issues

```bash
# 1. Run dry run with custom seed
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --dry-run --dry-run-episodes 5 --seed 42

# 2. Inspect frame artifacts
ls experiments/dqn_atari/runs/pong_42/frames/

# 3. Check rollout log for shape verification
python -m json.tool experiments/dqn_atari/runs/pong_42/rollout_log.json

# 4. Inspect specific frame
open experiments/dqn_atari/runs/pong_42/frames/reset_0_frame_0.png
```

### Parameter Sweeps

```bash
# Try different learning rates
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 1 --agent.learning_rate 0.0001

./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 2 --agent.learning_rate 0.00025

./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 3 --agent.learning_rate 0.0005

# Disable target network (2013 DQN)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 4 --agent.target_update_interval 0
```

<br><br>

## Configuration Quick Reference

### Override Config Parameters

```bash
# General syntax
./experiments/dqn_atari/scripts/run_dqn.sh \
  <config_file> \
  --<section>.<key> <value>

# Examples
--seed 42                           # Set random seed
--training.total_frames 5000000     # Change frame budget
--agent.learning_rate 0.0001        # Change learning rate
--agent.target_update_interval 0    # Disable target network
--training.episode_life false       # Full-episode termination
--eval.epsilon 0.0                  # Fully greedy evaluation
```

### Common Toggles

| Parameter | Default | Options | Purpose |
|-----------|---------|---------|---------|
| `--seed` | 0 | Any integer | Random seed for reproducibility |
| `--training.total_frames` | 10M (Pong) | Any integer | Total environment steps |
| `--training.episode_life` | true | true/false | Life loss as terminal (training) |
| `--agent.target_update_interval` | 10000 | >0 or 0 | Target net sync (0=disable) |
| `--training.reward_clip` | true | true/false | Clip rewards to {-1,0,+1} |
| `--env.max_noop_start` | 30 | 0-30 | Random no-op steps on reset |

See [config documentation](../experiments/dqn_atari/configs/README.md) for complete reference.

<br><br>

## Troubleshooting

### Setup Issues

**Problem:** `pip install` fails with dependency conflicts

**Solution:**
```bash
# Create fresh virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r setup/requirements.txt
```

**Problem:** ROM download fails or ROMs not found

**Solution:**
```bash
# Retry ROM installation
./setup/setup_roms.sh

# Verify installation
python -c 'import ale_py; print(ale_py.roms.list())'

# Manual installation (if script fails)
python -m AutoROM --accept-license
```

### Preprocessing Issues

**Problem:** Dry run fails with shape errors

**Solution:** Check [atari-env-wrapper.md](../reference/atari-env-wrapper.md#troubleshooting) for common wrapper issues.

**Problem:** Frames look wrong (distorted, wrong size, color artifacts)

**Solution:**
```bash
# Run dry run and inspect frames
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# Check frame artifacts
ls experiments/dqn_atari/runs/pong_0/frames/
# Frames should be 84×84 grayscale
```

### Training Issues

**Problem:** Training diverges (loss explodes, TD error increases)

**Solution:** See [dqn-training.md](../reference/dqn-training.md#debugging-unstable-training) for debugging strategies.

**Problem:** Training is slower than expected

**Solution:**
```bash
# Check if using GPU
python -c 'import torch; print(torch.cuda.is_available())'

# Force CPU (if GPU issues)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --device cpu

# Reduce batch size (if OOM)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --agent.batch_size 16
```

<br><br>

## Next Steps

### Learn More

- **Design Docs:** Detailed specifications in [docs/reference/](../reference/)
- **Config Reference:** All config options in [configs/README.md](../../experiments/dqn_atari/configs/README.md)

### Contributing

- Follow commit message style in [git-commit-guide.md](git-commit-guide.md)
- Document major changes in changelog
- Run dry-run validation before committing wrapper changes
- Run tests before committing: `pytest tests/`

### Getting Help

- **Setup issues:** See [dqn-setup.md](../reference/dqn-setup.md)
- **Preprocessing issues:** See [atari-env-wrapper.md](../reference/atari-env-wrapper.md)
- **Training issues:** See [dqn-training.md](../reference/dqn-training.md)
- **Config issues:** See [configs/README.md](../experiments/dqn_atari/configs/README.md)
- **General questions:** Check [docs/README.md](../README.md) for navigation
