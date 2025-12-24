# Scripts

Training and setup utilities for DQN Atari experiments. All scripts should be run from the repository root.

## Overview

This directory contains scripts for:
- **Training**: `run_dqn.sh` - Full training runs with config management
- **Validation**: `smoke_test.sh` - Fast end-to-end pipeline validation (~200K frames)
- **Setup**: `setup_roms.sh` - One-time ROM installation
- **Environment**: `capture_env.sh` - System info capture for reproducibility

**Recommended workflow for Subtask 6 validation:**
1. `setup_roms.sh` - Install Atari ROMs (one-time)
2. `capture_env.sh` - Document environment state
3. `smoke_test.sh` - Verify training loop works (~5-10 min)
4. `run_dqn.sh --dry-run` - Validate preprocessing with real env
5. `run_dqn.sh` - Start full training runs

## `run_dqn.sh`

Launch DQN training or dry-run validation with specified config.

**Purpose:** Convenience wrapper around `python train_dqn.py` that handles path resolution and config loading.

**Usage:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh <config> [options]
```

**Required arguments:**
- `<config>`: Path to YAML config file (e.g., `experiments/dqn_atari/configs/pong.yaml`)

**Common options:**
- `--dry-run`: Run random-policy rollout for 3 episodes (validates preprocessing, saves debug artifacts)
- `--dry-run-episodes N`: Number of episodes for dry run (default: 3)
- `--seed N`: Set random seed for reproducibility
- `--total_frames N`: Override total training frames
- `--device cuda/cpu`: Override device selection

**Examples:**

```bash
# Dry run with Pong (validates wrapper chain, saves frames)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# Dry run with custom seed and episode count
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml \
  --dry-run --seed 42 --dry-run-episodes 5

# Full training run with Pong
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --seed 123

# Training with custom parameters
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/beam_rider.yaml \
  --seed 456 --device cuda
```

**Config overrides** (use `--set` to adjust hyperparameters without editing YAML):

```bash
# Override learning rate
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 7 \
  --set training.optimizer.lr=0.001

# Multiple overrides (repeat --set flag for each override)
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 7 \
  --set training.total_frames=2000000 \
  --set training.gamma=0.95 \
  --set replay.capacity=500000

# Disable target network (2013 NIPS DQN mode)
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 7 \
  --set target_network.update_interval=null
```

**Note:** `--set` is repeatable - use one `--set` flag per KEY=VALUE override.

See [../configs/README.md](../configs/README.md) for complete CLI reference and more examples.

**Outputs:**
- Dry run: `experiments/dqn_atari/runs/{experiment_name}_{seed}/` (frames, logs, metadata)
- Training: `experiments/dqn_atari/runs/{experiment_name}_{seed}/` (checkpoints, logs, metrics)

### Resume a Run

Resume training from a saved checkpoint using the `--resume` flag.

**Basic Resume:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --resume experiments/dqn_atari/runs/pong_123/checkpoints/checkpoint_1000000.pt
```

**Resume with Strict Config Validation:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --resume experiments/dqn_atari/runs/pong_123/checkpoints/checkpoint_1000000.pt \
  --strict-resume
```

**Resume from Best Model:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --resume experiments/dqn_atari/runs/pong_123/checkpoints/best_model.pt
```

**What Gets Restored:**
- DONE Model weights (online and target Q-networks)
- DONE Optimizer state (momentum buffers, learning rate)
- DONE Training counters (step, episode, epsilon)
- DONE Replay buffer state (write index and size)
- DONE RNG states (Python, NumPy, PyTorch, CUDA, environment)

**Expected Output on Resume:**
```
================================================================================
RESUMING FROM CHECKPOINT
================================================================================
Checkpoint: experiments/dqn_atari/runs/pong_123/checkpoints/checkpoint_1000000.pt

Checkpoint Info:
  Step: 1,000,000
  Episode: 5,000
  Epsilon: 0.5000
  Saved at: 2025-01-15T10:30:45.123456
  Commit: a1b2c3d

Restoring epsilon scheduler...
  Setting epsilon to: 0.5000
  Setting frame counter to: 1000000

Restoring RNG states for reproducibility...
  DONE Python random state restored
  DONE NumPy random state restored
  DONE PyTorch random state restored

Replay buffer state:
  Size: 1,000,000 / 1,000,000
  Write index: 0

Optimizer state restored:
  Type: RMSprop
  Learning rate: 0.00025

Model weights restored:
  Online model parameters: 677,686
  Target model parameters: 677,686
  Device: cuda

================================================================================
RESUME COMPLETE - Starting from step 1,000,001
================================================================================
```

**Files Written by Checkpoint Manager:**

During training, the checkpoint manager creates these files:
```
experiments/dqn_atari/runs/pong_123/
├── checkpoints/
│   ├── checkpoint_1000000.pt     # Periodic checkpoint (every 1M steps)
│   ├── checkpoint_2000000.pt     # Periodic checkpoint
│   ├── checkpoint_3000000.pt     # Periodic checkpoint
│   └── best_model.pt             # Best model by eval score
├── logs/
│   ├── episodes.csv              # Episode metrics
│   ├── steps.csv                 # Step-level metrics
│   └── eval.csv                  # Evaluation results
└── meta.json                     # Run metadata (seed, git hash, config)
```

**Verification Checklist for Deterministic Resume:**

To verify a resume produces identical results:

1. DONE **Enable deterministic mode in config:**
   ```yaml
   experiment:
     deterministic:
       enabled: true
       strict: false
   ```

2. DONE **Use same seed and config:**
   ```bash
   # Original run
   ./run_dqn.sh config.yaml --seed 42

   # Resume must use same config
   ./run_dqn.sh config.yaml --resume checkpoint.pt
   ```

3. DONE **Check git commit hash:**
   - Resume warns if code version differs
   - Ensure working directory is clean (`git status`)

4. DONE **Verify RNG states restored:**
   - Check console output: "DONE RNG states restored"
   - Run smoke test: `pytest tests/test_save_resume_determinism.py -v -s`

5. DONE **Compare metrics after resume:**
   - Epsilon values should match exactly
   - Actions should be identical
   - Rewards should be identical (with tiny FP tolerance)

**Run Determinism Smoke Test:**
```bash
# Verify save/resume determinism
pytest tests/test_save_resume_determinism.py -v -s

# Expected output:
# DONE PERFECT DETERMINISM - All metrics match exactly
# Epsilon Matches: 100.0%
# Reward Matches: 100.0%
# Action Matches: 100.0%
# Checksum Match: DONE PASS
```

See [docs/design/checkpointing.md](../../../docs/design/checkpointing.md) for complete checkpoint/resume documentation.

## `setup_roms.sh`

Download and install Atari 2600 ROMs required for ALE environments.

**Purpose:** One-time setup to install legally-redistributable ROMs via AutoROM.

**Usage:**
```bash
./setup/setup_roms.sh
```

**Required environment:** Python environment with `AutoROM` package installed.

**What it does:**
1. Calls `python -m AutoROM --accept-license`
2. Downloads ~50 Atari 2600 ROMs from legally-redistributable sources
3. Installs ROMs to the location expected by `ale-py`

**Verification:**
```bash
# Check installed ROMs
python -c 'import ale_py; print(ale_py.roms.list())'
```

**When to run:**
- Initial setup on a new machine
- After fresh Python environment installation
- If ROM-related import errors occur

**Note:** You must accept the license terms for ROM redistribution. See AutoROM documentation for details.

## `smoke_test.sh`

Run end-to-end smoke test (~200K frames) to verify training loop stability.

**Purpose:** Quick validation that all training components work together correctly without running full multi-million frame experiments.

**Usage:**
```bash
./experiments/dqn_atari/scripts/smoke_test.sh [config] [seed]
```

**Arguments:**
- `[config]`: Optional config file path (default: `experiments/dqn_atari/configs/pong.yaml`)
- `[seed]`: Optional random seed (default: 0)

**Examples:**

```bash
# Default: Pong with seed 0, 200K frames
./experiments/dqn_atari/scripts/smoke_test.sh

# Custom config and seed
./experiments/dqn_atari/scripts/smoke_test.sh \
  experiments/dqn_atari/configs/breakout.yaml 42
```

**What it validates:**
- Training loop executes without errors
- Logs are created and grow (training_steps.csv, episodes.csv)
- Checkpoints appear (if save interval reached)
- Evaluation runs trigger and complete
- Reference-state Q logging works
- Metrics are recorded correctly

**Outputs:**
- `experiments/dqn_atari/runs/smoke_test_{seed}/`
  - `metadata.json` - Run metadata
  - `git_info.txt` - Git state snapshot
  - `logs/training_steps.csv` - Per-step metrics
  - `logs/episodes.csv` - Per-episode metrics
  - `logs/reference_q_values.csv` - Q-value tracking
  - `eval/evaluations.csv` - Evaluation results
  - `checkpoints/*.pt` - Model checkpoints (if save interval reached)

**Duration:** ~5-10 minutes on CPU (depends on hardware)

**When to run:**
- After implementing new training components
- Before starting long training runs
- After environment/dependency changes
- To verify end-to-end pipeline integrity
- As part of CI/CD validation

**Exit codes:**
- 0: All validations passed
- 1: Smoke test failed (missing outputs, errors during training)

## `capture_env.sh`

Capture system and environment information for reproducibility tracking.

**Purpose:** Record Python packages, Git state, hardware specs, and dependencies for experiment provenance.

**Usage:**
```bash
./setup/capture_env.sh
```

**Required environment:** Must be run from repository root with active Python environment.

**Output:** `experiments/dqn_atari/system_info.txt`

**Information captured:**
- System details (OS, kernel, architecture, hostname, timestamp)
- Python version and executable path
- Key package versions (PyTorch, NumPy, Gymnasium, ALE-py)
- CUDA availability and GPU device info
- Git commit hash, branch, and dirty state
- Complete `pip freeze` output

**When to run:**
- Before starting major training runs
- After changing environment setup or dependencies
- When reporting bugs or unexpected behavior
- For reproducibility documentation

**Example workflow:**
```bash
# Setup environment
pip install -r requirements.txt
./setup/setup_roms.sh

# Capture environment state
./setup/capture_env.sh

# Verify setup with dry run
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# Begin training
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --seed 123
```

## Inputs/Outputs Reference

Quick reference table for script inputs, outputs, and side effects.

### `run_dqn.sh`

| Category | Details |
|----------|---------|
| **Inputs** | • Config file path (required)<br>• CLI flags: --dry-run, --seed, --dry-run-episodes, --device, --training.*, --agent.* |
| **Outputs** | **Dry run mode:**<br>• `runs/{game}_{seed}/frames/*.png` - Preprocessed frame samples<br>• `runs/{game}_{seed}/rollout_log.json` - Debug log with shapes/stats<br>• `runs/{game}_{seed}/meta.json` - Git hash, config, seed<br>• `runs/{game}_{seed}/action_list.json` - Action space info<br><br>**Training mode:**<br>• `runs/{game}_{seed}/checkpoints/*.pt` - Model weights<br>• `runs/{game}_{seed}/logs/` - Training metrics<br>• `runs/{game}_{seed}/reference_states.pt` - Q-value tracking batch<br>• `runs/{game}_{seed}/config.yaml` - Merged config |
| **Side effects** | • Changes working directory to repository root<br>• Uses GPU if available (unless --device cpu)<br>• Creates timestamped run directory |
| **Exit codes** | • 0: Success<br>• Non-zero: Error (Python exception, config error, CUDA OOM) |
| **Environment vars** | None required |
| **Dependencies** | • Python packages from requirements.txt<br>• Atari ROMs installed via setup_roms.sh |

### `setup_roms.sh`

| Category | Details |
|----------|---------|
| **Inputs** | None (accepts AutoROM license automatically) |
| **Outputs** | • ROMs installed to ale-py package directory<br>• Terminal output showing installation progress |
| **Side effects** | • Downloads ~50 Atari 2600 ROMs (~10MB)<br>• Installs ROMs system-wide for ale-py<br>• Accepts license terms automatically |
| **Exit codes** | • 0: Success<br>• 1: AutoROM installation failed or not available |
| **Environment vars** | None |
| **Dependencies** | • Python with AutoROM package<br>• Internet connection for ROM download |

### `smoke_test.sh`

| Category | Details |
|----------|---------|
| **Inputs** | • Config file path (optional, default: pong.yaml)<br>• Random seed (optional, default: 0) |
| **Outputs** | • `runs/smoke_test_{seed}/metadata.json` - Run metadata<br>• `runs/smoke_test_{seed}/git_info.txt` - Git state<br>• `runs/smoke_test_{seed}/logs/*.csv` - Training/episode/Q metrics<br>• `runs/smoke_test_{seed}/eval/evaluations.csv` - Eval results<br>• `runs/smoke_test_{seed}/checkpoints/*.pt` - Model checkpoints |
| **Side effects** | • Cleans previous smoke test run with same seed<br>• Creates ~200K frame training run<br>• Uses CPU device for portability |
| **Exit codes** | • 0: All validations passed<br>• 1: Missing outputs or training errors |
| **Environment vars** | None |
| **Dependencies** | • All DQN training dependencies<br>• Mock environment (no ROMs needed) |

### `capture_env.sh`

| Category | Details |
|----------|---------|
| **Inputs** | None (reads system and Python environment state) |
| **Outputs** | • `experiments/dqn_atari/system_info.txt` - Complete environment snapshot |
| **Side effects** | • Overwrites existing system_info.txt if present<br>• Reads git repository state |
| **Exit codes** | • 0: Success<br>• 1: Git command failed or not in repository |
| **Environment vars** | None |
| **Dependencies** | • Git repository<br>• Python environment with packages installed |

### Output Directory Structure

```
experiments/dqn_atari/
├── runs/
│   └── {game}_{seed}/              # Run directory (e.g., pong_123)
│       ├── frames/                  # Dry run only
│       │   ├── reset_0_frame_0.png
│       │   ├── reset_0_frame_1.png
│       │   └── ...
│       ├── checkpoints/             # Training only
│       │   ├── step_1000000.pt
│       │   ├── step_2000000.pt
│       │   └── ...
│       ├── logs/                    # Training only
│       │   ├── training.log
│       │   └── metrics.csv
│       ├── rollout_log.json         # Dry run only
│       ├── action_list.json         # Dry run only
│       ├── meta.json                # Both modes
│       ├── config.yaml              # Both modes
│       └── reference_states.pt      # Training only (after 50K frames)
└── system_info.txt                  # From capture_env.sh
```

### Return Codes Summary

| Script | Success | Common Errors |
|--------|---------|---------------|
| `run_dqn.sh` | 0 | Config not found (2), Python error (1), CUDA OOM (137) |
| `setup_roms.sh` | 0 | AutoROM not installed (1), Network error (1) |
| `smoke_test.sh` | 0 | Training error (1), Missing outputs (1) |
| `capture_env.sh` | 0 | Not in git repo (1), Git command failed (1) |

### Standard Output/Error Behavior

**`run_dqn.sh`:**
- Dry run: Minimal output (episode counts, frame shapes)
- Training: Periodic metrics (loss, TD error, episode returns)
- Errors: Python tracebacks to stderr

**`setup_roms.sh`:**
- Progress messages to stdout
- AutoROM download progress
- Verification message on completion

**`smoke_test.sh`:**
- Progress updates every 10K steps
- Validation checklist at end
- Final pass/fail status

**`capture_env.sh`:**
- Brief status message to stdout
- Full output written to file (not terminal)

## Common Workflows

### First-time setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download ROMs
./setup/setup_roms.sh

# 3. Capture environment
./setup/capture_env.sh

# 4. Validate with dry run
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run
```

### Reproduce Subtasks 1-2
```bash
# Subtask 1: Environment setup and ROM installation
./setup/setup_roms.sh
./setup/capture_env.sh

# Subtask 2: Wrapper validation with dry runs
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run --seed 0

./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml --dry-run --seed 0

./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/beam_rider.yaml --dry-run --seed 0

# Inspect generated artifacts
ls experiments/dqn_atari/runs/*/frames/
cat experiments/dqn_atari/runs/*/rollout_log.json
```

### Debug preprocessing issues
```bash
# Run dry run with verbose output
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --dry-run --dry-run-episodes 5 --seed 42

# Check frame artifacts
ls experiments/dqn_atari/runs/pong_42/frames/

# Inspect rollout log
python -m json.tool experiments/dqn_atari/runs/pong_42/rollout_log.json
```

### Run evaluation on existing checkpoint

Evaluate a trained model from a checkpoint without continuing training.

**Basic evaluation (DQN paper protocol: ε=0.05, 10 episodes):**
```bash
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --eval-only \
  --checkpoint experiments/dqn_atari/runs/pong_123/checkpoints/step_1000000.pt \
  --seed 42
```

**Final reporting evaluation (30 episodes for paper results):**
```bash
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --eval-only \
  --checkpoint experiments/dqn_atari/runs/pong_123/checkpoints/best_model.pt \
  --set evaluation.num_episodes=30 \
  --seed 42
```

**Pure greedy evaluation (ε=0, no exploration):**
```bash
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --eval-only \
  --checkpoint experiments/dqn_atari/runs/pong_123/checkpoints/step_2000000.pt \
  --set evaluation.epsilon=0.0 \
  --seed 42
```

**Evaluation with video recording:**
```bash
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --eval-only \
  --checkpoint experiments/dqn_atari/runs/pong_123/checkpoints/best_model.pt \
  --set evaluation.record_video=true \
  --set evaluation.num_episodes=5 \
  --seed 42
```

**Expected outputs (in checkpoint's run directory):**

```
experiments/dqn_atari/runs/pong_123/
├── eval/
│   ├── evaluations.csv              # Summary: step, mean_return, median_return, std_return, etc.
│   ├── evaluations.jsonl            # Same data in JSONL format (streaming-friendly)
│   ├── per_episode_returns.jsonl    # Raw per-episode returns and lengths
│   └── detailed/
│       └── eval_step_<step>.json    # Complete evaluation details
└── videos/                          # If record_video=true
    └── Pong_step_<step>_best_ep<N>_r<return>.mp4  # Best episode recording (highest return)
```

**Troubleshooting evaluation issues:**

| Issue | Diagnosis | Solution |
|-------|-----------|----------|
| **Checkpoint not found** | Path incorrect or file doesn't exist | Check path with `ls experiments/dqn_atari/runs/*/checkpoints/*.pt` |
| **Config mismatch** | Checkpoint trained with different config | Use original config or disable strict validation with `--no-strict-config` |
| **CUDA out of memory** | Evaluation still uses GPU | Add `--set system.device=cpu` or use smaller batch |
| **Video file missing** | OpenCV not installed or render failed | `pip install opencv-python` and check `env.render()` works |
| **NaN in metrics** | Model producing NaN Q-values | Check model checkpoint integrity with `torch.load(checkpoint)` |
| **Evaluation hangs** | Episode never terminates | Check environment termination logic or add timeout |

**Evaluation metrics explained:**

The `evaluations.csv` file contains:
- `step`: Training step when checkpoint was saved
- `mean_return`: Average episode return across all evaluation episodes
- `median_return`: Median episode return (more robust to outliers)
- `std_return`: Standard deviation of returns (measures consistency)
- `min_return`, `max_return`: Range of episode performance
- `episodes`: Number of episodes evaluated (typically 10 or 30)
- `eval_epsilon`: Epsilon used during evaluation (typically 0.05)

**Batch evaluation across multiple checkpoints:**

```bash
# Evaluate all checkpoints from a run
for ckpt in experiments/dqn_atari/runs/pong_123/checkpoints/step_*.pt; do
    python train_dqn.py \
      --cfg experiments/dqn_atari/configs/pong.yaml \
      --eval-only \
      --checkpoint $ckpt \
      --seed 42
done

# Aggregate results
cat experiments/dqn_atari/runs/pong_123/eval/evaluations.csv
```

**Comparing models:**

```bash
# Evaluate best model
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --eval-only \
  --checkpoint experiments/dqn_atari/runs/pong_123/checkpoints/best_model.pt \
  --set evaluation.num_episodes=30 \
  --seed 0

# Evaluate latest checkpoint
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --eval-only \
  --checkpoint experiments/dqn_atari/runs/pong_123/checkpoints/step_10000000.pt \
  --set evaluation.num_episodes=30 \
  --seed 0

# Compare results
python -c "
import pandas as pd
df = pd.read_csv('experiments/dqn_atari/runs/pong_123/eval/evaluations.csv')
print(df[['step', 'mean_return', 'median_return', 'std_return']])
"
```

See [docs/design/eval_harness.md](../../../docs/design/eval_harness.md) for complete evaluation harness documentation.

---

## Logging & Plotting

Training metrics are logged to three backends simultaneously: **TensorBoard**, **Weights & Biases (W&B)**, and **CSV files**.

### Configure Logging Backends

Edit your config file (e.g., `experiments/dqn_atari/configs/pong.yaml`):

```yaml
logging:
  # TensorBoard (local event files)
  tensorboard:
    enabled: true
    flush_interval: 1000

  # CSV files (local structured logs)
  csv:
    enabled: true
    smoothing_window: 100

  # Weights & Biases (cloud logging)
  wandb:
    enabled: true
    project: "dqn-atari"
    entity: "my-team"  # optional
    upload_artifacts: true
    artifact_upload_interval: 1000000  # Upload every 1M steps
```

**Note:** Directories are automatically created by `setup_run_directory()` in `experiments/dqn_atari/runs/<game>_<seed>_<timestamp>/`. You don't need to specify `tensorboard_dir` or `csv_dir`.

### Enable W&B

**One-time setup:**
```bash
# Install W&B (if not already installed)
pip install wandb

# Login with your API key
wandb login

# Or set directly
export WANDB_API_KEY="your_api_key_here"
```

**Use offline mode** (sync later):
```bash
export WANDB_MODE=offline

# Run training...

# Later, sync offline runs
wandb sync experiments/dqn_atari/runs/pong_42_20251115/wandb/
```

**Disable W&B completely:**
```bash
export WANDB_DISABLED=true
```

### View Logs

**TensorBoard:**
```bash
# Launch TensorBoard
tensorboard --logdir experiments/dqn_atari/runs/

# Open browser to: http://localhost:6006
```

**W&B Dashboard:**
- Visit: `https://wandb.ai/<entity>/<project>`
- View runs, compare metrics, and download artifacts

**CSV Files:**
```bash
# Training steps (loss, epsilon, FPS)
cat experiments/dqn_atari/runs/pong_42_20251115/csv/training_steps.csv

# Episodes (return, length)
cat experiments/dqn_atari/runs/pong_42_20251115/csv/episodes.csv

# Watch live
tail -f experiments/dqn_atari/runs/pong_42_20251115/csv/episodes.csv
```

### File Locations

```
experiments/dqn_atari/runs/<game>_<seed>_<timestamp>/
├── config.yaml             # Frozen config snapshot
├── meta.json               # Run metadata (git hash, etc.)
├── tensorboard/
│   └── events.out.tfevents.*
├── csv/
│   ├── training_steps.csv
│   └── episodes.csv
├── eval/
│   └── evaluations.csv
├── videos/
│   └── <Env>_step_<step>_best_ep<N>_r<return>.mp4
├── checkpoints/            # Model checkpoints
└── artifacts/
```

### Generate Plots

Use `scripts/plot_results.py` to generate publication-quality figures:

**From local CSV files:**
```bash
python scripts/plot_results.py \
  --episodes experiments/dqn_atari/runs/pong_42_20251115/csv/episodes.csv \
  --steps experiments/dqn_atari/runs/pong_42_20251115/csv/training_steps.csv \
  --output plots/pong \
  --game-name pong \
  --formats png pdf svg
```

**From W&B artifacts:**
```bash
python scripts/plot_results.py \
  --wandb-project dqn-atari \
  --wandb-run abc123 \
  --wandb-artifact training_logs_step_10000000:latest \
  --output plots/pong \
  --game-name pong
```

**Multi-seed aggregation (with 95% CI):**
```bash
python scripts/plot_results.py \
  --multi-seed experiments/dqn_atari/runs/pong_42_20251115/csv/episodes.csv \
               experiments/dqn_atari/runs/pong_43_20251115/csv/episodes.csv \
               experiments/dqn_atari/runs/pong_44_20251115/csv/episodes.csv \
  --output plots/pong_multi_seed \
  --game-name pong \
  --smoothing 100
```

**Performance options for large files:**
```bash
# Downsample to 10K points (faster plotting)
python scripts/plot_results.py \
  --episodes results/logs/pong/run_123/csv/episodes.csv \
  --downsample 10000 \
  --warn-size-mb 100 \
  --output plots/pong
```

**Upload plots to W&B:**
```bash
python scripts/plot_results.py \
  --episodes results/logs/pong/run_123/csv/episodes.csv \
  --output plots/pong \
  --upload-wandb \
  --wandb-project dqn-atari \
  --wandb-upload-run abc123
```

### Export Results Tables

Generate summary tables from multiple runs:

```bash
# Scan all runs and generate Markdown/CSV tables
python scripts/export_results_table.py \
  --runs-dir results/logs/pong/ \
  --output results/summary

# Outputs:
# - results/summary/results_summary.md
# - results/summary/results_summary.csv
```

**With W&B upload:**
```bash
python scripts/export_results_table.py \
  --runs-dir results/logs/pong/ \
  --output results/summary \
  --upload-wandb \
  --wandb-project dqn-atari
```

### Plot Types Generated

| Plot | Filename | Description |
|------|----------|-------------|
| Episode Returns | `{game}_episode_returns.{fmt}` | Return over training steps |
| Training Loss | `{game}_training_loss.{fmt}` | TD loss progression |
| Evaluation Scores | `{game}_evaluation_scores.{fmt}` | Periodic eval performance |
| Epsilon Schedule | `{game}_epsilon_schedule.{fmt}` | Exploration decay |

All plots are 300 DPI publication-quality with configurable smoothing.

### Troubleshooting

**W&B not logging:**
```bash
# Check W&B is installed
pip install wandb

# Check login status
wandb login --verify

# Test connection
python -c "import wandb; wandb.init(project='test', mode='online')"
```

**TensorBoard not showing logs:**
```bash
# Check files exist
ls -la results/logs/pong/pong_seed42/tensorboard/

# Try different port
tensorboard --logdir results/logs/pong/ --port 6007
```

**CSV files missing:**
```bash
# Check training actually ran
ls -la results/logs/pong/pong_seed42/csv/

# Check file isn't empty
wc -l results/logs/pong/pong_seed42/csv/episodes.csv
```

### See Also

- **[docs/design/logging_pipeline.md](../../../docs/design/logging_pipeline.md)** - Complete logging architecture and API reference
- **[scripts/plot_results.py](../../../scripts/plot_results.py)** - Run with `--help` for full CLI reference
- **[scripts/export_results_table.py](../../../scripts/export_results_table.py)** - Results table generator
