# Scripts

Training and setup utilities for DQN Atari experiments. All scripts should be run from the repository root.

## `run_dqn.sh`

Launch DQN training or dry-run validation with specified config.

**Purpose:** Convenience wrapper around `python src/train_dqn.py` that handles path resolution and config loading.

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

**Outputs:**
- Dry run: `experiments/dqn_atari/runs/{experiment_name}_{seed}/` (frames, logs, metadata)
- Training: `experiments/dqn_atari/runs/{experiment_name}_{seed}/` (checkpoints, logs, metrics)

## `setup_roms.sh`

Download and install Atari 2600 ROMs required for ALE environments.

**Purpose:** One-time setup to install legally-redistributable ROMs via AutoROM.

**Usage:**
```bash
./experiments/dqn_atari/scripts/setup_roms.sh
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

## `capture_env.sh`

Capture system and environment information for reproducibility tracking.

**Purpose:** Record Python packages, Git state, hardware specs, and dependencies for experiment provenance.

**Usage:**
```bash
./experiments/dqn_atari/scripts/capture_env.sh
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
./experiments/dqn_atari/scripts/setup_roms.sh

# Capture environment state
./experiments/dqn_atari/scripts/capture_env.sh

# Verify setup with dry run
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# Begin training
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --seed 123
```

## Common Workflows

### First-time setup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download ROMs
./experiments/dqn_atari/scripts/setup_roms.sh

# 3. Capture environment
./experiments/dqn_atari/scripts/capture_env.sh

# 4. Validate with dry run
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run
```

### Reproduce Subtasks 1-2
```bash
# Subtask 1: Environment setup and ROM installation
./experiments/dqn_atari/scripts/setup_roms.sh
./experiments/dqn_atari/scripts/capture_env.sh

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
