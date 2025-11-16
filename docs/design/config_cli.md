# Configuration & CLI System

Complete guide to DQN training configuration management, command-line interface, and reproducibility features.

---

## Table of Contents

1. [Overview](#overview)
2. [File Hierarchy](#file-hierarchy)
3. [Configuration Loading](#configuration-loading)
4. [Override Precedence](#override-precedence)
5. [Command-Line Interface](#command-line-interface)
6. [Schema Validation](#schema-validation)
7. [Run Directory Structure](#run-directory-structure)
8. [Example Commands](#example-commands)
9. [Troubleshooting](#troubleshooting)
10. [Advanced Usage](#advanced-usage)

---

## Overview

The DQN configuration system provides:

- **Hierarchical YAML configs**: Base defaults + game-specific overrides
- **CLI overrides**: Modify any config value from command line
- **Strict validation**: Catch errors early with helpful messages
- **Automatic reproducibility**: Git hash, seed, full config snapshots saved automatically
- **Dynamic run management**: Organized directory structure created on startup

**Key Design Principles**:
- **Single source of truth**: Merged config snapshot saved with every run
- **Fail fast**: Validation happens at startup, before training begins
- **Traceability**: Every run folder contains complete reproducibility info
- **Flexibility**: Override any parameter without editing files

---

## File Hierarchy

### Configuration Directory Structure

```
experiments/dqn_atari/configs/
├── base.yaml              # Global defaults for all games
├── pong.yaml             # Pong-specific overrides
├── breakout.yaml         # Breakout-specific overrides
└── beam_rider.yaml       # BeamRider-specific overrides
```

### Base Configuration (`base.yaml`)

Contains **all** default hyperparameters:

```yaml
# experiments/dqn_atari/configs/base.yaml
experiment:
  name: "dqn_atari"
  notes: ""

environment:
  env_id: null  # Must be set in game configs
  action_repeat: 4
  preprocessing:
    frame_size: 84
    frame_stack: 4
    grayscale: true

network:
  architecture: "dqn"
  conv1_channels: 16
  conv1_kernel: 8
  fc_hidden: 256
  device: "cuda"

replay:
  capacity: 1000000
  batch_size: 32
  min_size: 50000

training:
  total_frames: 10000000
  train_every: 4
  gamma: 0.99
  optimizer:
    type: "rmsprop"
    lr: 0.00025

# ... (see full file for all parameters)
```

### Game Configuration (`pong.yaml`)

Overrides **only** game-specific values:

```yaml
# experiments/dqn_atari/configs/pong.yaml
base_config: "experiments/dqn_atari/configs/base.yaml"

experiment:
  name: "pong"
  notes: "Pong training with DQN (2013 paper reproduction)"

environment:
  env_id: "PongNoFrameskip-v4"

training:
  total_frames: 10000000  # Pong typically solves in 10M frames
```

**Key Points**:
- `base_config` references parent configuration
- Only override what changes for this game
- All other values inherited from base
- Comments explain game-specific choices

---

## Configuration Loading

### Loading Process

1. **Load game config** (`pong.yaml`)
2. **Resolve base reference** (if `base_config` key present)
3. **Recursively load base** (`base.yaml`)
4. **Deep merge**: Base values + game overrides
5. **Apply CLI overrides** (from `--set` flags)
6. **Validate schema**: Check all constraints
7. **Return merged config**: Single dictionary with all resolved values

### Deep Merge Behavior

```python
# Base config
{
  'training': {
    'gamma': 0.99,
    'optimizer': {
      'type': 'rmsprop',
      'lr': 0.00025
    }
  }
}

# Game override
{
  'training': {
    'optimizer': {
      'lr': 0.0005  # Only change learning rate
    }
  }
}

# Merged result
{
  'training': {
    'gamma': 0.99,           # Kept from base
    'optimizer': {
      'type': 'rmsprop',     # Kept from base
      'lr': 0.0005           # Overridden
    }
  }
}
```

**Merge Rules**:
- Dictionaries are merged recursively
- Lists are replaced entirely (not merged)
- Scalars (int, float, str, bool) are replaced
- `None` values are preserved (explicit nulls)

### Path Resolution

Config paths can be:
- **Absolute**: `/absolute/path/to/base.yaml`
- **Relative to config file**: `../shared/base.yaml`
- **Relative to working directory**: `experiments/dqn_atari/configs/base.yaml`

Resolution order:
1. If absolute path → use directly
2. Try relative to config file's directory
3. Fall back to relative to current working directory

---

## Override Precedence

Configuration values are resolved in this order (later overrides earlier):

```
1. base.yaml (lowest priority)
   ↓
2. game.yaml (overrides base)
   ↓
3. --set CLI flags (overrides game)
   ↓
4. --seed CLI flag (overrides config seed)
   ↓
5. --device CLI flag (highest priority)
```

### Example Precedence Chain

```bash
# Base config has:
training.optimizer.lr = 0.00025
seed.value = null

# Pong config overrides:
training.optimizer.lr = 0.00025  # Same as base (no override)

# CLI overrides:
python train_dqn.py \
  --cfg pong.yaml \
  --seed 42 \                      # Overrides seed.value
  --set training.optimizer.lr=0.001  # Overrides lr

# Final values:
training.optimizer.lr = 0.001  # From --set (highest)
seed.value = 42                # From --seed
```

### Dot Notation for Nested Keys

CLI overrides use dot notation to modify nested values:

```bash
# Modify deeply nested value
--set training.optimizer.rmsprop.alpha=0.99

# Multiple overrides (repeat --set flag for each)
--set training.gamma=0.95 --set training.lr=0.001 --set replay.capacity=500000

# Override list values (replaces entire list)
--set logging.step_metrics=loss,td_error,q_values
```

**Note:** `--set` is repeatable - use one `--set` flag per KEY=VALUE override.

---

## Command-Line Interface

### Basic Usage

```bash
python train_dqn.py --cfg <config_file> [OPTIONS]
```

### Required Flags

| Flag | Type | Description |
|------|------|-------------|
| `--cfg PATH` | str | Path to game configuration YAML file |

### Optional Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--seed N` | int | None | Random seed for reproducibility |
| `--resume PATH` | str | None | Resume training from checkpoint |
| `--set KEY=VALUE` | str[] | [] | Override config values (dot notation, repeatable) |
| `--tags TAG` | str[] | [] | W&B run tags (repeatable, merges with config tags) |
| `--device DEVICE` | str | None | Force device: cuda/cpu/mps |
| `--dry-run` | flag | False | Load config and exit (no training) |
| `--print-config` | flag | False | Print resolved config and exit |
| `--quiet` | flag | False | Suppress config printing at startup |
| `--verbose` | flag | False | Enable verbose logging |

### Config Lifecycle

```
1. Parse CLI arguments
   ↓
2. Load YAML config (with base merging)
   ↓
3. Apply CLI overrides (--set, --seed, --device)
   ↓
4. Validate schema (fail fast on errors)
   ↓
5. Print resolved config (unless --quiet)
   ↓
6. Create run directory
   ↓
7. Save config snapshot (config.yaml)
   ↓
8. Save metadata (meta.json)
   ↓
9. Start training
```

---

## Schema Validation

### Validation Rules

The configuration is validated against a comprehensive schema at startup:

#### Required Fields

| Section | Field | Constraint |
|---------|-------|------------|
| `experiment.name` | string | Non-empty string required |
| `environment.env_id` | string | Must be valid Atari environment |

#### Numeric Constraints

| Field | Constraint | Example Error |
|-------|------------|---------------|
| `training.gamma` | `[0.0, 1.0]` | "gamma: must be in range [0.0, 1.0], got 1.5" |
| `exploration.schedule.start_epsilon` | `[0.0, 1.0]` | "start_epsilon: must be in range [0.0, 1.0], got 2.0" |
| `training.optimizer.rmsprop.alpha` | `[0.0, 1.0]` | "alpha: must be in range [0.0, 1.0], got -0.1" |
| `replay.capacity` | `> 0` | "capacity: must be positive, got 0" |
| `training.total_frames` | `> 0` | "total_frames: must be positive, got -1000" |
| `environment.action_repeat` | `> 0` | "action_repeat: must be positive, got 0" (nonzero frameskip) |

#### Enumerated Values

| Field | Valid Values | Error Message |
|-------|--------------|---------------|
| `training.optimizer.type` | `rmsprop`, `adam` | "type: must be one of ['adam', 'rmsprop'], got 'sgd'" |
| `network.architecture` | `dqn` | "architecture: must be one of ['dqn'], got 'rainbow'" |
| `network.device` | `cuda`, `cpu`, `mps` | "device: must be one of ['cpu', 'cuda', 'mps'], got 'gpu'" |
| `training.loss.type` | `mse`, `huber` | "type: must be one of ['huber', 'mse'], got 'l1'" |

#### Valid Environment IDs

Valid Atari environments (NoFrameskip-v4 variants):
- `PongNoFrameskip-v4`
- `BreakoutNoFrameskip-v4`
- `BeamRiderNoFrameskip-v4`
- `QbertNoFrameskip-v4`
- `SpaceInvadersNoFrameskip-v4`
- `SeaquestNoFrameskip-v4`
- `EnduroNoFrameskip-v4`
- `MsPacmanNoFrameskip-v4`
- And 8 more (see `src/config/schema_validator.py`)

#### Unknown Field Detection

**Strict mode** (enabled by default) rejects unknown fields:

```yaml
# Invalid config
training:
  gamma: 0.99
  unknown_param: 123  # FAIL ERROR

# Error message:
# "Unknown fields in training: 'unknown_param'
#  Valid fields: 'gamma', 'loss', 'optimizer', 'total_frames', 'train_every', 'gradient_clip'"
```

### Validation Error Format

All validation errors follow this format:

```
Configuration validation failed:
<section>.<field>: <constraint description>, got <actual_value>

Example:
Configuration validation failed:
training.gamma: must be in range [0.0, 1.0], got 1.5
```

**Error includes**:
- DONE Exact field path (dot notation)
- DONE Constraint that was violated
- DONE Actual invalid value
- DONE List of valid options (for enums)

---

## Run Directory Structure

### Automatic Directory Creation

On every training run, a timestamped directory is created:

```
experiments/dqn_atari/runs/
└── pong_42_20250113_143022/          # <game>_<seed>_<timestamp>
    ├── config.yaml                   # Merged config snapshot
    ├── meta.json                     # Metadata (git hash, versions, etc.)
    ├── logs/                         # Training logs
    │   ├── train_metrics.csv
    │   └── episode_metrics.csv
    ├── checkpoints/                  # Model checkpoints
    │   ├── checkpoint_1000000.pt
    │   ├── checkpoint_2000000.pt
    │   └── best_model.pt
    ├── artifacts/                    # Debug artifacts
    │   └── plots/
    └── eval/                         # Evaluation results
        ├── eval_250k.json
        └── videos/
```

### Naming Convention

```
<experiment_name>_<seed>_<timestamp>

Examples:
pong_42_20250113_143022
breakout_123_20250113_150000
pong_noseed_20250113_160000  (if --seed not provided)
```

### Config Snapshot (`config.yaml`)

The **fully merged and resolved** configuration, saved at run start:

```yaml
# Exactly what was used for training
experiment:
  name: "pong"
  notes: "Pong training with DQN"

environment:
  env_id: "PongNoFrameskip-v4"
  action_repeat: 4
  # ... all values from base + game + CLI overrides

training:
  gamma: 0.99
  optimizer:
    type: "rmsprop"
    lr: 0.00025
  # ... all training params

# Complete config - no references, all values resolved
```

**Purpose**: Exact reproducibility - this file alone can recreate the run.

### Metadata File (`meta.json`)

Reproducibility metadata saved alongside config:

```json
{
  "created_at": "2025-01-13T14:30:22.123456",
  "python_version": "3.13.3",
  "pytorch_version": "2.1.0",
  "git": {
    "commit_hash": "a0ebb4c",
    "commit_hash_full": "a0ebb4c1ad8290572519d785affb06e14e48f129",
    "branch": "main",
    "dirty": false,
    "available": true
  },
  "seed": 42,
  "experiment": {
    "name": "pong",
    "notes": "Pong training with DQN"
  },
  "environment": {
    "env_id": "PongNoFrameskip-v4"
  },
  "training": {
    "total_frames": 10000000,
    "optimizer_lr": 0.00025
  },
  "cli": {
    "args": {
      "config_file": "experiments/dqn_atari/configs/pong.yaml",
      "seed": 42,
      "resume": null,
      "overrides": ["training.optimizer.lr=0.00025"],
      "dry_run": false,
      "device": null
    }
  }
}
```

**Key Fields**:
- `git.commit_hash`: Code version used for training
- `git.dirty`: Whether uncommitted changes existed
- `seed`: Random seed (for exact reproducibility)
- `cli.args`: Exact command-line arguments used
- `created_at`: ISO 8601 timestamp

### Subdirectory Purposes

| Directory | Purpose | Contents |
|-----------|---------|----------|
| `logs/` | Training metrics | CSV files, TensorBoard logs |
| `checkpoints/` | Model weights | `.pt` checkpoint files |
| `artifacts/` | Debug outputs | Plots, visualizations, debug info |
| `eval/` | Evaluation results | Metrics JSON, video recordings |

---

## Example Commands

### Basic Training

```bash
# Train Pong with default settings
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml

# Train with specific seed (for reproducibility)
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42

# Train Breakout
python train_dqn.py --cfg experiments/dqn_atari/configs/breakout.yaml --seed 123
```

### CLI Overrides

```bash
# Override learning rate
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set training.optimizer.lr=0.0005

# Multiple overrides
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set training.gamma=0.95 \
       training.total_frames=5000000 \
       replay.capacity=500000

# Override optimizer type and parameters
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --set training.optimizer.type=adam \
       training.optimizer.lr=0.0001
```

### Ablation Studies

```bash
# Disable target network (2013 NIPS DQN)
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set target_network.update_interval=null

# Different frame stack sizes
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml \
  --seed 42 --set environment.preprocessing.frame_stack=2

python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml \
  --seed 42 --set environment.preprocessing.frame_stack=8

# Huber loss instead of MSE
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --set training.loss.type=huber \
       training.loss.huber_delta=1.0
```

### Resuming Training

```bash
# Resume from checkpoint
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --resume experiments/dqn_atari/runs/pong_42_20250113_143022/checkpoints/checkpoint_1000000.pt

# Note: Config from checkpoint takes precedence over --cfg
# Seed is restored from checkpoint automatically
```

### Dry Runs & Debugging

```bash
# Test config loading without training
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --dry-run

# Print resolved config and exit
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --set training.gamma=0.95 \
  --print-config

# Quiet mode (suppress config printing)
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --quiet
```

### Device Selection

```bash
# Force CPU (even if CUDA available)
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --device cpu

# Force CUDA
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --device cuda

# Force MPS (Apple Silicon)
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --device mps
```

---

## Troubleshooting

### Schema Validation Errors

#### Error: "gamma: must be in range [0.0, 1.0], got 1.5"

**Cause**: Discount factor γ must be between 0 and 1.

**Fix**:
```bash
# Wrong
--set training.gamma=1.5

# Correct
--set training.gamma=0.99
```

#### Error: "optimizer.type: must be one of ['adam', 'rmsprop'], got 'sgd'"

**Cause**: Only RMSProp and Adam optimizers are supported.

**Fix**:
```bash
# Wrong
--set training.optimizer.type=sgd

# Correct
--set training.optimizer.type=adam
# or
--set training.optimizer.type=rmsprop
```

#### Error: "env_id: unknown environment 'Pong-v0'"

**Cause**: Only NoFrameskip-v4 Atari environments are supported.

**Fix**:
```yaml
# Wrong
environment:
  env_id: "Pong-v0"

# Correct
environment:
  env_id: "PongNoFrameskip-v4"
```

**Valid environments**: See [Schema Validation](#valid-environment-ids) section.

#### Error: "action_repeat: must be positive, got 0"

**Cause**: Frame skip (action_repeat) cannot be zero.

**Fix**:
```yaml
# Wrong
environment:
  action_repeat: 0

# Correct (DQN paper default)
environment:
  action_repeat: 4
```

#### Error: "Unknown fields in training: 'learning_rate'"

**Cause**: Field name is wrong (should be `optimizer.lr`, not `learning_rate`).

**Fix**:
```bash
# Wrong
--set training.learning_rate=0.001

# Correct
--set training.optimizer.lr=0.001
```

**Tip**: Check `experiments/dqn_atari/configs/base.yaml` for exact field names.

#### Error: "capacity: must be an integer, got 1e6"

**Cause**: Scientific notation parsed as float, not int.

**Fix**:
```yaml
# Wrong
replay:
  capacity: 1e6

# Correct
replay:
  capacity: 1000000
```

#### Error: "experiment: 'name' is required"

**Cause**: Every config must have an experiment name.

**Fix**:
```yaml
# Add to your game config
experiment:
  name: "pong"
```

### Config Loading Errors

#### Error: "Config file not found: pong.yaml"

**Cause**: Path is relative to current directory, not config directory.

**Fix**:
```bash
# Wrong (if running from project root)
python train_dqn.py --cfg pong.yaml

# Correct
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml
```

#### Error: "Could not load base config: base.yaml"

**Cause**: Base config path in game config is incorrect.

**Fix**:
```yaml
# In pong.yaml - use full path from project root
base_config: "experiments/dqn_atari/configs/base.yaml"

# Or relative to config file location
base_config: "./base.yaml"
```

### CLI Override Errors

#### Error: "Invalid override format: 'training.lr 0.001'"

**Cause**: Missing `=` between key and value.

**Fix**:
```bash
# Wrong
--set training.lr 0.001

# Correct
--set training.lr=0.001
```

#### Error: Type mismatch after override

**Cause**: CLI overrides are parsed as strings and auto-converted.

**Fix**: Explicitly type your overrides:
```bash
# For booleans
--set environment.preprocessing.grayscale=true   # lowercase

# For lists (comma-separated)
--set logging.step_metrics=loss,td_error,q_values

# For null
--set target_network.update_interval=null
```

### Run Directory Errors

#### Error: "Permission denied: experiments/dqn_atari/runs/"

**Cause**: No write permission in runs directory.

**Fix**:
```bash
# Create directory with correct permissions
mkdir -p experiments/dqn_atari/runs
chmod 755 experiments/dqn_atari/runs

# Or override base directory
--set logging.base_dir=/tmp/dqn_runs
```

#### Error: "Metadata save failed: git not found"

**Cause**: Git not installed or not in PATH.

**Fix**:
- Install git: `brew install git` (macOS) or `apt install git` (Linux)
- Git info will be marked as unavailable but training continues

### Debugging Tips

**1. Use `--dry-run` to test config loading:**
```bash
python train_dqn.py --cfg pong.yaml --seed 42 --dry-run
```

**2. Use `--print-config` to see resolved values:**
```bash
python train_dqn.py --cfg pong.yaml --set training.gamma=0.95 --print-config
```

**3. Check saved config snapshot:**
```bash
# After run starts, check what was actually used
cat experiments/dqn_atari/runs/pong_42_*/config.yaml
```

**4. Validate base config alone:**
```bash
# Load base config without game overrides
python -c "
from src.config import load_config, validate_config
config = load_config('experiments/dqn_atari/configs/base.yaml', resolve_base=False)
# Will fail if base.yaml has required fields missing - this is expected
"
```

**5. Check metadata for CLI args used:**
```bash
# See exact command that started the run
cat experiments/dqn_atari/runs/pong_42_*/meta.json | grep -A 10 '"cli"'
```

---

## Advanced Usage

### Creating Custom Game Configs

**Template** for new game:

```yaml
# experiments/dqn_atari/configs/my_game.yaml
base_config: "experiments/dqn_atari/configs/base.yaml"

experiment:
  name: "my_game"
  notes: "Custom game configuration"

environment:
  env_id: "MyGameNoFrameskip-v4"  # Must be in VALID_ENV_IDS

# Only override what differs from base
training:
  total_frames: 20000000  # If game needs more frames

# Everything else inherited from base
```

### Modifying Base Defaults

To change defaults for **all** games:

1. Edit `experiments/dqn_atari/configs/base.yaml`
2. Leave game configs unchanged (they inherit new defaults)
3. Commit base.yaml changes for reproducibility

**Example**: Change default optimizer for all games:
```yaml
# In base.yaml
training:
  optimizer:
    type: "adam"  # Changed from rmsprop
    lr: 0.0001
```

### Programmatic Config Access

```python
from src.config import load_config, validate_config

# Load and validate config
config = load_config('experiments/dqn_atari/configs/pong.yaml')
validate_config(config)

# Access nested values
gamma = config['training']['gamma']
env_id = config['environment']['env_id']

# Apply overrides
from src.config import merge_cli_overrides
overrides = ['training.gamma=0.95', 'replay.capacity=500000']
config = merge_cli_overrides(config, overrides)
```

### Custom Validation Rules

To add new validation rules, edit `src/config/schema_validator.py`:

```python
# Add new valid value
VALID_OPTIMIZERS = {"rmsprop", "adam", "sgd"}  # Add sgd

# Add new environment
VALID_ENV_IDS = {
    "PongNoFrameskip-v4",
    "MyNewGameNoFrameskip-v4",  # Add custom environment
}

# Add new field to known structure
KNOWN_STRUCTURE = {
    "training": {
        "gamma", "loss", "optimizer",
        "my_new_field"  # Add new field
    }
}
```

### Reproducibility Checklist

To ensure exact reproducibility:

DONE **Use specific seed**: `--seed 42`
DONE **Track git commit**: Check `meta.json` has `git.dirty = false`
DONE **Save config snapshot**: Automatically saved in run directory
DONE **Record package versions**: Check `meta.json` for PyTorch version
DONE **Note system info**: Document GPU model, CUDA version if needed

**To reproduce a run exactly**:
```bash
# 1. Checkout exact commit
git checkout <commit_hash_from_meta.json>

# 2. Use saved config snapshot
python train_dqn.py --cfg <run_dir>/config.yaml --seed <seed_from_meta>

# 3. Or use original command from meta.json
cat <run_dir>/meta.json | grep -A 5 '"cli"'
```

### Using Artifacts for Reproduction

Every run directory contains complete reproducibility information. Here's how to use these artifacts:

**1. Diff configs to understand what changed:**
```bash
# Compare your current config against what was actually used
diff experiments/dqn_atari/configs/pong.yaml \
     experiments/dqn_atari/runs/pong_42_20250113_143022/config.yaml

# Compare two runs to see parameter differences
diff experiments/dqn_atari/runs/pong_42_*/config.yaml \
     experiments/dqn_atari/runs/pong_123_*/config.yaml
```

**2. Extract exact CLI command used:**
```bash
# View CLI arguments from metadata
cat experiments/dqn_atari/runs/pong_42_*/meta.json | jq '.cli.args'

# Reconstruct original command
echo "python train_dqn.py --cfg $(jq -r '.cli.args.config_file' meta.json) --seed $(jq -r '.cli.args.seed' meta.json)"
```

**3. Verify code version match:**
```bash
# Check git commit used for run
jq -r '.git.commit_hash' experiments/dqn_atari/runs/pong_42_*/meta.json

# Check if there were uncommitted changes
jq -r '.git.dirty' experiments/dqn_atari/runs/pong_42_*/meta.json  # Should be false

# Checkout exact code version
git checkout $(jq -r '.git.commit_hash_full' meta.json)
```

**4. Reproduce run exactly:**
```bash
# Method 1: Use saved config snapshot (recommended)
RUN_DIR=experiments/dqn_atari/runs/pong_42_20250113_143022
python train_dqn.py --cfg $RUN_DIR/config.yaml --seed $(jq -r '.seed' $RUN_DIR/meta.json)

# Method 2: Reconstruct from original config + overrides
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \
  $(jq -r '.cli.args.overrides[]' meta.json | sed 's/^/--set /' | tr '\n' ' ')
```

**5. Compare package versions:**
```bash
# Check Python and PyTorch versions used
jq '.python_version, .pytorch_version' experiments/dqn_atari/runs/pong_42_*/meta.json

# Current versions
python -c "import sys; import torch; print(f'Python: {sys.version.split()[0]}, PyTorch: {torch.__version__}')"
```

**Why this matters:**
- **Config snapshot** = single source of truth (no base references, all values resolved)
- **Metadata** = exact code version, CLI args, package versions
- **Together** = complete reproducibility without reverse-engineering

**Common workflow:**
1. Train multiple runs with different seeds/hyperparameters
2. Find best-performing run
3. Use `config.yaml` from that run for production training
4. Check `meta.json` to verify clean git state and package versions

### Multi-Seed Sweeps

Run same config with multiple seeds:

```bash
#!/bin/bash
# sweep.sh
for seed in 42 123 456 789 1000; do
  python train_dqn.py \
    --cfg experiments/dqn_atari/configs/pong.yaml \
    --seed $seed
done
```

Each run creates separate directory:
```
runs/
├── pong_42_20250113_143022/
├── pong_123_20250113_150000/
├── pong_456_20250113_153000/
├── pong_789_20250113_160000/
└── pong_1000_20250113_163000/
```

### Hyperparameter Sweeps

```bash
#!/bin/bash
# lr_sweep.sh
for lr in 0.0001 0.00025 0.0005 0.001; do
  python train_dqn.py \
    --cfg experiments/dqn_atari/configs/pong.yaml \
    --seed 42 \
    --set training.optimizer.lr=$lr
done
```

**Tip**: Use experiment notes to identify sweeps:
```bash
--set experiment.notes="LR sweep: lr=$lr"
```

---

## Summary

### Quick Reference

| Task | Command |
|------|---------|
| Basic training | `python train_dqn.py --cfg pong.yaml --seed 42` |
| Override learning rate | `--set training.optimizer.lr=0.001` |
| Multiple overrides | `--set gamma=0.95 total_frames=5000000` |
| Resume training | `--resume path/to/checkpoint.pt` |
| Test config | `--dry-run` |
| Print config | `--print-config` |
| Force CPU | `--device cpu` |
| Quiet mode | `--quiet` |

### Key Files

| File | Purpose |
|------|---------|
| `base.yaml` | Global defaults for all games |
| `<game>.yaml` | Game-specific overrides |
| `<run_dir>/config.yaml` | Merged config snapshot (reproducibility) |
| `<run_dir>/meta.json` | Metadata (git hash, seed, versions) |

### Core Principles

1. **Hierarchical configs**: Base + game overrides
2. **CLI has highest priority**: Command-line overrides everything
3. **Validate early**: Fail at startup, not mid-training
4. **Save everything**: Config + metadata for reproducibility
5. **Fail fast**: Clear error messages guide fixes

---

## References

- Config loader implementation: `src/config/config_loader.py`
- CLI implementation: `src/config/cli.py`
- Schema validator: `src/config/schema_validator.py`
- Run manager: `src/config/run_manager.py`
- Base configuration: `experiments/dqn_atari/configs/base.yaml`
- Example game configs: `experiments/dqn_atari/configs/*.yaml`

**For issues or questions**, check:
1. This documentation first
2. Error message (includes field path and constraint)
3. `base.yaml` for field names and defaults
4. `schema_validator.py` for complete validation rules
