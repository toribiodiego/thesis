# DQN Atari Configurations

Configuration files for DQN training using hierarchical YAML configs with base + game-specific overrides.

## Quick Start

```bash
# Train Pong with default settings
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42

# Override learning rate
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \
  --set training.optimizer.lr=0.0005

# Multiple overrides
python train_dqn.py --cfg experiments/dqn_atari/configs/breakout.yaml --seed 123 \
  --set training.gamma=0.95 training.total_frames=20000000
```

## File Structure

```
experiments/dqn_atari/configs/
├── base.yaml           # Global defaults for all games
├── pong.yaml          # Pong-specific overrides
├── breakout.yaml      # Breakout-specific overrides
└── beam_rider.yaml    # BeamRider-specific overrides
```

### base.yaml

Contains **all** hyperparameters with DQN paper defaults:
- Network architecture (conv layers, FC hidden units)
- Replay buffer (capacity, batch size, warmup)
- Training (gamma, learning rate, optimizer, loss function)
- Target network (update interval, method)
- Exploration (epsilon schedule)
- Evaluation (frequency, episodes, epsilon)
- Logging and checkpointing

**You should NOT need to edit this file** unless changing global defaults for all games.

### Game Configs (pong.yaml, breakout.yaml, etc.)

Override **only** game-specific values:

```yaml
# pong.yaml
base_config: "experiments/dqn_atari/configs/base.yaml"

experiment:
  name: "pong"
  notes: "Pong training with DQN"

environment:
  env_id: "PongNoFrameskip-v4"  # Game-specific

training:
  total_frames: 10000000  # Override if game needs different frame budget
```

**Pattern**: Reference base, override minimal fields.

## Configuration Hierarchy

Configs are merged in this order (later overrides earlier):

```
1. base.yaml           (lowest priority)
   ↓
2. pong.yaml           (game overrides)
   ↓
3. --set CLI flags     (highest priority)
```

### Example Override Chain

```bash
# In base.yaml:
training.optimizer.lr = 0.00025

# In pong.yaml:
# (nothing - inherits from base)

# On command line:
--set training.optimizer.lr=0.001

# Final value:
training.optimizer.lr = 0.001  # CLI wins
```

## CLI Flags Reference

### Required

| Flag | Description |
|------|-------------|
| `--cfg PATH` | Path to game config YAML |

### Optional

| Flag | Default | Description |
|------|---------|-------------|
| `--seed N` | None | Random seed for reproducibility |
| `--resume PATH` | None | Resume from checkpoint |
| `--set KEY=VALUE` | [] | Override config values (dot notation) |
| `--device DEVICE` | auto | Force device: cuda/cpu/mps |
| `--dry-run` | False | Load config and exit (no training) |
| `--quiet` | False | Suppress config printing |

### CLI Override Syntax

Use dot notation to modify nested values:

```bash
# Single override
--set training.gamma=0.95

# Multiple overrides
--set training.gamma=0.95 replay.capacity=500000

# Nested parameters
--set training.optimizer.rmsprop.alpha=0.99

# Disable target network (set to null)
--set target_network.update_interval=null
```

## Common Workflows

### 1. Basic Training

```bash
# Train Pong with seed
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42

# Train Breakout
python train_dqn.py --cfg experiments/dqn_atari/configs/breakout.yaml --seed 123
```

### 2. Hyperparameter Tuning

```bash
# Try different learning rates
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \
  --set training.optimizer.lr=0.0001

# Change optimizer from RMSprop to Adam
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \
  --set training.optimizer.type=adam training.optimizer.lr=0.0001

# Adjust gamma
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \
  --set training.gamma=0.95
```

### 3. Ablation Studies

```bash
# Disable target network (2013 NIPS DQN)
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \
  --set target_network.update_interval=null

# Different frame stack sizes
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \
  --set environment.preprocessing.frame_stack=2

# Huber loss instead of MSE
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \
  --set training.loss.type=huber training.loss.huber_delta=1.0
```

### 4. Shorter Runs (Testing)

```bash
# Quick smoke test (1M frames)
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \
  --set training.total_frames=1000000

# Dry run (load config without training)
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --dry-run
```

### 5. Resume Training

```bash
# Resume from checkpoint
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml \
  --resume experiments/dqn_atari/runs/pong_42_*/checkpoints/checkpoint_1000000.pt
```

## Creating New Game Configs

**Template** for adding a new game:

```yaml
# experiments/dqn_atari/configs/space_invaders.yaml
base_config: "experiments/dqn_atari/configs/base.yaml"

experiment:
  name: "space_invaders"
  notes: "Space Invaders with DQN"

environment:
  env_id: "SpaceInvadersNoFrameskip-v4"

# Only override if game needs different budget
training:
  total_frames: 15000000  # Optional: if game needs more/less
```

**Steps**:
1. Copy template above
2. Change `experiment.name`
3. Set correct `environment.env_id`
4. Optionally adjust `training.total_frames`
5. That's it! Everything else inherits from base.yaml

**Valid Environment IDs**:
- PongNoFrameskip-v4
- BreakoutNoFrameskip-v4
- BeamRiderNoFrameskip-v4
- QbertNoFrameskip-v4
- SpaceInvadersNoFrameskip-v4
- SeaquestNoFrameskip-v4
- And 10 more (see docs/design/config_cli.md)

## Run Artifacts

Every training run creates a timestamped directory with **complete reproducibility information**:

```
experiments/dqn_atari/runs/pong_42_20250113_143022/
├── config.yaml                   # Merged config snapshot (exact settings used)
├── meta.json                     # Metadata (git hash, seed, versions)
├── logs/                         # Training metrics (CSV, TensorBoard)
├── checkpoints/                  # Model checkpoints
├── artifacts/                    # Debug outputs
└── eval/                         # Evaluation results
```

### config.yaml

The **fully merged and resolved** configuration - exactly what was used for training:

```yaml
# No more base_config reference - all values resolved
experiment:
  name: "pong"

environment:
  env_id: "PongNoFrameskip-v4"
  action_repeat: 4
  # ... all values from base + game + CLI overrides

training:
  gamma: 0.99
  optimizer:
    type: "rmsprop"
    lr: 0.00025
  # ... complete config
```

**Purpose**: Use this file to reproduce runs exactly - it has no references, all values are final.

### meta.json

Reproducibility metadata:

```json
{
  "created_at": "2025-01-13T14:30:22.123456",
  "git": {
    "commit_hash": "a0ebb4c",
    "branch": "main",
    "dirty": false
  },
  "seed": 42,
  "experiment": {"name": "pong"},
  "environment": {"env_id": "PongNoFrameskip-v4"},
  "cli": {
    "args": {
      "config_file": "experiments/dqn_atari/configs/pong.yaml",
      "seed": 42,
      "overrides": ["training.gamma=0.99"]
    }
  }
}
```

**Purpose**: Track exact code version (git hash), CLI arguments, and environment info.

## Configuration Validation

The config system validates all settings at startup with helpful error messages.

### Common Validation Errors

**Invalid gamma:**
```
Error: training.gamma: must be in range [0.0, 1.0], got 1.5
Fix: --set training.gamma=0.99
```

**Invalid optimizer:**
```
Error: optimizer.type: must be one of ['adam', 'rmsprop'], got 'sgd'
Fix: --set training.optimizer.type=adam
```

**Unknown environment:**
```
Error: env_id: unknown environment 'Pong-v0'
Fix: Use "PongNoFrameskip-v4" (must be NoFrameskip-v4 variant)
```

**Zero frameskip:**
```
Error: action_repeat: must be positive, got 0
Fix: environment.action_repeat: 4
```

### Validation Rules

| Field | Constraint | Error if violated |
|-------|-----------|-------------------|
| `training.gamma` | [0.0, 1.0] | "must be in range [0.0, 1.0]" |
| `exploration.*.epsilon` | [0.0, 1.0] | "must be in range [0.0, 1.0]" |
| `environment.action_repeat` | > 0 | "must be positive" (nonzero frameskip) |
| `training.optimizer.type` | rmsprop, adam | "must be one of ['adam', 'rmsprop']" |
| `environment.env_id` | Valid Atari ID | "unknown environment" |
| `replay.capacity` | > 0 | "must be positive" |

**See [docs/design/config_cli.md](../../../docs/design/config_cli.md) for complete validation reference.**

## Debugging Config Issues

**1. Print resolved config:**
```bash
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --print-config
```

**2. Dry run (no training):**
```bash
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --dry-run
```

**3. Check saved snapshot:**
```bash
# After run starts, verify exact config used
cat experiments/dqn_atari/runs/pong_42_*/config.yaml
```

**4. Verify base config loads:**
```python
from src.config import load_config
config = load_config('experiments/dqn_atari/configs/base.yaml', resolve_base=False)
print(config.keys())
```

## Complete Reference

For comprehensive documentation on:
- Override precedence rules
- Schema validation details
- CLI flag reference
- Troubleshooting validation errors
- Advanced usage patterns
- Reproducibility best practices

**See: [docs/design/config_cli.md](../../../docs/design/config_cli.md)**

## Example Command Summary

```bash
# Basic
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42

# Override LR
python train_dqn.py --cfg pong.yaml --seed 42 --set training.optimizer.lr=0.001

# Multiple overrides
python train_dqn.py --cfg pong.yaml --seed 42 \
  --set training.gamma=0.95 training.total_frames=5000000

# Ablation: disable target network
python train_dqn.py --cfg pong.yaml --seed 42 --set target_network.update_interval=null

# Resume training
python train_dqn.py --cfg pong.yaml --resume runs/pong_42_*/checkpoints/checkpoint_1000000.pt

# Dry run
python train_dqn.py --cfg pong.yaml --dry-run
```
