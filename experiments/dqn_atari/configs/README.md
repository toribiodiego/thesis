# Configs

Configuration management for DQN Atari experiments using OmegaConf. All configs use YAML format with hierarchical structure and interpolation support.

## Structure

- **`base.yaml`** – Default hyperparameters and runtime settings for DQN reproduction
- **`pong.yaml`, `breakout.yaml`, `beam_rider.yaml`** – Game-specific overrides (env ID, frame budgets)

## OmegaConf Basics

Configs are loaded using OmegaConf, which provides:

1. **Variable interpolation:** Reference other config values
   ```yaml
   experiment:
     name: dqn_pong
     seed: 42
     output_dir: runs/${experiment.name}_${experiment.seed}
     # Resolves to: runs/dqn_pong_42
   ```

2. **Hierarchical defaults:** Per-game configs inherit from `base.yaml`
   ```yaml
   # pong.yaml only needs to override specific fields
   defaults:
     - base

   env:
     id: ALE/Pong-v5  # Override only the game ID
   ```

3. **CLI overrides:** Override any key at runtime
   ```bash
   ./experiments/dqn_atari/scripts/run_dqn.sh \
     experiments/dqn_atari/configs/pong.yaml \
     --seed 123 --agent.learning_rate 0.0001
   ```

## Creating Per-Game Configs

**Method 1: Inherit from base (recommended)**
```yaml
# breakout.yaml
defaults:
  - base

experiment:
  name: dqn_breakout

env:
  id: ALE/Breakout-v5

training:
  total_frames: 20000000  # Breakout needs more frames
```

**Method 2: Copy and modify**
```bash
cp experiments/dqn_atari/configs/base.yaml experiments/dqn_atari/configs/my_game.yaml
# Edit my_game.yaml to change env.id and other fields
```

## Key Configuration Sections

### Environment (`env`)

Controls Atari environment setup and preprocessing.

**Must override:**
- `id`: ALE game identifier (e.g., `ALE/Pong-v5`, `ALE/Breakout-v5`)

**Common toggles:**
- `max_noop_start`: Random no-op steps on reset (0-30, default: 30)
  - Set to `0` to disable no-op randomization (deterministic starts)
- `frameskip`: Action repeat count (default: 4, DQN paper standard)
- `repeat_action_probability`: Stochastic frame-skip (default: 0.0 for deterministic)

**Example:**
```yaml
env:
  id: ALE/BeamRider-v5
  max_noop_start: 30  # Diverse initial states
  frameskip: 4         # Repeat actions 4 times
```

### Training (`training`)

Training protocol and termination behavior.

**Key options:**
- `total_frames`: Total environment steps (default: 10M for Pong, 20M+ for harder games)
- `episode_life`: Treat life loss as episode termination
  - `false` (default): Full-episode termination only (DQN paper standard)
  - `true`: Life loss ends episode (optional training optimization)
- `reward_clip`: Clip rewards to {-1, 0, +1} (default: true, DQN paper standard)
- `train_frequency`: Train every N steps (default: 4, matches frameskip)

**Episode life toggle:**
```yaml
training:
  # Pure DQN paper approach (full episodes only)
  episode_life: false

  # OR

  # Optional optimization (helps agent preserve lives)
  episode_life: true
```

**When to use `episode_life=true`:**
- Agent struggles to learn long-term strategy
- Game has multiple lives and life preservation is important
- Common in many DQN implementations (but not required by paper)

**When to use `episode_life=false`:**
- Pure reproduction of DQN 2013 paper
- Evaluation (ALWAYS use false for eval)
- Prefer full-episode returns without artificial termination

### Agent (`agent`)

DQN hyperparameters and architecture.

**Core hyperparameters:**
- `gamma`: Discount factor (default: 0.99)
- `learning_rate`: Optimizer learning rate (default: 0.00025)
- `batch_size`: Replay batch size (default: 32)
- `replay_capacity`: Replay buffer size (default: 1M transitions)
- `replay_min_transitions`: Minimum transitions before training (default: 50K)

**Target network:**
- `target_update_interval`: Hard-copy every N steps (default: 10000)
  - Set to `0` or `null` to disable target network (2013 purist mode)
  - See "Target Network Toggle" section below

**Optimizer settings:**
- `optimizer`: `rmsprop` or `adam` (default: rmsprop, DQN paper standard)
- `optimizer_eps`: Epsilon for numerical stability (default: 1e-5)
- `optimizer_alpha`: RMSprop decay (default: 0.95)
- `gradient_clip_norm`: Max gradient norm (default: 10.0)

**Example ablation:**
```yaml
agent:
  optimizer: adam  # Try Adam instead of RMSprop
  learning_rate: 0.0001  # Lower LR
  target_update_interval: 0  # Disable target network (2013 mode)
```

### Exploration (`exploration`)

ε-greedy schedule for action selection.

**Linear schedule (default):**
```yaml
exploration:
  schedule: linear
  epsilon_start: 1.0    # Start fully random
  epsilon_end: 0.1      # End at 10% random
  epsilon_decay_frames: 1000000  # Decay over 1M frames
```

**Fixed epsilon:**
```yaml
exploration:
  schedule: constant
  epsilon: 0.1  # Always 10% random
```

### Evaluation (`eval`)

Evaluation protocol settings.

**Key options:**
- `episodes`: Number of eval episodes per checkpoint (default: 10)
- `epsilon`: ε-greedy during eval (default: 0.05, set to 0.0 for fully greedy)
- `episode_life`: MUST be `false` for correct episode returns
- `deterministic`: Override epsilon to 0.0 (fully greedy)

**Example:**
```yaml
eval:
  episodes: 30  # More episodes for stable estimates
  epsilon: 0.0  # Fully greedy
  episode_life: false  # REQUIRED: Full episodes
```

### Intervals

Logging and checkpointing frequencies (in frames).

```yaml
intervals:
  log_frames: 10000         # Log training metrics every 10K frames
  eval_frames: 250000       # Evaluate every 250K frames
  checkpoint_frames: 1000000  # Save checkpoint every 1M frames
```

## Target Network Toggle (Subtask 5)

The target network can be disabled for a purist 2013 DQN reproduction.

**Enable target network (default, 2015 Nature DQN):**
```yaml
agent:
  target_update_interval: 10000  # Hard-copy every 10K steps
```

**Disable target network (2013 NIPS DQN):**
```yaml
agent:
  target_update_interval: 0  # or null
```

**Implications:**
- **2013 mode (disabled):** TD targets computed using online Q-network
  - Simpler, matches original 2013 NIPS paper
  - May have less stable training (moving target problem)
  - Useful for ablation studies

- **2015 mode (enabled):** TD targets computed using frozen target network
  - More stable training (fixed target for C update steps)
  - Standard in most DQN implementations
  - Recommended for best performance

See [docs/design/dqn_training.md](../../../docs/design/dqn_training.md) for detailed explanation of target network behavior.

## Common Workflows

### Quick smoke test
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --training.total_frames 100000 --seed 0
```

### Dry run validation
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run
```

### Override multiple parameters
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml \
  --seed 123 \
  --agent.learning_rate 0.0001 \
  --training.episode_life false \
  --eval.epsilon 0.0
```

### Disable target network
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --agent.target_update_interval 0
```

## Config Validation

Before training, verify your config:

```python
from omegaconf import OmegaConf

# Load and inspect
cfg = OmegaConf.load("experiments/dqn_atari/configs/pong.yaml")
print(OmegaConf.to_yaml(cfg))

# Check interpolation
print(f"Output dir: {cfg.experiment.output_dir}")

# Validate required fields
assert cfg.env.id.startswith("ALE/")
assert cfg.agent.replay_capacity > cfg.agent.replay_min_transitions
```

## Sample Usage

```bash
# Use base config directly (Pong by default)
python src/train_dqn.py --config experiments/dqn_atari/configs/base.yaml

# Use game-specific config
python src/train_dqn.py --config experiments/dqn_atari/configs/breakout.yaml

# Or use convenience script
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --seed 42
```
