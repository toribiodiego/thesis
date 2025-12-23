# DQN Architecture Overview

High-level system architecture and component interactions for DQN Atari implementation. This document provides a bird's-eye view before diving into detailed design docs.

---

## System Components

The DQN implementation consists of 5 core subsystems:

```
┌─────────────────────────────────────────────────────────────────┐
│                         DQN TRAINING SYSTEM                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Environment    │    │    Q-Network     │    │  Replay Buffer   │
│  (Atari Game)    │───▶│   (DQN Model)    │◀───│  (Experience)    │
│                  │    │                  │    │                  │
│ - Frame Stack    │    │ - Online Model   │    │ - Circular Ring  │
│ - Preprocessing  │    │ - Target Model   │    │ - Episode Track  │
│ - Reward Clip    │    │ - Action Select  │    │ - Uniform Sample │
└──────────────────┘    └──────────────────┘    └──────────────────┘
        │                       │                         │
        │                       │                         │
        └───────────────────────┼─────────────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │   Training Loop      │
                    │  (Orchestration)     │
                    │                      │
                    │ - ε-greedy Explore   │
                    │ - TD Loss Update     │
                    │ - Target Sync        │
                    │ - Logging/Eval       │
                    │ - Checkpointing      │
                    └──────────────────────┘
```

---

## Data Flow

### Training Step (Single Iteration)

```
1. SELECT ACTION
   ┌─────────────────────────────────────────────┐
   │ Epsilon Scheduler                           │
   │   ↓ current_epsilon = 0.85                  │
   │ ε-greedy Policy                             │
   │   if random() < ε: action = random()        │
   │   else: action = argmax(Q(state))           │
   └─────────────────────────────────────────────┘
                    ↓
2. EXECUTE ACTION
   ┌─────────────────────────────────────────────┐
   │ Environment (4-frame repeat)                │
   │   for i in range(4):                        │
   │     obs, reward, done = env.step(action)    │
   │     total_reward += reward                  │
   │   Preprocessing: grayscale, resize, stack   │
   └─────────────────────────────────────────────┘
                    ↓
3. STORE TRANSITION
   ┌─────────────────────────────────────────────┐
   │ Replay Buffer                               │
   │   buffer.append(s, a, r, s', done)          │
   │   write_index = (write_index + 1) % capacity│
   │   Track episode boundaries                  │
   └─────────────────────────────────────────────┘
                    ↓
4. OPTIMIZE (every 4 steps after 50K warm-up)
   ┌─────────────────────────────────────────────┐
   │ Sample Batch                                │
   │   batch = buffer.sample(32)                 │
   │   (s, a, r, s', done) x 32                  │
   └─────────────────────────────────────────────┘
                    ↓
   ┌─────────────────────────────────────────────┐
   │ Compute TD Targets                          │
   │   with torch.no_grad():                     │
   │     max_q_next = Q_target(s').max(dim=1)    │
   │     y = r + γ * (1 - done) * max_q_next     │
   └─────────────────────────────────────────────┘
                    ↓
   ┌─────────────────────────────────────────────┐
   │ Compute Loss                                │
   │   q_pred = Q_online(s).gather(a)            │
   │   loss = MSE(q_pred, y)                     │
   └─────────────────────────────────────────────┘
                    ↓
   ┌─────────────────────────────────────────────┐
   │ Backprop & Optimize                         │
   │   loss.backward()                           │
   │   clip_grad_norm(params, max_norm=10.0)     │
   │   optimizer.step()                          │
   └─────────────────────────────────────────────┘
                    ↓
5. UPDATE TARGET (every 10K steps)
   ┌─────────────────────────────────────────────┐
   │ Hard Sync Target Network                    │
   │   Q_target.load_state_dict(                 │
   │       Q_online.state_dict()                 │
   │   )                                         │
   └─────────────────────────────────────────────┘
                    ↓
6. LOG & EVALUATE
   ┌─────────────────────────────────────────────┐
   │ Periodic Logging (every 10K frames)         │
   │   Log: loss, epsilon, grad_norm, FPS        │
   │ Periodic Eval (every 250K frames)           │
   │   Run 10 episodes with ε=0.05               │
   │ Checkpointing (every 1M frames)             │
   │   Save models, optimizer, replay, RNG       │
   └─────────────────────────────────────────────┘
```

---

## Component Details

### 1. Environment (src/envs/)

**Purpose:** Atari game simulation with DQN-specific preprocessing

**Wrapper Chain:**
```
ALE/Pong-v5
  ↓ NoopResetEnv (0-30 random no-ops at episode start)
  ↓ MaxAndSkipEnv (action repeat 4x + max-pooling last 2 frames)
  ↓ EpisodeLifeEnv (OPTIONAL: treat life-loss as terminal)
  ↓ RewardClipper (clip rewards to {-1, 0, +1})
  ↓ AtariPreprocessing (grayscale + resize to 84x84)
  ↓ FrameStack (stack last 4 frames)
  ↓ Final: (4, 84, 84) uint8 → float32 [0,1]
```

**Key Files:**
- `src/envs/atari_wrappers.py` - Wrapper implementations
- `src/envs/make_env.py` - Environment factory

**Docs:** [atari-env-wrapper.md](../reference/atari-env-wrapper.md)

---

### 2. Q-Network (src/models/)

**Purpose:** Deep CNN mapping frames to Q-values

**Architecture:**
```
Input: (B, 4, 84, 84) float32

Conv1: 16 filters, 8x8 kernel, stride=4
  ↓ ReLU
  → (B, 16, 20, 20)

Conv2: 32 filters, 4x4 kernel, stride=2
  ↓ ReLU
  → (B, 32, 9, 9)

Flatten
  → (B, 2592)

FC1: 256 units
  ↓ ReLU
  → (B, 256)

FC2: num_actions units
  → (B, 6)  [for Pong]

Output: Q-values for each action
```

**Two Instances:**
- **Online Network:** Trained every step (parameters updated)
- **Target Network:** Frozen copy (hard-synced every 10K steps)

**Key Files:**
- `src/models/dqn.py` - DQN model implementation

**Docs:** [dqn-model.md](../reference/dqn-model.md)

---

### 3. Replay Buffer (src/replay/)

**Purpose:** Store and sample past experiences for off-policy learning

**Storage:**
```
Capacity: 1,000,000 transitions
Memory Layout:
  observations:    (1M, 4, 84, 84) uint8   ~27 GB
  actions:         (1M,) int64              ~8 MB
  rewards:         (1M,) float32            ~4 MB
  dones:           (1M,) bool               ~1 MB
  episode_starts:  (1M,) bool               ~1 MB
Total: ~27 GB
```

**Sampling:**
- Uniform random sampling (no prioritization)
- Episode boundary tracking (no cross-episode samples)
- Warm-up: 50K transitions before training starts
- Batch size: 32

**Key Files:**
- `src/replay/uniform_buffer.py` - Circular replay buffer

**Docs:** [replay-buffer.md](../reference/replay-buffer.md)

---

### 4. Training Loop (src/training/)

**Purpose:** Orchestrate all components for end-to-end learning

**Schedulers:**
- **Epsilon Scheduler:** 1.0 → 0.1 over 1M frames
- **Training Frequency:** Every 4 environment steps
- **Target Sync:** Every 10,000 steps
- **Logging:** Every 10,000 frames
- **Evaluation:** Every 250,000 frames
- **Checkpointing:** Every 1,000,000 frames

**Key Components:**
- `DQNTrainer` - Main training orchestrator
- `EpsilonScheduler` - Exploration decay
- `TargetNetworkUpdater` - Periodic hard sync
- `MetricsLogger` - Step/episode/eval logging
- `CheckpointManager` - Save/load system

**Key Files:**
- `src/training/dqn_trainer.py` - Main trainer
- `src/training/schedulers.py` - Epsilon and update schedulers
- `src/training/checkpoint.py` - Checkpoint/resume system

**Docs:** [training-loop-runtime.md](../reference/training-loop-runtime.md), [dqn-training.md](../reference/dqn-training.md)

---

### 5. Checkpointing System (src/training/)

**Purpose:** Save/restore complete training state for resumption and reproducibility

**What Gets Saved:**
```python
checkpoint = {
    # Metadata
    'schema_version': '1.0.0',
    'timestamp': '2025-01-15T10:30:45',
    'commit_hash': 'a1b2c3d',

    # Training State
    'step': 1000000,
    'episode': 5000,
    'epsilon': 0.5,

    # Models
    'online_model_state_dict': {...},
    'target_model_state_dict': {...},

    # Optimizer
    'optimizer_state_dict': {...},

    # Replay Buffer
    'replay_buffer_state': {
        'index': 250000,
        'size': 250000,
        'capacity': 1000000
    },

    # RNG States (for determinism)
    'rng_states': {
        'python_random': (...),
        'numpy_random': (...),
        'torch_cpu': tensor(...),
        'torch_cuda': [tensor(...)],
        'env': {...}
    }
}
```

**Key Files:**
- `src/training/checkpoint.py` - Save/load logic
- `src/training/resume.py` - Resume validation

**Docs:** [checkpointing.md](../reference/checkpointing.md)

---

## Configuration System

**Hierarchical YAML Configs:**

```
experiments/dqn_atari/configs/
├── base.yaml              # Global defaults
├── pong.yaml              # Game-specific overrides
├── breakout.yaml
└── beam_rider.yaml
```

**Config Structure:**
```yaml
experiment:
  name: "pong"
  seed: 42
  deterministic:
    enabled: false

env:
  env_id: "ALE/Pong-v5"
  episode_life: false
  reward_clip: true
  frameskip: 4

agent:
  num_actions: 6

replay:
  capacity: 1000000
  warmup_steps: 50000
  batch_size: 32

training:
  total_frames: 10000000
  train_every: 4
  gamma: 0.99
  optimizer:
    type: "RMSprop"
    lr: 0.00025
    rho: 0.95
    eps: 0.01
  grad_clip_norm: 10.0
  target_update_interval: 10000

exploration:
  epsilon_start: 1.0
  epsilon_end: 0.1
  epsilon_frames: 1000000

eval:
  eval_every: 250000
  eval_episodes: 10
  eval_epsilon: 0.05

logging:
  log_every: 10000
  checkpoint_every: 1000000
```

**CLI Overrides:**
```bash
./run_dqn.sh config.yaml \
  --seed 123 \
  --set training.optimizer.lr=1e-4 \
  --set exploration.epsilon_end=0.01
```

**Docs:** [Config README](../experiments/dqn_atari/configs/README.md)

---

## Directory Structure

```
thesis/
├── src/                          # Reusable RL components
│   ├── models/                   # Neural network architectures
│   │   └── dqn.py                # DQN CNN model
│   ├── replay/                   # Experience replay
│   │   └── uniform_buffer.py     # Circular buffer
│   ├── envs/                     # Environment wrappers
│   │   ├── atari_wrappers.py     # Preprocessing wrappers
│   │   └── make_env.py           # Environment factory
│   ├── training/                 # DQN trainer
│   │   ├── dqn_trainer.py        # Main training loop
│   │   ├── schedulers.py         # Epsilon/update schedulers
│   │   ├── checkpoint.py         # Save/load system
│   │   ├── resume.py             # Resume validation
│   │   └── loggers.py            # Metrics logging
│   └── utils/                    # Utilities
│       ├── repro.py              # Seeding utilities
│       └── determinism.py        # Deterministic mode
│
├── experiments/dqn_atari/        # DQN experiment configs
│   ├── configs/                  # YAML configurations
│   │   ├── base.yaml             # Default settings
│   │   └── pong.yaml             # Game-specific
│   ├── scripts/                  # Training scripts
│   │   ├── run_dqn.sh            # Main launcher
│   │   ├── setup_roms.sh         # ROM installation
│   │   └── smoke_test.sh         # Fast validation
│   └── runs/                     # Training outputs
│       └── pong_42/              # Run directory
│           ├── logs/             # Step/episode logs
│           ├── checkpoints/      # Model checkpoints
│           ├── eval/             # Evaluation results
│           └── meta.json         # Run metadata
│
├── tests/                        # Unit tests
│   ├── test_dqn_model.py         # Model tests
│   ├── test_replay_buffer.py     # Replay tests
│   ├── test_dqn_trainer.py       # Training tests
│   ├── test_checkpoint.py        # Checkpoint tests
│   └── test_resume.py            # Resume tests
│
└── docs/                         # Documentation
    ├── README.md                 # Documentation hub
    ├── workflows.md              # Task guides
    ├── troubleshooting.md        # Problem solving
    ├── roadmap.md                # Project plan
    └── design/                   # Design specs
        ├── dqn-setup.md
        ├── dqn-model.md
        ├── replay-buffer.md
        ├── dqn-training.md
        └── training-loop-runtime.md
```

---

## Key Algorithms

### Q-Learning Update

```python
# Sample batch from replay buffer
batch = replay_buffer.sample(batch_size=32)
states, actions, rewards, next_states, dones = batch

# Compute TD targets (no gradient)
with torch.no_grad():
    max_next_q = target_net(next_states).max(dim=1)[0]
    td_targets = rewards + gamma * (1 - dones) * max_next_q

# Compute current Q-values
q_values = online_net(states)
q_selected = q_values.gather(1, actions.unsqueeze(1)).squeeze()

# Compute loss and optimize
loss = F.mse_loss(q_selected, td_targets)
loss.backward()
torch.nn.utils.clip_grad_norm_(online_net.parameters(), max_norm=10.0)
optimizer.step()

# Periodic target network sync (every 10K steps)
if step % target_update_interval == 0:
    target_net.load_state_dict(online_net.state_dict())
```

**Docs:** [dqn-training.md](../reference/dqn-training.md#complete-update-pipeline)

---

### Epsilon-Greedy Exploration

```python
# Linear decay: 1.0 → 0.1 over 1M frames
epsilon = epsilon_start + (epsilon_end - epsilon_start) * min(1.0, step / epsilon_frames)

# Action selection
if random.random() < epsilon:
    action = env.action_space.sample()  # Explore
else:
    with torch.no_grad():
        q_values = online_net(state)
        action = q_values.argmax().item()  # Exploit
```

**Docs:** [training-loop-runtime.md](../reference/training-loop-runtime.md#component-orchestration)

---

## Testing Strategy

**Unit Tests (287+ tests):**
- Component isolation
- Fast execution (< 5 minutes)
- Run frequently during development

**Smoke Test:**
- End-to-end validation
- 200K frames (~5-10 minutes)
- Verifies all components integrate

**Determinism Test:**
- Save/resume verification
- Checks bit-for-bit reproducibility
- 5K step comparison

**Run Tests:**
```bash
# All tests
pytest tests/ -v

# Component tests
pytest tests/test_dqn_model.py -v
pytest tests/test_replay_buffer.py -v
pytest tests/test_dqn_trainer.py -v

# Smoke test
./experiments/dqn_atari/scripts/smoke_test.sh

# Determinism test
pytest tests/test_save_resume_determinism.py -v -s
```

**Docs:** [tests/README.md](../tests/README.md)

---

## Execution Flow Summary

```
1. Setup
   └─ Install deps → Download ROMs → Verify

2. Training
   ├─ Initialize: env, model, replay, optimizer
   ├─ Warm-up: fill replay buffer (50K steps)
   └─ Main loop:
      ├─ Select action (ε-greedy)
      ├─ Execute in env (4-frame repeat)
      ├─ Store transition in replay
      ├─ Optimize (every 4 steps):
      │  ├─ Sample batch (32)
      │  ├─ Compute TD targets
      │  ├─ Compute loss
      │  └─ Backprop + gradient clip + step
      ├─ Sync target net (every 10K steps)
      ├─ Log metrics (every 10K frames)
      ├─ Evaluate (every 250K frames)
      └─ Checkpoint (every 1M frames)

3. Evaluation
   ├─ Run 10 episodes with ε=0.05
   ├─ Compute mean/std return
   └─ Save results to CSV

4. Checkpoint/Resume
   ├─ Save: models, optimizer, replay, RNG states
   └─ Resume: restore all state, continue training
```

---

## Next Steps

**New contributors should:**

1. **Read in this order:**
   - This architecture overview (you are here)
   - [Quick Start Guide](quick-start.md) - Get environment running
   - [Workflows](workflows.md) - Learn common tasks
   - Component design docs (as needed)

2. **Run validation:**
   ```bash
   # Setup
   bash envs/setup_env.sh
   source .venv/bin/activate

   # Dry run
   ./experiments/dqn_atari/scripts/run_dqn.sh \
     experiments/dqn_atari/configs/pong.yaml --dry-run

   # Smoke test
   ./experiments/dqn_atari/scripts/smoke_test.sh

   # Unit tests
   pytest tests/ -v
   ```

3. **Explore codebase:**
   - Read component source in `src/`
   - Check test files for usage examples
   - Review config files for parameter tuning

**Docs:** [README.md](../README.md) for full documentation index

---

**Last Updated:** 2025-11-13
