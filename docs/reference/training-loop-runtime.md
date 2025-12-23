# Training Loop Runtime & Orchestration

Comprehensive guide to DQN training loop control flow, component orchestration, logging schema, evaluation cadence, and troubleshooting. This document describes how all training components work together during execution.

---

**Prerequisites:**
- Completed all component docs: [Model](dqn-model.md), [Replay](replay-buffer.md), [Training](dqn-training.md)
- Understand [Episode Handling](episode-handling.md) - Training vs evaluation modes
- Read [Atari Wrappers](atari-env-wrapper.md) - Frame preprocessing pipeline

**Related Docs:**
- [Checkpointing](checkpointing.md) - Checkpoint/resume integration
- [Scripts README](../../experiments/dqn_atari/scripts/README.md) - CLI usage
- [Test README](../../tests/README.md) - Running smoke tests

---

## Table of Contents

1. [Overview](#overview)
2. [Control Flow](#control-flow)
3. [Component Orchestration](#component-orchestration)
4. [Logging Schema](#logging-schema)
5. [Evaluation Cadence](#evaluation-cadence)
6. [Command Reference](#command-reference)
7. [Smoke Test Procedure](#smoke-test-procedure)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Configuration Knobs](#configuration-knobs)

---

## Overview

The DQN training loop orchestrates multiple components to implement the complete reinforcement learning pipeline:

1. **Action Selection**: ε-greedy policy with scheduled exploration decay
2. **Environment Interaction**: Frame-skip execution with reward accumulation
3. **Experience Storage**: Replay buffer management with episode boundaries
4. **Optimization**: Periodic TD-loss minimization with gradient clipping
5. **Target Synchronization**: Scheduled hard-updates of target network
6. **Logging**: Structured metrics for steps, episodes, and Q-values
7. **Evaluation**: Periodic performance assessment with low-ε policy
8. **Checkpointing**: Periodic and best-model saves with metadata

### Key Design Principles

- **Scheduled Operations**: Most components trigger on fixed intervals (train every 4 steps, eval every 250K frames)
- **Decoupled Counters**: Separate tracking for steps (decisions), frames (environment steps with skip), and episodes
- **Structured Logging**: CSV-based metrics for easy plotting and analysis
- **Reproducibility**: Metadata capture (git hash, config, seed) with every run
- **Fail-Fast Validation**: Smoke tests verify end-to-end stability before long runs

---

## Control Flow

### High-Level Training Loop

```
Initialize:
  - Create environment with preprocessing wrappers
  - Build online and target Q-networks
  - Configure optimizer and replay buffer
  - Initialize schedulers (epsilon, target sync, training, eval)
  - Create loggers (step, episode, eval, reference Q)
  - Write metadata (config, seed, git hash)

Warm-up Phase (until replay buffer has min_size transitions):
  - Select random actions
  - Step environment with frame-skip
  - Append transitions to replay buffer
  - No training yet

Main Training Loop (until total_frames reached):
  For each step:
    1. Action Selection
       - Get current epsilon from scheduler
       - Sample action via ε-greedy from online Q-network

    2. Environment Interaction
       - Execute action for k frames (default k=4)
       - Accumulate clipped rewards from wrapper
       - Observe next state and termination flags

    3. Experience Storage
       - Append (s, a, r, s', done) to replay buffer
       - Track episode boundaries for safe sampling

    4. Training (if step % train_every == 0 and buffer ready):
       - Sample batch from replay buffer
       - Compute TD targets: y = r + γ(1-done)×max Q_target(s',a')
       - Compute loss (MSE or Huber)
       - Backpropagate and clip gradients
       - Optimizer step
       - Log metrics (loss, TD-error, grad norm)

    5. Target Network Sync (if step % target_update == 0):
       - Hard-copy online weights to target network
       - Log sync event

    6. Step Logging (if step % log_interval == 0):
       - Write metrics to training_steps.csv
       - Compute loss moving average

    7. Episode Handling (if terminated or truncated):
       - Log episode return and length to episodes.csv
       - Compute rolling statistics (mean/std over last N episodes)
       - Reset environment

    8. Evaluation (if step % eval_interval == 0):
       - Run K episodes with eval_epsilon (default 0.05)
       - Log mean/median/std returns
       - Save best model if new best return
       - Write detailed results to eval/evaluations.csv

    9. Reference Q Logging (if step % q_log_interval == 0):
       - Compute Q-values on fixed reference batch
       - Track avg_max_q, max_q, min_q over time
       - Write to logs/reference_q_values.csv

    10. Checkpointing (if step % checkpoint_interval == 0):
        - Save model, optimizer, metadata
        - Clean up old checkpoints (keep last N)

Finalization:
  - Save final checkpoint
  - Close loggers
  - Write summary statistics
```

### Detailed Step Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     training_step()                          │
│                                                              │
│  Input: current state, all components                       │
│  Output: next state, metrics, flags                         │
└─────────────────────────────────────────────────────────────┘
                              ║
                              ▼
         ┌────────────────────────────────────┐
         │ 1. Epsilon Scheduler                │
         │    - Get current epsilon            │
         │    - Decay based on frame count     │
         └────────────────────────────────────┘
                              ║
                              ▼
         ┌────────────────────────────────────┐
         │ 2. Action Selection                 │
         │    - ε-greedy: random with prob ε   │
         │    - Greedy: argmax Q(s,a)          │
         └────────────────────────────────────┘
                              ║
                              ▼
         ┌────────────────────────────────────┐
         │ 3. Environment Step                 │
         │    - Execute action k frames        │
         │    - Accumulate rewards             │
         │    - Observe s', terminated, etc.   │
         └────────────────────────────────────┘
                              ║
                              ▼
         ┌────────────────────────────────────┐
         │ 4. Replay Buffer Append             │
         │    - Store (s, a, r, s', done)      │
         │    - Track episode boundaries       │
         └────────────────────────────────────┘
                              ║
                              ▼
         ┌────────────────────────────────────┐
         │ 5. Training Scheduler Check         │
         │    - Step % train_every == 0?       │
         │    - Buffer has min_size?           │
         └────────────────────────────────────┘
                              ║
                    ┌─────────┴─────────┐
                    │                   │
                 Yes│                   │No (skip)
                    ▼                   │
         ┌────────────────────┐         │
         │ 6. Sample Batch    │         │
         │    - Uniform sample│         │
         └────────────────────┘         │
                    ║                   │
                    ▼                   │
         ┌────────────────────┐         │
         │ 7. Compute Loss    │         │
         │    - TD targets    │         │
         │    - MSE/Huber     │         │
         └────────────────────┘         │
                    ║                   │
                    ▼                   │
         ┌────────────────────┐         │
         │ 8. Optimize        │         │
         │    - Backward      │         │
         │    - Clip grads    │         │
         │    - Optimizer step│         │
         └────────────────────┘         │
                    ║                   │
                    └───────────────────┘
                              ║
                              ▼
         ┌────────────────────────────────────┐
         │ 9. Target Network Updater Check     │
         │    - Step % target_update == 0?     │
         │    - If yes: hard-copy weights      │
         └────────────────────────────────────┘
                              ║
                              ▼
         ┌────────────────────────────────────┐
         │ 10. Frame Counter Update            │
         │     - Increment step count          │
         │     - Increment frame count (×k)    │
         │     - Update FPS                    │
         └────────────────────────────────────┘
                              ║
                              ▼
         ┌────────────────────────────────────┐
         │ 11. Return Result Dictionary        │
         │     - next_state, reward            │
         │     - terminated, truncated         │
         │     - epsilon, metrics              │
         └────────────────────────────────────┘
```

---

## Component Orchestration

### 1. Epsilon Scheduler

**Purpose**: Decay exploration rate from 1.0 → 0.1 over first 1M frames

**Implementation**: `src/training/dqn_trainer.py:EpsilonScheduler`

```python
epsilon_scheduler = EpsilonScheduler(
    epsilon_start=1.0,
    epsilon_end=0.1,
    decay_frames=1_000_000
)

# Each step
epsilon = epsilon_scheduler.get_epsilon(frame_counter.frames)
```

**Key Methods**:
- `get_epsilon(frames)`: Returns current epsilon based on frame count
- Linear decay: `ε = max(ε_end, ε_start - (frames / decay_frames) × (ε_start - ε_end))`

**Config Keys**:
- `exploration.epsilon_start`: Initial epsilon (default: 1.0)
- `exploration.epsilon_end`: Final epsilon (default: 0.1)
- `exploration.decay_frames`: Decay period (default: 1M)

---

### 2. Frame Counter

**Purpose**: Track steps (decisions), frames (env steps with skip), and FPS

**Implementation**: `src/training/dqn_trainer.py:FrameCounter`

```python
frame_counter = FrameCounter(frameskip=4)

# Each step
frame_counter.step()  # Increments steps by 1, frames by frameskip
fps = frame_counter.fps(elapsed_time)
```

**Key Attributes**:
- `steps`: Number of decisions made (actions selected)
- `frames`: Number of environment frames (steps × frameskip)
- `frameskip`: Frames per action (default: 4)

**Why Both Counters?**:
- Training triggers based on `steps` (every 4 decisions)
- Exploration decay based on `frames` (1M environment frames)
- Target sync based on `steps` (every 10K decisions)
- Evaluation based on `frames` (every 250K environment frames)

---

### 3. Training Scheduler

**Purpose**: Trigger optimization every k steps after warm-up

**Implementation**: `src/training/dqn_trainer.py:TrainingScheduler`

```python
training_scheduler = TrainingScheduler(train_every=4)

# Each step
if training_scheduler.should_train(step, replay_buffer):
    # Perform optimization
    ...
```

**Key Methods**:
- `should_train(step, buffer)`: Returns True if should optimize this step
- Checks: `buffer.can_sample()` and `step % train_every == 0`

**Config Keys**:
- `training.train_every`: Training frequency in steps (default: 4)

---

### 4. Target Network Updater

**Purpose**: Synchronize target network every N steps

**Implementation**: `src/training/dqn_trainer.py:TargetNetworkUpdater`

```python
target_updater = TargetNetworkUpdater(update_interval=10_000)

# Each step
if target_updater.should_update(step):
    target_updater.update(online_net, target_net)
```

**Key Methods**:
- `should_update(step)`: Returns True if should sync this step
- `update(online, target)`: Hard-copies weights from online to target

**Config Keys**:
- `agent.target_update_interval`: Sync interval in steps (default: 10K)
- Set to 0 to disable target network (2013 DQN variant)

---

### 5. Step Logger

**Purpose**: Log per-step metrics (loss, epsilon, replay size, FPS)

**Implementation**: `src/training/dqn_trainer.py:StepLogger`

```python
step_logger = StepLogger(
    log_dir='runs/pong_0/logs',
    log_interval=1000,
    moving_avg_window=100
)

# Each step
step_logger.log_step(
    step=frame_counter.steps,
    epsilon=epsilon,
    metrics=update_metrics,  # Contains loss, td_error, grad_norm
    replay_size=len(replay_buffer),
    fps=frame_counter.fps(elapsed_time)
)
```

**Output**: `csv/training_steps.csv`

**CSV Schema**:
```csv
step,epsilon,loss,loss_ma,td_error_mean,grad_norm,lr,replay_size,fps
1000,0.95,0.125,0.130,0.089,2.45,0.00025,10000,235.7
2000,0.90,0.118,0.122,0.084,2.31,0.00025,14000,241.3
...
```

**Fields**:
- `step`: Decision count (not frames)
- `epsilon`: Current exploration rate
- `loss`: Current batch loss
- `loss_ma`: Moving average of loss over last N steps
- `td_error_mean`: Mean absolute TD error in batch
- `grad_norm`: Global gradient norm after clipping
- `lr`: Current learning rate
- `replay_size`: Number of transitions in buffer
- `fps`: Frames per second

---

### 6. Episode Logger

**Purpose**: Log per-episode metrics (return, length, rolling stats)

**Implementation**: `src/training/dqn_trainer.py:EpisodeLogger`

```python
episode_logger = EpisodeLogger(
    log_dir='runs/pong_0/logs',
    rolling_window=100
)

# Each episode end
episode_logger.log_episode(
    step=frame_counter.steps,
    episode_return=total_return,
    episode_length=total_length,
    fps=fps,
    epsilon=epsilon
)
```

**Output**: `logs/episodes.csv`

**CSV Schema**:
```csv
step,episode_return,episode_length,fps,epsilon,rolling_mean,rolling_std
982,12.0,892,235.7,0.951,8.3,5.2
2145,-3.0,1156,241.3,0.902,9.1,4.8
...
```

**Fields**:
- `step`: Step when episode ended
- `episode_return`: Sum of rewards (clipped to {-1, 0, +1})
- `episode_length`: Number of steps in episode
- `fps`: Current FPS
- `epsilon`: Epsilon when episode ended
- `rolling_mean`: Mean return over last N episodes
- `rolling_std`: Std of return over last N episodes

---

### 7. Evaluation Scheduler & Logger

**Purpose**: Periodic performance assessment with low-ε policy

**Implementation**: `src/training/dqn_trainer.py:EvaluationScheduler`, `EvaluationLogger`

```python
eval_scheduler = EvaluationScheduler(
    eval_interval=250_000,
    num_episodes=10,
    eval_epsilon=0.05
)

eval_logger = EvaluationLogger(log_dir='runs/pong_0/eval')

# Each step
if eval_scheduler.should_evaluate(frame_counter.frames):
    results = evaluate(
        env=env,
        model=online_net,
        num_episodes=eval_scheduler.num_episodes,
        eval_epsilon=eval_scheduler.eval_epsilon,
        device=device
    )
    eval_scheduler.record_evaluation(frame_counter.frames, results)
    eval_logger.log_evaluation(
        step=frame_counter.frames,
        results=results,
        epsilon=epsilon
    )
```

**Output**: `eval/evaluations.csv` + `eval/eval_{step}.json`

**CSV Schema**:
```csv
step,mean_return,median_return,std_return,min_return,max_return,mean_length,num_episodes,eval_epsilon,train_epsilon
250000,15.2,16.0,3.1,9.0,19.0,1203.4,10,0.05,0.75
500000,18.7,19.0,2.8,13.0,21.0,1156.2,10,0.05,0.50
...
```

**JSON Format** (detailed per-episode results):
```json
{
  "step": 250000,
  "num_episodes": 10,
  "eval_epsilon": 0.05,
  "train_epsilon": 0.75,
  "summary": {
    "mean_return": 15.2,
    "median_return": 16.0,
    "std_return": 3.1,
    "min_return": 9.0,
    "max_return": 19.0,
    "mean_length": 1203.4
  },
  "episodes": [
    {"return": 16.0, "length": 1245},
    {"return": 14.0, "length": 1189},
    ...
  ]
}
```

**Config Keys**:
- `eval.interval`: Frames between evaluations (default: 250K)
- `eval.num_episodes`: Episodes per evaluation (default: 10)
- `eval.epsilon`: Exploration during eval (default: 0.05)

---

### 8. Reference Q Tracker & Logger

**Purpose**: Monitor learning progress via Q-values on fixed states

**Implementation**: `src/training/dqn_trainer.py:ReferenceStateQTracker`, `ReferenceQLogger`

```python
q_tracker = ReferenceStateQTracker(
    log_interval=10_000,
    device=device
)

q_logger = ReferenceQLogger(log_dir='runs/pong_0/logs')

# After buffer warm-up (once)
if q_tracker.reference_states is None and len(replay_buffer) >= 100:
    ref_batch = replay_buffer.sample(100)
    q_tracker.set_reference_states(ref_batch['states'])

# Each step
if q_tracker.should_log(frame_counter.steps):
    q_tracker.log_q_values(step=frame_counter.steps, model=online_net)
    q_stats = {
        'avg_max_q': q_tracker.avg_max_q[-1],
        'max_q': q_tracker.max_q[-1],
        'min_q': q_tracker.min_q[-1]
    }
    q_logger.log(step=frame_counter.steps, q_stats=q_stats)
```

**Output**: `logs/reference_q_values.csv`

**CSV Schema**:
```csv
step,avg_max_q,max_q,min_q
10000,0.12,0.45,-0.08
20000,0.18,0.62,-0.03
...
```

**Fields**:
- `step`: Decision count when logged
- `avg_max_q`: Average of max Q-values across reference states
- `max_q`: Maximum Q-value across all (state, action) pairs
- `min_q`: Minimum Q-value across all (state, action) pairs

**Why Track Reference Q?**:
- Reward signal is noisy and sparse in Atari
- Q-values provide smooth learning signal
- Should increase monotonically if learning progresses
- Helps diagnose value overestimation/underestimation

---

### 9. Checkpoint Manager

**Purpose**: Save model/optimizer state periodically and on best eval

**Implementation**: `src/training/dqn_trainer.py:CheckpointManager`

```python
checkpoint_manager = CheckpointManager(
    checkpoint_dir='runs/pong_0/checkpoints',
    save_interval=1_000_000,
    keep_last_n=3,
    save_best=True
)

# Periodic saves
if checkpoint_manager.should_save(frame_counter.steps):
    checkpoint_manager.save_checkpoint(
        step=frame_counter.steps,
        model=online_net,
        optimizer=optimizer,
        metadata={'epsilon': epsilon}
    )

# Best model saves
if eval_results:
    checkpoint_manager.save_best(
        step=frame_counter.steps,
        eval_return=eval_results['mean_return'],
        model=online_net,
        optimizer=optimizer,
        metadata={'epsilon': epsilon}
    )
```

**Output**: `checkpoints/step_{step}.pt`, `checkpoints/best_model.pt`

**Checkpoint Format**:
```python
{
    'step': 1000000,
    'model_state_dict': {...},
    'optimizer_state_dict': {...},
    'metadata': {'epsilon': 0.5, 'eval_return': 18.2}
}
```

**Config Keys**:
- `training.checkpoint_interval`: Steps between saves (default: 1M)
- `training.keep_last_n`: Number of periodic checkpoints to keep (default: 3)

---

### 10. Metadata Writer

**Purpose**: Capture reproducibility information (config, seed, git state)

**Implementation**: `src/training/dqn_trainer.py:MetadataWriter`

```python
metadata_writer = MetadataWriter(run_dir='runs/pong_0')

metadata_writer.write_metadata(
    config={'total_frames': 10_000_000, ...},
    seed=42,
    extra={'device': 'cuda:0', 'gpu_count': 1}
)
```

**Output**: `metadata.json`, `git_info.txt`

**metadata.json Format**:
```json
{
  "timestamp": "2025-11-13T21:00:00",
  "seed": 42,
  "python_version": "3.10.13",
  "pytorch_version": "2.4.1",
  "device": "cuda:0",
  "git": {
    "commit_hash": "14153d1",
    "commit_hash_full": "14153d1b082ec917dcc04f0020305b2095d10e9e",
    "branch": "main",
    "dirty": false
  },
  "config": {...}
}
```

**git_info.txt Format**:
```
Git Information
===============
Commit: 14153d1b082ec917dcc04f0020305b2095d10e9e
Branch: main
Status: clean
```

---

## Logging Schema

### Directory Structure

```
experiments/dqn_atari/runs/{game}_{seed}/
├── metadata.json                    # Run metadata (git, config, seed)
├── git_info.txt                     # Git state snapshot
├── config.yaml                      # Merged config snapshot
├── logs/
│   ├── training_steps.csv          # Per-step metrics
│   ├── episodes.csv                # Per-episode metrics
│   └── reference_q_values.csv      # Q-value tracking
├── eval/
│   ├── evaluations.csv             # Eval summary stats
│   └── eval_{step}.json            # Detailed eval results
└── checkpoints/
    ├── step_1000000.pt             # Periodic checkpoints
    ├── step_2000000.pt
    └── best_model.pt               # Best eval model
```

### CSV Format Guidelines

All CSV files follow these conventions:

1. **Header Row**: Column names in snake_case
2. **Numeric Precision**: Floats to 3-6 decimal places
3. **Step Column**: Always first column (for easy plotting)
4. **Append-Only**: New rows appended during training
5. **No Gaps**: Write every interval (no sparse rows)

### Reading Logs for Plotting

```python
import pandas as pd

# Training curves
steps = pd.read_csv('csv/training_steps.csv')
steps.plot(x='step', y=['loss_ma', 'td_error_mean'])

# Episode returns
episodes = pd.read_csv('logs/episodes.csv')
episodes.plot(x='step', y=['rolling_mean', 'rolling_std'])

# Evaluation results
evals = pd.read_csv('eval/evaluations.csv')
evals.plot(x='step', y=['mean_return', 'std_return'])

# Reference Q values
q_vals = pd.read_csv('logs/reference_q_values.csv')
q_vals.plot(x='step', y=['avg_max_q', 'max_q', 'min_q'])
```

---

## Evaluation Cadence

### When to Evaluate

**Default Schedule**: Every 250,000 frames

**Rationale**:
- Frequent enough to track learning progress (40 evals in 10M frames)
- Infrequent enough to not slow training significantly
- Matches Nature DQN evaluation protocol

**Alternative Schedules**:
- **Quick debugging**: Every 50K frames (200 evals in 10M)
- **Long runs**: Every 500K frames (20 evals in 10M)
- **Final evaluation**: Every 1M frames (10 evals in 10M)

### Evaluation vs. Training Differences

| Aspect | Training | Evaluation |
|--------|----------|------------|
| **Epsilon** | 1.0 → 0.1 (decaying) | 0.05 (fixed) or greedy |
| **Episodes** | Life-loss as terminal | Full episodes (all lives) |
| **Learning** | Enabled (gradients) | Disabled (`model.eval()`) |
| **No-op starts** | Yes (0-30 no-ops) | Yes (0-30 no-ops) |
| **Reward clipping** | Yes ({-1, 0, +1}) | Yes (for consistency) |
| **Logging** | Step-level metrics | Episode-level summaries |

**Why Different Epsilon?**:
- Training: Need exploration to discover new strategies
- Evaluation: Want to measure current policy performance (mostly greedy)

**Why Full Episodes?**:
- Training: Life-loss terminal speeds up learning
- Evaluation: Want true game score (all lives matter)

### Interpreting Evaluation Results

**Good Learning**:
- Mean return increases over time
- Std return decreases (more consistent)
- Matches or exceeds paper benchmarks

**Poor Learning**:
- Mean return flat or decreasing
- High variance in returns
- Far below paper benchmarks

**Possible Issues**:
- **No improvement**: Check if training is happening (loss decreasing?)
- **High variance**: Increase num_episodes (10 → 30)
- **Sudden drops**: Check for target sync issues or learning rate problems

---

## Command Reference

### Running Training

```bash
# Full training with Pong (10M frames)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --seed 42

# Breakout with custom settings
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml \
  --seed 123 \
  --device cuda \
  --training.total_frames 20000000
```

### Dry Run Validation

```bash
# Validate preprocessing and wrappers (3 episodes, random policy)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --dry-run --seed 0

# Custom dry run (5 episodes)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml \
  --dry-run --dry-run-episodes 5 --seed 42
```

### Smoke Test

```bash
# Quick end-to-end validation (~200K frames, ~5-10 min)
./experiments/dqn_atari/scripts/smoke_test.sh

# Custom smoke test
./experiments/dqn_atari/scripts/smoke_test.sh \
  experiments/dqn_atari/configs/breakout.yaml 42
```

### Monitoring Training

```bash
# Watch training progress
tail -f experiments/dqn_atari/runs/pong_42/csv/training_steps.csv

# Check episode returns
tail -f experiments/dqn_atari/runs/pong_42/logs/episodes.csv

# View evaluation results
cat experiments/dqn_atari/runs/pong_42/eval/evaluations.csv

# Check for checkpoints
ls -lh experiments/dqn_atari/runs/pong_42/checkpoints/
```

### Resuming Training

```bash
# Resume from checkpoint (future feature - Subtask 7)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --resume runs/pong_42/checkpoints/step_5000000.pt
```

---

## Smoke Test Procedure

### Purpose

Validate end-to-end training stability **before** launching long experiments.

### What It Checks

- [pass] Training loop executes without crashes
- [pass] Logs are created and grow over time
- [pass] Checkpoints appear (if interval reached)
- [pass] Evaluation runs trigger and complete
- [pass] Reference Q logging works
- [pass] Metrics are recorded correctly

### Running the Smoke Test

```bash
# Default: Pong, seed 0, 200K frames
./experiments/dqn_atari/scripts/smoke_test.sh

# Output: experiments/dqn_atari/runs/smoke_test_0/
```

### Expected Output

```
========================================
DQN Training Smoke Test
========================================

Config: experiments/dqn_atari/configs/pong.yaml
Seed: 0
Total frames: 200000
Run directory: experiments/dqn_atari/runs/smoke_test_0

Running Python smoke test...

Starting smoke test...
  Total frames: 200,000
  Seed: 0
  Run dir: experiments/dqn_atari/runs/smoke_test_0

Creating environment...
Creating model...
Creating optimizer...
Creating replay buffer...
Creating schedulers...
Creating loggers...
Creating checkpoint manager...
Creating evaluation scheduler...
Creating reference Q tracker...
Writing metadata...

Starting training loop...
============================================================
  Step 10,000 | Frames 40,000/200,000 (20.0%) | FPS: 235.7 | Epsilon: 0.960 | Replay: 10000
  Saving checkpoint at step 12,500...
  Step 20,000 | Frames 80,000/200,000 (40.0%) | FPS: 241.3 | Epsilon: 0.920 | Replay: 20000
  Step 30,000 | Frames 120,000/200,000 (60.0%) | FPS: 238.9 | Epsilon: 0.880 | Replay: 30000
  Step 40,000 | Frames 160,000/200,000 (80.0%) | FPS: 240.1 | Epsilon: 0.840 | Replay: 40000
  Running evaluation at step 50,000...
    Mean return: 8.3
  Step 50,000 | Frames 200,000/200,000 (100.0%) | FPS: 239.5 | Epsilon: 0.800 | Replay: 50000
============================================================
Training complete!
Total time: 215.3s

Validating outputs...

  [x] metadata.json
  [x] git_info.txt
  [x] training_steps.csv
  [x] episodes.csv
  [x] reference_q_values.csv
  [x] evaluations.csv
  [x] checkpoints

[x] Smoke test passed!

========================================
Smoke test PASSED [x]
========================================
```

### Validation Criteria

The smoke test checks for these files:

```bash
runs/smoke_test_0/
├── metadata.json          # [x] Required
├── git_info.txt           # [x] Required
├── logs/
│   ├── training_steps.csv # [x] Required
│   ├── episodes.csv       # [x] Required
│   └── reference_q_values.csv # [x] Required
├── eval/
│   └── evaluations.csv    # [x] Required
└── checkpoints/           # [x] Required (non-empty)
    └── step_50000.pt
```

### When to Run Smoke Tests

- [pass] After implementing new training components
- [pass] Before starting long training runs (10M+ frames)
- [pass] After changing environment/dependency versions
- [pass] After modifying training loop logic
- [pass] As part of CI/CD pipeline

### Smoke Test vs. Full Training

| Aspect | Smoke Test | Full Training |
|--------|------------|---------------|
| **Frames** | 200K (~5-10 min) | 10M+ (~2-10 hours) |
| **Environment** | Mock (no ROMs) | Real Atari |
| **Purpose** | Validate pipeline | Train agent |
| **Checkpoints** | 1-2 checkpoints | 10-20 checkpoints |
| **Evaluations** | 1-4 evals | 40+ evals |

---

## Testing

### Unit Test Suites

The training loop components are covered by comprehensive unit tests in `tests/test_dqn_trainer.py`.

**Running all training tests:**
```bash
# Full training module test suite (163+ tests)
pytest tests/test_dqn_trainer.py -v

# With coverage report
pytest tests/test_dqn_trainer.py --cov=src.training --cov-report=html
```

**Targeted test suites:**

```bash
# Scheduler tests (epsilon, training, target sync)
pytest tests/test_dqn_trainer.py -k "scheduler" -v

# Training step orchestration
pytest tests/test_dqn_trainer.py -k "training_step" -v

# Logging components (step, episode, checkpoint)
pytest tests/test_dqn_trainer.py -k "logger" -v

# Evaluation system
pytest tests/test_dqn_trainer.py -k "evaluation" -v

# Reference Q tracking
pytest tests/test_dqn_trainer.py -k "reference" -v

# Metadata and git utilities
pytest tests/test_dqn_trainer.py -k "metadata or git" -v

# Stability checks (NaN/Inf detection)
pytest tests/test_dqn_trainer.py -k "stability or nan or inf" -v
```

**Test coverage by component:**

| Component | Tests | What's Covered |
|-----------|-------|----------------|
| **Target Network** | 6 | hard_update, init, scheduler |
| **Loss Functions** | 12 | TD targets, Q-selection, MSE/Huber loss |
| **Optimization** | 8 | Optimizer config, gradient clipping |
| **Schedulers** | 18 | Epsilon decay, training frequency, target sync |
| **Stability Checks** | 21 | NaN/Inf detection, loss validation, target sync verification |
| **Metrics** | 15 | UpdateMetrics, perform_update_step |
| **Training Loop** | 22 | training_step, action selection, frame counter |
| **Logging** | 24 | StepLogger, EpisodeLogger, CheckpointManager |
| **Evaluation** | 18 | evaluate(), EvaluationScheduler, EvaluationLogger |
| **Q Tracking** | 9 | ReferenceStateQTracker, ReferenceQLogger |
| **Metadata** | 8 | Git utilities, MetadataWriter |
| **Integration** | 2 | End-to-end training step with all components |

**Fast subset for CI:**
```bash
# Quick smoke test of core functionality (~30s)
pytest tests/test_dqn_trainer.py -k "not slow" -v

# Just the integration tests
pytest tests/test_dqn_trainer.py::test_training_step_integration -v
```

### Relationship to Smoke Test

**Unit tests** verify individual components in isolation:
- Fast (seconds to minutes)
- No environment interaction
- Mocked dependencies
- Test edge cases and error handling

**Smoke test** validates end-to-end integration:
- Slower (~5-10 minutes)
- Uses mock environment
- All components working together
- Validates file outputs and logging

**Recommended testing workflow:**
1. Run unit tests during development: `pytest tests/test_dqn_trainer.py -k component`
2. Run smoke test before commits: `./experiments/dqn_atari/scripts/smoke_test.sh`
3. Run full test suite before PRs: `pytest tests/ -v`

---

## Troubleshooting Guide

### Issue: Training Not Starting

**Symptoms**:
- Script hangs after "Starting training loop..."
- No log files created
- No progress updates

**Diagnosis**:
```bash
# Check if replay buffer warm-up is stuck
tail -f experiments/dqn_atari/runs/{game}_{seed}/csv/training_steps.csv

# Should see initial random exploration
# If nothing appears, check environment creation
```

**Common Causes**:
1. **Environment not responding**: Check ROM installation
2. **Infinite warm-up**: Check `min_size` vs `capacity` in config
3. **Device issues**: Check CUDA availability if using GPU

**Fixes**:
```bash
# Verify ROMs installed
python -c "import ale_py; print(ale_py.roms.list())"

# Check replay buffer config
grep -A5 "replay" experiments/dqn_atari/configs/base.yaml

# Force CPU mode
./experiments/dqn_atari/scripts/run_dqn.sh ... --device cpu
```

---

### Issue: Loss Not Decreasing

**Symptoms**:
- `loss` and `loss_ma` stay constant or increase
- `td_error_mean` stays high
- No improvement in episode returns

**Diagnosis**:
```python
import pandas as pd

steps = pd.read_csv('csv/training_steps.csv')
print(steps[['step', 'loss', 'loss_ma', 'td_error_mean']].tail(20))

# Check if loss is stuck at high value
# Check if loss_ma is actually changing
```

**Common Causes**:
1. **Learning rate too low**: Optimizer not making progress
2. **Gradient clipping too aggressive**: Clipping at 0.1 or lower
3. **Target network not syncing**: Check sync schedule
4. **Replay buffer too small**: Not enough diversity in samples

**Fixes**:
```yaml
# Increase learning rate (carefully!)
optimizer:
  learning_rate: 0.00025  # Try 0.0001 or 0.0005

# Relax gradient clipping
optimizer:
  max_grad_norm: 10.0  # From default 10.0 to 20.0

# Check target sync
agent:
  target_update_interval: 10000  # Should not be 0

# Increase replay capacity
agent:
  replay_capacity: 1000000  # From 100K to 1M
```

---

### Issue: NaN/Inf in Losses

**Symptoms**:
- `loss` becomes NaN or Inf
- Training crashes with "RuntimeError: Function AddBackward0 returned nan values"
- Gradients explode

**Diagnosis**:
```python
# Check for NaN/Inf in logs
steps = pd.read_csv('csv/training_steps.csv')
print(steps[steps['loss'].isna() | (steps['loss'] == float('inf'))])

# Check gradient norms
print(steps[['step', 'grad_norm']].tail(20))
# If grad_norm > 100, gradients are exploding
```

**Common Causes**:
1. **Learning rate too high**: Optimizer takes too large steps
2. **No gradient clipping**: Gradients not bounded
3. **Target network issue**: Stale or corrupted targets
4. **Reward scale too large**: Rewards not clipped properly

**Fixes**:
```yaml
# Lower learning rate
optimizer:
  learning_rate: 0.0001  # From 0.00025

# Enable gradient clipping (should already be on)
optimizer:
  max_grad_norm: 10.0

# Verify reward clipping
training:
  reward_clip: true  # Should be true

# Reset target network more frequently
agent:
  target_update_interval: 5000  # From 10000
```

---

### Issue: Target Network Not Syncing

**Symptoms**:
- No "Target network synced" messages in logs
- TD error stays very high
- Learning unstable

**Diagnosis**:
```bash
# Check target update schedule
grep "target_update" experiments/dqn_atari/configs/base.yaml

# Should be 10000, not 0
# If 0, target network disabled (2013 DQN variant)
```

**Common Causes**:
1. **Config set to 0**: Intentionally disabled
2. **Never reaching interval**: Training ends before first sync
3. **Scheduler bug**: Implementation error

**Fixes**:
```yaml
# Enable target network (2015 DQN)
agent:
  target_update_interval: 10000  # Must be > 0

# Or reduce interval for debugging
agent:
  target_update_interval: 5000
```

---

### Issue: Evaluation Not Triggering

**Symptoms**:
- No files in `eval/` directory
- No evaluation messages during training
- `evaluations.csv` not created

**Diagnosis**:
```bash
# Check eval config
grep -A5 "eval:" experiments/dqn_atari/configs/base.yaml

# Check if training reached eval interval
# Default: 250K frames, so need at least 250K to see first eval
```

**Common Causes**:
1. **Interval too large**: Training ends before first eval
2. **Frame count incorrect**: Checking wrong counter (steps vs frames)
3. **Eval disabled in config**: `interval: 0`

**Fixes**:
```yaml
# Reduce eval interval for debugging
eval:
  interval: 50000  # From 250000 (frames, not steps!)

# Verify eval enabled
eval:
  interval: 250000  # Must be > 0
```

---

### Issue: Checkpoints Not Saving

**Symptoms**:
- Empty `checkpoints/` directory
- No "Saving checkpoint" messages
- Training completes but no model saved

**Diagnosis**:
```bash
# Check checkpoint config
grep -A3 "checkpoint" experiments/dqn_atari/configs/base.yaml

# Check permissions
ls -ld experiments/dqn_atari/runs/{game}_{seed}/checkpoints/
```

**Common Causes**:
1. **Interval too large**: Training ends before first checkpoint
2. **Disk space full**: No room to save
3. **Permission error**: Can't write to directory

**Fixes**:
```yaml
# Reduce checkpoint interval for debugging
training:
  checkpoint_interval: 50000  # From 1000000 (steps, not frames!)

# Check disk space
df -h

# Fix permissions
chmod -R u+w experiments/dqn_atari/runs/
```

---

### Issue: High Memory Usage

**Symptoms**:
- OOM (out of memory) errors
- System slows down during training
- Replay buffer causing issues

**Diagnosis**:
```python
# Check replay buffer size
import sys
import numpy as np

capacity = 1_000_000
obs_shape = (4, 84, 84)
mem_bytes = capacity * np.prod(obs_shape) * 5  # 5 arrays (s, a, r, s', done)
mem_gb = mem_bytes / 1e9

print(f"Replay buffer memory: {mem_gb:.2f} GB")
# With 1M capacity: ~1.4 GB (uint8 storage)
```

**Common Causes**:
1. **Capacity too large**: 1M+ transitions with float32 storage
2. **Batch size too large**: Sampling 256+ per batch
3. **Memory leak**: Not releasing old checkpoints

**Fixes**:
```yaml
# Reduce replay capacity
agent:
  replay_capacity: 500000  # From 1000000

# Reduce batch size
agent:
  batch_size: 32  # Default, don't increase

# Enable checkpoint cleanup
training:
  keep_last_n: 3  # Only keep 3 recent checkpoints
```

---

### Issue: Slow Training

**Symptoms**:
- FPS < 100 on CPU
- FPS < 500 on GPU
- Training takes much longer than expected

**Diagnosis**:
```bash
# Check FPS in logs
tail -n 20 experiments/dqn_atari/runs/{game}_{seed}/csv/training_steps.csv | cut -d',' -f8

# Expected FPS:
# CPU: 200-300 FPS
# GPU: 500-1000+ FPS
```

**Common Causes**:
1. **Device mismatch**: Model on GPU, data on CPU
2. **Frequent syncing**: Too many checkpoints/evals
3. **Disk I/O**: Writing logs too frequently
4. **Inefficient env**: Wrappers not optimized

**Fixes**:
```yaml
# Reduce logging frequency
logging:
  step_interval: 10000  # From 1000

# Reduce checkpoint frequency
training:
  checkpoint_interval: 2000000  # From 1000000

# Reduce eval frequency
eval:
  interval: 500000  # From 250000

# Verify device consistency
device: cuda  # Ensure all components use same device
```

---

### Issue: Episode Returns Not Improving

**Symptoms**:
- `rolling_mean` stays flat or decreases
- Evaluation returns don't increase
- Agent seems stuck at random policy performance

**Diagnosis**:
```python
import pandas as pd

episodes = pd.read_csv('logs/episodes.csv')
print(episodes[['step', 'rolling_mean']].tail(20))

evals = pd.read_csv('eval/evaluations.csv')
print(evals[['step', 'mean_return']])

# Check if returns improving over time
# If flat, learning may not be happening
```

**Common Causes**:
1. **Training not happening**: Check `train_every` schedule
2. **Learning rate too low**: Optimizer not making progress
3. **Epsilon too high**: Too much exploration, not exploiting
4. **Target network stale**: Not syncing properly

**Fixes**:
```yaml
# Verify training frequency
training:
  train_every: 4  # Should train every 4 steps after warm-up

# Check epsilon schedule
exploration:
  decay_frames: 1000000  # Should reach 0.1 after 1M frames

# Verify target sync
agent:
  target_update_interval: 10000  # Should sync every 10K steps

# Check if warm-up completed
agent:
  replay_min_transitions: 50000  # Should be < total frames
```

---

## Configuration Knobs

### Epsilon Schedule

Control exploration-exploitation tradeoff:

```yaml
exploration:
  epsilon_start: 1.0      # Initial exploration (100% random)
  epsilon_end: 0.1        # Final exploration (10% random)
  decay_frames: 1000000   # Decay period (1M frames)
```

**Effect**:
- Higher `epsilon_end` → More exploration → Slower convergence, better final policy
- Lower `epsilon_end` → Less exploration → Faster convergence, risk of suboptimal policy
- Longer `decay_frames` → Gradual transition → More stable learning
- Shorter `decay_frames` → Quick transition → Faster exploitation

---

### Frame Counters

Track training progress:

```yaml
env:
  frameskip: 4  # Frames per action (action repeat)
```

**Key Relationships**:
- `steps = total_decisions`
- `frames = steps × frameskip`
- Most schedules based on `frames` (epsilon, eval)
- Some schedules based on `steps` (training, target sync)

**Why Frameskip?**:
- Reduces computation (4× fewer decisions)
- Increases temporal granularity
- Matches Nature DQN protocol

---

### Training Frequency

Control how often optimization happens:

```yaml
training:
  train_every: 4  # Train every 4 steps (decisions)
```

**Effect**:
- Lower `train_every` → More frequent updates → Faster learning, higher compute
- Higher `train_every` → Less frequent updates → Slower learning, lower compute

**Typical Values**:
- CartPole: `train_every: 1` (every step)
- Atari: `train_every: 4` (Nature DQN default)

---

### Target Network Sync

Control target network update frequency:

```yaml
agent:
  target_update_interval: 10000  # Hard-copy every 10K steps
  # Set to 0 to disable (2013 DQN variant)
```

**Effect**:
- Smaller interval → More frequent sync → Less stable targets, faster adaptation
- Larger interval → Less frequent sync → More stable targets, slower adaptation
- `interval: 0` → No target network → 2013 DQN (less stable)

**Typical Values**:
- 2013 DQN: `target_update_interval: 0` (disabled)
- 2015 Nature DQN: `target_update_interval: 10000` (10K steps)
- Alternative: `target_update_interval: 8000` (more frequent)

---

### Evaluation Interval

Control how often to assess performance:

```yaml
eval:
  interval: 250000      # Evaluate every 250K frames
  num_episodes: 10      # Run 10 episodes per evaluation
  epsilon: 0.05         # Small epsilon during eval (mostly greedy)
```

**Effect**:
- Smaller `interval` → More frequent eval → Better learning curve, slower training
- Larger `interval` → Less frequent eval → Faster training, coarser curve
- More `num_episodes` → More accurate estimates → Longer eval time

**Typical Values**:
- Quick debugging: `interval: 50000`, `num_episodes: 3`
- Standard: `interval: 250000`, `num_episodes: 10`
- Final runs: `interval: 500000`, `num_episodes: 30`

---

### Logging Frequency

Control how often to write metrics:

```yaml
logging:
  step_interval: 1000   # Log step metrics every 1K steps
  episode_interval: 1   # Log every episode
```

**Effect**:
- Smaller `step_interval` → More log entries → Smoother curves, larger files
- Larger `step_interval` → Fewer log entries → Coarser curves, smaller files

**Typical Values**:
- Debugging: `step_interval: 100`
- Standard: `step_interval: 1000`
- Long runs: `step_interval: 10000`

---

### Checkpoint Frequency

Control how often to save models:

```yaml
training:
  checkpoint_interval: 1000000  # Save every 1M steps
  keep_last_n: 3                # Keep 3 most recent
```

**Effect**:
- Smaller `interval` → More checkpoints → Better resume points, more disk space
- Larger `interval` → Fewer checkpoints → Less disk space, coarser resume points
- Higher `keep_last_n` → More saved models → More disk space

**Typical Values**:
- Short runs (<1M frames): `checkpoint_interval: 100000`, `keep_last_n: 5`
- Standard runs (10M frames): `checkpoint_interval: 1000000`, `keep_last_n: 3`
- Long runs (50M+ frames): `checkpoint_interval: 5000000`, `keep_last_n: 2`

---

## Summary

The DQN training loop orchestrates 10+ components to implement end-to-end reinforcement learning:

1. **Epsilon Scheduler**: Decay exploration from 1.0 → 0.1
2. **Frame Counter**: Track steps, frames, and FPS
3. **Training Scheduler**: Trigger optimization every 4 steps
4. **Target Network Updater**: Sync every 10K steps
5. **Step Logger**: Write per-step metrics to CSV
6. **Episode Logger**: Write per-episode metrics with rolling stats
7. **Evaluation Scheduler & Logger**: Assess performance every 250K frames
8. **Reference Q Tracker & Logger**: Monitor learning via Q-values
9. **Checkpoint Manager**: Save models periodically and on best eval
10. **Metadata Writer**: Capture reproducibility info (git, config, seed)

**Key Principles**:
- Scheduled operations (train every 4, eval every 250K)
- Structured logging (CSV for metrics, JSON for details)
- Reproducibility (metadata capture, deterministic seeding)
- Fail-fast validation (smoke tests before long runs)

**Troubleshooting Workflow**:
1. Run smoke test to validate pipeline
2. Check logs for anomalies (NaN, constant loss, no improvement)
3. Adjust config knobs (epsilon, learning rate, intervals)
4. Monitor FPS and resource usage
5. Compare eval returns to paper benchmarks

**Next Steps**:
- Implement checkpointing and resume (Subtask 7)
- Add config system and CLI (Subtask 8)
- Run full training experiments (Subtasks 9-10)
- Generate plots and reports (Subtasks 11-14)
