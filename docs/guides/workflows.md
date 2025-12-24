# Common Workflows

Task-oriented guides for common development and experimentation workflows. Each workflow provides quick commands with links to detailed documentation.

---

## Table of Contents

### Getting Started
- [First-Time Setup](#first-time-setup)
- [Verify Installation](#verify-installation)
- [Run Dry Run Test](#run-dry-run-test)

### Training
- [Train From Scratch](#train-from-scratch)
- [Resume Training](#resume-training)
- [Monitor Training Progress](#monitor-training-progress)

### Debugging
- [Debug Unstable Training](#debug-unstable-training)
- [Verify Determinism](#verify-determinism)
- [Inspect Checkpoint Contents](#inspect-checkpoint-contents)

### Testing
- [Run Unit Tests](#run-unit-tests)
- [Run Smoke Test](#run-smoke-test)
- [Validate Component](#validate-component)

---

## First-Time Setup

**Goal:** Set up development environment and verify everything works.

```bash
# 1. Create virtual environment and install dependencies
bash setup/setup_env.sh

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Verify core imports
python -c "import torch, gymnasium, ale_py; print('Success!')"

# 4. Install ROMs (accept license)
./experiments/dqn_atari/scripts/setup_roms.sh

# 5. Capture system info
./experiments/dqn_atari/scripts/capture_env.sh
```

**Expected output:**
- `.venv/` directory created
- All packages installed without errors
- ROMs available: `python -c "import ale_py; print(len(ale_py.roms.list()))"`
- `experiments/dqn_atari/system_info.txt` generated

**Docs:** [DQN Setup](../reference/dqn-setup.md), [Quick Start](quick-start.md)

---

## Verify Installation

**Goal:** Confirm environment is working correctly.

```bash
# Activate venv
source .venv/bin/activate

# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check Gymnasium and ALE
python -c "import gymnasium as gym; env = gym.make('ALE/Pong-v5'); print('Environment OK')"

# Run pytest (install if needed: pip install pytest)
pytest --version
```

**Expected:**
- PyTorch 2.4.1
- CUDA: True (if GPU available)
- Environment creates without errors
- pytest 8.x or higher

**Docs:** [DQN Setup](../reference/dqn-setup.md#troubleshooting)

---

## Run Dry Run Test

**Goal:** Validate preprocessing pipeline without training.

```bash
# Basic dry run (Pong, 3 episodes, random policy)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# Custom episodes and seed
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml \
  --dry-run --dry-run-episodes 5 --seed 42
```

**Expected output:**
```
experiments/dqn_atari/runs/pong_<timestamp>/
├── frames/frame_*.npy        # Sample preprocessed frames
├── action_list.json          # Available actions
├── dry_run_report.json       # Episode statistics
└── meta.json                 # Run metadata
```

**Docs:** [Atari Wrappers](../reference/atari-env-wrapper.md), [Scripts README](../experiments/dqn_atari/scripts/README.md)

---

## Train From Scratch

**Goal:** Start a full training run from the beginning.

```bash
# Basic training (Pong, 10M frames)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42

# With deterministic mode enabled
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set experiment.deterministic.enabled=true

# Different game
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml \
  --seed 123
```

**Expected output:**
```
experiments/dqn_atari/runs/pong_42/
├── config.yaml                      # Full resolved configuration
├── meta.json                        # Run metadata
├── csv/
│   ├── training_steps.csv           # Per-step metrics (loss, epsilon, etc.)
│   └── episodes.csv                 # Per-episode returns
├── checkpoints/
│   ├── checkpoint_1000000.pt
│   ├── checkpoint_2000000.pt
│   └── best_model.pt
├── eval/
│   ├── evaluations.csv              # Periodic evaluation summaries
│   └── evaluations.jsonl            # Same data in JSONL format
├── tensorboard/                     # TensorBoard event files
└── videos/                          # Evaluation episode recordings
    └── Pong_step_250000_best_ep3_r21.mp4
```

**Docs:** [Training Loop](../reference/training-loop-runtime.md), [Scripts README](../experiments/dqn_atari/scripts/README.md)

---

## Resume Training

**Goal:** Continue training from a saved checkpoint.

```bash
# Find latest checkpoint
ls experiments/dqn_atari/runs/pong_42/checkpoints/

# Resume from specific checkpoint
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --resume experiments/dqn_atari/runs/pong_42/checkpoints/checkpoint_1000000.pt

# Resume with strict config validation (recommended)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --resume experiments/dqn_atari/runs/pong_42/checkpoints/checkpoint_1000000.pt \
  --strict-resume
```

**What gets restored:**
- Model weights (online and target networks)
- Optimizer state (momentum, learning rate)
- Training counters (step, episode, epsilon)
- Replay buffer state (write index, size)
- RNG states (Python, NumPy, PyTorch, CUDA, env)

**Docs:** [Checkpointing](../reference/checkpointing.md), [Scripts README](../experiments/dqn_atari/scripts/README.md#resume-a-run)

---

## Monitor Training Progress

**Goal:** Track training metrics in real-time.

```bash
# Monitor episode returns (updates every episode)
tail -f experiments/dqn_atari/runs/pong_42/csv/episodes.csv

# Monitor step metrics (loss, epsilon, etc.)
tail -f experiments/dqn_atari/runs/pong_42/csv/training_steps.csv

# Check evaluation results (CSV or JSONL format)
cat experiments/dqn_atari/runs/pong_42/eval/evaluations.csv
cat experiments/dqn_atari/runs/pong_42/eval/evaluations.jsonl

# Quick stats
tail -n 20 experiments/dqn_atari/runs/pong_42/csv/episodes.csv | \
  awk -F',' '{sum+=$2; count++} END {print "Avg return:", sum/count}'

# List video recordings
ls -la experiments/dqn_atari/runs/pong_42/videos/
```

**Key metrics to watch:**
- **Episode return:** Should increase over time
- **Loss:** Should decrease and stabilize
- **Epsilon:** Should decay from 1.0 to 0.1
- **Eval score:** Periodic assessment with low-epsilon policy

**Docs:** [Training Loop](../reference/training-loop-runtime.md#logging-schema)

---

## Debug Unstable Training

**Goal:** Diagnose and fix training instability (NaN loss, divergence).

```bash
# 1. Check for NaN/Inf in logs
grep -i "nan\|inf" experiments/dqn_atari/runs/pong_42/logs/steps.csv

# 2. Run stability checks (unit tests)
pytest tests/test_dqn_trainer.py -k "stability or nan" -v

# 3. Enable deterministic mode for reproducibility
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set experiment.deterministic.enabled=true \
  --set experiment.deterministic.strict=true

# 4. Reduce learning rate if exploding gradients
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set training.optimizer.lr=1e-4
```

**Common causes:**
- Exploding gradients: Check gradient norm in logs
- Stale target network: Verify target sync frequency
- Replay buffer issues: Check episode boundary handling
- Bad initialization: Try different random seed

**Docs:** [DQN Training](../reference/dqn-training.md#debugging-unstable-training), [Training Loop](../reference/training-loop-runtime.md#troubleshooting-guide)

---

## Verify Determinism

**Goal:** Confirm that save/resume produces identical results.

```bash
# Run determinism smoke test (5000 steps)
pytest tests/test_save_resume_determinism.py -v -s

# Expected output:
# DONE PERFECT DETERMINISM - All metrics match exactly
# Epsilon Matches: 100.0%
# Reward Matches: 100.0%
# Action Matches: 100.0%
# Checksum Match: DONE PASS

# Manual verification (two runs with same seed)
# Run 1
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set experiment.deterministic.enabled=true \
  --set training.total_frames=10000

# Run 2 (same seed, should be identical)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set experiment.deterministic.enabled=true \
  --set training.total_frames=10000

# Compare logs
diff experiments/dqn_atari/runs/pong_42_*/logs/episodes.csv
```

**Docs:** [Checkpointing](../reference/checkpointing.md#deterministic-seeding), [DQN Setup](../reference/dqn-setup.md#deterministic-mode-configuration)

---

## Inspect Checkpoint Contents

**Goal:** Examine what's saved in a checkpoint file.

```python
# In Python interpreter
import torch

# Load checkpoint
ckpt = torch.load('experiments/dqn_atari/runs/pong_42/checkpoints/checkpoint_1000000.pt',
                   weights_only=False)

# Inspect metadata
print(f"Schema version: {ckpt['schema_version']}")
print(f"Step: {ckpt['step']}")
print(f"Episode: {ckpt['episode']}")
print(f"Epsilon: {ckpt['epsilon']}")
print(f"Timestamp: {ckpt['timestamp']}")
print(f"Commit: {ckpt['commit_hash']}")

# Check what's saved
print(f"Keys: {list(ckpt.keys())}")

# Check RNG states
print(f"RNG states: {list(ckpt['rng_states'].keys())}")

# Replay buffer state
print(f"Replay size: {ckpt['replay_buffer_state']['size']}")
print(f"Replay index: {ckpt['replay_buffer_state']['index']}")
```

**Docs:** [Checkpointing](../reference/checkpointing.md#checkpoint-structure)

---

## Run Unit Tests

**Goal:** Validate specific components.

```bash
# Activate venv
source .venv/bin/activate

# Install pytest if needed
pip install pytest

# All tests
pytest tests/ -v

# Specific component tests
pytest tests/test_dqn_model.py -v          # Model architecture
pytest tests/test_replay_buffer.py -v      # Replay buffer
pytest tests/test_dqn_trainer.py -v        # Training loop
pytest tests/test_atari_wrappers.py -v     # Preprocessing

# Checkpoint/resume tests
pytest tests/test_checkpoint.py -v         # Save/load
pytest tests/test_resume.py -v             # Resume logic
pytest tests/test_seeding.py -v            # Deterministic seeding
pytest tests/test_determinism.py -v        # Determinism config

# Targeted test
pytest tests/test_dqn_trainer.py::test_epsilon_scheduler -v
```

**Docs:** [Test README](../tests/README.md)

---

## Run Smoke Test

**Goal:** Quick validation of end-to-end training pipeline.

```bash
# Run smoke test script (~5-10 minutes, 200K frames)
./experiments/dqn_atari/scripts/smoke_test.sh

# What it validates:
# - Environment creation and preprocessing
# - Replay buffer filling
# - Model forward/backward passes
# - Optimizer steps
# - Target network updates
# - Logging and checkpointing
# - Evaluation loop
```

**Expected output:**
- No errors or warnings
- Checkpoints created
- Logs populated
- Eval results generated

**Docs:** [Training Loop](../reference/training-loop-runtime.md#smoke-test-procedure), [Scripts README](../experiments/dqn_atari/scripts/README.md#smoke-test)

---

## Validate Component

**Goal:** Test a specific component in isolation.

```bash
# Model
pytest tests/test_dqn_model.py -k "forward" -v

# Replay buffer sampling
pytest tests/test_replay_buffer.py -k "sample" -v

# Loss computation
pytest tests/test_dqn_trainer.py -k "loss" -v

# Target network sync
pytest tests/test_dqn_trainer.py -k "target" -v

# Epsilon scheduler
pytest tests/test_dqn_trainer.py -k "scheduler" -v
```

**Docs:** [Test README](../tests/README.md), individual design docs for each component

---

## Related Documentation

**Setup and Installation:**
- [DQN Setup](../reference/dqn-setup.md) - Environment and dependencies
- [Quick Start](quick-start.md) - Step-by-step walkthrough
- [Scripts README](../experiments/dqn_atari/scripts/README.md) - CLI reference

**Training:**
- [Training Loop](../reference/training-loop-runtime.md) - Orchestration and runtime behavior
- [DQN Training](../reference/dqn-training.md) - Q-learning update details
- [Checkpointing](../reference/checkpointing.md) - Save/resume system

**Debugging:**
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions
- [Test README](../tests/README.md) - Running tests

---

**Last Updated:** 2025-11-13
