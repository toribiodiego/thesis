# Troubleshooting Guide

Quick reference for diagnosing and fixing common issues. Organized by symptom for fast problem resolution.

---

## Table of Contents

### Setup Issues
- [ImportError: No module named 'ale_py'](#importerror-no-module-named-ale_py)
- [ROM not found errors](#rom-not-found-errors)
- [CUDA not available](#cuda-not-available)
- [ModuleNotFoundError](#modulenotfounderror)

### Training Issues
- [NaN loss after training starts](#nan-loss-after-training-starts)
- [Loss not decreasing](#loss-not-decreasing)
- [Training diverges](#training-diverges)
- [Different results with same seed](#different-results-with-same-seed)

### Performance Issues
- [Training too slow](#training-too-slow)
- [Out of memory errors](#out-of-memory-errors)
- [High CPU usage](#high-cpu-usage)

### Checkpoint/Resume Issues
- [Resume fails with config mismatch](#resume-fails-with-config-mismatch)
- [Resume warning about git commit](#resume-warning-about-git-commit)
- [RNG states not restored](#rng-states-not-restored)

### Environment Issues
- [Frame shapes incorrect](#frame-shapes-incorrect)
- [Action space mismatch](#action-space-mismatch)
- [Episode termination issues](#episode-termination-issues)

---

## Setup Issues

### ImportError: No module named 'ale_py'

**Symptom:**
```
ImportError: No module named 'ale_py'
```

**Cause:** ALE-py package not installed.

**Solution:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Install ale-py
pip install ale-py==0.8.1

# Verify installation
python -c "import ale_py; print(ale_py.__version__)"
```

**Docs:** [DQN Setup](../reference/dqn_setup.md#dependencies--environment)

---

### ROM not found errors

**Symptom:**
```
gymnasium.error.NameNotFound: Environment ALE/Pong-v5 doesn't exist
```
or
```
ale_py._ale_py.ALEException: Unable to find ROM file
```

**Cause:** Atari ROMs not installed.

**Solution:**
```bash
# Run ROM setup script
./experiments/dqn_atari/scripts/setup_roms.sh

# Or manually
python -m AutoROM --accept-license

# Verify ROMs installed
python -c "import ale_py; print(len(ale_py.roms.list()))"
# Should print 60+
```

**Docs:** [DQN Setup](../reference/dqn_setup.md#rom-installation)

---

### CUDA not available

**Symptom:**
```python
torch.cuda.is_available()  # Returns False
```

**Causes:**
1. GPU not detected
2. CUDA version mismatch
3. PyTorch CPU-only version installed

**Solution:**

**Check GPU detection:**
```bash
nvidia-smi
```

**Check PyTorch CUDA version:**
```python
import torch
print(torch.__version__)
print(torch.version.cuda)
```

**Reinstall PyTorch with CUDA:**
```bash
pip uninstall torch
pip install torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

**CPU-only fallback:**
Training will still work, just slower. No action needed.

**Docs:** [DQN Setup](../reference/dqn_setup.md#troubleshooting)

---

### ModuleNotFoundError

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

**Cause:** Running from wrong directory or PYTHONPATH not set.

**Solution:**
```bash
# Always run from repository root
cd /path/to/thesis

# Set PYTHONPATH (if needed)
export PYTHONPATH=.

# Verify
python -c "from src.models import DQN; print('OK')"
```

**Docs:** [DQN Setup](../reference/dqn_setup.md#import-errors)

---

## Training Issues

### TypeError: Parameter mismatch errors

**Symptom:**
```
TypeError: __init__() got an unexpected keyword argument 'start_epsilon'
TypeError: __init__() missing 1 required positional argument: 'obs_shape'
```

**Cause:** API mismatch between training script and component implementations.

**Common Issues:**

**1. EpsilonScheduler parameter names:**
```python
# Wrong:
epsilon_scheduler = EpsilonScheduler(
    start_epsilon=1.0,
    end_epsilon=0.1,
    decay_frames=1_000_000
)

# Correct:
epsilon_scheduler = EpsilonScheduler(
    epsilon_start=1.0,
    epsilon_end=0.1,
    decay_frames=1_000_000
)
```

**2. MetricsLogger parameter names:**
```python
# Wrong:
logger = MetricsLogger(
    tensorboard_enabled=True,
    wandb_enabled=False,
    csv_enabled=True
)

# Correct:
logger = MetricsLogger(
    enable_tensorboard=True,
    enable_wandb=False,
    enable_csv=True
)
```

**3. ReplayBuffer initialization:**
```python
# Wrong:
buffer = ReplayBuffer(capacity=1000000, batch_size=32)

# Correct:
buffer = ReplayBuffer(capacity=1000000, obs_shape=(84, 84))
```

**4. EvaluationScheduler parameter names:**
```python
# Wrong:
eval_scheduler = EvaluationScheduler(
    eval_every=250000,
    eval_enabled=True
)

# Correct:
eval_scheduler = EvaluationScheduler(
    eval_interval=250000,
    num_episodes=10,
    eval_epsilon=0.05
)
```

**Solution:**
Check component API documentation and ensure all parameter names match current implementation.

**Docs:** [Logging Pipeline](../reference/logging_pipeline.md), [Training Loop](../reference/training_loop_runtime.md)

---

### ImportError: cannot import name

**Symptom:**
```
ImportError: cannot import name 'hard_update_target' from 'src.training.target_network'
```

**Cause:** Missing import statement in module.

**Solution:**
```python
# In src/training/schedulers.py, add:
from .target_network import hard_update_target
```

**Common missing imports:**
- `hard_update_target` in `schedulers.py`
- `torch` in `metadata.py`

---

### CSV schema errors

**Symptom:**
```
ValueError: dict contains fields not in fieldnames
```

**Cause:** Dynamic CSV schema causes issues when fields change between writes.

**Solution:**
Define schema upfront in CSVBackend:

```python
# In metrics_logger.py CSVBackend:
self._step_fieldnames = [
    'step', 'epsilon', 'replay_size', 'fps',
    'loss', 'td_error', 'grad_norm', 'learning_rate',
    'loss_ma'  # moving average
]

# Filter log entries to only include defined fields:
filtered_entry = {k: v for k, v in log_entry.items()
                  if k in self._step_fieldnames}
writer.writerow(filtered_entry)
```

**Docs:** [Logging Pipeline](../reference/logging_pipeline.md#csv-backend)

---

### Device detection failures

**Symptom:**
Training fails to start or uses wrong device (CPU when GPU available).

**Cause:** Hardcoded device or missing device detection logic.

**Solution:**
Use automatic device detection with fallback:

```python
def setup_device(config):
    """Setup compute device with automatic fallback."""
    requested_device = config.network.device

    # Try CUDA first
    if requested_device in ['auto', 'cuda']:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return device

    # Try MPS (Apple Silicon)
    if requested_device in ['auto', 'mps']:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device (Apple Silicon GPU)")
            return device

    # Fallback to CPU
    device = torch.device('cpu')
    print("Using CPU device")
    return device
```

**Config:**
```yaml
network:
  device: auto  # or 'cuda', 'mps', 'cpu'
```

**Docs:** [DQN Setup](../reference/dqn_setup.md#device-configuration)

---

### NaN loss after training starts

**Symptom:**
Loss becomes NaN after some training steps.

**Causes:**
1. Exploding gradients
2. Learning rate too high
3. Numerical instability in loss computation

**Diagnosis:**
```bash
# Check gradient norms in logs
grep "grad_norm" experiments/dqn_atari/runs/pong_42/logs/steps.csv | tail -n 20

# Check TD errors
grep "td_error" experiments/dqn_atari/runs/pong_42/logs/steps.csv | tail -n 20
```

**Solutions:**

**1. Reduce learning rate:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set training.optimizer.lr=1e-4
```

**2. Increase gradient clipping:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set training.grad_clip_norm=5.0
```

**3. Use Huber loss instead of MSE:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set training.loss_fn=huber
```

**Docs:** [DQN Training](../reference/dqn_training.md#debugging-unstable-training)

---

### Loss not decreasing

**Symptom:**
Loss stays high and flat over many training steps.

**Causes:**
1. Replay buffer not filled (still in warm-up)
2. Target network never updated
3. Learning rate too low
4. Exploration epsilon stuck at 1.0

**Diagnosis:**
```bash
# Check replay buffer size
grep "replay_size" experiments/dqn_atari/runs/pong_42/logs/steps.csv | tail -n 5

# Check epsilon decay
grep "epsilon" experiments/dqn_atari/runs/pong_42/logs/steps.csv | tail -n 10

# Check if training updates happening
grep "loss" experiments/dqn_atari/runs/pong_42/logs/steps.csv | wc -l
```

**Solutions:**

**1. Wait for warm-up:**
Replay buffer needs 50K transitions before training starts. Check `replay_size` in logs.

**2. Verify target network updates:**
```python
# Run test
pytest tests/test_dqn_trainer.py -k "target_update" -v
```

**3. Check epsilon schedule:**
```bash
# Should decay from 1.0 to 0.1
grep "epsilon" experiments/dqn_atari/runs/pong_42/logs/steps.csv | head -n 20
```

**Docs:** [Training Loop](../reference/training_loop_runtime.md#troubleshooting-guide)

---

### MetricsLogger.log_evaluation() missing arguments (FIXED)

**Symptom:**
```
TypeError: MetricsLogger.log_evaluation() missing 3 required positional arguments:
'median_return', 'min_return', and 'max_return'
```

**Cause:** Training loop was not computing or passing all required statistics to the MetricsLogger.

**Fix Applied:** Added missing metric computations to `train_dqn.py`:
```python
mean_return = np.mean(eval_results['episode_returns'])
median_return = np.median(eval_results['episode_returns'])  # Added
std_return = np.std(eval_results['episode_returns'])
min_return = np.min(eval_results['episode_returns'])  # Added
max_return = np.max(eval_results['episode_returns'])  # Added
mean_length = np.mean(eval_results['episode_lengths'])

metrics_logger.log_evaluation(
    step=frame_counter.frames,
    mean_return=mean_return,
    median_return=median_return,  # Added
    std_return=std_return,
    min_return=min_return,  # Added
    max_return=max_return,  # Added
    mean_length=mean_length,
    num_episodes=config.evaluation.num_episodes
)
```

**Status:** Fixed in train_dqn.py

---

### TensorBoard not available

**Symptom:**
```
Warning: torch.utils.tensorboard not available. TensorBoard logging disabled.
```

**Cause:** Missing tensorboard dependency in Python environment.

**Solution:**
```bash
source .venv/bin/activate
pip install tensorboard
```

**Note:** Training continues without TensorBoard, but CSV logging still works.

---

### Training diverges

**Symptom:**
Q-values or loss grow unbounded, or performance suddenly collapses.

**Causes:**
1. Target network updates too frequent
2. Reward clipping disabled
3. Replay buffer sampling bug

**Solutions:**

**1. Verify target update interval:**
```bash
# Should be 10,000 steps by default
grep "target_sync_interval" experiments/dqn_atari/runs/pong_42/meta.json
```

**2. Enable reward clipping:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set env.reward_clip=true
```

**3. Run replay buffer tests:**
```bash
pytest tests/test_replay_buffer.py -v
```

**Docs:** [DQN Training](../reference/dqn_training.md#debugging-unstable-training)

---

### Different results with same seed

**Symptom:**
Two runs with identical seed produce different metrics.

**Cause:** Non-deterministic mode or incomplete RNG seeding.

**Solution:**

**Enable deterministic mode:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set experiment.deterministic.enabled=true
```

**Verify with smoke test:**
```bash
pytest tests/test_save_resume_determinism.py -v -s
```

**Check for CUDA non-determinism:**
```bash
# Use CPU for debugging
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set experiment.deterministic.enabled=true \
  --set training.device=cpu
```

**Docs:** [Checkpointing](../reference/checkpointing.md#deterministic-seeding), [DQN Setup](../reference/dqn_setup.md#deterministic-mode-configuration)

---

## Performance Issues

### Training too slow

**Symptom:**
FPS much lower than expected (< 500 FPS on GPU).

**Causes:**
1. Running on CPU instead of GPU
2. Deterministic mode enabled
3. Inefficient data loading
4. Debug mode enabled

**Diagnosis:**
```bash
# Check device
grep "device" experiments/dqn_atari/runs/pong_42/meta.json

# Check FPS in logs
grep "fps" experiments/dqn_atari/runs/pong_42/logs/episodes.csv | tail -n 10

# Check deterministic mode
grep "deterministic" experiments/dqn_atari/runs/pong_42/meta.json
```

**Solutions:**

**1. Verify GPU usage:**
```bash
# Watch GPU utilization
nvidia-smi -l 1

# Force GPU device
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --set training.device=cuda
```

**2. Disable deterministic mode:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --set experiment.deterministic.enabled=false
```

**3. Increase batch size:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --set training.batch_size=64
```

**Expected FPS:**
- GPU (deterministic off): ~1000 FPS
- GPU (deterministic on): ~850 FPS
- CPU: ~100-200 FPS

**Docs:** [DQN Setup](../reference/dqn_setup.md#performance-impact)

---

### Out of memory errors

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Causes:**
1. Batch size too large
2. Replay buffer too large
3. Multiple models on GPU

**Solutions:**

**1. Reduce batch size:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --set training.batch_size=16
```

**2. Reduce replay capacity:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --set replay.capacity=500000
```

**3. Use CPU for replay buffer:**
```python
# In replay buffer config
replay_buffer = ReplayBuffer(
    capacity=1000000,
    device='cpu',  # Keep on CPU
    pin_memory=True  # Fast transfer to GPU
)
```

**Docs:** [Replay Buffer](../reference/replay_buffer.md#memory-layout)

---

### High CPU usage

**Symptom:**
All CPU cores at 100% utilization.

**Cause:** Environment simulation and preprocessing on CPU (expected).

**Solution:**
This is normal. Atari environments run on CPU. To reduce:

```bash
# Reduce number of parallel environments (if using vectorized env)
--set env.num_envs=1

# Increase train_every to reduce environment steps
--set training.train_every=8
```

**Docs:** [Training Loop](../reference/training_loop_runtime.md#performance)

---

## Checkpoint/Resume Issues

### Resume fails with config mismatch

**Symptom:**
```
ValueError: Config mismatch between checkpoint and current config
```

**Cause:** Configuration changed between save and resume.

**Solutions:**

**1. Use strict resume mode to see details:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --resume path/to/checkpoint.pt \
  --strict-resume
```

**2. Allow config mismatch (not recommended):**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --resume path/to/checkpoint.pt
# (without --strict-resume)
```

**3. Use original config:**
```bash
# Config is saved in run directory
cp experiments/dqn_atari/runs/pong_42/meta.json \
   experiments/dqn_atari/configs/pong_resume.yaml

./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong_resume.yaml \
  --resume path/to/checkpoint.pt
```

**Docs:** [Checkpointing](../reference/checkpointing.md#config-validation-on-resume)

---

### Resume warning about git commit

**Symptom:**
```
Warning: Checkpoint commit hash (abc123) differs from current (def456)
```

**Cause:** Code changed since checkpoint was saved.

**Solutions:**

**1. Checkout original commit:**
```bash
# Find commit hash in checkpoint
python -c "import torch; ckpt=torch.load('checkpoint.pt', weights_only=False); print(ckpt['commit_hash'])"

# Checkout that commit
git checkout abc123
```

**2. Continue anyway (may have unexpected behavior):**
Resume will work but results may differ if code changed.

**3. Suppress warning:**
```bash
# Only if you know code is compatible
export IGNORE_COMMIT_MISMATCH=1
./experiments/dqn_atari/scripts/run_dqn.sh ...
```

**Docs:** [Checkpointing](../reference/checkpointing.md#checkpoint-validation)

---

### RNG states not restored

**Symptom:**
Resume produces different results than continuing original run.

**Cause:** RNG states not saved or not restored properly.

**Diagnosis:**
```python
# Check if RNG states in checkpoint
import torch
ckpt = torch.load('checkpoint.pt', weights_only=False)
print('RNG states saved:' in ckpt)
print(ckpt.get('rng_states', {}).keys())
```

**Solution:**

**1. Enable RNG state saving:**
Ensure checkpoint includes RNG states (should be default).

**2. Verify deterministic mode:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set experiment.deterministic.enabled=true
```

**3. Run determinism test:**
```bash
pytest tests/test_save_resume_determinism.py -v -s
```

**Docs:** [Checkpointing](../reference/checkpointing.md#rng-state-management)

---

## Environment Issues

### Frame shapes incorrect

**Symptom:**
Model expects (4, 84, 84) but receives different shape.

**Diagnosis:**
```python
import gymnasium as gym
from src.envs import make_atari_env

env = make_atari_env('ALE/Pong-v5')
obs, info = env.reset()
print(f"Observation shape: {obs.shape}")
# Should be (4, 84, 84)
```

**Causes:**
1. Missing FrameStack wrapper
2. Incorrect preprocessing
3. Wrong wrapper order

**Solution:**

**Check wrapper chain:**
```python
print(env)
# Should show: FrameStack -> AtariPreprocessing -> ...
```

**Run dry run to verify:**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run
```

**Docs:** [Atari Wrappers](../reference/atari_env_wrapper.md#wrapper-chain)

---

### Action space mismatch

**Symptom:**
```
IndexError: index N is out of bounds for axis 0 with size M
```

**Cause:** Model output size doesn't match environment action space.

**Diagnosis:**
```python
import gymnasium as gym
env = gym.make('ALE/Pong-v5')
print(f"Action space: {env.action_space.n}")
# Pong: 6 actions (minimal set)

from src.models import DQN
model = DQN(num_actions=6)
```

**Solution:**

**Verify config matches game:**
```yaml
env:
  env_id: ALE/Pong-v5
  full_action_space: false  # Use minimal action set

agent:
  num_actions: 6  # Must match game's minimal action set
```

**Get correct action count:**
```python
import gymnasium as gym
env = gym.make('ALE/Pong-v5', full_action_space=False)
print(env.action_space.n)  # Use this value
```

**Docs:** [DQN Setup](../reference/dqn_setup.md#selected-games)

---

### Episode termination issues

**Symptom:**
Episodes end unexpectedly or don't end when they should.

**Causes:**
1. Life-loss termination enabled when it shouldn't be
2. Missing EpisodeLifeEnv wrapper
3. Terminal flag handling incorrect

**Diagnosis:**
```bash
# Check episode_life setting
grep "episode_life" experiments/dqn_atari/runs/pong_42/meta.json

# Check episode lengths in logs
tail -n 20 experiments/dqn_atari/runs/pong_42/logs/episodes.csv
```

**Solutions:**

**1. Disable life-loss termination (evaluation):**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --set env.episode_life=false
```

**2. Enable life-loss termination (training):**
```bash
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --set env.episode_life=true
```

**3. Check wrapper:**
```python
from src.envs import make_atari_env
env = make_atari_env('ALE/Pong-v5', episode_life=True)
print(env)  # Should show EpisodeLifeEnv in chain
```

**Docs:** [Episode Handling](../reference/episode_handling.md), [Atari Wrappers](../reference/atari_env_wrapper.md)

---

## Getting More Help

If your issue isn't covered here:

1. **Search documentation:**
   - [Index](../index.md) - All documentation
   - [Workflows](workflows.md) - Common tasks
   - Design docs for specific components

2. **Run diagnostics:**
   ```bash
   # Environment info
   ./experiments/dqn_atari/scripts/capture_env.sh
   cat experiments/dqn_atari/system_info.txt

   # Test suite
   pytest tests/ -v
   ```

3. **Check logs:**
   ```bash
   # Training logs
   tail -f experiments/dqn_atari/runs/*/logs/*.csv

   # Run metadata
   cat experiments/dqn_atari/runs/*/meta.json
   ```

4. **Enable debug mode:**
   ```bash
   # Verbose logging
   python -m pdb src/train_dqn.py ...
   ```

---

**Last Updated:** 2025-11-13
