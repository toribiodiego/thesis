# Evaluation Harness Design

---

**Prerequisites:**
- Completed [DQN Model](dqn_model.md) - Understanding Q-network architecture
- Completed [Training Loop](training_loop_runtime.md) - Training flow and step counters
- Completed [Atari Wrappers](atari_env_wrapper.md) - Environment preprocessing

**Related Docs:**
- [Config & CLI](config_cli.md) - Evaluation configuration and CLI overrides
- [Checkpointing](checkpointing.md) - Saving best models from evaluation
- [Episode Handling](episode_handling.md) - Full-episode vs life-loss termination

---

## Overview

This document describes the evaluation harness implementation for periodic performance assessment during DQN training. The evaluation system follows the DQN paper protocol (Mnih et al., 2015) and provides:

- **Standalone evaluation loop** with ε-greedy or greedy policy
- **Periodic scheduling** (frame-based or wall-clock triggers)
- **Video capture** (MP4/GIF recording of episodes)
- **Structured outputs** (CSV/JSONL metrics, per-episode data)
- **Summary statistics** (mean/median/std/min/max returns)

**Key characteristics:**
- Evaluation runs in `model.eval()` mode with `torch.no_grad()`
- Uses separate `eval_epsilon` (default 0.05) from training epsilon
- Supports full-episode evaluation (no life-loss termination)
- Records first episode per evaluation interval for video inspection
- Persists results to multiple formats for analysis and plotting

---

## Architecture Components

### 1. Core Evaluation Function

**Location:** `src/training/evaluation.py::evaluate()`

**Purpose:** Standalone evaluation loop that runs N episodes and computes summary statistics.

**Signature:**
```python
def evaluate(
    env,
    model: torch.nn.Module,
    num_episodes: int = 10,
    eval_epsilon: float = 0.05,
    num_actions: int = None,
    device: str = 'cpu',
    seed: int = None,
    step: int = None,
    track_lives: bool = False,
    record_video: bool = False,
    video_dir: str = None,
    video_fps: int = 30,
    export_gif: bool = False,
    render_mode: str = 'rgb_array'
) -> dict
```

**Key Features:**
- Switches model to eval mode, restores training mode on exit
- Uses `torch.no_grad()` to disable gradient computation
- Supports ε-greedy (eval_epsilon > 0) or pure greedy (eval_epsilon = 0)
- Optional lives tracking via `info['lives']` or ALE API
- Optional video recording of first episode
- Includes run metadata (seed, step) in results

**Returns:**
```python
{
    'mean_return': float,           # Mean episode return
    'median_return': float,         # Median episode return
    'std_return': float,            # Standard deviation
    'min_return': float,            # Minimum return
    'max_return': float,            # Maximum return
    'mean_length': float,           # Mean episode length
    'episode_returns': list,        # Per-episode returns
    'episode_lengths': list,        # Per-episode lengths
    'num_episodes': int,            # Number of episodes run
    'eval_epsilon': float,          # Epsilon used
    # Optional fields:
    'seed': int,                    # Random seed (if provided)
    'step': int,                    # Training step (if provided)
    'episode_lives_lost': list,     # Lives lost (if track_lives=True)
    'video_info': dict              # Video metadata (if record_video=True)
}
```

### 2. Evaluation Scheduler

**Location:** `src/training/evaluation.py::EvaluationScheduler`

**Purpose:** Triggers periodic evaluations and tracks evaluation history.

**Initialization:**
```python
scheduler = EvaluationScheduler(
    eval_interval=250_000,        # Steps between evaluations
    num_episodes=10,              # Episodes per evaluation
    eval_epsilon=0.05,            # Epsilon during eval
    wall_clock_interval=None      # Optional time-based scheduling (seconds)
)
```

**Key Methods:**

- `should_evaluate(step: int) -> bool`: Check if evaluation should trigger
  - Frame-based: Triggers every `eval_interval` steps
  - Wall-clock: Triggers every `wall_clock_interval` seconds

- `record_evaluation(step: int, results: dict)`: Record evaluation with timestamp
  - Tracks steps, returns, timestamps
  - Updates last eval step/time

- `get_best_return() -> float`: Get best mean return across all evaluations

- `get_recent_trend(n: int = 3) -> str`: Detect performance trend
  - Returns: `'improving'`, `'declining'`, `'stable'`, or `'insufficient_data'`

- `get_schedule_metadata() -> dict`: Export schedule config and history

**Scheduling Modes:**

1. **Frame-based (default):**
   ```python
   # Evaluate every 250K environment frames
   scheduler = EvaluationScheduler(eval_interval=250_000)
   ```

2. **Wall-clock:**
   ```python
   # Evaluate every 30 minutes
   scheduler = EvaluationScheduler(wall_clock_interval=1800)
   ```

3. **Combined:**
   ```python
   # Both modes can be active (evaluation triggers on first condition met)
   scheduler = EvaluationScheduler(
       eval_interval=250_000,
       wall_clock_interval=3600
   )
   ```

### 3. Video Recorder

**Location:** `src/training/evaluation.py::VideoRecorder`

**Purpose:** Capture and save evaluation episode frames to MP4 (and optional GIF).

**Initialization:**
```python
recorder = VideoRecorder(
    output_path='videos/pong_250000.mp4',
    fps=30,
    export_gif=False
)
```

**Usage:**
```python
# During episode
frame = env.render()  # rgb_array mode
recorder.capture_frame(frame)

# After episode
video_info = recorder.save()
# Returns: {'video_path': '...', 'gif_path': '...', 'num_frames': N, 'fps': 30}
```

**Implementation Details:**
- Uses OpenCV (`cv2.VideoWriter`) with `'mp4v'` codec
- Converts RGB → BGR for OpenCV compatibility
- Handles uint8 and float32 frames (normalizes if needed)
- Optional GIF export via PIL/Pillow
- Creates output directory automatically

**Frame Rate:**
- Default: 30 FPS
- Configurable via `video_fps` parameter
- Matches natural game speed with 4-frame action repeat

### 4. Evaluation Logger

**Location:** `src/training/evaluation.py::EvaluationLogger`

**Purpose:** Persist evaluation results to multiple file formats for analysis.

**Initialization:**
```python
logger = EvaluationLogger(log_dir='runs/pong_123/eval')
```

**Output Files:**

1. **`evaluations.csv`** - Summary statistics (tabular format)
   ```csv
   step,mean_return,median_return,std_return,min_return,max_return,episodes,eval_epsilon
   250000,15.3,16.0,2.1,10.0,18.0,10,0.05
   500000,18.7,19.0,1.8,15.0,21.0,10,0.05
   ```

2. **`evaluations.jsonl`** - Summary statistics (streaming format)
   ```jsonl
   {"step": 250000, "mean_return": 15.3, "median_return": 16.0, ...}
   {"step": 500000, "mean_return": 18.7, "median_return": 19.0, ...}
   ```

3. **`per_episode_returns.jsonl`** - Raw per-episode data
   ```jsonl
   {"step": 250000, "episode_returns": [16.0, 14.0, ...], "episode_lengths": [1234, 1456, ...]}
   {"step": 500000, "episode_returns": [19.0, 18.0, ...], "episode_lengths": [1345, 1289, ...]}
   ```

4. **`detailed/eval_step_<step>.json`** - Complete evaluation details
   ```json
   {
     "step": 250000,
     "statistics": {
       "mean_return": 15.3,
       "median_return": 16.0,
       "std_return": 2.1,
       "min_return": 10.0,
       "max_return": 18.0,
       "mean_length": 1345.6
     },
     "episode_returns": [16.0, 14.0, 15.0, ...],
     "episode_lengths": [1234, 1456, 1345, ...],
     "num_episodes": 10,
     "eval_epsilon": 0.05
   }
   ```

**Key Methods:**

- `log_evaluation(step: int, results: dict, epsilon: float = None)`: Log evaluation
  - Writes to CSV, JSONL, per-episode sidecar, and detailed JSON
  - Auto-creates CSV header on first write
  - Appends incrementally for streaming analysis

- `get_all_results() -> list`: Load all results from CSV
  - Returns list of dicts for plotting/analysis

---

## Evaluation Loop Structure

### Control Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                     Training Loop (main)                         │
│                                                                   │
│  for step in range(total_steps):                                 │
│      1. Select action (ε-greedy, training epsilon)               │
│      2. Step environment                                         │
│      3. Store transition in replay buffer                        │
│      4. Perform optimization (if step % train_every == 0)        │
│      5. Update target network (if step % target_update == 0)     │
│                                                                   │
│      ┌──────────────────────────────────────────────────┐       │
│      │ 6. Check if evaluation should trigger            │       │
│      │    if scheduler.should_evaluate(step):           │       │
│      │                                                   │       │
│      │    ┌──────────────────────────────────────────┐  │       │
│      │    │   Evaluation Loop (eval mode)            │  │       │
│      │    │                                           │  │       │
│      │    │   • model.eval()                         │  │       │
│      │    │   • torch.no_grad()                      │  │       │
│      │    │                                           │  │       │
│      │    │   for episode in range(num_episodes):    │  │       │
│      │    │       • Reset environment                │  │       │
│      │    │       • Run full episode with ε_eval     │  │       │
│      │    │       • Record return, length, lives     │  │       │
│      │    │       • Capture video (first episode)    │  │       │
│      │    │                                           │  │       │
│      │    │   • Compute summary statistics           │  │       │
│      │    │   • Save video (if recording)            │  │       │
│      │    │   • Log results (CSV/JSONL/JSON)         │  │       │
│      │    │   • Update scheduler history             │  │       │
│      │    │                                           │  │       │
│      │    │   • model.train()  # Restore             │  │       │
│      │    └──────────────────────────────────────────┘  │       │
│      └──────────────────────────────────────────────────┘       │
│                                                                   │
│      7. Continue training...                                     │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### Pseudocode

```python
# Setup
scheduler = EvaluationScheduler(eval_interval=250_000, num_episodes=10)
logger = EvaluationLogger(log_dir='runs/pong_123/eval')

# Training loop
for step in range(total_steps):
    # ... training logic ...

    # Periodic evaluation
    if scheduler.should_evaluate(step):
        # Run evaluation
        results = evaluate(
            env=eval_env,
            model=model,
            num_episodes=scheduler.num_episodes,
            eval_epsilon=scheduler.eval_epsilon,
            device=device,
            step=step,
            record_video=True,
            video_dir=f'runs/videos/{step}'
        )

        # Log results
        logger.log_evaluation(step, results, epsilon=current_epsilon)
        scheduler.record_evaluation(step, results)

        # Optional: Save best model
        if results['mean_return'] > scheduler.get_best_return():
            save_checkpoint(model, f'checkpoints/best_model.pt')
```

### Train/Eval Mode Switching

The `evaluate()` function ensures proper mode switching:

```python
def evaluate(env, model, ...):
    # Save original mode
    was_training = model.training

    # Switch to eval mode
    model.eval()

    try:
        # Evaluation logic with torch.no_grad()
        with torch.no_grad():
            # Run episodes...
            pass
    finally:
        # Restore original mode
        if was_training:
            model.train()
```

**Why this matters:**
- `model.eval()`: Disables dropout, freezes batch norm statistics
- `torch.no_grad()`: Disables gradient computation (faster inference)
- Automatic restoration ensures training continues correctly

---

## Metric Definitions

### Summary Statistics

| Metric | Definition | Computation |
|--------|------------|-------------|
| `mean_return` | Average episode return | `np.mean(episode_returns)` |
| `median_return` | Median episode return | `np.median(episode_returns)` |
| `std_return` | Standard deviation of returns | `np.std(episode_returns)` |
| `min_return` | Minimum episode return | `np.min(episode_returns)` |
| `max_return` | Maximum episode return | `np.max(episode_returns)` |
| `mean_length` | Average episode length (steps) | `np.mean(episode_lengths)` |

### Per-Episode Metrics

| Metric | Definition | Source |
|--------|------------|--------|
| `episode_return` | Total reward for episode | Sum of step rewards |
| `episode_length` | Number of steps in episode | Step counter |
| `lives_lost` | Lives lost during episode | `info['lives']` or ALE API |

### Metadata Fields

| Field | Description | Example |
|-------|-------------|---------|
| `step` | Training step when eval occurred | `250000` |
| `num_episodes` | Number of episodes evaluated | `10` |
| `eval_epsilon` | Epsilon used during evaluation | `0.05` |
| `seed` | Random seed (if provided) | `42` |
| `training_epsilon` | Current training epsilon (optional) | `0.15` |

---

## Output File Schemas

### CSV Schema (`evaluations.csv`)

**Headers:**
```
step,mean_return,median_return,std_return,min_return,max_return,episodes,eval_epsilon
```

**Data Types:**
- `step`: int - Environment step when evaluation occurred
- `mean_return`: float - Mean episode return
- `median_return`: float - Median episode return
- `std_return`: float - Standard deviation of returns
- `min_return`: float - Minimum return across episodes
- `max_return`: float - Maximum return across episodes
- `episodes`: int - Number of episodes evaluated (typically 10)
- `eval_epsilon`: float - Epsilon used during evaluation (typically 0.05)

**Example:**
```csv
step,mean_return,median_return,std_return,min_return,max_return,episodes,eval_epsilon
250000,15.3,16.0,2.1,10.0,18.0,10,0.05
500000,18.7,19.0,1.8,15.0,21.0,10,0.05
750000,20.1,20.5,1.5,17.0,22.0,10,0.05
```

### JSONL Schema (`evaluations.jsonl`)

**Format:** One JSON object per line (newline-delimited)

**Schema:**
```json
{
  "step": int,
  "mean_return": float,
  "median_return": float,
  "std_return": float,
  "min_return": float,
  "max_return": float,
  "episodes": int,
  "eval_epsilon": float,
  "training_epsilon": float  // Optional
}
```

**Usage:** Stream-friendly format for incremental analysis
```python
import json

# Read line-by-line
with open('evaluations.jsonl', 'r') as f:
    for line in f:
        eval_data = json.loads(line)
        print(f"Step {eval_data['step']}: {eval_data['mean_return']}")
```

### Per-Episode Sidecar (`per_episode_returns.jsonl`)

**Format:** One JSON object per evaluation

**Schema:**
```json
{
  "step": int,
  "episode_returns": [float, float, ...],  // List of length num_episodes
  "episode_lengths": [int, int, ...]       // List of length num_episodes
}
```

**Purpose:** Raw per-episode data for post-hoc analysis
- Histogram plotting
- Statistical tests (t-tests, ANOVA)
- Learning curve variance analysis

### Detailed JSON (`detailed/eval_step_<step>.json`)

**Format:** Complete evaluation details

**Schema:**
```json
{
  "step": int,
  "statistics": {
    "mean_return": float,
    "median_return": float,
    "std_return": float,
    "min_return": float,
    "max_return": float,
    "mean_length": float
  },
  "episode_returns": [float, ...],
  "episode_lengths": [int, ...],
  "num_episodes": int,
  "eval_epsilon": float,
  "training_epsilon": float  // Optional
}
```

**Purpose:** Archival format with full evaluation context

---

## Video Capture Settings

### Configuration

**Config file:** `experiments/dqn_atari/configs/base.yaml`

```yaml
evaluation:
  # Video capture
  record_video: true          # Enable video recording
  video_frequency: 1          # Record first N episodes (1 = first episode only)
  video_format: "mp4"         # Video format: 'mp4' or 'gif'
```

**Runtime parameters:**
```python
evaluate(
    env=eval_env,
    model=model,
    record_video=True,
    video_dir='runs/videos',
    video_fps=30,
    export_gif=False,
    render_mode='rgb_array'
)
```

### Video File Organization

**Directory structure:**
```
runs/
└── pong_seed42_20250114/
    └── videos/
        ├── step_250000.mp4      # First evaluation
        ├── step_500000.mp4      # Second evaluation
        ├── step_750000.mp4      # Third evaluation
        └── ...
```

**Naming convention:** `step_<training_step>.mp4`

### Technical Specifications

| Setting | Default | Description |
|---------|---------|-------------|
| **Codec** | `'mp4v'` | MPEG-4 codec (broad compatibility) |
| **FPS** | `30` | Frames per second |
| **Resolution** | Native | Matches environment render resolution |
| **Color** | RGB → BGR | Converted for OpenCV |
| **Recording** | First episode only | Minimizes overhead |

### Video Metadata

When `record_video=True`, results include `video_info`:

```python
{
    'video_path': 'runs/videos/step_250000.mp4',
    'gif_path': 'runs/videos/step_250000.gif',  # If export_gif=True
    'num_frames': 1234,
    'fps': 30
}
```

### Optional GIF Export

**Enable GIF export:**
```python
evaluate(..., export_gif=True)
```

**Requirements:** PIL/Pillow library
```bash
pip install Pillow
```

**Output:** Generates both MP4 and GIF with same filename base

---

## Scheduling Triggers

### Frame-Based Scheduling (Default)

**Trigger condition:** Every `eval_interval` environment frames

```python
scheduler = EvaluationScheduler(eval_interval=250_000)

# Evaluates at steps: 250000, 500000, 750000, 1000000, ...
```

**Implementation:**
```python
def should_evaluate(self, step: int) -> bool:
    if step >= self.eval_interval and step % self.eval_interval == 0:
        return step != self.last_eval_step  # Avoid duplicates
    return False
```

**DQN paper protocol:** Evaluate every 250K frames (Mnih et al., 2015)

### Wall-Clock Scheduling

**Trigger condition:** Every `wall_clock_interval` seconds

```python
# Evaluate every 30 minutes
scheduler = EvaluationScheduler(wall_clock_interval=1800)
```

**Use cases:**
- Long runs with variable FPS
- Debugging/development (frequent checks)
- Resource-constrained environments

**Implementation:**
```python
def should_evaluate(self, step: int) -> bool:
    import time
    current_time = time.time()

    if self.last_eval_time is None:
        return True  # First evaluation

    elapsed = current_time - self.last_eval_time
    return elapsed >= self.wall_clock_interval
```

### Combined Scheduling

**Both modes active:** Evaluation triggers on first condition met

```python
scheduler = EvaluationScheduler(
    eval_interval=250_000,      # Frame-based
    wall_clock_interval=3600    # Wall-clock (1 hour)
)
```

**Behavior:**
- If training is fast: Triggers every 250K frames (whichever comes first)
- If training is slow: Triggers every hour (ensures periodic checks)

### Schedule Metadata

Get schedule configuration and history:

```python
metadata = scheduler.get_schedule_metadata()

# Returns:
{
    'schedule_type': 'frame_based',  # or 'wall_clock'
    'eval_interval': 250000,
    'wall_clock_interval': None,
    'num_episodes': 10,
    'eval_epsilon': 0.05,
    'total_evaluations': 5,
    'eval_steps': [250000, 500000, 750000, 1000000, 1250000],
    'eval_returns': [15.3, 18.7, 20.1, 21.5, 22.0],
    'eval_timestamps': [1234567.8, 1234987.6, ...],
    'elapsed_times': [1234.5, 2345.6, ...],
    'total_elapsed_time': 12345.6
}
```

---

## CLI Examples

### Manual Evaluation Run

**Basic evaluation:**
```bash
python scripts/evaluate_model.py \
    --checkpoint runs/pong_seed42/checkpoints/step_1000000.pt \
    --env ALE/Pong-v5 \
    --num-episodes 30 \
    --eval-epsilon 0.05 \
    --device cuda
```

**Pure greedy evaluation:**
```bash
python scripts/evaluate_model.py \
    --checkpoint runs/pong_seed42/checkpoints/best_model.pt \
    --env ALE/Pong-v5 \
    --num-episodes 30 \
    --eval-epsilon 0.0 \
    --device cuda
```

**With video recording:**
```bash
python scripts/evaluate_model.py \
    --checkpoint runs/pong_seed42/checkpoints/step_1000000.pt \
    --env ALE/Pong-v5 \
    --num-episodes 10 \
    --record-video \
    --video-dir results/videos/pong_final \
    --export-gif
```

### Override Evaluation Config During Training

**Change num_episodes for final run:**
```bash
python train_dqn.py \
    --cfg experiments/dqn_atari/configs/pong.yaml \
    --seed 42 \
    --set evaluation.num_episodes=30
```

**Disable video recording:**
```bash
python train_dqn.py \
    --cfg experiments/dqn_atari/configs/pong.yaml \
    --seed 42 \
    --set evaluation.record_video=false
```

**Change evaluation frequency:**
```bash
python train_dqn.py \
    --cfg experiments/dqn_atari/configs/pong.yaml \
    --seed 42 \
    --set evaluation.eval_every=100000
```

### Re-rendering Videos from Checkpoint

**Re-evaluate with video recording:**
```bash
python scripts/evaluate_model.py \
    --checkpoint runs/pong_seed42/checkpoints/step_1000000.pt \
    --env ALE/Pong-v5 \
    --num-episodes 5 \
    --record-video \
    --video-dir results/videos/pong_step1M_rerender \
    --video-fps 30
```

**Generate GIF from existing MP4:**
```bash
# Using ffmpeg
ffmpeg -i runs/videos/step_250000.mp4 \
       -vf "fps=10,scale=320:-1:flags=lanczos" \
       -loop 0 \
       runs/videos/step_250000.gif
```

### Batch Evaluation Across Seeds

**Evaluate multiple checkpoints:**
```bash
#!/bin/bash
for seed in 0 1 2; do
    python scripts/evaluate_model.py \
        --checkpoint runs/pong_seed${seed}/checkpoints/best_model.pt \
        --env ALE/Pong-v5 \
        --num-episodes 30 \
        --eval-epsilon 0.05 \
        --output results/final_eval/pong_seed${seed}.json
done
```

### Re-generating Metrics from Logs

**Load and re-compute statistics:**
```python
import pandas as pd

# Load CSV
df = pd.read_csv('runs/pong_seed42/eval/evaluations.csv')

# Compute rolling average
df['mean_return_rolling'] = df['mean_return'].rolling(window=5).mean()

# Plot
import matplotlib.pyplot as plt
plt.plot(df['step'], df['mean_return'], label='Mean Return')
plt.plot(df['step'], df['mean_return_rolling'], label='Rolling Avg (5 evals)')
plt.xlabel('Training Steps')
plt.ylabel('Mean Evaluation Return')
plt.legend()
plt.savefig('results/plots/pong_eval_curve.png')
```

**Aggregate per-episode data:**
```python
import json

# Load per-episode returns
episodes = []
with open('runs/pong_seed42/eval/per_episode_returns.jsonl', 'r') as f:
    for line in f:
        episodes.append(json.loads(line))

# Extract all returns
all_returns = []
for eval_data in episodes:
    all_returns.extend(eval_data['episode_returns'])

# Compute overall statistics
print(f"Overall mean: {np.mean(all_returns)}")
print(f"Overall std: {np.std(all_returns)}")
```

---

## Debugging Guide

### Common Issues

#### 1. Evaluation Not Triggering

**Symptoms:**
- No evaluation logs appearing
- `should_evaluate()` always returns `False`

**Diagnosis:**
```python
# Check scheduler state
print(f"Current step: {step}")
print(f"Eval interval: {scheduler.eval_interval}")
print(f"Last eval step: {scheduler.last_eval_step}")
print(f"Should evaluate: {scheduler.should_evaluate(step)}")
```

**Common causes:**
- `step < eval_interval` (first evaluation hasn't occurred yet)
- `step % eval_interval != 0` (not on interval boundary)
- Duplicate check triggered (`step == last_eval_step`)

**Fix:**
```python
# Force evaluation for testing
results = evaluate(env, model, num_episodes=1)
scheduler.record_evaluation(step, results)
```

#### 2. Video Corruption or Missing Frames

**Symptoms:**
- Video file exists but won't play
- Video is shorter than expected
- Green/black frames in video

**Diagnosis:**
```python
# Check frame capture
recorder = VideoRecorder('test.mp4', fps=30)
for i in range(10):
    frame = env.render()
    print(f"Frame {i} shape: {frame.shape}, dtype: {frame.dtype}")
    recorder.capture_frame(frame)
video_info = recorder.save()
print(f"Captured {video_info['num_frames']} frames")
```

**Common causes:**

1. **Environment not in rgb_array mode:**
   ```python
   # Wrong
   env = gym.make('ALE/Pong-v5')

   # Correct
   env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
   ```

2. **OpenCV not installed or wrong version:**
   ```bash
   pip install opencv-python
   # Or for headless servers:
   pip install opencv-python-headless
   ```

3. **Incorrect color space:**
   - VideoRecorder handles RGB→BGR conversion
   - Check frame is RGB (not BGR or grayscale)

**Fix:**
```python
# Verify environment rendering
env = gym.make('ALE/Pong-v5', render_mode='rgb_array')
frame = env.reset()[0]
print(f"Frame shape: {frame.shape}")  # Should be (H, W, 3)
print(f"Frame dtype: {frame.dtype}")  # Should be uint8
```

#### 3. NaN or Inf in Evaluation Metrics

**Symptoms:**
- CSV contains `NaN` or `inf` values
- Plotting fails with value errors

**Diagnosis:**
```python
# Check returns
print(f"Episode returns: {results['episode_returns']}")
print(f"Any NaN: {any(np.isnan(results['episode_returns']))}")
print(f"Any Inf: {any(np.isinf(results['episode_returns']))}")
```

**Common causes:**
- Model producing NaN Q-values (training instability)
- Division by zero in metric computation
- Empty episode_returns list

**Fix:**
```python
# Add validation in evaluate()
if len(episode_returns) == 0:
    raise ValueError("No episodes completed during evaluation")

# Check for NaN Q-values
q_values = model(state)['q_values']
if torch.isnan(q_values).any():
    raise ValueError("Model producing NaN Q-values")
```

#### 4. Evaluation Slowing Down Training

**Symptoms:**
- Training FPS drops during evaluation
- Evaluation takes too long

**Diagnosis:**
```python
import time

start = time.time()
results = evaluate(env, model, num_episodes=10)
elapsed = time.time() - start
print(f"Evaluation took {elapsed:.1f}s for {results['num_episodes']} episodes")
```

**Common causes:**
- `num_episodes` too high (use 10 for interim, 30 only for final)
- Video recording enabled for all episodes (should only record first)
- Not using GPU for inference
- Environment rendering overhead

**Fix:**
```python
# Reduce episodes for interim checks
scheduler = EvaluationScheduler(num_episodes=10)  # Not 30

# Disable video for frequent checks
evaluate(env, model, record_video=False)

# Use GPU if available
evaluate(env, model, device='cuda')

# Reduce eval frequency
scheduler = EvaluationScheduler(eval_interval=500_000)  # Instead of 250K
```

#### 5. Desync Between CSV and JSONL

**Symptoms:**
- CSV and JSONL have different number of entries
- Metrics don't match between files

**Diagnosis:**
```bash
# Count lines
wc -l runs/pong_seed42/eval/evaluations.csv
wc -l runs/pong_seed42/eval/evaluations.jsonl

# Check last entries
tail -n 1 runs/pong_seed42/eval/evaluations.csv
tail -n 1 runs/pong_seed42/eval/evaluations.jsonl
```

**Common causes:**
- Interrupted evaluation (crashed mid-write)
- Manual file editing
- Race condition with concurrent writes

**Fix:**
```python
# Re-generate from detailed JSON files
import glob
import json
import csv

# Load all detailed results
detailed_dir = 'runs/pong_seed42/eval/detailed'
json_files = sorted(glob.glob(f'{detailed_dir}/eval_step_*.json'))

# Re-create CSV
csv_path = 'runs/pong_seed42/eval/evaluations_fixed.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'step', 'mean_return', 'median_return', 'std_return',
        'min_return', 'max_return', 'episodes', 'eval_epsilon'
    ])
    writer.writeheader()

    for json_file in json_files:
        with open(json_file, 'r') as jf:
            data = json.load(jf)
            writer.writerow({
                'step': data['step'],
                'mean_return': data['statistics']['mean_return'],
                'median_return': data['statistics']['median_return'],
                'std_return': data['statistics']['std_return'],
                'min_return': data['statistics']['min_return'],
                'max_return': data['statistics']['max_return'],
                'episodes': data['num_episodes'],
                'eval_epsilon': data['eval_epsilon']
            })
```

#### 6. Lives Tracking Not Working

**Symptoms:**
- `episode_lives_lost` is empty or missing
- All values are `None`

**Diagnosis:**
```python
# Check info dict
obs, info = env.reset()
print(f"Info keys: {info.keys()}")
print(f"Lives in info: {'lives' in info}")

# Try ALE API
try:
    lives = env.unwrapped.ale.lives()
    print(f"Lives from ALE: {lives}")
except AttributeError:
    print("ALE API not available")
```

**Common causes:**
- Environment doesn't expose lives in `info` dict
- Not an ALE environment (lives only for Atari)
- Wrapper hiding `info['lives']`

**Fix:**
```python
# Use track_lives=True only for Atari environments
if 'ALE' in env.spec.id:
    results = evaluate(env, model, track_lives=True)
else:
    results = evaluate(env, model, track_lives=False)
```

#### 7. Train/Eval Mode Not Switching

**Symptoms:**
- Model stays in eval mode after evaluation
- Batch norm statistics frozen during training

**Diagnosis:**
```python
# Check mode before/after evaluation
print(f"Before eval: model.training = {model.training}")
results = evaluate(env, model)
print(f"After eval: model.training = {model.training}")
```

**Common causes:**
- Exception during evaluation prevents mode restoration
- Manual mode override after evaluation
- Multiple models with different states

**Fix:**
```python
# The evaluate() function uses try/finally to ensure restoration
def evaluate(...):
    was_training = model.training
    model.eval()
    try:
        # Evaluation logic
        pass
    finally:
        if was_training:
            model.train()  # Always restores
```

**Verify:**
```python
# Force check
assert model.training, "Model should be in training mode"
```

---

## DQN Paper Evaluation Protocol

### Protocol Summary (Mnih et al., 2015)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **eval_epsilon** | `0.05` | Mostly greedy with 5% exploration |
| **num_episodes** | `10` | Interim checks during training |
| **num_episodes (final)** | `30` | Final reporting for paper results |
| **eval_interval** | `250000` | Evaluate every 250K frames |
| **termination** | Full episode | No life-loss termination during eval |

### Rationale

**Why ε=0.05 (not 0.0)?**
- Pure greedy (ε=0) can be brittle and overfit to specific trajectories
- Small exploration (5%) provides more representative performance estimates
- Matches paper standard for comparability

**Why 10 episodes (not 1 or 30)?**
- **10 episodes:** Good balance between statistical reliability and runtime
  - Sufficient for tracking learning progress
  - Fast enough for frequent interim checks (every 250K frames)
- **30 episodes:** Only for final reporting
  - More stable statistics for paper results
  - Too expensive for frequent evaluation

**Why every 250K frames?**
- Frequent enough to catch learning dynamics
- Infrequent enough to minimize training overhead
- Matches paper protocol for reproducibility

**Why full episodes (no life-loss termination)?**
- Evaluation should measure final policy performance
- Life-loss termination is a training trick, not part of the game
- Full episodes provide true measure of game performance

### Implementation

**Default configuration already follows paper:**
```yaml
# experiments/dqn_atari/configs/base.yaml
evaluation:
  enabled: true
  eval_every: 250000        # ✓ Every 250K frames
  num_episodes: 10          # ✓ 10 for interim
  epsilon: 0.05             # ✓ ε=0.05 (not 0.0)
  deterministic: false      # ✓ Full episodes
```

**For final reporting:**
```bash
python train_dqn.py \
    --cfg experiments/dqn_atari/configs/pong.yaml \
    --seed 42 \
    --set evaluation.num_episodes=30
```

---

## Regenerating Artifacts

### Re-evaluate from Checkpoint

```bash
python scripts/evaluate_model.py \
    --checkpoint runs/pong_seed42/checkpoints/step_1000000.pt \
    --env ALE/Pong-v5 \
    --num-episodes 30 \
    --output results/re_eval/pong_1M.json
```

### Re-render Video

```bash
python scripts/evaluate_model.py \
    --checkpoint runs/pong_seed42/checkpoints/best_model.pt \
    --env ALE/Pong-v5 \
    --num-episodes 1 \
    --record-video \
    --video-dir results/videos/best_model \
    --video-fps 30 \
    --export-gif
```

### Re-compute Metrics from JSONL

```python
import json
import numpy as np

# Load per-episode data
episodes = []
with open('runs/pong_seed42/eval/per_episode_returns.jsonl', 'r') as f:
    for line in f:
        episodes.append(json.loads(line))

# Re-compute statistics for specific step
step = 1000000
eval_data = next(e for e in episodes if e['step'] == step)

returns = np.array(eval_data['episode_returns'])
print(f"Mean: {np.mean(returns)}")
print(f"Median: {np.median(returns)}")
print(f"Std: {np.std(returns)}")
print(f"Min: {np.min(returns)}")
print(f"Max: {np.max(returns)}")
```

### Generate Summary Report

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load evaluation history
df = pd.read_csv('runs/pong_seed42/eval/evaluations.csv')

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(df['step'], df['mean_return'], marker='o', label='Mean Return')
plt.fill_between(
    df['step'],
    df['mean_return'] - df['std_return'],
    df['mean_return'] + df['std_return'],
    alpha=0.3,
    label='±1 Std Dev'
)
plt.xlabel('Training Steps')
plt.ylabel('Evaluation Return')
plt.title('Pong Evaluation Performance')
plt.legend()
plt.grid(True)
plt.savefig('results/plots/pong_eval_curve.png', dpi=150)

# Summary table
print(df[['step', 'mean_return', 'median_return', 'std_return']].to_string())
```

---

## References

### Code Locations

- **Core evaluation:** `src/training/evaluation.py::evaluate()`
- **Scheduler:** `src/training/evaluation.py::EvaluationScheduler`
- **Video recording:** `src/training/evaluation.py::VideoRecorder`
- **Logging:** `src/training/evaluation.py::EvaluationLogger`
- **Configuration:** `experiments/dqn_atari/configs/base.yaml`

### Related Documents

- [Training Loop](training_loop_runtime.md) - Integration with main training loop
- [Config & CLI](config_cli.md) - Configuration and CLI overrides
- [Checkpointing](checkpointing.md) - Saving best models from evaluation
- [Episode Handling](episode_handling.md) - Life-loss vs full-episode termination

### Paper Reference

Mnih, V., et al. (2015). Human-level control through deep reinforcement learning.
Nature, 518(7540), 529-533.

**Evaluation protocol details:** Supplementary Methods, Section "Evaluation procedure"

---

## Quick Reference

### Minimal Evaluation Example

```python
from src.training import evaluate, EvaluationScheduler, EvaluationLogger

# Setup
scheduler = EvaluationScheduler(eval_interval=250_000, num_episodes=10)
logger = EvaluationLogger(log_dir='runs/eval')

# In training loop
if scheduler.should_evaluate(step):
    results = evaluate(env, model,
                      num_episodes=scheduler.num_episodes,
                      eval_epsilon=scheduler.eval_epsilon,
                      device='cuda')
    logger.log_evaluation(step, results)
    scheduler.record_evaluation(step, results)
```

### Output Files Checklist

After evaluation, verify these files exist:

```bash
runs/pong_seed42/eval/
├── evaluations.csv                    # Summary statistics (CSV)
├── evaluations.jsonl                  # Summary statistics (JSONL)
├── per_episode_returns.jsonl          # Per-episode raw data
└── detailed/
    ├── eval_step_250000.json          # Detailed results
    ├── eval_step_500000.json
    └── ...

runs/pong_seed42/videos/
├── step_250000.mp4                    # Video recordings
├── step_500000.mp4
└── ...
```

### Common CLI Patterns

```bash
# Standard evaluation during training
python train_dqn.py --cfg configs/pong.yaml --seed 42

# Final evaluation with 30 episodes
python train_dqn.py --cfg configs/pong.yaml --seed 42 --set evaluation.num_episodes=30

# Re-evaluate from checkpoint
python scripts/evaluate_model.py --checkpoint best_model.pt --env ALE/Pong-v5 --num-episodes 30

# Generate video
python scripts/evaluate_model.py --checkpoint best_model.pt --env ALE/Pong-v5 --record-video
```
