# Logging & Plotting Pipeline

**Status**: DONE Complete
**Location**: `src/training/metrics_logger.py`, `scripts/plot_results.py`, `scripts/export_results_table.py`

## Overview

The logging and plotting pipeline provides a comprehensive system for recording, visualizing, and analyzing training metrics. It supports multiple logging backends (TensorBoard, W&B, CSV), publication-quality plotting, and multi-seed statistical analysis.

**Key Features**:
- Multi-backend logging with unified interface
- Standardized metric keys across all backends
- Periodic flushing and artifact uploads
- Publication-quality plotting (PNG, PDF, SVG)
- Multi-seed aggregation with confidence intervals
- Results table generation (Markdown/CSV)
- Performance safeguards for large logs

---

## Architecture

### Components

```
Logging Pipeline
├── MetricsLogger          # Unified logging interface
│   ├── TensorBoardBackend # TensorBoard logs
│   ├── WandBBackend       # W&B cloud logging
│   └── CSVBackend         # CSV files for offline analysis
├── plot_results.py        # Visualization script
├── export_results_table.py # Summary table generator
└── Artifact Management    # W&B artifact uploads
```

### Backend Interaction Model

The three backends operate **independently** and **in parallel**:

```
MetricsLogger.log_step(step=1000, loss=0.5, ...)
    │
    ├──→ TensorBoardBackend.log_scalar("train/loss", 0.5, step=1000)
    │    └─→ writes to: experiments/dqn_atari/runs/<game>_<seed>_<timestamp>/tensorboard/events.out.tfevents.*
    │
    ├──→ WandBBackend.log({"train/loss": 0.5}, step=1000)
    │    └─→ uploads to: W&B cloud (wandb.ai/<entity>/<project>/<run_id>)
    │
    └──→ CSVBackend.log_step_metric(step=1000, loss=0.5, ...)
         └─→ writes to: experiments/dqn_atari/runs/<game>_<seed>_<timestamp>/csv/training_steps.csv
```

**Key Properties**:
1. **Fail-safe**: If one backend fails (e.g., W&B network error), others continue
2. **Async**: TensorBoard/W&B write asynchronously; CSV writes synchronously
3. **No dependencies**: Backends don't communicate with each other
4. **Unified keys**: All backends receive identical metric names (train/loss, episode/return, etc.)

### File System Layout

```
experiments/dqn_atari/runs/
└── <game>_<seed>_<timestamp>/   # e.g., "pong_42_20251115_230409"
    ├── config.yaml              # Frozen config snapshot
    ├── meta.json                # Run metadata (git hash, python version, etc.)
    ├── tensorboard/
    │   └── events.out.tfevents.1699977600.hostname
    ├── csv/
    │   ├── training_steps.csv   # Per-step metrics
    │   └── episodes.csv         # Per-episode metrics
    ├── eval/
    │   └── evaluations.csv      # Periodic evaluation results
    ├── videos/
    │   └── <Env>_step_<step>_best_ep<N>_r<return>.mp4
    ├── checkpoints/             # Model checkpoints
    │   └── checkpoint_250000.pt
    ├── artifacts/               # Additional artifacts
    └── logs/                    # Reserved for future use
```

**Directory Creation**:
- `MetricsLogger.__init__()` creates all directories on first initialization
- If W&B is disabled, `wandb/` directory is not created
- Parent directories are created recursively (`mkdir -p` behavior)

### Data Flow

```
Training Loop
    ↓
MetricsLogger.log_step()     # Per-step metrics (loss, epsilon, FPS)
MetricsLogger.log_episode()  # Per-episode metrics (return, length)
MetricsLogger.log_evaluation() # Evaluation metrics (mean return, std)
    ↓
    ├─→ TensorBoard: Async write to event file
    ├─→ W&B: Async upload to cloud (batched)
    └─→ CSV: Sync write to file (buffered)
    ↓
Periodic Flush (every 1K steps)
    │
    ├─→ TensorBoard: writer.flush()
    ├─→ W&B: No-op (already batched)
    └─→ CSV: file.flush() + os.fsync()
    ↓
Artifact Upload (every 1M steps) → W&B
    │
    └─→ Upload CSV files as W&B artifact
        Artifact name: "training_logs_step_{step}"
        Contains: training_steps.csv, episodes.csv, metadata
    ↓
Training Complete
    ↓
logger.close() - Final flush + cleanup
    ↓
plot_results.py              # Generate plots from CSV/W&B artifacts
export_results_table.py      # Summary tables from run directories
```

### Backend Configuration

Each backend is configured independently:

#### TensorBoard Backend

```python
# Enable/disable
enable_tensorboard=True

# Output directory (automatically created)
log_dir = "experiments/dqn_atari/runs/<game>_<seed>_<timestamp>/tensorboard"

# Usage: view with TensorBoard
# $ tensorboard --logdir experiments/dqn_atari/runs/
```

**Config Fields**:
```yaml
logging:
  enable_tensorboard: true
  tensorboard_dir: "results/logs/{game}/{run_id}/tensorboard"
```

#### W&B Backend

```python
# Enable/disable
enable_wandb=True

# Configuration
wandb_config={
    "project": "dqn-atari",      # W&B project name
    "entity": "my-team",          # W&B team/username (optional)
    "run_id": "pong_seed42",      # Unique run identifier
    "config": {...}               # Training hyperparameters
}
```

**Config Fields**:
```yaml
logging:
  enable_wandb: true
  wandb_project: "dqn-atari"
  wandb_entity: "my-team"      # optional
  wandb_run_id: null           # auto-generated if null
  upload_artifacts: true       # Enable artifact uploads
  artifact_upload_interval: 1000000  # Upload every 1M steps
```

**Environment Variables**:
```bash
# W&B API key (required)
export WANDB_API_KEY="your_key_here"

# Optional: offline mode (no cloud sync)
export WANDB_MODE=offline

# Optional: disable W&B entirely
export WANDB_DISABLED=true
```

#### CSV Backend

```python
# Enable/disable
enable_csv=True

# Output directory (automatically created)
log_dir = "results/logs/<game>/<run_id>/csv"
```

**Config Fields**:
```yaml
logging:
  enable_csv: true
  csv_dir: "results/logs/{game}/{run_id}/csv"
  flush_interval: 1000  # Flush every N steps
```

**Output Files**:
- `training_steps.csv` - Per-step metrics (step, loss, epsilon, learning_rate, grad_norm, fps)
- `episodes.csv` - Per-episode metrics (step, episode, return, length)

---

## MetricsLogger

### Initialization

```python
from src.training.metrics_logger import MetricsLogger

logger = MetricsLogger(
    log_dir="runs/pong_123/logs",
    enable_tensorboard=True,
    enable_wandb=True,
    enable_csv=True,
    flush_interval=1000,         # Flush every 1000 steps
    upload_artifacts=True,       # Upload to W&B
    wandb_config={
        "project": "dqn-atari",
        "entity": "my-team",
        "run_id": "pong_123",
        "config": {...}           # Training config
    }
)
```

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_dir` | str | Required | Base directory for logs |
| `enable_tensorboard` | bool | True | Enable TensorBoard backend |
| `enable_wandb` | bool | False | Enable W&B backend |
| `enable_csv` | bool | True | Enable CSV backend |
| `flush_interval` | int | 1000 | Steps between flushes |
| `upload_artifacts` | bool | False | Upload logs to W&B |
| `wandb_config` | dict | None | W&B configuration |
| `wandb_tags` | list | [] | W&B run tags for filtering and grouping |

**W&B Tagging:**

Tags can be specified via config file (`logging.wandb.tags`) or CLI (`--tags`). CLI tags **merge** with config tags (they don't overwrite):

```bash
# Config file: tags: ["baseline", "pong"]
# CLI: --tags experiment-v2 --tags ablation
# Result: ["baseline", "pong", "experiment-v2", "ablation"]

python train_dqn.py --cfg configs/pong.yaml \
  --tags experiment-v2 --tags ablation
```

### Logging Methods

#### Per-Step Metrics

```python
logger.log_step(
    step=1000,
    loss=0.5,
    epsilon=0.95,
    learning_rate=0.00025,
    grad_norm=1.2,
    fps=120.0,
    extra_metrics={"custom_metric": 42.0}
)
```

**Logs to**:
- `train/loss`
- `train/epsilon`
- `train/learning_rate`
- `train/grad_norm`
- `train/fps`
- Custom metrics with `train/` prefix

#### Per-Episode Metrics

```python
logger.log_episode(
    step=1000,
    episode=5,
    episode_return=21.0,
    episode_length=500,
    extra_metrics={"lives_lost": 2}
)
```

**Logs to**:
- `episode/return`
- `episode/length`
- Episode number
- Custom metrics with `episode/` prefix

#### Evaluation Metrics

```python
logger.log_evaluation(
    step=250000,
    mean_return=35.5,
    std_return=5.2,
    min_return=18.0,
    max_return=45.0,
    num_episodes=10
)
```

**Logs to**:
- `eval/mean_return`
- `eval/std_return`
- `eval/min_return`
- `eval/max_return`
- `eval/num_episodes`

#### Q-Value Metrics

```python
logger.log_q_values(
    step=1000,
    q_mean=10.5,
    q_std=2.3,
    q_max=25.0
)
```

**Logs to**:
- `q_values/mean`
- `q_values/std`
- `q_values/max`

### Flushing and Artifacts

#### Periodic Flush

```python
# Automatically flushes every flush_interval steps
logger.maybe_flush_and_upload(step=1000)

# Force immediate flush
logger.maybe_flush_and_upload(step=1000, force=True)
```

#### W&B Artifact Uploads

Artifacts are uploaded at **1M step intervals** (1M, 2M, 3M, ...):

```python
# Automatically uploads at 1M, 2M, 3M steps
logger.maybe_flush_and_upload(step=1_000_000)

# Manual upload
logger.upload_logs_as_artifacts(
    step=1_000_000,
    metadata={"game": "pong", "seed": 42}
)
```

**Artifact Contents**:
- `csv/training_steps.csv` - Per-step training metrics (loss, epsilon, replay size, etc.)
- `csv/episodes.csv` - Per-episode metrics (returns, lengths)
- `config.yaml` - Full resolved configuration for reproducibility
- `meta.json` - Run metadata (environment, seed, timestamps)
- `eval/evaluations.csv` - Periodic evaluation summaries
- `eval/evaluations.jsonl` - Same evaluation data in JSONL format
- `videos/*.mp4` - Best episode recordings from evaluations
- Artifact metadata includes step, total_size_mb, and any custom fields

**Artifact Naming**: `training_logs_step_{step}`

**Guaranteed Final Upload**: When training completes, a forced artifact upload is performed via `maybe_flush_and_upload(..., force=True)`, ensuring all final results are captured in W&B even if the normal upload interval hasn't been reached. This final upload includes the complete evaluation history and all generated videos.

### Closing

```python
# Perform final flush and cleanup
logger.close()
```

---

## Standardized Metric Keys

All metrics follow a consistent naming scheme:

### Training Metrics (`train/`)

| Key | Description |
|-----|-------------|
| `train/loss` | TD loss |
| `train/epsilon` | Exploration rate |
| `train/learning_rate` | Optimizer LR |
| `train/grad_norm` | Gradient norm |
| `train/fps` | Training throughput |

### Episode Metrics (`episode/`)

| Key | Description |
|-----|-------------|
| `episode/return` | Episode return |
| `episode/length` | Episode length |
| `episode/number` | Episode index |

### Evaluation Metrics (`eval/`)

| Key | Description |
|-----|-------------|
| `eval/mean_return` | Mean eval return |
| `eval/median_return` | Median eval return |
| `eval/std_return` | Std eval return |
| `eval/min_return` | Min eval return |
| `eval/max_return` | Max eval return |
| `eval/mean_length` | Mean episode length |
| `eval/num_episodes` | # eval episodes |

### Q-Value Metrics (`q_values/`)

| Key | Description |
|-----|-------------|
| `q_values/mean` | Mean Q-value |
| `q_values/std` | Q-value std dev |
| `q_values/max` | Max Q-value |

---

## Implementation Decisions and Bug Fixes

### CSV Schema Management (Fixed 2025-11-14)

**Problem:** Dynamic CSV schemas caused failures when fieldnames changed between writes.

**Original Implementation:**
```python
# Bug: Dynamic schema based on first write
self._step_fieldnames = list(log_entry.keys())
writer.writeheader()
```

**Issue:**
- If later log entries had different fields, CSV writes would fail
- Error: `ValueError: dict contains fields not in fieldnames`

**Fixed Implementation:**
```python
# Define all expected fields upfront
self._step_fieldnames = [
    'step', 'epsilon', 'replay_size', 'fps',
    'loss', 'td_error', 'grad_norm', 'learning_rate',
    'loss_ma'  # moving average
]

# Filter log entries to only include defined fields
filtered_entry = {k: v for k, v in log_entry.items()
                  if k in self._step_fieldnames}
writer.writerow(filtered_entry)
```

**Rationale:**
- Predefined schema prevents runtime errors
- Missing fields are simply not written (graceful degradation)
- Extra fields are filtered out automatically
- Schema is self-documenting

**Location:** `src/training/metrics_logger.py:263-278`

**Related Commit:** `3e67aa6` - fix: Resolve all integration bugs and test failures

---

### Device Auto-Detection (Added 2025-11-14)

**Problem:** Training script used hardcoded device, failing when CUDA unavailable.

**Original Implementation:**
```python
if config.network.device == 'cuda' and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
```

**Issues:**
- No support for MPS (Apple Silicon GPU)
- No automatic fallback
- Unclear error messages when device unavailable

**Fixed Implementation:**
```python
def setup_device(config):
    """
    Setup compute device with automatic fallback.
    Checks availability in order: CUDA > MPS > CPU
    """
    requested_device = config.network.device

    # Try CUDA first
    if requested_device in ['auto', 'cuda']:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        elif requested_device == 'cuda':
            print("Warning: CUDA requested but not available. Falling back to CPU.")

    # Try MPS (Apple Silicon)
    if requested_device in ['auto', 'mps']:
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device (Apple Silicon GPU)")
            return device
        elif requested_device == 'mps':
            print("Warning: MPS requested but not available. Falling back to CPU.")

    # Default to CPU
    device = torch.device('cpu')
    print("Using CPU device")
    return device
```

**Features:**
- Auto-detection mode (`device: auto`)
- Explicit device requests with fallback warnings
- MPS support for Apple Silicon
- Clear user feedback on device selection

**Location:** `train_dqn.py:61-90`

**Related Commit:** `3e67aa6` - fix: Resolve all integration bugs and test failures

---

### API Parameter Naming Consistency

**Problem:** Inconsistent parameter naming across components caused integration failures.

**Fixed APIs:**

**1. EpsilonScheduler:**
```python
# Before:
EpsilonScheduler(start_epsilon=1.0, end_epsilon=0.1, decay_frames=1M)

# After:
EpsilonScheduler(epsilon_start=1.0, epsilon_end=0.1, decay_frames=1M)
```

**Rationale:** Prefix with parameter type for clarity and consistency with other schedulers.

**2. MetricsLogger:**
```python
# Before:
MetricsLogger(tensorboard_enabled=True, wandb_enabled=False, csv_enabled=True)

# After:
MetricsLogger(enable_tensorboard=True, enable_wandb=False, enable_csv=True)
```

**Rationale:** Verb-first naming (`enable_X`) is clearer than adjective-first (`X_enabled`).

**3. EvaluationScheduler:**
```python
# Before:
EvaluationScheduler(eval_every=250K, eval_enabled=True)

# After:
EvaluationScheduler(eval_interval=250K, num_episodes=10, eval_epsilon=0.05)
```

**Rationale:**
- `eval_interval` is more descriptive than `eval_every`
- All eval parameters now specified upfront (no hidden defaults)

**4. CheckpointManager.save_checkpoint:**
```python
# Before:
save_checkpoint(step=1M, model=model, optimizer=optimizer)

# After:
save_checkpoint(step=1M, episode=100, epsilon=0.5,
                online_model=model, target_model=target, optimizer=optimizer)
```

**Rationale:**
- Explicit online/target model separation
- Include episode and epsilon for better checkpoint context

**Location:** `train_dqn.py:146-178`

**Related Commit:** `3e67aa6` - fix: Resolve all integration bugs and test failures

---

### Missing Imports Resolution

**Problem:** Missing imports caused ImportErrors at runtime.

**Fixes:**

**1. schedulers.py:**
```python
# Added:
from .target_network import hard_update_target
```

**Why missed:** TargetNetworkUpdater uses hard_update_target but import was missing.

**2. metadata.py:**
```python
# Added:
import torch
```

**Why missed:** Module uses torch.cuda.is_available() but torch import was missing.

**Prevention:**
- Run full integration tests before commits
- Use static analysis tools (pyright, mypy)
- Import what you use, even for type checking

**Location:**
- `src/training/schedulers.py:8`
- `src/training/metadata.py:7`

**Related Commit:** `3e67aa6` - fix: Resolve all integration bugs and test failures

---

## CSV Log Layout

### Directory Structure

```
runs/pong_123/logs/
├── tensorboard/           # TensorBoard event files
│   └── events.out.tfevents.*
├── csv/
│   ├── training_steps.csv  # Per-step metrics
│   └── episodes.csv        # Per-episode metrics
└── wandb/                 # W&B local files (optional)
```

### training_steps.csv

| Column | Type | Description |
|--------|------|-------------|
| `step` | int | Training step |
| `loss` | float | TD loss |
| `epsilon` | float | Exploration rate |
| `learning_rate` | float | Optimizer LR |
| `grad_norm` | float | Gradient norm |
| `fps` | float | Training FPS |

### episodes.csv

| Column | Type | Description |
|--------|------|-------------|
| `step` | int | Training step |
| `episode` | int | Episode number |
| `return` | float | Episode return |
| `length` | int | Episode length |

---

## Plotting Script

### Basic Usage

```bash
# Plot from local CSV files
python scripts/plot_results.py \
  --episodes runs/pong_123/logs/csv/episodes.csv \
  --steps runs/pong_123/logs/csv/training_steps.csv \
  --output plots/pong \
  --game-name pong

# With custom smoothing and formats
python scripts/plot_results.py \
  --episodes runs/pong_123/logs/csv/episodes.csv \
  --smoothing 200 \
  --formats png pdf svg \
  --output plots/pong
```

### W&B Artifact Download

```bash
# Download and plot from W&B artifact
python scripts/plot_results.py \
  --wandb-project dqn-atari \
  --wandb-run abc123 \
  --wandb-artifact training_logs_step_1000000:latest \
  --output plots/pong \
  --game-name pong
```

### Multi-Seed Aggregation

```bash
# Aggregate multiple seed runs
python scripts/plot_results.py \
  --multi-seed runs/pong_seed1/logs/csv/episodes.csv \
               runs/pong_seed2/logs/csv/episodes.csv \
               runs/pong_seed3/logs/csv/episodes.csv \
  --output plots/pong_multi_seed \
  --game-name pong
```

**Outputs**:
- Mean curve with 95% confidence intervals
- Individual seed curves (optional)
- Statistics: mean, std, min, max, CI

### Plot Metadata Bundle

Every plot generation creates a **metadata bundle** for full reproducibility. The bundle consists of:

1. **Plot files** (PNG/PDF/SVG)
2. **Metadata JSON** (parameters and provenance)

#### Metadata JSON Schema

```json
{
  "game_name": "pong",
  "smoothing_window": 100,
  "smoothing_method": "moving_average",
  "formats": ["png", "pdf", "svg"],
  "commit_hash": "abc123def456789",
  "generated_at": "2025-11-14T12:34:56",
  "data_sources": {
    "episodes_csv": "results/logs/pong/pong_seed42/csv/episodes.csv",
    "steps_csv": "results/logs/pong/pong_seed42/csv/training_steps.csv"
  },
  "num_episodes": 1250,
  "num_steps": 500000,
  "total_frames": 10000000
}
```

**File Location**: `{output_dir}/{game_name}_plot_metadata.json`

**Example**: `plots/pong/pong_plot_metadata.json`

#### Why Metadata Matters

1. **Reproducibility**: Know exactly how plots were generated
2. **Provenance**: Track which data sources were used
3. **Version Control**: Git commit hash links plots to code version
4. **Comparison**: Compare smoothing parameters across experiments
5. **Publication**: Document figure generation for papers

#### Metadata Fields

| Field | Type | Description |
|-------|------|-------------|
| `game_name` | string | Game identifier (e.g., "pong") |
| `smoothing_window` | int | Window size for smoothing |
| `smoothing_method` | string | "moving_average" or "exponential" |
| `formats` | list[str] | Output formats generated |
| `commit_hash` | string | Git commit hash (or "unknown") |
| `generated_at` | string | ISO 8601 timestamp |
| `data_sources` | dict | Paths to input CSV files |
| `num_episodes` | int | Number of episodes in data |
| `num_steps` | int | Number of training steps |
| `total_frames` | int | Total environment frames |

#### Automatic Generation

Metadata is generated automatically by `plot_all_metrics()`:

```python
from scripts.plot_results import plot_all_metrics, load_csv_data

# Load data
episodes_data = load_csv_data("runs/pong/csv/episodes.csv")
steps_data = load_csv_data("runs/pong/csv/training_steps.csv")

# Generate plots with metadata
plot_files, metadata_file = plot_all_metrics(
    episodes_data=episodes_data,
    steps_data=steps_data,
    output_dir=Path("plots/pong"),
    game_name="pong",
    smoothing_window=100,
    formats=["png", "pdf"],
    save_metadata=True  # Default: True
)

print(f"Saved {len(plot_files)} plots")
print(f"Metadata: {metadata_file}")
```

#### Disabling Metadata

To disable metadata saving (not recommended):

```bash
python scripts/plot_results.py \
  --episodes data.csv \
  --output plots/ \
  --no-metadata
```

### W&B Artifact Upload Workflow

The W&B artifact system provides **versioned storage** for plots, logs, and metadata.

#### Artifact Types

1. **Log Artifacts** (`training_logs_step_{step}`)
   - Contains: CSV files (training_steps.csv, episodes.csv)
   - Upload interval: Every 1M steps (configurable)
   - Purpose: Incremental log backups

2. **Plot Artifacts** (`{game_name}_plots`)
   - Contains: Plot files (PNG/PDF/SVG) + metadata JSON
   - Upload: On-demand via `--upload-wandb` flag
   - Purpose: Share publication-quality figures

3. **Summary Artifacts** (`results_summary`)
   - Contains: Aggregated results tables (CSV/Markdown)
   - Upload: Via `export_results_table.py --upload-wandb`
   - Purpose: Compare multiple runs/seeds

#### Upload Plot Bundle to W&B

**Basic upload:**
```bash
python scripts/plot_results.py \
  --episodes results/logs/pong/run_123/csv/episodes.csv \
  --output plots/pong \
  --upload-wandb \
  --wandb-project dqn-atari \
  --wandb-upload-run abc123
```

**What gets uploaded:**
- `pong_episode_returns.png`
- `pong_training_loss.png`
- `pong_evaluation_scores.png`
- `pong_epsilon_schedule.png`
- `pong_plot_metadata.json`

**Artifact name**: `pong_plots` (or custom via script)

**Multi-format upload:**
```bash
python scripts/plot_results.py \
  --episodes results/logs/pong/run_123/csv/episodes.csv \
  --output plots/pong \
  --formats png pdf svg \
  --upload-wandb \
  --wandb-project dqn-atari \
  --wandb-upload-run abc123
```

Uploads all formats: PNG, PDF, SVG (total: 12 files + metadata)

#### Viewing Artifacts in W&B

1. Navigate to: `https://wandb.ai/<entity>/<project>/runs/<run_id>`
2. Click "Artifacts" tab
3. Find artifact (e.g., `pong_plots:v0`)
4. Download individual files or entire artifact

#### Download Artifacts

**Via W&B CLI:**
```bash
# Download specific artifact
wandb artifact get <entity>/<project>/pong_plots:latest

# Download to specific directory
wandb artifact get <entity>/<project>/pong_plots:v2 \
  --root plots/downloaded/
```

**Via Python API:**
```python
import wandb

api = wandb.Api()
artifact = api.artifact('entity/project/pong_plots:latest')
artifact_dir = artifact.download()

print(f"Downloaded to: {artifact_dir}")
```

**Via plot_results.py:**
```bash
# Download and plot from W&B artifact
python scripts/plot_results.py \
  --wandb-project dqn-atari \
  --wandb-run abc123 \
  --wandb-artifact training_logs_step_10000000:latest \
  --output plots/pong
```

#### Artifact Versioning

W&B automatically versions artifacts:

- First upload: `pong_plots:v0`
- Second upload: `pong_plots:v1`
- Third upload: `pong_plots:v2`
- Latest: `pong_plots:latest` (alias)

**View version history** in W&B dashboard to track plot evolution over time.

#### Metadata in Artifacts

Artifacts automatically include:

1. **Files**: Plot images + metadata JSON
2. **Artifact Metadata**: Upload timestamp, uploader
3. **Custom Metadata**: Can be added via script

**Example**: View metadata in W&B UI:
```
Artifact: pong_plots:v2
Files: 5
Size: 2.3 MB
Created: 2025-11-14 12:34:56
Metadata:
  - generated_at: 2025-11-14T12:34:56
  - smoothing_window: 100
  - commit_hash: abc123def
```

### Performance Options

```bash
# Downsample large logs for faster plotting
python scripts/plot_results.py \
  --episodes runs/pong_123/logs/csv/episodes.csv \
  --downsample 10000 \
  --warn-size-mb 100 \
  --output plots/pong
```

**Options**:
- `--downsample MAX_POINTS`: Downsample to N points (uniform sampling)
- `--warn-size-mb SIZE`: Warn if CSV exceeds SIZE MB (default: 50)

### Generated Plots

| Plot | Filename | Description |
|------|----------|-------------|
| Episode Returns | `{game}_episode_returns.{fmt}` | Returns over training |
| Training Loss | `{game}_training_loss.{fmt}` | TD loss progression |
| Evaluation Scores | `{game}_evaluation_scores.{fmt}` | Periodic eval results |
| Epsilon Schedule | `{game}_epsilon_schedule.{fmt}` | Exploration decay |

**Format**: 300 DPI, publication-quality (PNG/PDF/SVG)

---

## Results Table Exporter

### Usage

```bash
# Export all runs to summary tables
python scripts/export_results_table.py \
  --runs-dir experiments/dqn_atari/runs \
  --output results/summary

# Upload to W&B
python scripts/export_results_table.py \
  --runs-dir experiments/dqn_atari/runs \
  --output results/summary \
  --upload-wandb \
  --wandb-project dqn-atari
```

### Run Directory Scan

Scans for directories containing:
- `meta.json` - Game, seed, commit hash
- `eval/evaluations.csv` - Final performance
- `config.yaml` - Training config
- `checkpoints/*.pt` - Wall time estimation

### Markdown Output

```markdown
# DQN Training Results Summary

Generated: 2025-11-14 12:00:00

## Pong

| Run ID | Seed | Mean Return | Std | Frames | Wall Time (hrs) | Commit |
|--------|------|-------------|-----|--------|----------------|--------|
| pong_123 | 42 | 21.50 | 3.20 | 50,000,000 | 24.5 | abc123d |
| pong_456 | 43 | 20.80 | 3.50 | 50,000,000 | 24.2 | abc123d |

## Breakout

| Run ID | Seed | Mean Return | Std | Frames | Wall Time (hrs) | Commit |
...
```

### CSV Output

```csv
run_id,game,seed,mean_eval_return,std_eval_return,final_step,total_frames,wall_time_hours,commit_hash
pong_123,pong,42,21.50,3.20,50000000,50000000,24.5,abc123d
pong_456,pong,43,20.80,3.50,50000000,50000000,24.2,abc123d
```

---

## W&B Artifact Workflow Reference

Complete reference for W&B artifact uploads, naming conventions, and retrieval.

### Artifact Naming Conventions

All artifacts follow deterministic naming patterns for consistency:

#### 1. Training Log Artifacts

**Pattern**: `training_logs_step_{step}`

**Examples**:
- `training_logs_step_1000000` - Logs at 1M steps
- `training_logs_step_2000000` - Logs at 2M steps
- `training_logs_step_10000000` - Logs at 10M steps

**Contains**:
- `training_steps.csv` - Per-step metrics
- `episodes.csv` - Per-episode metrics
- Metadata: `{"step": 1000000, "game": "pong", "seed": 42}`

**Upload Schedule**: Every 1M steps (configurable via `artifact_upload_interval`)

#### 2. Plot Artifacts

**Pattern**: `{game_name}_plots` or custom name

**Examples**:
- `pong_plots:v0` - First upload
- `pong_plots:v1` - Second upload
- `pong_plots:latest` - Latest version (alias)

**Contains**:
- Plot files: `{game}_episode_returns.{fmt}`, `{game}_training_loss.{fmt}`, etc.
- Metadata: `{game}_plot_metadata.json`

**Upload**: On-demand via `--upload-wandb` flag

#### 3. Summary Table Artifacts

**Pattern**: `results_summary`

**Examples**:
- `results_summary:v0` - First export
- `results_summary:latest` - Latest export

**Contains**:
- `results_summary.csv` - All runs in CSV format
- `results_summary.md` - Markdown table grouped by game

**Upload**: Via `export_results_table.py --upload-wandb`

### Artifact Versioning

W&B automatically versions artifacts on each upload:

```
First upload:    pong_plots:v0
Second upload:   pong_plots:v1
Third upload:    pong_plots:v2
Latest alias:    pong_plots:latest
```

**Accessing versions**:
```bash
# Download latest version
wandb artifact get entity/project/pong_plots:latest

# Download specific version
wandb artifact get entity/project/pong_plots:v1

# Compare versions in W&B UI
# Navigate to: Artifacts → pong_plots → Version History
```

### Artifact Structure in W&B

**Organization**:
```
W&B Project: dqn-atari
├── Run: pong_seed42_20231114_120000
│   ├── Artifacts (Produced)
│   │   ├── training_logs_step_1000000:v0
│   │   ├── training_logs_step_2000000:v0
│   │   └── pong_plots:v0
│   └── Metrics
│       ├── train/loss
│       ├── episode/return
│       └── eval/mean_return
└── Run: pong_seed43_20231114_120001
    ├── Artifacts (Produced)
    │   ├── training_logs_step_1000000:v0
    │   └── pong_plots:v0
    └── Metrics
        └── ...
```

### Viewing Artifacts in W&B Dashboard

**Navigate to artifacts**:
1. Open project: `https://wandb.ai/<entity>/<project>`
2. Click specific run
3. Click "Artifacts" tab
4. View "Artifacts produced by this run"

**Artifact details view**:
- Files list (with download links)
- Metadata (upload time, size, uploader)
- Version history
- Usage (which runs consumed this artifact)

### Downloading Artifacts

#### Via W&B CLI

**Download latest:**
```bash
wandb artifact get <entity>/<project>/training_logs_step_1000000:latest
```

**Download to specific directory:**
```bash
wandb artifact get <entity>/<project>/pong_plots:v2 \
  --root plots/downloaded/
```

**Download all versions:**
```bash
for v in v0 v1 v2; do
  wandb artifact get <entity>/<project>/pong_plots:$v \
    --root plots/version_$v/
done
```

#### Via Python API

```python
import wandb

# Initialize API
api = wandb.Api()

# Download specific artifact
artifact = api.artifact('entity/project/pong_plots:latest')
artifact_dir = artifact.download()

print(f"Downloaded to: {artifact_dir}")

# Access files
import os
files = os.listdir(artifact_dir)
print(f"Files: {files}")

# Load metadata
import json
with open(os.path.join(artifact_dir, 'pong_plot_metadata.json')) as f:
    metadata = json.load(f)
    print(f"Commit: {metadata['commit_hash']}")
```

#### Via Plotting Script

```bash
# Download and re-generate plots from W&B logs
python scripts/plot_results.py \
  --wandb-project dqn-atari \
  --wandb-run abc123 \
  --wandb-artifact training_logs_step_10000000:latest \
  --output plots/pong
```

### Upload Workflow

#### Automatic Log Uploads (During Training)

**Configuration**:
```python
logger = MetricsLogger(
    log_dir="results/logs/pong/run_123",
    enable_wandb=True,
    upload_artifacts=True,
    wandb_config={
        "project": "dqn-atari",
        "entity": "my-team",
        "artifact_upload_interval": 1_000_000  # Upload every 1M steps
    }
)
```

**Triggers**:
- Automatic: Every `artifact_upload_interval` steps
- On demand: `logger.upload_logs_as_artifacts(step=1000000)`
- Final: On `logger.close()` (uploads final state)

**What happens**:
1. Check if step is upload interval (e.g., 1M, 2M, 3M)
2. Flush all backends to ensure CSVs are up-to-date
3. Collect CSV file paths
4. Create artifact with name `training_logs_step_{step}`
5. Upload to W&B (async, doesn't block training)
6. Continue training

#### Manual Plot Uploads (After Training)

**Upload plots from local files:**
```bash
python scripts/plot_results.py \
  --episodes results/logs/pong/run_123/csv/episodes.csv \
  --steps results/logs/pong/run_123/csv/training_steps.csv \
  --output plots/pong \
  --upload-wandb \
  --wandb-project dqn-atari \
  --wandb-upload-run abc123  # W&B run ID
```

**Upload results table:**
```bash
python scripts/export_results_table.py \
  --runs-dir results/logs/pong/ \
  --output results/summary \
  --upload-wandb \
  --wandb-project dqn-atari
```

### Artifact Metadata

Each artifact includes custom metadata for searchability:

**Training logs metadata**:
```json
{
  "step": 1000000,
  "game": "pong",
  "seed": 42,
  "commit_hash": "abc123def",
  "upload_time": "2025-11-14T12:34:56"
}
```

**Plot metadata**:
```json
{
  "generated_at": "2025-11-14T12:34:56",
  "smoothing_window": 100,
  "commit_hash": "abc123def",
  "num_episodes": 1250,
  "total_frames": 10000000
}
```

**Access metadata in W&B UI**:
- Artifact page → "Metadata" section
- Filter/search artifacts by metadata fields

### Best Practices

1. **Consistent naming**: Always use deterministic artifact names
2. **Version tracking**: Use version aliases (`latest`, `best`, `v0`, `v1`)
3. **Metadata**: Include game, seed, step, commit hash
4. **Upload frequency**: Balance storage costs vs. recovery granularity
   - Logs: Every 1M steps (default)
   - Plots: On-demand after training
   - Summaries: Once per experiment batch

5. **Storage management**:
   - W&B artifacts are versioned (old versions retained)
   - Use W&B storage quota monitoring
   - Delete old artifact versions if needed

6. **Offline workflows**:
   - Use `WANDB_MODE=offline` during training
   - Sync later with `wandb sync results/logs/pong/run_123/wandb/`

### Naming Convention Summary

| Artifact Type | Pattern | Example | Upload Trigger |
|---------------|---------|---------|----------------|
| Training Logs | `training_logs_step_{step}` | `training_logs_step_1000000` | Every 1M steps |
| Plots | `{game}_plots` | `pong_plots:v0` | On-demand |
| Multi-format Plots | `{game}_plots_{format}` | `pong_plots_all:v0` | On-demand |
| Summary Tables | `results_summary` | `results_summary:v0` | On-demand |
| Checkpoints* | `checkpoint_step_{step}` | `checkpoint_step_1000000` | Per checkpoint save |

*Checkpoint uploads require separate integration (not covered in this document)

### Viewing and Comparing Artifacts

**Compare runs via artifacts**:
1. Navigate to project → "Artifacts" tab
2. Select artifact type (e.g., "training_logs")
3. View which runs produced this artifact
4. Compare metadata across runs

**Search artifacts**:
```python
import wandb

api = wandb.Api()

# Find all log artifacts at 1M steps
artifacts = api.artifacts(
    type_name="logs",
    name="training_logs_step_1000000"
)

for artifact in artifacts:
    print(f"Run: {artifact.logged_by().name}")
    print(f"Metadata: {artifact.metadata}")
```

### Troubleshooting

**Artifact upload fails**:
- Check network connection
- Verify W&B login: `wandb login --verify`
- Check file sizes (>1GB may timeout)
- Use `WANDB_SILENT=true` to reduce log noise

**Artifact not found**:
- Verify artifact name (case-sensitive)
- Check W&B project/entity name
- Ensure artifact was actually uploaded (check run page)

**Version conflicts**:
- W&B auto-increments versions (v0, v1, v2)
- Use `:latest` alias for most recent
- Check version history in W&B UI

---

## Integration with Training Loop

### Example: DQN Trainer

```python
from src.training.metrics_logger import MetricsLogger

class DQNTrainer:
    def __init__(self, config):
        # Initialize logger
        self.logger = MetricsLogger(
            log_dir=config.log_dir,
            enable_tensorboard=config.enable_tensorboard,
            enable_wandb=config.enable_wandb,
            enable_csv=True,
            flush_interval=1000,
            upload_artifacts=config.upload_artifacts,
            wandb_config={
                "project": config.wandb_project,
                "entity": config.wandb_entity,
                "run_id": config.run_id,
                "config": config.to_dict()
            } if config.enable_wandb else None
        )

    def train_step(self, step):
        # Perform training step
        loss = self._compute_loss()

        # Log step metrics
        self.logger.log_step(
            step=step,
            loss=loss,
            epsilon=self.epsilon,
            learning_rate=self.optimizer.param_groups[0]['lr'],
            grad_norm=self._get_grad_norm(),
            fps=self.fps
        )

        # Periodic flush and upload
        self.logger.maybe_flush_and_upload(step)

    def on_episode_end(self, step, episode, episode_return, episode_length):
        # Log episode metrics
        self.logger.log_episode(
            step=step,
            episode=episode,
            episode_return=episode_return,
            episode_length=episode_length
        )

    def evaluate(self, step):
        # Run evaluation
        returns = [self._run_episode(eval=True) for _ in range(10)]

        # Log evaluation metrics
        self.logger.log_evaluation(
            step=step,
            mean_return=np.mean(returns),
            std_return=np.std(returns),
            min_return=np.min(returns),
            max_return=np.max(returns),
            num_episodes=len(returns)
        )

    def close(self):
        # Final flush and cleanup
        self.logger.close()
```

---

## Performance Considerations

### Large Log Files

**Problem**: Training runs can generate multi-GB CSV files.

**Solutions**:
1. **Downsampling**: Use `--downsample` flag to reduce points
2. **Periodic Cleanup**: Archive old logs after artifact upload
3. **Selective Logging**: Log episodes less frequently

### W&B Upload Bandwidth

**Problem**: Large artifact uploads can be slow.

**Solutions**:
1. **Upload Intervals**: Default 1M steps (adjust as needed)
2. **Artifact Versioning**: Only upload incremental changes
3. **Local-First**: Always save CSV locally first

### Memory Usage

**Problem**: Loading large CSVs into memory.

**Solutions**:
1. **Streaming**: Read CSV in chunks (not yet implemented)
2. **Downsampling**: Reduce data before plotting
3. **File Size Warnings**: Alert user to large files

---

## Best Practices

### Logging Frequency

| Metric Type | Frequency | Rationale |
|-------------|-----------|-----------|
| Training Loss | Every step | Detailed learning curve |
| Episode Return | Every episode | Complete episode history |
| Evaluation | Every 250K steps | Periodic checkpoints |
| Q-Values | Every 1K steps | Representative sample |

### Smoothing Windows

| Plot Type | Window Size | Method |
|-----------|-------------|--------|
| Episode Returns | 100 | Moving Average |
| Training Loss | 100 | Moving Average |
| Evaluation | None | Raw data |
| Epsilon | None | Raw data |

### Artifact Strategy

1. **Local CSV**: Always enabled (minimal overhead)
2. **TensorBoard**: Enabled for local monitoring
3. **W&B**: Enabled for cloud storage and collaboration
4. **Artifacts**: Upload at 1M step intervals

### Multi-Seed Analysis

1. Run 3-5 seeds per game/config
2. Use consistent hyperparameters
3. Aggregate with `--multi-seed` flag
4. Report mean ± 95% CI in papers

---

## Troubleshooting

### Backend Failure Handling

#### What Happens When a Backend Fails?

The logging system is designed to **fail gracefully**. If one backend fails, the others continue operating:

```python
# Example: W&B fails due to network error
logger.log_step(step=1000, loss=0.5, ...)
    ├─→ TensorBoard: DONE Success
    ├─→ W&B: FAIL Network timeout (logs warning, continues)
    └─→ CSV: DONE Success
```

**Behavior**:
1. **Exception Handling**: Each backend wraps operations in try/except
2. **Warning Messages**: Failures print warnings to stderr but don't raise exceptions
3. **Disable on Repeated Failures**: After 10 consecutive failures, backend auto-disables
4. **CSV Always Works**: CSV backend has no external dependencies

#### TensorBoard Backend Issues

**Symptoms**: No TensorBoard logs / `tensorboard` command fails.

**Causes**:
- TensorBoard not installed
- Corrupted event files
- Wrong log directory

**Fix**:
```bash
# Install TensorBoard
pip install tensorboard

# Verify files exist
ls -la results/logs/pong/pong_seed42/tensorboard/

# Launch TensorBoard
tensorboard --logdir results/logs/pong/

# If corrupted, delete and restart training
rm -rf results/logs/pong/pong_seed42/tensorboard/*
```

**Disabling TensorBoard**:
```python
logger = MetricsLogger(
    log_dir="results/logs/pong/run_123",
    enable_tensorboard=False  # Disable if not needed
)
```

#### W&B Backend Issues

**Symptoms**: No W&B logs appear in dashboard.

**Common Causes & Fixes**:

1. **Not Installed**:
```bash
pip install wandb
```

2. **Not Logged In**:
```bash
wandb login
# Or set API key directly:
export WANDB_API_KEY="your_key_here"
```

3. **Network Issues** (transient):
- W&B retries automatically
- If persistent, check firewall/proxy settings
- Consider using `WANDB_MODE=offline` (see below)

4. **Offline Mode** (no internet):
```bash
# Run in offline mode
export WANDB_MODE=offline

# Later, sync offline runs
wandb sync results/logs/pong/pong_seed42/wandb/
```

5. **Wrong Project/Entity**:
```python
# Verify configuration
wandb_config = {
    "project": "dqn-atari",     # Must exist in W&B
    "entity": "my-team",         # Optional; your username by default
    "run_id": "pong_seed42"
}
```

6. **Artifact Upload Failures**:
- Large files may timeout (>100MB warning is issued)
- Check network bandwidth
- Consider increasing `artifact_upload_interval`

**Disabling W&B**:
```python
# In code:
logger = MetricsLogger(
    log_dir="results/logs/pong/run_123",
    enable_wandb=False
)

# Or via environment variable:
export WANDB_DISABLED=true
```

#### CSV Backend Issues

**Symptoms**: Missing CSV files / corrupted data.

**Causes**:
- Disk full
- Permission errors
- Process killed mid-write

**Fix**:
```bash
# Check disk space
df -h results/

# Check permissions
ls -la results/logs/pong/pong_seed42/csv/

# Check file integrity (should have header + data rows)
head -5 results/logs/pong/pong_seed42/csv/training_steps.csv

# If corrupted, delete and restart
rm results/logs/pong/pong_seed42/csv/*.csv
```

**Recovery**:
- If training crashes mid-write, CSVs may be incomplete
- Use `flush_interval=1000` (default) to minimize data loss
- Consider more frequent flushing for critical experiments:
  ```python
  logger = MetricsLogger(
      log_dir="...",
      flush_interval=500  # Flush more frequently
  )
  ```

### Missing Plots

**Symptoms**: Plots not generated.

**Causes**:
- CSV files not found
- Empty CSV files
- Missing required columns

**Diagnosis**:
```bash
# Check CSV exists
ls -la results/logs/pong/pong_seed42/csv/episodes.csv

# Check file is not empty
wc -l results/logs/pong/pong_seed42/csv/episodes.csv

# Check columns
head -1 results/logs/pong/pong_seed42/csv/episodes.csv
# Should see: step,episode,return,length
```

**Fix**:
- Verify CSV paths in command
- Ensure training ran long enough to generate episodes
- Check for correct column names (case-sensitive)

### Large File Warnings

**Symptoms**: "Large CSV file (XX.X MB). Consider using --downsample for faster plotting."

**When This Happens**:
- Long training runs (50M+ frames)
- High logging frequency
- Multiple seeds aggregated

**Fix**:
```bash
# Downsample to 10K points (faster plotting, minimal information loss)
python scripts/plot_results.py \
  --episodes large_file.csv \
  --downsample 10000 \
  --output plots/

# Adjust warning threshold
python scripts/plot_results.py \
  --episodes large_file.csv \
  --warn-size-mb 200 \  # Only warn if >200 MB
  --output plots/
```

**Performance Impact**:
- Files >100 MB: Plotting may take 10-30 seconds
- Files >500 MB: May cause memory issues; use `--downsample`
- Files >1 GB: Strongly recommend downsampling

### W&B Artifact Upload Slow

**Symptoms**: "Warning: Uploading large artifact (XXX.X MB). This may take some time."

**Causes**:
- Large CSV files (>100 MB)
- Slow network connection
- Frequent uploads

**Fix**:
```python
# Increase upload interval (default: 1M steps)
logger = MetricsLogger(
    log_dir="...",
    upload_artifacts=True,
    # Upload every 5M steps instead of 1M
    wandb_config={
        "artifact_upload_interval": 5_000_000
    }
)
```

**Alternative**:
- Disable automatic uploads during training
- Upload manually after training completes:
  ```bash
  python scripts/export_results_table.py \
    --runs-dir results/logs/pong/ \
    --output results/summary \
    --upload-wandb \
    --wandb-project dqn-atari
  ```

### Permission Denied Errors

**Symptoms**: `PermissionError: [Errno 13] Permission denied: 'results/logs/...'`

**Causes**:
- Running as different user
- Incorrect file permissions
- NFS/shared filesystem issues

**Fix**:
```bash
# Check ownership
ls -la results/logs/pong/

# Fix permissions
chmod -R u+rwX results/logs/pong/

# On NFS: disable file locking for CSV (use with caution)
# Only if above fixes don't work
```

---

## Testing

### Metrics Logger Tests

```bash
# Run full test suite
pytest tests/test_metrics_logger.py -v

# Test specific backend
pytest tests/test_metrics_logger.py::test_csv_backend_initialization -v
```

**Coverage**: 28 tests covering:
- TensorBoard backend
- W&B backend (with graceful degradation)
- CSV backend
- MetricsLogger interface
- Periodic flush
- Artifact uploads

### Plotting Tests

```bash
# Run plotting tests
pytest tests/test_plot_results.py -v

# Test specific plot type
pytest tests/test_plot_results.py::test_plot_episode_returns -v
```

**Coverage**: 20 tests covering:
- CSV loading
- Data smoothing
- All plot types
- Multi-format output
- Metadata saving

---

## References

### Related Documentation

- [Training Loop Runtime](training_loop_runtime.md) - Training loop integration
- [Checkpointing](checkpointing.md) - Checkpoint metadata
- [Evaluation Harness](eval_harness.md) - Evaluation logging
- [Config & CLI](config_cli.md) - Configuration parameters

### External Resources

- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)
- [Weights & Biases Documentation](https://docs.wandb.ai/)
- [Matplotlib Documentation](https://matplotlib.org/stable/contents.html)

---

## Summary

The logging and plotting pipeline provides:

DONE **Multi-backend logging** - TensorBoard, W&B, CSV
DONE **Standardized metrics** - Consistent naming across backends
DONE **Periodic flush** - Automatic buffer flushing
DONE **W&B artifacts** - Incremental log uploads
DONE **Publication plots** - High-quality figures (300 DPI)
DONE **Multi-seed analysis** - Statistical aggregation with CI
DONE **Results tables** - Markdown/CSV summaries
DONE **Performance safeguards** - Downsampling and warnings

**Next Steps**: Integrate with training loop and run experiments!
