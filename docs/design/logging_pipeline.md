# Logging & Plotting Pipeline

**Status**: ✅ Complete
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

### Data Flow

```
Training Loop
    ↓
MetricsLogger.log_step()     # Per-step metrics
MetricsLogger.log_episode()  # Per-episode metrics
MetricsLogger.log_evaluation() # Evaluation metrics
    ↓
[TensorBoard] [W&B] [CSV]    # Multi-backend logging
    ↓
Periodic Flush (every 1K steps)
    ↓
Artifact Upload (every 1M steps) → W&B
    ↓
plot_results.py              # Generate plots
export_results_table.py      # Summary tables
```

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
- `training_steps.csv` - Per-step metrics
- `episodes.csv` - Per-episode metrics
- Metadata with step, timestamp, custom fields

**Artifact Naming**: `training_logs_step_{step}`

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
| `eval/std_return` | Std eval return |
| `eval/min_return` | Min eval return |
| `eval/max_return` | Max eval return |
| `eval/num_episodes` | # eval episodes |

### Q-Value Metrics (`q_values/`)

| Key | Description |
|-----|-------------|
| `q_values/mean` | Mean Q-value |
| `q_values/std` | Q-value std dev |
| `q_values/max` | Max Q-value |

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

### Plot Metadata

Plots automatically include metadata for reproducibility:

```json
{
  "game_name": "pong",
  "smoothing_window": 100,
  "formats": ["png", "pdf"],
  "commit_hash": "abc123def",
  "generated_at": "2025-11-14T12:00:00"
}
```

Saved to: `{output_dir}/{game_name}_plot_metadata.json`

### W&B Plot Upload

```bash
# Upload plots as W&B artifacts
python scripts/plot_results.py \
  --episodes runs/pong_123/logs/csv/episodes.csv \
  --output plots/pong \
  --upload-wandb \
  --wandb-project dqn-atari \
  --wandb-upload-run abc123
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

### W&B Not Logging

**Symptoms**: No W&B logs appear in dashboard.

**Causes**:
- W&B package not installed
- Not logged in (`wandb login`)
- Network issues

**Fix**:
```bash
pip install wandb
wandb login
```

### Missing Plots

**Symptoms**: Plots not generated.

**Causes**:
- CSV files not found
- Empty CSV files
- Missing required columns

**Fix**:
- Check CSV paths
- Verify `step`, `return`, `loss` columns exist

### Large File Warnings

**Symptoms**: "Large CSV file" warning.

**Fix**:
```bash
# Downsample to 10K points
python scripts/plot_results.py \
  --episodes large_file.csv \
  --downsample 10000 \
  --output plots/
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

✅ **Multi-backend logging** - TensorBoard, W&B, CSV
✅ **Standardized metrics** - Consistent naming across backends
✅ **Periodic flush** - Automatic buffer flushing
✅ **W&B artifacts** - Incremental log uploads
✅ **Publication plots** - High-quality figures (300 DPI)
✅ **Multi-seed analysis** - Statistical aggregation with CI
✅ **Results tables** - Markdown/CSV summaries
✅ **Performance safeguards** - Downsampling and warnings

**Next Steps**: Integrate with training loop and run experiments!
