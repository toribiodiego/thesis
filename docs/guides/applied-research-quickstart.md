# Applied Research Quickstart

> **Status**: ACTIVE | Minimal workflow for validating DQN foundation and running thesis experiments.
> **Purpose**: Fast-track guide from setup to validation to production training runs.

## Purpose

This guide provides a streamlined workflow for applied research tasks: validating the DQN implementation foundation, running preliminary experiments, and launching production training runs for thesis results.

Unlike [Quick Start](quick-start.md) (detailed step-by-step setup), this focuses on the research workflow after initial setup is complete.

## Inputs/Outputs

**Inputs** (prerequisites):
- Completed setup ([Quick Start](quick-start.md) steps 1-2)
- Understanding of DQN architecture ([Architecture](architecture.md))
- Access to compute resources (CPU or GPU)

**Outputs** (deliverables):
- Validated implementation (smoke tests passed)
- Baseline training runs for thesis
- Performance metrics and plots
- Reproducible experiment artifacts

---

## Workflow Overview

```
1. Validate Foundation
   └─ Smoke test → Unit tests → Performance baseline

2. Run Baseline Experiments
   └─ Single-seed test → Multi-seed production → Checkpoints

3. Analyze Results
   └─ Generate plots → Export metrics → Compare to paper

4. Document Findings
   └─ Update reports → Archive artifacts → Record decisions
```

**Est. time**: 1 hour (validation) + days-weeks (training) + hours (analysis)

---

## Phase 1: Validate Foundation

### 1.1 Smoke Test (5 minutes)

Verify end-to-end pipeline before committing to long training runs:

```bash
# Fast validation: 200K frames (~5-10 minutes)
./experiments/dqn_atari/scripts/smoke_test.sh
```

**What it tests**:
- Environment creation and preprocessing
- Model initialization and forward pass
- Replay buffer sampling
- Training step execution
- Logging and checkpointing
- Evaluation routine

**Success criteria**:
- Script completes without errors
- Loss decreases from initial values
- Checkpoints saved at expected intervals
- CSV logs created with correct columns

**If smoke test fails**: See [Troubleshooting](troubleshooting.md)

### 1.2 Unit Tests (5-10 minutes)

Run comprehensive test suite to verify components:

```bash
# All tests (335+ tests)
pytest tests/ -v

# Faster: Skip slow integration tests
pytest tests/ -v -m "not slow"

# Specific components
pytest tests/test_dqn_model.py -v
pytest tests/test_replay_buffer.py -v
pytest tests/test_dqn_trainer.py -v
```

**Success criteria**: All tests pass (green output)

**Common failures**:
- ROM not found → Run `./setup/setup_roms.sh`
- Import errors → Check `pip install -r requirements.txt`
- GPU tests fail → Expected if no CUDA available

### 1.3 Performance Baseline (30 minutes)

Measure throughput to estimate training time:

```bash
# Run 1M frames and measure FPS
python train_dqn.py \
  experiments/dqn_atari/configs/pong.yaml \
  --set training.total_frames=1000000 \
  --set logging.log_every=10000
```

**Extract FPS**:
```bash
# From training output
grep "FPS" <run_dir>/logs/training.log | tail -5
```

**Estimate total time**:
- **Pong (10M frames)**: 10M ÷ FPS ÷ 3600 = hours
- **Breakout (50M frames)**: 50M ÷ FPS ÷ 3600 = hours

**Typical FPS**:
- CPU (M1/M2): 100-200 FPS → Pong ~14-28 hours
- CPU (Intel): 50-100 FPS → Pong ~28-56 hours
- GPU (RTX 3080): 1000-2000 FPS → Pong ~1.4-2.8 hours

**Document baseline**: Save system info and FPS to `results/summary/performance_baseline.txt`

---

## Phase 2: Run Baseline Experiments

### 2.1 Single-Seed Test Run (hours-days)

Validate learning on one seed before multi-seed production:

```bash
# Pong single seed (10M frames)
./scripts/reproduce_dqn.sh --game pong --seed 42
```

**Monitor progress**:
```bash
# Check latest eval results
tail -5 experiments/dqn_atari/runs/pong_42_*/csv/evaluation.csv

# Watch FPS and loss
tail -f experiments/dqn_atari/runs/pong_42_*/logs/training.log
```

**Success criteria** (after 10M frames):
- Mean eval return > 18 (≥90% of paper score 20)
- Loss stabilizes (no divergence)
- Q-values are reasonable (not NaN or exploding)
- Checkpoints saved every 1M frames

**If learning fails**:
- Check epsilon decay (should reach 0.1 after 1M frames)
- Verify reward clipping enabled
- Inspect replay buffer (50K warmup before training)
- Review [Stability Notes](../reference/stability-notes.md)

### 2.2 Multi-Seed Production Runs

For statistical robustness, run 3 seeds:

```bash
# Sequential (one at a time)
for seed in 42 123 456; do
    ./scripts/reproduce_dqn.sh --game pong --seed $seed
done

# Or parallel (if multiple GPUs/machines available)
# Launch each in separate terminal/screen session
```

**Parallel execution tip**:
```bash
# Terminal 1
./scripts/reproduce_dqn.sh --game pong --seed 42

# Terminal 2
./scripts/reproduce_dqn.sh --game pong --seed 123

# Terminal 3
./scripts/reproduce_dqn.sh --game pong --seed 456
```

**Checkpoint management**:
- Each run creates ~5-10 checkpoints (1M frame intervals)
- Disk usage: ~500MB-1GB per checkpoint
- Best checkpoint saved automatically
- Use W&B artifacts for cloud backup (enabled by default)

### 2.3 Extended Games (Optional)

If GPU available and time permits:

```bash
# Breakout (50M frames, ~5x longer than Pong)
./scripts/reproduce_dqn.sh --game breakout --seed 42

# Beam Rider (50M frames)
./scripts/reproduce_dqn.sh --game beam_rider --seed 42
```

**Note**: Breakout/Beam Rider each take 150-300 hours on CPU, 25-50 hours on GPU per seed. Consider deferring if GPU not available.

---

## Phase 3: Analyze Results

### 3.1 Generate Plots

Create learning curves and diagnostics:

```bash
# Single run plots
python scripts/plot_results.py \
  --episodes experiments/dqn_atari/runs/pong_42_*/csv/episodes.csv \
  --output results/plots/pong_42

# Multi-seed aggregation (manual for now)
# TODO: Implement multi-seed plotting script
```

**Expected outputs**:
- `returns.png` - Episode returns over time
- `episode_length.png` - Episode lengths
- `loss.png` - Training loss curve
- `q_values.png` - Q-value evolution

### 3.2 Export Metrics

Create summary tables for thesis:

```bash
# Aggregate all runs
python scripts/export_results_table.py \
  --runs-dir experiments/dqn_atari/runs \
  --output results/summary/metrics.csv
```

**Output format**:
```
game,seed,final_mean_return,final_std,paper_score,percent_of_paper
pong,42,19.5,0.8,20,97.5
pong,123,20.2,0.5,20,101.0
pong,456,18.8,1.2,20,94.0
```

### 3.3 Compare to Paper

Validate reproduction quality:

```python
import pandas as pd
import numpy as np

# Load results
df = pd.read_csv("results/summary/metrics.csv")

# Aggregate seeds
pong_results = df[df['game'] == 'pong']
mean_score = pong_results['final_mean_return'].mean()
std_score = pong_results['final_mean_return'].std()
paper_score = 20

# Statistical test
from scipy import stats
ci_95 = 1.96 * std_score / np.sqrt(len(pong_results))

print(f"Our score: {mean_score:.1f} ± {ci_95:.1f} (95% CI)")
print(f"Paper score: {paper_score}")
print(f"Achievement: {100 * mean_score / paper_score:.1f}%")

# Success if ≥90% of paper score
success = mean_score >= 0.9 * paper_score
print(f"Reproduction: {'SUCCESS' if success else 'NEEDS INVESTIGATION'}")
```

---

## Phase 4: Document Findings

### 4.1 Update Reports

Record results in thesis documentation:

1. **Add run metadata** to [Reporting Requirements](../reports/reporting-requirements.md)
   - Mark completed items (e.g., Priority 1.1 Pong 3-seed runs)
   - Update timeline estimates based on actual FPS

2. **Summarize findings** in [DQN Results](../reports/report-dqn-results.md)
   - Final scores with confidence intervals
   - Learning curves analysis
   - Comparison to paper scores

3. **Document issues** in [Stability Notes](../reference/stability-notes.md)
   - Any hyperparameter adjustments
   - Observed failure modes
   - Environment-specific quirks

### 4.2 Archive Artifacts

Preserve reproducibility:

```bash
# Organize run outputs
mkdir -p results/final_runs/pong
cp -r experiments/dqn_atari/runs/pong_*/ results/final_runs/pong/

# Save system provenance
cp experiments/dqn_atari/runs/pong_42_*/system_info.txt \
   results/summary/pong_system_info.txt

# Archive to W&B (if enabled)
# Artifacts uploaded automatically during training
```

**What to keep**:
- Final checkpoints (best model per seed)
- Evaluation CSVs
- Training plots
- System info and configs
- Git commit hash

**What to exclude** (gitignored):
- Intermediate checkpoints (every 1M frames)
- Full replay buffer snapshots
- Video recordings (unless needed for presentation)

### 4.3 Record Decisions

Update research log:

```bash
# Experiment notes
vim experiments/dqn_atari/notes.md
```

**Document**:
- Why certain hyperparameters were chosen/changed
- Any deviations from original DQN paper
- Follow-up experiments planned
- Issues encountered and resolutions

---

## Common Workflows

### Quick Validation (1 hour)
```bash
./experiments/dqn_atari/scripts/smoke_test.sh
pytest tests/ -v -m "not slow"
```

### Test Run (1 day)
```bash
./scripts/reproduce_dqn.sh --game pong --seed 42 --frames 1000000
python scripts/plot_results.py --episodes <run>/csv/episodes.csv
```

### Production Run (1-4 weeks)
```bash
# Multi-seed Pong
for seed in 42 123 456; do
    ./scripts/reproduce_dqn.sh --game pong --seed $seed
done

# Analysis
python scripts/export_results_table.py --runs-dir experiments/dqn_atari/runs
python scripts/plot_results.py --episodes <runs>/csv/episodes.csv
```

### Full Suite (2-6 months)
```bash
# All games, all seeds
for game in pong breakout beam_rider; do
    for seed in 42 123 456; do
        ./scripts/reproduce_dqn.sh --game $game --seed $seed
    done
done
```

---

## Troubleshooting Quick Reference

| Issue | Solution |
|-------|----------|
| Smoke test fails | Check [Troubleshooting](troubleshooting.md) |
| Unit tests fail | Verify ROMs installed, dependencies up to date |
| Low FPS (<50) | Check CPU usage, consider GPU, close background apps |
| No learning after 1M frames | Check epsilon decay, reward clipping, replay warmup |
| Loss diverges (NaN) | Reduce learning rate, check gradient clipping |
| Out of memory | Reduce replay buffer size in config |
| W&B login required | `wandb login` or `export WANDB_API_KEY=<key>` |
| Checkpoints too large | Expected (~500MB each), exclude intermediate ones |

---

## Next Steps

**After validation**:
- Launch production runs for thesis (Priority 1 from [Reporting Requirements](../reports/reporting-requirements.md))
- Monitor W&B dashboard for progress
- Plan ablation studies (Priority 3)

**For deeper understanding**:
- Read component specs in [reference/](../reference/)
- Review [DQN 2013 paper notes](../research/papers/dqn-2013-notes.md)
- Study [Architecture Overview](architecture.md)

**For advanced topics**:
- GPU acceleration: [Colab Guide](colab-guide.md)
- Reproducibility: [Checkpointing](../reference/checkpointing.md)
- Results analysis: [Results Comparison](../reports/report-results-comparison.md)

---

## Related Documents

- [Quick Start](quick-start.md) - Detailed step-by-step setup guide
- [Architecture](architecture.md) - System design overview
- [Workflows](workflows.md) - Task-oriented guides
- [Reporting Requirements](../reports/reporting-requirements.md) - Prioritized thesis backlog
- [Reproduce Recipe](../plans/plan-reproduce-recipe.md) - Detailed reproduction commands
- [Troubleshooting](troubleshooting.md) - Problem diagnosis and fixes

---

**Last Updated**: 2025-12-23
