# Reporting Pipeline

> **Status**: REFERENCE | Technical specification for result generation and analysis workflow.
> **Purpose**: Document the end-to-end pipeline from training artifacts to thesis-ready outputs.

## Purpose

This document specifies the reporting pipeline that transforms raw training artifacts (logs, checkpoints, configs) into thesis-ready outputs (plots, tables, metrics). It covers inputs, outputs, processing steps, and common failure modes.

For operational usage, see [Applied Research Quickstart](../guides/applied-research-quickstart.md). For logging implementation, see [Logging Pipeline](logging-pipeline.md).

## Inputs/Outputs

**Inputs** (prerequisites):
- Completed training runs in `experiments/dqn_atari/runs/`
- CSV logs (episodes, training_steps, evaluations)
- Checkpoints and config files
- System provenance (git hash, environment info)

**Outputs** (deliverables):
- Publication-quality plots (PNG, PDF, SVG)
- Summary metrics tables (CSV, Markdown)
- Statistical analyses (confidence intervals, paper comparisons)
- Thesis-ready figures and tables

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    REPORTING PIPELINE                        │
└─────────────────────────────────────────────────────────────┘

1. DATA COLLECTION
   experiments/dqn_atari/runs/<game>_<seed>_<timestamp>/
   ├── csv/episodes.csv           # Per-episode metrics
   ├── csv/training_steps.csv     # Per-step metrics
   ├── csv/evaluations.csv        # Periodic eval results
   ├── config.yaml                # Training configuration
   ├── meta.json                  # Run metadata
   └── checkpoints/best_model.pt  # Best checkpoint

2. VISUALIZATION (plot_results.py)
   Input:  episodes.csv, evaluations.csv
   Output: results/plots/<run>/
           ├── returns.png         # Episode returns over time
           ├── episode_length.png  # Episode lengths
           ├── loss.png           # Training loss
           └── q_values.png       # Q-value evolution

3. AGGREGATION (export_results_table.py)
   Input:  Multiple runs (3+ seeds)
   Output: results/summary/metrics.csv
           game,seed,final_return,std,paper_score,percent

4. ANALYSIS (analyze_results.py)
   Input:  metrics.csv
   Output: Statistical tests, confidence intervals, paper comparison

5. THESIS INTEGRATION
   Input:  Plots, tables, W&B URLs
   Output: Markdown sections, LaTeX figures, result summaries
```

---

## Data Collection

### Input Artifacts

Training runs produce standardized artifacts:

**Required files**:
- `csv/episodes.csv` - Episode-level metrics (return, length, epsilon)
- `csv/evaluations.csv` - Periodic evaluation results (mean, std, min, max)
- `config.yaml` - Frozen configuration snapshot
- `meta.json` - Run metadata (seed, git hash, timestamp)

**Optional files**:
- `csv/training_steps.csv` - Step-level metrics (loss, FPS, grad norm)
- `checkpoints/*.pt` - Model checkpoints for analysis
- `videos/*.mp4` - Gameplay recordings
- `system_info.txt` - Hardware/software provenance

### Artifact Locations

```
experiments/dqn_atari/runs/
├── pong_42_20251115_230409/    # Seed 42
│   ├── csv/
│   │   ├── episodes.csv
│   │   ├── training_steps.csv
│   │   └── evaluations.csv
│   ├── config.yaml
│   └── meta.json
├── pong_123_20251116_010205/   # Seed 123
└── pong_456_20251116_143012/   # Seed 456
```

**Discovery pattern**:
```bash
# Find all Pong runs
find experiments/dqn_atari/runs -name "pong_*" -type d

# Find evaluation CSVs
find experiments/dqn_atari/runs -name "evaluations.csv"
```

---

## Visualization (plot_results.py)

### Purpose

Generate publication-quality plots from training logs.

### Inputs

```bash
python scripts/plot_results.py \
  --episodes experiments/dqn_atari/runs/pong_42_*/csv/episodes.csv \
  --output results/plots/pong_42
```

**Required**:
- `--episodes` - Path to episodes.csv
- `--output` - Output directory for plots

**Optional**:
- `--training-steps` - Path to training_steps.csv (for loss/FPS plots)
- `--evaluations` - Path to evaluations.csv (for eval metrics)
- `--format` - Output format (png, pdf, svg) [default: png]
- `--smooth` - Smoothing window for curves [default: 10]

### Outputs

**Generated files**:
```
results/plots/pong_42/
├── returns.png           # Episode returns over time
├── episode_length.png    # Episode durations
├── loss.png             # Training loss (if training_steps.csv provided)
├── q_values.png         # Q-value estimates (if available)
└── metadata.json        # Plot metadata (smoothing, commit hash)
```

**Plot specifications**:
- DPI: 300 (publication quality)
- Size: 8x6 inches (default)
- Format: PNG, PDF, or SVG
- Smoothing: Exponential moving average (configurable window)
- Metadata: Embedded in metadata.json

### Failure Modes

| Error | Cause | Solution |
|-------|-------|----------|
| `FileNotFoundError: episodes.csv` | Path incorrect or run incomplete | Verify run completed, check path |
| `ValueError: Empty CSV` | Training run failed early | Check training logs for errors |
| `KeyError: 'return'` | CSV schema mismatch | Verify CSV has expected columns |
| `MemoryError` | Large CSV (>100M rows) | Use `--subsample` flag (future) |
| Permission denied | Output dir not writable | Check permissions, create dir |

---

## Aggregation (export_results_table.py)

### Purpose

Aggregate multi-seed runs into summary tables for thesis.

### Inputs

```bash
python scripts/export_results_table.py \
  --runs-dir experiments/dqn_atari/runs \
  --output results/summary/metrics.csv
```

**Required**:
- `--runs-dir` - Directory containing all runs
- `--output` - Output CSV path

**Optional**:
- `--paper-scores` - JSON file with paper baseline scores
- `--format` - Output format (csv, markdown) [default: csv]

### Processing

1. **Discovery**: Find all runs in `runs-dir`
2. **Extraction**: Read final eval metrics from `evaluations.csv`
3. **Grouping**: Group by game, aggregate across seeds
4. **Statistics**: Compute mean, std, min, max per game
5. **Comparison**: Calculate % of paper score (if provided)

### Outputs

**CSV format** (`metrics.csv`):
```csv
game,seed,final_mean_return,final_std,paper_score,percent_of_paper
pong,42,19.5,0.8,20,97.5
pong,123,20.2,0.5,20,101.0
pong,456,18.8,1.2,20,94.0
breakout,42,165.3,12.4,168,98.4
...
```

**Markdown format** (`metrics.md`):
```markdown
| Game | Seeds | Mean Return | Std | Paper Score | % of Paper |
|------|-------|-------------|-----|-------------|------------|
| Pong | 3 | 19.5 ± 0.4 | - | 20 | 97.5% |
| Breakout | 3 | 165.3 ± 8.1 | - | 168 | 98.4% |
```

### Failure Modes

| Error | Cause | Solution |
|-------|-------|----------|
| `No runs found` | Empty runs-dir or wrong path | Verify runs completed, check path |
| `Missing evaluations.csv` | Run incomplete | Check run status, re-run if needed |
| `Inconsistent seeds` | Different games have different seed counts | Document in notes, proceed with available |
| `Parse error` | Malformed CSV | Inspect CSV, fix manually or re-run |

---

## Analysis (analyze_results.py)

### Purpose

Perform statistical tests and generate paper comparisons.

### Inputs

```bash
python scripts/analyze_results.py \
  --metrics results/summary/metrics.csv \
  --paper-scores paper_baselines.json \
  --output results/summary/analysis.txt
```

**Required**:
- `--metrics` - Path to metrics.csv (from export_results_table.py)

**Optional**:
- `--paper-scores` - JSON with paper baseline scores
- `--confidence` - Confidence level (default: 0.95)
- `--test` - Statistical test (t-test, bootstrap) [default: bootstrap]

### Processing

1. **Load data**: Read metrics.csv
2. **Group by game**: Aggregate seeds per game
3. **Compute statistics**:
   - Mean and standard deviation
   - 95% confidence intervals (bootstrap or t-distribution)
   - Min/max across seeds
4. **Paper comparison** (if paper scores provided):
   - Percent of paper score
   - Statistical significance tests
5. **Generate report**: Text summary with findings

### Outputs

**Analysis report** (`analysis.txt`):
```
DQN Baseline Results Analysis
==============================

Pong (3 seeds):
  Mean: 19.5 ± 0.4 (95% CI: [18.8, 20.1])
  Paper: 20
  Achievement: 97.5% of paper score
  Verdict: SUCCESS (≥90% threshold)

Breakout (3 seeds):
  Mean: 165.3 ± 8.1 (95% CI: [151.2, 179.4])
  Paper: 168
  Achievement: 98.4% of paper score
  Verdict: SUCCESS

Overall: 2/2 games meet ≥90% threshold
```

### Failure Modes

| Error | Cause | Solution |
|-------|-------|----------|
| `Insufficient samples` | <3 seeds per game | Run more seeds or reduce confidence |
| `High variance` | Unstable training | Check hyperparams, add more seeds |
| `Invalid paper scores` | Malformed JSON | Verify JSON format, check paper reference |

---

## Thesis Integration

### Workflow

1. **Generate plots** for each game/seed
2. **Export summary table** across all runs
3. **Run statistical analysis** to validate reproduction
4. **Embed in thesis**:
   - Copy plots to `thesis/figures/`
   - Convert tables to LaTeX or Markdown
   - Reference W&B URLs for interactive exploration
   - Document provenance (git hash, system info)

### Thesis Artifacts

**Figures**:
```
thesis/figures/
├── pong_learning_curve.png      # Main learning curve
├── pong_multi_seed.png          # Aggregate across seeds
├── breakout_learning_curve.png
└── comparison_all_games.png     # Bar chart comparison
```

**Tables**:
```
thesis/tables/
├── hyperparameters.tex          # Training config
├── results_summary.tex          # Reproduction results
└── paper_comparison.tex         # Our scores vs paper
```

**References**:
- W&B run URLs in bibliography
- Git commit hash in appendix
- System provenance in methods

---

## Common Failure Modes

### Data Collection Issues

**Missing artifacts**:
- **Cause**: Training run crashed or interrupted
- **Detection**: `evaluations.csv` not found or incomplete
- **Solution**: Resume from checkpoint or re-run

**Corrupt CSV**:
- **Cause**: Disk full, process killed during write
- **Detection**: Parse errors, truncated rows
- **Solution**: Remove last line, resume training, or re-run

**Inconsistent schema**:
- **Cause**: Code version mismatch between runs
- **Detection**: Missing columns, type errors
- **Solution**: Re-run with consistent code version

### Visualization Issues

**Plot quality poor**:
- **Cause**: High noise, insufficient smoothing
- **Detection**: Jagged curves, hard to interpret
- **Solution**: Increase `--smooth` parameter, use PDF format

**Out of memory**:
- **Cause**: Very long runs (>50M steps), large CSVs
- **Detection**: `MemoryError` during plotting
- **Solution**: Subsample data (every Nth row), increase RAM

**Missing data**:
- **Cause**: Evaluation not run (eval_every too large)
- **Detection**: Empty evaluations.csv
- **Solution**: Check config, ensure eval_every reasonable

### Aggregation Issues

**Seed mismatch**:
- **Cause**: Different games have different numbers of seeds
- **Detection**: Uneven sample sizes in metrics.csv
- **Solution**: Document limitation, proceed with available

**Performance outlier**:
- **Cause**: One seed diverged (hyperparameter instability)
- **Detection**: Very high std, one seed much lower/higher
- **Solution**: Investigate logs, consider excluding outlier, add note

### Analysis Issues

**Non-reproducible**:
- **Cause**: Insufficient seeds, high variance
- **Detection**: Wide confidence intervals, <90% of paper
- **Solution**: Add more seeds, check hyperparameters, document

**Statistical test failure**:
- **Cause**: Assumptions violated (e.g., non-normal distribution)
- **Detection**: Warning messages, suspicious p-values
- **Solution**: Use non-parametric tests (bootstrap), increase samples

---

## Automation

### Batch Processing

Process all runs automatically:

```bash
# Generate all plots
for run in experiments/dqn_atari/runs/*/; do
    python scripts/plot_results.py \
        --episodes "$run/csv/episodes.csv" \
        --output "results/plots/$(basename $run)"
done

# Export summary table
python scripts/export_results_table.py \
    --runs-dir experiments/dqn_atari/runs \
    --output results/summary/metrics.csv

# Run analysis
python scripts/analyze_results.py \
    --metrics results/summary/metrics.csv \
    --output results/summary/analysis.txt
```

### Makefile Integration

```makefile
.PHONY: plots tables analysis all

plots:
	@for run in experiments/dqn_atari/runs/*/; do \
		python scripts/plot_results.py \
			--episodes "$$run/csv/episodes.csv" \
			--output "results/plots/$$(basename $$run)"; \
	done

tables:
	python scripts/export_results_table.py \
		--runs-dir experiments/dqn_atari/runs \
		--output results/summary/metrics.csv

analysis: tables
	python scripts/analyze_results.py \
		--metrics results/summary/metrics.csv \
		--output results/summary/analysis.txt

all: plots tables analysis
```

---

## Quality Checks

### Pre-submission Checklist

Before including results in thesis:

- [ ] All runs completed to target frames
- [ ] All evaluations.csv files present and non-empty
- [ ] Plots generated for all runs (check results/plots/)
- [ ] Summary table exported with all seeds
- [ ] Statistical analysis shows ≥90% of paper scores
- [ ] W&B run URLs accessible and linked
- [ ] Git commit hash documented in all metadata
- [ ] System provenance saved (Python version, dependencies)
- [ ] Plots are publication quality (300 DPI, clear labels)
- [ ] Tables formatted correctly (LaTeX or Markdown)

### Validation Commands

```bash
# Check all runs completed
find experiments/dqn_atari/runs -name "evaluations.csv" | wc -l
# Expected: 3 per game × N games

# Verify plot generation
find results/plots -name "returns.png" | wc -l
# Expected: Same as number of runs

# Check metrics table
wc -l results/summary/metrics.csv
# Expected: N_runs + 1 (header)

# Validate analysis output
grep "SUCCESS" results/summary/analysis.txt
# Expected: At least 1 game meets threshold
```

---

## Next Steps

**For generating reports**:
1. Follow [Applied Research Quickstart](../guides/applied-research-quickstart.md) Phase 3
2. Use automation scripts for batch processing
3. Review [Reporting Requirements](../reports/reporting-requirements.md) for thesis priorities

**For troubleshooting**:
- Consult failure modes section above
- Check [Troubleshooting Guide](../guides/troubleshooting.md)
- Review [Logging Pipeline](logging-pipeline.md) for data collection issues

**For customization**:
- Modify plot styles in `scripts/plot_results.py`
- Add new metrics to `scripts/export_results_table.py`
- Extend analysis in `scripts/analyze_results.py`

---

## Related Documents

- [Logging Pipeline](logging-pipeline.md) - Implementation of training metrics collection
- [Applied Research Quickstart](../guides/applied-research-quickstart.md) - Workflow guide using this pipeline
- [Reporting Requirements](../reports/reporting-requirements.md) - Thesis reporting backlog
- [Report Outline](../plans/plan-report-outline.md) - Detailed thesis report structure
- [Results Comparison](../reports/report-results-comparison.md) - Multi-seed comparison methodology

---

**Last Updated**: 2025-12-23
