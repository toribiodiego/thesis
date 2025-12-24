# Report Outline

> **Status**: ACTIVE | Living outline used to structure thesis report and track progress.

This document tracks the structure and source artifacts for the main results report (`docs/reports/report-dqn-results.md`).

## Report Structure

### 1. Executive Summary
- **Content**: High-level reproduction results vs paper
- **Data sources**:
  - `output/summary/metrics.csv` - aggregated scores
  - `experiments/dqn_atari/runs/*/eval/evaluations.csv` - final evals
- **Scripts**: `scripts/export_results_table.py`
- **Status**: [ ] Draft | [ ] Complete

### 2. Introduction
- **Content**: Project motivation, DQN 2013 paper context
- **References**:
  - arXiv:1312.5602 (DQN 2013 paper)
  - `plan-game-suite.md` - game selection rationale
- **Status**: [ ] Draft | [ ] Complete

### 3. Methods
- **Content**: Implementation details and deviations from paper
- **Data sources**:
  - `experiments/dqn_atari/configs/*.yaml` - training configs
  - `../reference/environment-notes.md` - toolchain differences
  - `src/models/dqn_model.py` - network architecture
- **Key sections**:
  - 3.1 Network Architecture
  - 3.2 Training Procedure
  - 3.3 Preprocessing Pipeline
  - 3.4 Evaluation Protocol
- **Status**: [ ] Draft | [ ] Complete

### 4. Results

#### 4.1 Pong
- **Content**: Learning curves, final scores, videos
- **Artifacts**:
  - Learning curves: `output/plots/pong_*/returns.png`
  - Final eval: `experiments/dqn_atari/runs/pong_*/eval/evaluations.csv`
  - Videos: `experiments/dqn_atari/runs/pong_*/videos/*.mp4`
- **Paper comparison**: Score 20 (Table 1)
- **Scripts**: `scripts/plot_results.py`
- **Status**: [ ] Data collected | [ ] Analysis complete | [ ] Written

#### 4.2 Breakout
- **Content**: Learning curves, final scores, videos
- **Artifacts**:
  - Learning curves: `output/plots/breakout_*/returns.png`
  - Final eval: `experiments/dqn_atari/runs/breakout_*/eval/evaluations.csv`
  - Videos: `experiments/dqn_atari/runs/breakout_*/videos/*.mp4`
- **Paper comparison**: Score 168 (Table 1)
- **Scripts**: `scripts/plot_results.py`
- **Status**: [ ] Data collected | [ ] Analysis complete | [ ] Written

#### 4.3 Beam Rider
- **Content**: Learning curves, final scores, videos
- **Artifacts**:
  - Learning curves: `output/plots/beam_rider_*/returns.png`
  - Final eval: `experiments/dqn_atari/runs/beam_rider_*/eval/evaluations.csv`
  - Videos: `experiments/dqn_atari/runs/beam_rider_*/videos/*.mp4`
- **Paper comparison**: Score 4092 (Table 1)
- **Scripts**: `scripts/plot_results.py`
- **Status**: [ ] Data collected | [ ] Analysis complete | [ ] Written

#### 4.4 Aggregate Comparison
- **Content**: Summary table comparing all games
- **Artifacts**:
  - `output/summary/metrics.csv` - consolidated scores
  - `output/summary/metrics.md` - markdown table
  - `output/summary/plots/comparison.png` - bar chart
- **Scripts**:
  - `scripts/export_results_table.py`
  - `scripts/analyze_results.py` (planned)
- **Status**: [ ] Data collected | [ ] Analysis complete | [ ] Written

### 5. Discussion

#### 5.1 Comparison to Paper
- **Content**: Match/gap analysis with root causes
- **Data sources**:
  - `../reports/report-results-comparison.md` - methodology
  - `../reference/environment-notes.md` - known differences
- **Key points**:
  - Percentage of paper score achieved
  - Environment version differences
  - Reward clipping impact
  - Hardware/software variations
- **Status**: [ ] Draft | [ ] Complete

#### 5.2 Learning Dynamics
- **Content**: Analysis of training behavior
- **Artifacts**:
  - Loss curves: `output/plots/*/loss.png`
  - Q-value evolution: `output/plots/*/q_values.png`
  - Episode length: `output/plots/*/episode_length.png`
- **Key insights**:
  - Convergence speed
  - Stability (loss variance)
  - Q-value magnitudes
- **Status**: [ ] Draft | [ ] Complete

#### 5.3 Ablation Studies (if completed)
- **Content**: Impact of key design choices
- **Artifacts**:
  - `experiments/dqn_atari/runs/ablation_*/`
  - `output/ablations/*/`
  - `plan-ablations.md` - hypotheses
- **Ablations**:
  - Reward clipping disabled
  - Frame stack reduced to 2
  - No target network
- **Status**: [ ] Runs complete | [ ] Analysis complete | [ ] Written

### 6. Lessons Learned
- **Content**: Reproducibility insights and challenges
- **Sources**: Development experience, troubleshooting logs
- **Key lessons**:
  - ROM version importance
  - Epsilon decay timing
  - Memory management for replay buffer
  - Seed sensitivity
- **Status**: [ ] Draft | [ ] Complete

### 7. Future Work
- **Content**: Next algorithmic steps
- **References**:
  - [TODO](../../TODO) - planned extensions
  - Double DQN (Van Hasselt et al., 2016)
  - Prioritized Experience Replay (Schaul et al., 2016)
  - Dueling DQN (Wang et al., 2016)
- **Status**: [ ] Draft | [ ] Complete

### 8. Appendix

#### A. Hyperparameters
- **Content**: Complete configuration table
- **Source**: `experiments/dqn_atari/configs/*.yaml`
- **Format**: Table with parameter names, values, and paper references

#### B. Compute Resources
- **Content**: Hardware specs, runtime statistics
- **Sources**:
  - `experiments/dqn_atari/runs/*/system_info.txt`
  - `output/summary/cpu_performance_baseline.md` (gitignored)
- **Metrics**: FPS, memory usage, GPU utilization

#### C. Reproduction Commands
- **Content**: Exact commands to reproduce
- **Source**:
  - `experiments/dqn_atari/scripts/reproduce_dqn.sh`
  - `plan-reproduce-recipe.md`

---

## Artifact Inventory

### Essential Artifacts (must have before report)

| Artifact | Location | Script to Generate | Status |
|----------|----------|-------------------|--------|
| Pong seed 42 run | `experiments/dqn_atari/runs/pong_42_*` | `train_dqn.py` | [ ] |
| Pong seed 123 run | `experiments/dqn_atari/runs/pong_123_*` | `train_dqn.py` | [ ] |
| Pong seed 456 run | `experiments/dqn_atari/runs/pong_456_*` | `train_dqn.py` | [ ] |
| Pong learning curves | `output/plots/pong_*` | `scripts/plot_results.py` | [ ] |
| Summary metrics CSV | `output/summary/metrics.csv` | `scripts/export_results_table.py` | [ ] |
| Summary metrics MD | `output/summary/metrics.md` | Manual/script | [ ] |

### Optional Artifacts (enhance report)

| Artifact | Location | Script to Generate | Status |
|----------|----------|-------------------|--------|
| Breakout runs (3 seeds) | `experiments/dqn_atari/runs/breakout_*` | `train_dqn.py` | [ ] |
| Beam Rider runs (3 seeds) | `experiments/dqn_atari/runs/beam_rider_*` | `train_dqn.py` | [ ] |
| Ablation runs | `experiments/dqn_atari/runs/ablation_*` | `train_dqn.py` | [ ] |
| Comparison bar chart | `output/summary/plots/comparison.png` | Custom script | [ ] |
| W&B dashboard | wandb.ai URL | W&B UI | [ ] |

---

## Figure List

| Figure # | Title | Source File | Status |
|----------|-------|-------------|--------|
| 1 | Pong Learning Curve | `output/plots/pong_42/returns.png` | [ ] |
| 2 | Pong Loss Curve | `output/plots/pong_42/loss.png` | [ ] |
| 3 | Pong Q-Values | `output/plots/pong_42/q_values.png` | [ ] |
| 4 | Multi-Seed Pong Comparison | `output/plots/pong_aggregate.png` | [ ] |
| 5 | Paper vs Reproduction Scores | `output/summary/plots/comparison.png` | [ ] |

---

## Table List

| Table # | Title | Source Data | Status |
|---------|-------|-------------|--------|
| 1 | Hyperparameters | Config YAML files | [ ] |
| 2 | Paper Target Scores | DQN 2013 Table 1 | [ ] |
| 3 | Reproduction Results | `output/summary/metrics.csv` | [ ] |
| 4 | Comparison Summary | Computed from Table 2 & 3 | [ ] |
| 5 | Environment Versions | `../reference/environment-notes.md` | [ ] |

---

## Open Questions / TODOs

- [ ] Decide on confidence interval calculation method (bootstrap vs analytical)
- [ ] Determine minimum seeds required for statistical significance
- [ ] Establish threshold for "successful" reproduction (90% of paper?)
- [ ] Plan video embedding approach for W&B report
- [ ] Choose plot style (seaborn, matplotlib, custom)
- [ ] Define episode window for "final performance" (last 100? last 5 evals?)

---

## Reporting Pipeline Mapping

This section maps the [Reporting Pipeline](../reference/reporting-pipeline.md) stages to specific report sections.

### Pipeline Stage → Report Section Map

| Pipeline Stage | Script | Input | Output | Report Section(s) |
|----------------|--------|-------|--------|-------------------|
| **Data Collection** | Training runs | `experiments/dqn_atari/runs/*/csv/*.csv` | Logs, metrics, configs | All sections (raw data) |
| **Visualization** | `scripts/plot_results.py` | `episodes.csv`, `evaluations.csv` | `output/plots/*/returns.png`, `loss.png`, etc. | 4.1-4.3 (Results - per game) |
| **Aggregation** | `scripts/export_results_table.py` | All runs in `runs/` | `output/summary/metrics.csv` | 4.4 (Aggregate Comparison), Appendix |
| **Analysis** | `scripts/analyze_results.py` | `metrics.csv` | `analysis.txt`, statistical tests | 5.1 (Comparison to Paper), 5.2 (Learning Dynamics) |
| **Thesis Integration** | Manual/LaTeX | All above outputs | Formatted figures/tables | All sections (final thesis) |

### Detailed Script-to-Section Mapping

**Section 1: Executive Summary**
- Script: `scripts/export_results_table.py`
- Input: `experiments/dqn_atari/runs/*/csv/evaluations.csv`
- Output: `output/summary/metrics.csv` (final row per game)
- Usage: Extract mean return, % of paper score for summary statement

**Section 3: Methods**
- Script: None (manual extraction from docs + configs)
- Input: `experiments/dqn_atari/configs/*.yaml`, `docs/reference/*.md`
- Output: Hyperparameters table, architecture description
- Usage: Copy config values, cite design docs

**Section 4.1-4.3: Per-Game Results**
- Script: `scripts/plot_results.py`
- Input: `experiments/dqn_atari/runs/<game>_<seed>_*/csv/episodes.csv`
- Output: `output/plots/<game>_<seed>/returns.png`, `loss.png`, `q_values.png`
- Usage: Embed learning curves in Results chapter

**Section 4.4: Aggregate Comparison**
- Script: `scripts/export_results_table.py` → `scripts/analyze_results.py`
- Input: All runs (multi-seed)
- Output: `output/summary/metrics.csv`, `metrics.md` (table), `analysis.txt`
- Usage: Summary table showing all games, seeds, paper comparison

**Section 5.1: Comparison to Paper**
- Script: `scripts/analyze_results.py`
- Input: `output/summary/metrics.csv`, paper scores (hardcoded or JSON)
- Output: `output/summary/analysis.txt` (statistical tests, CI, % of paper)
- Usage: Justify reproduction quality, discuss gaps

**Section 5.2: Learning Dynamics**
- Script: `scripts/plot_results.py` (loss, Q-values outputs)
- Input: `experiments/dqn_atari/runs/*/csv/training_steps.csv`
- Output: `output/plots/*/loss.png`, `q_values.png`, `episode_length.png`
- Usage: Analyze convergence, stability, Q-value magnitudes

**Appendix: Hyperparameters Table**
- Script: Manual extraction or YAML→Markdown converter
- Input: `experiments/dqn_atari/configs/pong.yaml`
- Output: Markdown/LaTeX table
- Usage: Document exact hyperparameters used

**Appendix: Compute Resources**
- Script: Manual extraction
- Input: `experiments/dqn_atari/runs/*/system_info.txt`, training logs
- Output: Hardware specs, FPS, runtime table
- Usage: Document computational requirements

**Appendix: Reproduction Commands**
- Script: None (copy from docs)
- Input: `docs/plans/plan-reproduce-recipe.md`
- Output: Bash commands for full reproduction
- Usage: Enable others to reproduce results

### Quick Reference

For full pipeline documentation, see [Reporting Pipeline](../reference/reporting-pipeline.md).

For thesis artifact index and regeneration steps, see [Thesis Artifact Index](../thesis/README.md).

---

## Regeneration Commands

### Complete report regeneration
```bash
# 1. Generate all plots
for run in experiments/dqn_atari/runs/*/; do
    python scripts/plot_results.py \
        --episodes "$run/csv/episodes.csv" \
        --output "output/plots/$(basename $run)"
done

# 2. Export summary table
python scripts/export_results_table.py \
    --runs-dir experiments/dqn_atari/runs \
    --output output/summary/metrics.csv

# 3. Generate markdown report sections
# (manual or semi-automated with templates)

# 4. Upload to W&B
# python scripts/upload_to_wandb.py --report output/summary/
```

### Quick update (new run added)
```bash
# Generate plots for new run
python scripts/plot_results.py \
    --episodes experiments/dqn_atari/runs/<new_run>/csv/episodes.csv \
    --output output/plots/<new_run>

# Re-export summary table
python scripts/export_results_table.py \
    --runs-dir experiments/dqn_atari/runs
```

---

## Review Checklist

Before finalizing report:

- [ ] All plots render correctly in markdown
- [ ] All tables have consistent formatting
- [ ] All artifact paths are valid
- [ ] All scripts mentioned can be run successfully
- [ ] Numbers in tables match source data
- [ ] Paper citations are accurate
- [ ] W&B links are accessible
- [ ] Figures have proper captions
- [ ] Statistical claims are justified
- [ ] Code snippets are syntax-highlighted

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1 | TBD | Initial outline created |
| 0.2 | TBD | Pong results added |
| 0.3 | TBD | Multi-game results added |
| 0.4 | 2025-12-23 | Added reporting pipeline mapping |
| 1.0 | TBD | Final report complete |

---

## References

**Report Structure**:
- Main report: `docs/reports/report-dqn-results.md` (to be created)
- Results comparison guide: `../reports/report-results-comparison.md`
- Reproduction recipe: `plan-reproduce-recipe.md`

**Reporting Infrastructure**:
- [Reporting Pipeline](../reference/reporting-pipeline.md) - End-to-end workflow from artifacts to thesis
- [Thesis Artifact Index](../thesis/README.md) - Index of figures, tables, and regeneration steps
- [Logging Pipeline](../reference/logging-pipeline.md) - Training metrics collection

**Implementation References**:
- Environment notes: `../reference/environment-notes.md`
- Training configs: `experiments/dqn_atari/configs/`
- Plotting tools: `scripts/plot_results.py`
- Analysis tools: `scripts/export_results_table.py`, `scripts/analyze_results.py`

**Source Material**:
- DQN 2013 paper: arXiv:1312.5602
