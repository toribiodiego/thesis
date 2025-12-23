# Report Outline

> **Status**: ACTIVE | Living outline used to structure thesis report and track progress.

This document tracks the structure and source artifacts for the main results report (`docs/reports/report-dqn-results.md`).

## Report Structure

### 1. Executive Summary
- **Content**: High-level reproduction results vs paper
- **Data sources**:
  - `results/summary/metrics.csv` - aggregated scores
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
  - Learning curves: `results/plots/pong_*/returns.png`
  - Final eval: `experiments/dqn_atari/runs/pong_*/eval/evaluations.csv`
  - Videos: `experiments/dqn_atari/runs/pong_*/videos/*.mp4`
- **Paper comparison**: Score 20 (Table 1)
- **Scripts**: `scripts/plot_results.py`
- **Status**: [ ] Data collected | [ ] Analysis complete | [ ] Written

#### 4.2 Breakout
- **Content**: Learning curves, final scores, videos
- **Artifacts**:
  - Learning curves: `results/plots/breakout_*/returns.png`
  - Final eval: `experiments/dqn_atari/runs/breakout_*/eval/evaluations.csv`
  - Videos: `experiments/dqn_atari/runs/breakout_*/videos/*.mp4`
- **Paper comparison**: Score 168 (Table 1)
- **Scripts**: `scripts/plot_results.py`
- **Status**: [ ] Data collected | [ ] Analysis complete | [ ] Written

#### 4.3 Beam Rider
- **Content**: Learning curves, final scores, videos
- **Artifacts**:
  - Learning curves: `results/plots/beam_rider_*/returns.png`
  - Final eval: `experiments/dqn_atari/runs/beam_rider_*/eval/evaluations.csv`
  - Videos: `experiments/dqn_atari/runs/beam_rider_*/videos/*.mp4`
- **Paper comparison**: Score 4092 (Table 1)
- **Scripts**: `scripts/plot_results.py`
- **Status**: [ ] Data collected | [ ] Analysis complete | [ ] Written

#### 4.4 Aggregate Comparison
- **Content**: Summary table comparing all games
- **Artifacts**:
  - `results/summary/metrics.csv` - consolidated scores
  - `results/summary/metrics.md` - markdown table
  - `results/summary/plots/comparison.png` - bar chart
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
  - Loss curves: `results/plots/*/loss.png`
  - Q-value evolution: `results/plots/*/q_values.png`
  - Episode length: `results/plots/*/episode_length.png`
- **Key insights**:
  - Convergence speed
  - Stability (loss variance)
  - Q-value magnitudes
- **Status**: [ ] Draft | [ ] Complete

#### 5.3 Ablation Studies (if completed)
- **Content**: Impact of key design choices
- **Artifacts**:
  - `experiments/dqn_atari/runs/ablation_*/`
  - `results/ablations/*/`
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
  - `results/summary/cpu_performance_baseline.md` (gitignored)
- **Metrics**: FPS, memory usage, GPU utilization

#### C. Reproduction Commands
- **Content**: Exact commands to reproduce
- **Source**:
  - `scripts/reproduce_dqn.sh`
  - `plan-reproduce-recipe.md`

---

## Artifact Inventory

### Essential Artifacts (must have before report)

| Artifact | Location | Script to Generate | Status |
|----------|----------|-------------------|--------|
| Pong seed 42 run | `experiments/dqn_atari/runs/pong_42_*` | `train_dqn.py` | [ ] |
| Pong seed 123 run | `experiments/dqn_atari/runs/pong_123_*` | `train_dqn.py` | [ ] |
| Pong seed 456 run | `experiments/dqn_atari/runs/pong_456_*` | `train_dqn.py` | [ ] |
| Pong learning curves | `results/plots/pong_*` | `scripts/plot_results.py` | [ ] |
| Summary metrics CSV | `results/summary/metrics.csv` | `scripts/export_results_table.py` | [ ] |
| Summary metrics MD | `results/summary/metrics.md` | Manual/script | [ ] |

### Optional Artifacts (enhance report)

| Artifact | Location | Script to Generate | Status |
|----------|----------|-------------------|--------|
| Breakout runs (3 seeds) | `experiments/dqn_atari/runs/breakout_*` | `train_dqn.py` | [ ] |
| Beam Rider runs (3 seeds) | `experiments/dqn_atari/runs/beam_rider_*` | `train_dqn.py` | [ ] |
| Ablation runs | `experiments/dqn_atari/runs/ablation_*` | `train_dqn.py` | [ ] |
| Comparison bar chart | `results/summary/plots/comparison.png` | Custom script | [ ] |
| W&B dashboard | wandb.ai URL | W&B UI | [ ] |

---

## Figure List

| Figure # | Title | Source File | Status |
|----------|-------|-------------|--------|
| 1 | Pong Learning Curve | `results/plots/pong_42/returns.png` | [ ] |
| 2 | Pong Loss Curve | `results/plots/pong_42/loss.png` | [ ] |
| 3 | Pong Q-Values | `results/plots/pong_42/q_values.png` | [ ] |
| 4 | Multi-Seed Pong Comparison | `results/plots/pong_aggregate.png` | [ ] |
| 5 | Paper vs Reproduction Scores | `results/summary/plots/comparison.png` | [ ] |

---

## Table List

| Table # | Title | Source Data | Status |
|---------|-------|-------------|--------|
| 1 | Hyperparameters | Config YAML files | [ ] |
| 2 | Paper Target Scores | DQN 2013 Table 1 | [ ] |
| 3 | Reproduction Results | `results/summary/metrics.csv` | [ ] |
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

## Regeneration Commands

### Complete report regeneration
```bash
# 1. Generate all plots
for run in experiments/dqn_atari/runs/*/; do
    python scripts/plot_results.py \
        --episodes "$run/csv/episodes.csv" \
        --output "results/plots/$(basename $run)"
done

# 2. Export summary table
python scripts/export_results_table.py \
    --runs-dir experiments/dqn_atari/runs \
    --output results/summary/metrics.csv

# 3. Generate markdown report sections
# (manual or semi-automated with templates)

# 4. Upload to W&B
# python scripts/upload_to_wandb.py --report results/summary/
```

### Quick update (new run added)
```bash
# Generate plots for new run
python scripts/plot_results.py \
    --episodes experiments/dqn_atari/runs/<new_run>/csv/episodes.csv \
    --output results/plots/<new_run>

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
| 1.0 | TBD | Final report complete |

---

## References

- Main report: `docs/reports/report-dqn-results.md` (to be created)
- Results comparison guide: `../reports/report-results-comparison.md`
- Environment notes: `../reference/environment-notes.md`
- Reproduction recipe: `plan-reproduce-recipe.md`
- Training configs: `experiments/dqn_atari/configs/`
- Plotting tools: `scripts/plot_results.py`
- DQN 2013 paper: arXiv:1312.5602
