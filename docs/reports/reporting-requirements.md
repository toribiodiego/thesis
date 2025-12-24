# Reporting Requirements

> **Status**: ACTIVE | Prioritized backlog of thesis reporting gaps and requirements.
> **Purpose**: Track what's needed to complete the DQN reproduction thesis report.

## Purpose

This document maintains a prioritized list of missing artifacts, analyses, and documentation needed for the final thesis report. It serves as a focused backlog complementing the detailed structure in [plan-report-outline.md](../plans/plan-report-outline.md).

## Inputs/Outputs

**Inputs** (prerequisites):
- [Report Outline](../plans/plan-report-outline.md) - Detailed report structure
- [DQN Results Report](report-dqn-results.md) - Main results document
- Training runs in `experiments/dqn_atari/runs/`

**Outputs** (deliverables):
- Clear priority order for completing thesis report
- Actionable list of missing artifacts and analyses
- Timeline estimates for report completion

---

## Priority 1: Essential for Minimal Viable Report

These must be completed before any report submission:

### 1.1 Baseline Training Data
- [ ] **Pong 3-seed runs** (10M frames each)
  - Seeds: 42, 123, 456
  - Est. time: 90-180 hours CPU or 15-30 hours GPU
  - Blocking: All results sections depend on this
  - Script: `./experiments/dqn_atari/scripts/reproduce_dqn.sh --game pong --seed <SEED>`

### 1.2 Core Visualizations
- [ ] **Pong learning curves** (returns over time)
  - Script: `python scripts/plot_results.py`
  - Depends on: 1.1 Pong runs
  - Output: `output/plots/pong_*/returns.png`

- [ ] **Pong multi-seed comparison**
  - Aggregate 3 seeds with confidence intervals
  - Script: Custom aggregation (needs creation)
  - Output: `output/plots/pong_aggregate.png`

### 1.3 Summary Tables
- [ ] **Hyperparameters table**
  - Source: `experiments/dqn_atari/configs/*.yaml`
  - Format: Markdown table for appendix
  - Action: Extract from YAML to markdown

- [ ] **Results comparison table**
  - Columns: Game, Paper Score, Our Score, % of Paper, CI
  - Script: `python scripts/export_results_table.py`
  - Depends on: 1.1 Training runs

### 1.4 Methods Section
- [ ] **Implementation details writeup**
  - Network architecture (from [dqn-model.md](../reference/dqn-model.md))
  - Training procedure (from [dqn-training.md](../reference/dqn-training.md))
  - Preprocessing pipeline (from [atari-env-wrapper.md](../reference/atari-env-wrapper.md))
  - Known deviations from paper (from [environment-notes.md](../reference/environment-notes.md))

---

## Priority 2: Important for Strong Report

Should be included if time permits before submission:

### 2.1 Extended Game Suite
- [ ] **Breakout 3-seed runs** (50M frames each)
  - Est. time: 450-900 hours CPU or 75-150 hours GPU
  - Value: Demonstrates generalization beyond Pong
  - Defer if: GPU not available

- [ ] **Beam Rider 3-seed runs** (50M frames each)
  - Est. time: Same as Breakout
  - Value: Third game for statistical robustness
  - Defer if: GPU not available

### 2.2 Analysis Depth
- [ ] **Learning dynamics analysis**
  - Loss curves over training
  - Q-value evolution
  - Episode length trends
  - Script: Extend `scripts/plot_results.py`

- [ ] **Failure case analysis**
  - When/why does Pong fail?
  - Q-value distribution on failure episodes
  - Requires: Episode-level inspection

### 2.3 Reproducibility Documentation
- [ ] **Compute resource appendix**
  - Hardware specs from `system_info.txt`
  - Runtime statistics (FPS, memory, wall-clock time)
  - Cost estimates for reproduction

- [ ] **Exact reproduction commands**
  - Complete bash script walkthrough
  - Environment setup verification
  - Source: [reproduce-recipe.md](../plans/plan-reproduce-recipe.md)

---

## Priority 3: Nice-to-Have Enhancements

Optional content that strengthens the report:

### 3.1 Ablation Studies
- [ ] **Reward clipping ablation**
  - Train Pong without reward clipping
  - Compare stability and final performance
  - Est. time: 1 run × 10M frames

- [ ] **Frame stack ablation**
  - Reduce stack from 4 to 2 frames
  - Measure impact on learning speed
  - Est. time: 1 run × 10M frames

- [ ] **Target network ablation**
  - Train without target network
  - Expect instability, quantify impact
  - Est. time: 1 run × 10M frames (may diverge early)

### 3.2 Advanced Visualizations
- [ ] **Aggregate comparison bar chart**
  - Multi-game comparison to paper
  - Error bars for confidence intervals
  - Output: `output/summary/plots/comparison.png`

- [ ] **Video recordings**
  - Best episode videos for each game
  - Before/after learning comparison
  - Upload to W&B or YouTube

### 3.3 Extended Discussion
- [ ] **Environment version impact**
  - ALE version differences analysis
  - ROM version considerations
  - Nondeterminism sources

- [ ] **Hyperparameter sensitivity**
  - Which params are most critical?
  - Literature survey on DQN variants
  - Link to future work (Double DQN, etc.)

---

## Bottlenecks and Blockers

**Current blockers**:
1. **GPU access** - Limits ability to run Breakout/Beam Rider in reasonable time
2. **Training time** - Even Pong 3-seeds takes weeks on CPU
3. **W&B artifacts** - Some checkpoint uploads may be incomplete

**Mitigation strategies**:
1. Focus on Priority 1 (Pong-only report is acceptable)
2. Use Google Colab for GPU acceleration (see [colab-guide.md](../guides/colab-guide.md))
3. Run training overnight/weekends to maximize throughput
4. Consider 2-seed runs instead of 3 if time-constrained

---

## Timeline Estimates

**Minimum viable report** (Priority 1 only):
- Training: 2-4 weeks (CPU) or 3-5 days (GPU)
- Analysis: 3-5 days
- Writing: 1-2 weeks
- **Total**: 4-7 weeks (CPU) or 3-4 weeks (GPU)

**Strong report** (Priority 1 + 2):
- Training: 6-12 weeks (CPU) or 2-3 weeks (GPU)
- Analysis: 1-2 weeks
- Writing: 2-3 weeks
- **Total**: 10-17 weeks (CPU) or 5-8 weeks (GPU)

**Comprehensive report** (All priorities):
- Training: 8-16 weeks (CPU) or 3-4 weeks (GPU)
- Analysis: 2-3 weeks
- Writing: 3-4 weeks
- **Total**: 13-23 weeks (CPU) or 8-11 weeks (GPU)

---

## Next Steps

**Immediate actions**:
1. Start Priority 1.1 Pong 3-seed runs immediately
2. Set up automated plotting pipeline for Priority 1.2
3. Draft Methods section (Priority 1.4) while training runs

**Weekly checkpoints**:
- Monitor training progress (FPS, loss trends)
- Generate intermediate plots to validate learning
- Update this document as items are completed

**Decision points**:
- Week 2: Assess GPU availability → Decide on Priority 2 scope
- Week 4: Review Pong results → Decide on ablation studies
- Week 6: Finalize report scope based on available data

---

## Related Documents

- [Report Outline](../plans/plan-report-outline.md) - Detailed report structure and artifact inventory
- [DQN Results](report-dqn-results.md) - Main results document (currently incomplete)
- [Reproduce Recipe](../plans/plan-reproduce-recipe.md) - Training run commands and workflows
- [GPU Validation](report-gpu-validation.md) - GPU performance analysis (completed)
- [Results Comparison](report-results-comparison.md) - Multi-seed comparison methodology

---

**Last Updated**: 2025-12-23
