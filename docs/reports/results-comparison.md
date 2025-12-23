# Results Comparison

This document provides the authoritative methodology for comparing reproduction results against the DQN 2013 paper (arXiv:1312.5602).

## Overview

We compare our results to the original DQN paper using:
1. Mean episode returns over final evaluation window
2. Percentage of paper-reported scores
3. Learning curve shape and convergence time
4. Statistical significance across seeds

## Paper Reference Scores (DQN 2013)

From Table 1 of "Playing Atari with Deep Reinforcement Learning":

| Game | Random | Sarsa | Contingency | **DQN** | Human |
|------|--------|-------|-------------|---------|-------|
| Pong | -20.4 | -19 | -17 | **20** | -3 |
| Breakout | 1.2 | 5.2 | 6 | **168** | 31 |
| Beam Rider | 354 | 996 | 1743 | **4092** | 7456 |

**Important**: These are 2013 arXiv paper scores, NOT the 2015 Nature paper scores.

---

## Data Collection

### Training Outputs

Each training run produces:
```
experiments/dqn_atari/runs/<run_name>/
├── csv/
│   ├── episodes.csv          # Per-episode data (return, length)
│   ├── training_steps.csv    # Per-step metrics (loss, Q-values)
│   └── evaluation.csv        # Evaluation results (30 episodes each)
├── eval/
│   └── *.npz                 # Raw evaluation episode data
├── checkpoints/
│   └── checkpoint_*.pt       # Model weights
└── config.yaml               # Resolved configuration
```

### Key Metrics

1. **Episode Return**: Total undiscounted reward per episode
2. **Evaluation Return**: Mean return over 30 episodes (epsilon=0.05)
3. **Learning Frames**: Total environment frames processed
4. **Training Steps**: Gradient updates performed

---

## Analysis Scripts

### 1. Export Results Table

Generate summary CSV/Markdown from all runs:

```bash
python scripts/export_results_table.py \
  --runs-dir experiments/dqn_atari/runs \
  --output results/summary/metrics.csv
```

Output format:
```csv
run_name,game,seed,frames,mean_return,std_return,max_return,final_loss
pong_42_...,Pong,42,10000000,20.5,0.3,21.0,1.82
```

### 2. Generate Comparison Plots

Create publication-quality figures:

```bash
python scripts/plot_results.py \
  --episodes experiments/dqn_atari/runs/<run>/csv/episodes.csv \
  --output results/summary/plots/ \
  --game-name "Pong (DQN 2013 Reproduction)"
```

Produces:
- `returns.png` - Episode returns with rolling average
- `episode_length.png` - Episode lengths over time
- `loss.png` - Training loss (smoothed)
- `q_values.png` - Average Q-value estimates

### 3. Compute Statistics

For final evaluation window (last 5 evaluations or 100 episodes):

```bash
python scripts/analyze_results.py \
  --eval-csv experiments/dqn_atari/runs/<run>/csv/evaluation.csv \
  --paper-score 20 \
  --output results/summary/<game>_analysis.json
```

Output includes:
- Mean, median, std of final returns
- Percentage of paper score achieved
- Confidence intervals (95%)
- Number of seeds aggregated

---

## Comparison Methodology

### 1. Final Performance

Compare mean evaluation return over final evaluation window:

```python
# Load evaluation data
import pandas as pd
import numpy as np

df = pd.read_csv("evaluation.csv")
# Use last 5 evaluation checkpoints
final_evals = df.tail(5)
mean_return = final_evals["mean_return"].mean()
std_return = final_evals["mean_return"].std()

# Compare to paper
paper_score = 20  # Pong
percentage = (mean_return / paper_score) * 100
print(f"Achievement: {percentage:.1f}% of paper score")
```

### 2. Multi-Seed Aggregation

For robust comparison, aggregate across seeds:

```python
# Combine results from seeds 42, 123, 456
seeds_data = []
for seed in [42, 123, 456]:
    df = pd.read_csv(f"pong_{seed}/evaluation.csv")
    final_return = df.tail(5)["mean_return"].mean()
    seeds_data.append(final_return)

mean_across_seeds = np.mean(seeds_data)
std_across_seeds = np.std(seeds_data)
ci_95 = 1.96 * std_across_seeds / np.sqrt(len(seeds_data))
```

### 3. Learning Curve Shape

Key checkpoints to verify:
- **250K frames**: Random policy (-20 to -21)
- **1M frames**: Early learning (> -20)
- **5M frames**: Significant improvement (> 0)
- **10M frames**: Near-optimal (close to 20)

### 4. Convergence Time

Measure frames to reach threshold performance:

```python
# Frames to reach 50% of paper score
threshold = paper_score * 0.5
frames_to_threshold = df[df["mean_return"] >= threshold]["frame"].iloc[0]
```

---

## Result Classification

### Match (Green)
- Score >= 90% of paper score
- Stable convergence
- Consistent across seeds

### Partial (Yellow)
- Score 50-90% of paper score
- Convergence achieved but slower
- Higher variance across seeds

### Lag (Red)
- Score < 50% of paper score
- Divergence or instability
- Missing critical learning milestones

---

## Common Discrepancies

### 1. Environment Differences

**Gymnasium vs ALE versions**:
- Frame skip implementation
- Action repeat stochasticity
- ROM versions
- Terminal signal handling

Document in `results/summary/notes.md`:
```markdown
## Environment Configuration

- gymnasium==0.29.1
- ale-py==0.11.2
- ROM version: Pong - Video Olympics
- NoFrameskip-v4 (deterministic)
```

### 2. Preprocessing Differences

- Grayscale conversion method
- Resize interpolation (bilinear vs nearest)
- Frame stacking order
- Max-pooling vs last frame

### 3. Training Parameters

- RMSprop vs Adam optimizer
- Learning rate schedule
- Batch size (32 standard)
- Target network update frequency

### 4. Evaluation Protocol

- Number of evaluation episodes (30)
- Evaluation epsilon (0.05)
- No-op start (up to 30)
- Episode termination (life loss vs game over)

---

## Regeneration Steps

### Local Reproduction

1. **Verify environment setup**:
```bash
python -c "import gymnasium; print(gymnasium.__version__)"
python -c "import ale_py; print(ale_py.__version__)"
```

2. **Check configuration matches paper**:
```bash
cat experiments/dqn_atari/configs/pong.yaml | grep -A5 "preprocessing:"
```

3. **Run training**:
```bash
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/pong.yaml \
  --seed 42
```

4. **Generate results**:
```bash
python scripts/export_results_table.py --runs-dir experiments/dqn_atari/runs
python scripts/plot_results.py --episodes <run>/csv/episodes.csv
```

5. **Validate percentage calculation**:
```python
# Manual verification
our_score = 19.5  # From evaluation.csv
paper_score = 20.0  # From Table 1
percentage = (our_score / paper_score) * 100
assert abs(percentage - 97.5) < 0.1
```

### W&B Reproduction

1. **Access run history**:
```python
import wandb
api = wandb.Api()
run = api.run("Cooper-Union/dqn-atari/<run_id>")
history = run.history()
```

2. **Download artifacts**:
```bash
wandb artifact get Cooper-Union/dqn-atari/model:<version>
```

3. **Query final metrics**:
```python
eval_data = history[["_step", "eval/mean_return"]].dropna()
final_return = eval_data.tail(5)["eval/mean_return"].mean()
```

4. **Compare to paper in W&B Tables**:
```python
table = wandb.Table(columns=["Game", "Ours", "Paper", "Percentage"])
table.add_data("Pong", 19.5, 20.0, 97.5)
wandb.log({"results_comparison": table})
```

---

## Output Files

### Required Deliverables

1. **Metrics table** (`results/summary/metrics.csv`):
```csv
game,mean_score_ours,paper_score,percentage,std_dev,frames,seeds,notes
Pong,19.5,20.0,97.5,0.8,10000000,"42,123,456",Matches paper
Breakout,155,168,92.3,12.4,50000000,"42,123,456",Slight lag
Beam Rider,3680,4092,89.9,342,50000000,"42,123,456",Within variance
```

2. **Markdown table** (`results/summary/metrics.md`):
```markdown
| Game | Mean Score (Ours) | Paper Score | % of Paper | Std Dev | Notes |
|------|-------------------|-------------|------------|---------|-------|
| Pong | 19.5 | 20 | 97.5% | 0.8 | Matches |
| Breakout | 155 | 168 | 92.3% | 12.4 | Slight lag |
| Beam Rider | 3680 | 4092 | 89.9% | 342 | Within variance |
```

3. **Comparison plots** (`results/summary/plots/`):
- `pong_comparison.png`
- `breakout_comparison.png`
- `beam_rider_comparison.png`

4. **Environment notes** (`results/summary/notes.md`):
- Software versions
- Hardware specs
- Known differences from paper

---

## Validation Checklist

Before finalizing results:

- [ ] All seeds completed (42, 123, 456)
- [ ] Evaluation CSVs have expected rows
- [ ] Loss curves show convergence (not divergence)
- [ ] Q-values are reasonable (not exploding)
- [ ] Percentage calculations verified manually
- [ ] Environment versions documented
- [ ] W&B artifacts uploaded (if enabled)
- [ ] Plots generated and reviewed
- [ ] Notes file updated with any anomalies

---

## Troubleshooting

### Low Scores

1. **Check epsilon decay**: Should reach 0.1 by 1M frames
2. **Verify reward clipping**: Enabled in config
3. **Inspect replay buffer**: Ensure diverse samples
4. **Monitor Q-values**: Should increase over training

### High Variance

1. **Increase evaluation episodes**: From 30 to 50
2. **Extend final window**: Use last 10 evals instead of 5
3. **Add more seeds**: 5 seeds for tighter confidence intervals

### Training Instability

1. **Check loss spikes**: May indicate gradient issues
2. **Monitor Q-value magnitudes**: Should be bounded
3. **Verify target network updates**: Every 10K steps

---

## References

- Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. arXiv:1312.5602
- Game Suite Plan: `../plans/game-suite-plan.md`
- Training Configuration: `experiments/dqn_atari/configs/`
- Plotting Scripts: `scripts/plot_results.py`
- Export Scripts: `scripts/export_results_table.py`
