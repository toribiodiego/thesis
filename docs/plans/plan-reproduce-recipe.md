# Reproduce Recipe

> **Status**: ACTIVE | Actively used workflow for reproducing baseline DQN results.

This document provides the complete guide for reproducing DQN 2013 paper results using the automated reproduction script.

## Quick Start

**One-command reproduction:**
```bash
./experiments/dqn_atari/scripts/reproduce_dqn.sh --game pong --seed 42
```

This will:
1. Set up Python environment (if needed)
2. Install Atari ROMs
3. Run training (10M frames for Pong)
4. Generate result plots
5. Save system provenance

---

## Prerequisites

### System Requirements

**Minimum:**
- Python 3.11+
- 16 GB RAM (for replay buffer)
- 20 GB disk space
- macOS, Linux, or Windows with WSL

**Recommended:**
- NVIDIA GPU (RTX 2080+ or equivalent)
- 32 GB RAM
- 50 GB disk space
- Ubuntu 20.04+ or macOS 13+

### Software Dependencies

The script will install these automatically, but ensure:
```bash
# Python version
python3 --version  # 3.11 or higher

# pip available
pip --version

# Git (for provenance tracking)
git --version
```

---

## Usage

### Basic Usage

```bash
# Full Pong reproduction (10M frames, ~30-60 hours CPU)
./experiments/dqn_atari/scripts/reproduce_dqn.sh --game pong --seed 42

# Quick test (1M frames, ~3-6 hours CPU)
./experiments/dqn_atari/scripts/reproduce_dqn.sh --game pong --seed 42 --frames 1000000

# Different game (50M frames for Breakout)
./experiments/dqn_atari/scripts/reproduce_dqn.sh --game breakout --seed 42

# Different seed
./experiments/dqn_atari/scripts/reproduce_dqn.sh --game pong --seed 123
```

### Advanced Options

```bash
# Skip environment setup (already configured)
./experiments/dqn_atari/scripts/reproduce_dqn.sh --skip-setup

# Skip ROM installation (already have ROMs)
./experiments/dqn_atari/scripts/reproduce_dqn.sh --skip-roms

# Disable Weights & Biases logging
./experiments/dqn_atari/scripts/reproduce_dqn.sh --disable-wandb

# Skip plot generation
./experiments/dqn_atari/scripts/reproduce_dqn.sh --skip-plots

# Preview what would be executed
./experiments/dqn_atari/scripts/reproduce_dqn.sh --dry-run

# Show help
./experiments/dqn_atari/scripts/reproduce_dqn.sh --help
```

### Multi-Seed Experiments

For statistical significance, run multiple seeds:

```bash
# Sequential (one at a time)
for seed in 42 123 456; do
    ./experiments/dqn_atari/scripts/reproduce_dqn.sh --game pong --seed $seed
done

# Or use the multi-seed launcher
./experiments/dqn_atari/scripts/run_multi_seed.sh pong "42 123 456"
```

---

## Configuration

### Game Configurations

| Game | Config File | Default Frames | Paper Score |
|------|-------------|----------------|-------------|
| Pong | `pong.yaml` | 10,000,000 | 20 |
| Breakout | `breakout.yaml` | 50,000,000 | 168 |
| Beam Rider | `beam_rider.yaml` | 50,000,000 | 4092 |

### Override Parameters

The script uses YAML configs but allows overrides:

```bash
# Change total frames
./experiments/dqn_atari/scripts/reproduce_dqn.sh --frames 5000000

# Other parameters via config files
cat experiments/dqn_atari/configs/pong.yaml
```

### Environment Variables

```bash
# Weights & Biases API key
export WANDB_API_KEY="your-key-here"

# Or use .env file
echo "WANDB_API_KEY=your-key" >> .env
```

---

## Output Structure

After running, outputs are organized as:

```
experiments/dqn_atari/runs/<game>_<seed>_<timestamp>/
├── config.yaml           # Resolved configuration
├── meta.json            # Run metadata
├── system_info.txt      # Environment provenance
├── git_info.txt         # Git repository state
├── training.log         # Full training output
├── csv/
│   ├── episodes.csv     # Per-episode statistics
│   ├── training_steps.csv  # Per-step metrics
│   └── evaluation.csv   # Evaluation results
├── checkpoints/
│   ├── checkpoint_250000.pt
│   ├── checkpoint_500000.pt
│   └── best_model.pt
├── videos/
│   └── *.mp4            # Evaluation gameplay
└── logs/
    └── *.txt            # Detailed logs

output/plots/<game>_<seed>/
├── returns.png          # Episode returns
├── episode_length.png   # Episode lengths
├── loss.png            # Training loss
└── q_values.png        # Q-value estimates
```

---

## Runtime Estimates

### CPU Training

| Game | Frames | Seeds | Est. Time | Total |
|------|--------|-------|-----------|-------|
| Pong | 10M | 3 | ~30-60 hrs/seed | 90-180 hrs |
| Breakout | 50M | 3 | ~150-300 hrs/seed | 450-900 hrs |
| Beam Rider | 50M | 3 | ~150-300 hrs/seed | 450-900 hrs |

### GPU Training (RTX 3080)

| Game | Frames | Seeds | Est. Time | Total |
|------|--------|-------|-----------|-------|
| Pong | 10M | 3 | ~5-10 hrs/seed | 15-30 hrs |
| Breakout | 50M | 3 | ~25-50 hrs/seed | 75-150 hrs |
| Beam Rider | 50M | 3 | ~25-50 hrs/seed | 75-150 hrs |

**Note:** Times are estimates. Actual performance depends on hardware.

---

## Verification

### Check Training Progress

```bash
# View latest frame count
tail -1 experiments/dqn_atari/runs/<run>/csv/training_steps.csv

# Monitor FPS
tail -f experiments/dqn_atari/runs/<run>/training.log | grep "FPS"

# Check evaluation results
cat experiments/dqn_atari/runs/<run>/csv/evaluation.csv
```

### Validate Results

After training completes:

```bash
# Generate summary table
python scripts/export_results_table.py \
    --runs-dir experiments/dqn_atari/runs

# Compare to paper scores
python scripts/analyze_results.py \
    --eval-csv <run>/csv/evaluation.csv \
    --paper-score 20  # For Pong
```

### Expected Results

**Pong (10M frames):**
- Random policy: -21 to -20
- After 5M frames: > 0
- After 10M frames: 18-21
- Paper score: 20
- Success: >= 90% of paper (>= 18)

**Breakout (50M frames):**
- Random policy: ~1-2
- After 25M frames: > 50
- After 50M frames: 150-180
- Paper score: 168
- Success: >= 90% of paper (>= 151)

---

## Troubleshooting

### Common Issues

**1. Out of memory:**
```bash
# Reduce replay buffer size in config
# Edit experiments/dqn_atari/configs/<game>.yaml
replay_buffer:
    capacity: 500000  # Reduce from 1M
```

**2. ROMs not found:**
```bash
# Install ROMs manually
AutoROM --accept-license

# Or via pip
pip install ale-py[roms]
```

**3. CUDA not available:**
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Falls back to CPU automatically
```

**4. W&B login required:**
```bash
# Interactive login
wandb login

# Or use API key
export WANDB_API_KEY=your-key
```

**5. Script permission denied:**
```bash
chmod +x experiments/dqn_atari/scripts/reproduce_dqn.sh
```

### Performance Issues

**Slow training (< 100 FPS on CPU):**
- Normal for CPU training
- GPU provides 10-100x speedup
- Check for background processes

**Memory usage growing:**
- Replay buffer fills up (expected)
- Should stabilize at ~10-12 GB for 1M buffer

**No learning after 1M frames:**
- Check epsilon decay schedule
- Verify reward clipping enabled
- Monitor Q-values for explosion

---

## Advanced Usage

### Custom Configurations

Create custom config by copying and modifying:

```bash
cp experiments/dqn_atari/configs/pong.yaml \
   experiments/dqn_atari/configs/pong_custom.yaml

# Edit the new config
# Then run with custom config
python train_dqn.py --cfg experiments/dqn_atari/configs/pong_custom.yaml
```

### Resuming Training

If training was interrupted:

```bash
python train_dqn.py \
    --cfg experiments/dqn_atari/configs/pong.yaml \
    --resume experiments/dqn_atari/runs/<run>/checkpoints/latest.pt
```

### W&B Integration

```bash
# View runs
wandb runs Cooper-Union/dqn-atari

# Download artifacts
wandb artifact get Cooper-Union/dqn-atari/model:latest

# Compare runs
# https://wandb.ai/Cooper-Union/dqn-atari
```

---

## Post-Reproduction Analysis

### Generate Comparison Report

```bash
# Create summary tables
python scripts/export_results_table.py \
    --runs-dir experiments/dqn_atari/runs \
    --output output/summary/metrics.csv

# Create comparison plots
python scripts/plot_results.py \
    --episodes <run>/csv/episodes.csv \
    --output output/plots/
```

### Calculate Paper Percentage

```python
import pandas as pd

# Load evaluation results
df = pd.read_csv("experiments/dqn_atari/runs/<run>/csv/evaluation.csv")

# Get final performance
final_mean = df.tail(5)["mean_return"].mean()
paper_score = 20  # Pong

percentage = (final_mean / paper_score) * 100
print(f"Achievement: {percentage:.1f}% of paper score")
```

### Multi-Seed Statistics

```python
import numpy as np

# Aggregate across seeds
scores = [19.5, 20.2, 18.8]  # From different seeds
mean = np.mean(scores)
std = np.std(scores)
ci_95 = 1.96 * std / np.sqrt(len(scores))

print(f"Mean: {mean:.2f} +/- {ci_95:.2f} (95% CI)")
```

---

## References

- Reproduction script: `experiments/dqn_atari/scripts/reproduce_dqn.sh`
- Results comparison: `../reports/report-results-comparison.md`
- Environment notes: `docs/reference/environment-notes.md`
- Game suite plan: `plan-game-suite.md`
- Training configs: `experiments/dqn_atari/configs/`
- DQN 2013 paper: arXiv:1312.5602
