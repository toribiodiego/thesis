# Game Suite Plan

This document outlines the target games, baselines, frame budgets, and evaluation schedule for reproducing the DQN 2013 paper results.

## Selected Games

We reproduce results on three representative Atari games from the original paper:

| Game | Environment ID | Action Space | Difficulty | Paper Notes |
|------|----------------|--------------|------------|-------------|
| **Pong** | `PongNoFrameskip-v4` | 6 actions | Easy | Solved quickly, good sanity check |
| **Breakout** | `BreakoutNoFrameskip-v4` | 4 actions | Medium | Requires timing and strategy |
| **Beam Rider** | `BeamRiderNoFrameskip-v4` | 9 actions | Hard | Needs longer-term planning |

### Game Rationale

1. **Pong**: Simplest game, fastest learning curve. Agent should converge quickly and beat human performance. Good for validating infrastructure.

2. **Breakout**: More complex strategy (angle shots, break through ceiling). DQN significantly outperforms human baseline.

3. **Beam Rider**: Requires long-term planning and multi-step strategies. DQN approaches but doesn't match human performance.

## Target Scores (DQN 2013 Paper)

From Table 1 of Mnih et al. (2013) arXiv:1312.5602:

| Game | Random | Sarsa | Contingency | **DQN** | Human |
|------|--------|-------|-------------|---------|-------|
| Pong | -20.4 | -19 | -17 | **20** | -3 |
| Breakout | 1.2 | 5.2 | 6 | **168** | 31 |
| Beam Rider | 354 | 996 | 1743 | **4092** | 7456 |

### Success Criteria

- **Pong**: Score >= 18 (90% of paper's 20)
- **Breakout**: Score >= 150 (89% of paper's 168)
- **Beam Rider**: Score >= 3500 (85% of paper's 4092)

We allow some margin due to environment version differences (ALE v5 vs original) and potential hyperparameter sensitivity.

## Training Configuration

### Frame Budgets

| Game | Training Frames | Training Steps | Expected Wall Time (GPU) | Expected Wall Time (CPU) |
|------|----------------|----------------|---------------------------|---------------------------|
| Pong | 10,000,000 | 2,500,000 | ~2-3 hours | ~12-24 hours |
| Breakout | 50,000,000 | 12,500,000 | ~10-15 hours | ~60-120 hours |
| Beam Rider | 50,000,000 | 12,500,000 | ~10-15 hours | ~60-120 hours |

**Note**: The paper trained for 50M frames per game. For Pong, 10M frames is sufficient as it converges quickly.

### Multi-Seed Runs

Each game will be trained with 3 seeds for statistical significance:
- Seed 42 (primary)
- Seed 123
- Seed 456

Total runs: 3 games x 3 seeds = 9 training runs

### Hardware Requirements

- **GPU (Recommended)**: NVIDIA GPU with 4GB+ VRAM
  - Expected FPS: 300-500 during training
  - Memory: ~2-3GB for replay buffer (1M transitions)

- **CPU (Baseline)**: Apple M1/M2 or modern Intel/AMD
  - Expected FPS: 100-200 during training
  - Memory: Same as GPU

- **Storage**: ~10GB per run (checkpoints, logs, videos)
  - Total for all runs: ~90GB

## Evaluation Protocol

### Evaluation Schedule

| Checkpoint | Frames | Purpose |
|------------|--------|---------|
| 250K | 250,000 | Early learning check |
| 500K | 500,000 | Quarter progress |
| 1M | 1,000,000 | Pong convergence check |
| 2.5M | 2,500,000 | Half progress (50M runs) |
| 5M | 5,000,000 | Breakout/Beam Rider baseline |
| 10M | 10,000,000 | Pong final |
| 25M | 25,000,000 | Three-quarter progress |
| 50M | 50,000,000 | Final evaluation |

### Evaluation Parameters

```yaml
evaluation:
  frequency: 250000  # Every 250K frames
  episodes: 30       # 30 episodes per evaluation
  epsilon: 0.05      # Low exploration during eval
  render_video: true # Record best episode
```

### Metrics Collected

Per evaluation:
- Mean return (primary metric)
- Median return
- Standard deviation
- Min/max return
- Mean episode length
- Best episode video

## Checkpointing Strategy

### Checkpoint Schedule

```yaml
checkpoints:
  interval: 1000000  # Every 1M frames
  keep_best: true    # Save best model separately
  keep_last: 3       # Keep last 3 checkpoints
```

### Saved Artifacts

Per checkpoint:
- `checkpoint_{step}.pt`: Full training state
- `best_model.pt`: Best performing model
- `config.yaml`: Frozen configuration
- `meta.json`: Run metadata and commit hash

## Monitoring and Logging

### Real-Time Monitoring

```bash
# Watch episode returns
tail -f experiments/dqn_atari/runs/<run_id>/csv/episodes.csv

# Watch training metrics
tail -f experiments/dqn_atari/runs/<run_id>/csv/training_steps.csv

# View TensorBoard
tensorboard --logdir experiments/dqn_atari/runs/<run_id>/tensorboard/
```

### W&B Integration

- **Project**: `dqn-atari`
- **Tags**: `[game, seed, baseline]`
- **Artifacts**: CSV logs, eval results, videos, final model

## Expected Results Timeline

### Phase 1: Pong Validation (Week 1)
1. Run 1M frame test on CPU (current)
2. Verify all systems work
3. Launch full 10M Pong run on GPU
4. Expected result: Score ~20, matches paper

### Phase 2: Extended Training (Week 2-3)
1. Launch Breakout (50M frames, 3 seeds)
2. Launch Beam Rider (50M frames, 3 seeds)
3. Monitor for stability (NaN/Inf, gradient explosions)
4. Collect intermediate results

### Phase 3: Analysis (Week 4)
1. Generate multi-seed aggregated plots
2. Compare to paper baselines
3. Document any discrepancies
4. Write results report

## Known Considerations

### Environment Differences

The original DQN paper used ALE 0.4.x. We use:
- Gymnasium 1.1.1
- ALE-py 0.11.2 (ALE v0.11)
- NoFrameskip-v4 environments

Potential differences:
- ROM versions may vary slightly
- Reward structure standardized in newer ALE
- Deterministic vs stochastic frame skip

### Hyperparameter Sensitivity

From literature, these parameters may need adjustment:
- **Learning rate**: Paper uses 2.5e-4, some find 1e-4 more stable
- **Epsilon decay**: Over 1M frames vs 10% of training
- **Target network update**: Every 10K steps vs 10K gradient updates

We use paper defaults initially, document any changes.

## Commands Reference

### Launch Training

```bash
# Pong (10M frames)
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml \
  --seed 42 \
  --set logging.wandb.enabled=true

# Breakout (50M frames)
python train_dqn.py --cfg experiments/dqn_atari/configs/breakout.yaml \
  --seed 42 \
  --set training.total_frames=50000000

# Beam Rider (50M frames)
python train_dqn.py --cfg experiments/dqn_atari/configs/beam_rider.yaml \
  --seed 42 \
  --set training.total_frames=50000000
```

### Generate Plots

```bash
python scripts/plot_results.py \
  --run-dir experiments/dqn_atari/runs/<run_id> \
  --output results/plots/
```

### Export Results

```bash
python scripts/export_results_table.py \
  --runs experiments/dqn_atari/runs/ \
  --output results/summary/
```

## References

- Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. arXiv:1312.5602
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature, 518(7540).
- Bellemare, M. G., et al. (2013). The Arcade Learning Environment. JAIR.
