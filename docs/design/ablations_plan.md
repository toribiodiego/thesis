# DQN Ablation Study Plan

This document outlines ablation experiments to quantify the impact of key DQN design choices.

## Objective

Validate the importance of specific DQN components by systematically disabling or modifying them:
- Reward clipping
- Frame stacking
- Target network

Each ablation runs on Pong (easiest game) with 5M frames and 3 seeds for statistical significance.

## Ablation Configurations

### 1. Reward Clipping Disabled

**Config**: `experiments/dqn_atari/configs/ablations/reward_clip_off.yaml`

**Hypothesis**: Reward clipping to {-1, 0, +1} normalizes gradients and stabilizes learning. Without clipping, games with large reward magnitudes may exhibit gradient explosions.

**Key Change**:
```yaml
environment:
  preprocessing:
    clip_rewards: false
```

**Expected Impact**:
- Unstable training (high gradient variance)
- Possible divergence on games with large score ranges
- More volatile loss curves

**Severity**: Medium-High (critical for stability)

---

### 2. Reduced Frame Stack (2 frames)

**Config**: `experiments/dqn_atari/configs/ablations/stack_2.yaml`

**Hypothesis**: 4-frame stacking provides temporal context for velocity estimation. With 2 frames, the agent may struggle to infer object motion.

**Key Change**:
```yaml
environment:
  preprocessing:
    frame_stack: 2
```

**Expected Impact**:
- Reduced performance on motion-dependent games
- Pong should show ~10-30% performance drop
- Network input changes from (4,84,84) to (2,84,84)

**Severity**: Medium (degrades but doesn't break)

---

### 3. No Target Network

**Config**: `experiments/dqn_atari/configs/ablations/no_target_net.yaml`

**Hypothesis**: The target network stabilizes learning by preventing "moving target" problem. Without it, Q-targets change with each update.

**Key Change**:
```yaml
training:
  target_network:
    update_interval: 1  # Effectively sync every step
```

**Expected Impact**:
- Oscillating or diverging loss curves
- Slower convergence or complete failure
- Matches original 2013 DQN behavior (before Nature paper)

**Severity**: High (fundamental stability mechanism)

---

## Experimental Protocol

### Training Setup

| Parameter | Value |
|-----------|-------|
| Game | Pong (PongNoFrameskip-v4) |
| Frame Budget | 5,000,000 (5M) |
| Seeds | 42, 123, 456 |
| Evaluation | Every 250K frames |
| Episodes per Eval | 30 |
| Baseline | Pong 10M frames, seed 42 |

### Metrics Collected

Per ablation:
1. **Learning Curves**: Episode return vs frames
2. **Loss Stability**: TD error variance, gradient norms
3. **Final Performance**: Mean eval return (last 5 evals)
4. **Convergence Time**: Frames to reach threshold performance
5. **Stability Indicators**: NaN/Inf occurrences, loss spikes

### Success Criteria

| Ablation | Pass if... | Fail if... |
|----------|-----------|-----------|
| Reward Clip Off | Completes without NaN | Diverges or NaN |
| Stack 2 | Score > 10 (50% of baseline) | Score < 0 |
| No Target Net | Converges (any positive score) | Complete divergence |

## Execution Plan

### Phase 1: Sequential Runs (~15-30 hours CPU each)

```bash
# Baseline (already done)
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42

# Ablation 1: Reward clipping off
python train_dqn.py --cfg experiments/dqn_atari/configs/ablations/reward_clip_off.yaml --seed 42
python train_dqn.py --cfg experiments/dqn_atari/configs/ablations/reward_clip_off.yaml --seed 123
python train_dqn.py --cfg experiments/dqn_atari/configs/ablations/reward_clip_off.yaml --seed 456

# Ablation 2: Frame stack 2
python train_dqn.py --cfg experiments/dqn_atari/configs/ablations/stack_2.yaml --seed 42
# ... repeat for seeds 123, 456

# Ablation 3: No target network
python train_dqn.py --cfg experiments/dqn_atari/configs/ablations/no_target_net.yaml --seed 42
# ... repeat for seeds 123, 456
```

### Phase 2: Analysis

1. **Generate comparison plots**:
```bash
python scripts/plot_results.py \
  --episodes <baseline_csv> \
  --output results/ablations/reward_clip_off/ \
  --game-name "Pong (No Reward Clipping)"
```

2. **Create summary table**:
```markdown
| Ablation | Mean Score | % of Baseline | Stability | Notes |
|----------|-----------|---------------|-----------|-------|
| Baseline | 20.0 | 100% | Stable | Reference |
| No Reward Clip | ??? | ???% | ??? | ??? |
| Stack 2 | ??? | ???% | ??? | ??? |
| No Target Net | ??? | ???% | ??? | ??? |
```

3. **Diagnose failures**: Document root causes in `docs/design/ablation_results.md`

## Expected Findings

Based on literature and DQN paper analysis:

1. **Reward Clipping**: Critical for stability. Without it:
   - Gradient magnitudes vary wildly
   - May work on Pong (scores -21 to +21)
   - Would likely fail on Breakout (scores 0-400+)

2. **Frame Stacking**: Important for temporal reasoning:
   - 2 frames capture basic motion
   - 4 frames capture acceleration/deceleration
   - Pong should work with 2 frames but score lower

3. **Target Network**: Key stabilization mechanism:
   - Without it, TD targets are non-stationary
   - May oscillate but could eventually converge
   - Original 2013 DQN worked without it (different architecture)

## Runtime Estimates

| Ablation | Seeds | Frames | Est. Time (CPU) | Est. Time (GPU) |
|----------|-------|--------|-----------------|-----------------|
| Reward Clip Off | 3 | 5M x 3 | ~30 hrs | ~5 hrs |
| Stack 2 | 3 | 5M x 3 | ~30 hrs | ~5 hrs |
| No Target Net | 3 | 5M x 3 | ~30 hrs | ~5 hrs |
| **Total** | 9 | 45M | **~90 hrs** | **~15 hrs** |

## File Structure

```
experiments/dqn_atari/
├── configs/
│   └── ablations/
│       ├── reward_clip_off.yaml
│       ├── stack_2.yaml
│       └── no_target_net.yaml
└── runs/
    └── ablation_*/
        ├── csv/
        ├── eval/
        ├── checkpoints/
        └── config.yaml

results/ablations/
├── reward_clip_off/
│   ├── plots/
│   └── summary.csv
├── stack_2/
└── no_target_net/
```

## References

- Mnih, V., et al. (2013). Playing Atari with Deep Reinforcement Learning. arXiv:1312.5602
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double Q-learning.

## Future Ablations (Optional)

- **Epsilon schedule**: Test faster/slower decay rates
- **Replay buffer size**: 100K vs 1M capacity
- **Target update frequency**: 1K vs 10K vs 100K steps
- **Network architecture**: Deeper/shallower networks
- **Optimizer**: Adam vs RMSProp
