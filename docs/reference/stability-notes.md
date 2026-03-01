# Stability Notes

This document tracks hyperparameter choices, stability observations, and any tuning performed during DQN reproduction.

## Paper-Default Hyperparameters

The following hyperparameters match the 2013 DQN paper (arXiv:1312.5602):

| Parameter | Value | Source |
|-----------|-------|--------|
| Replay buffer size | 1,000,000 | Section 4 |
| Minibatch size | 32 | Section 4 |
| Learning rate | 2.5e-4 | Section 4 (RMSProp) |
| Discount factor (gamma) | 0.99 | Section 4 |
| Target network update | 10,000 steps | Nature 2015 |
| Optimizer | RMSProp | Section 4 |
| RMSProp rho | 0.95 | Section 4 |
| RMSProp epsilon | 0.01 | Section 4 |
| Initial epsilon | 1.0 | Section 4 |
| Final epsilon | 0.1 | Section 4 |
| Epsilon decay frames | 1,000,000 | Section 4 |
| Training starts | 50,000 frames | Section 4 |
| Frame skip | 4 | Section 4 |
| Frame stack | 4 | Section 4 |

## Stability Observations

### What Works Out of the Box

1. **Pong**: Most stable game for initial testing
   - Consistent reward range (-21 to +21)
   - Clear success/failure signal
   - Relatively simple dynamics

2. **Loss Convergence**: TD loss stabilizes well
   - Initial loss: 10-15 (random policy)
   - Converged loss: 0.3-0.5
   - No divergence observed with paper defaults

3. **Epsilon Decay**: Linear schedule works well
   - 1M frame decay provides gradual exploration
   - Agent explores sufficiently early
   - Final epsilon=0.1 maintains some exploration

### Potential Instabilities

1. **Learning Rate Sensitivity**
   - Too high (>5e-4): Q-values may diverge
   - Too low (<1e-4): Slow convergence
   - Paper default 2.5e-4 is stable

2. **Target Network Update Frequency**
   - Frequent (1K steps): Some oscillation
   - Infrequent (100K steps): Slow adaptation
   - 10K steps balances stability and adaptation

3. **Replay Buffer Size**
   - Small buffer (<100K): High correlation, unstable
   - Large buffer (>1M): Memory intensive but stable
   - 1M is standard, provides good decorrelation

## Games with Known Challenges

### Breakout
- Reward range: 0-400+ (much larger than Pong)
- Reward clipping critical for stability
- May need longer training (>10M frames)

### Beam Rider
- High score variance
- Complex visual dynamics
- May show slower initial learning

### Montezuma's Revenge
- Extremely sparse rewards
- Not suitable for vanilla DQN
- Would require intrinsic motivation or hierarchical methods

## Debugging Checklist

If training is unstable, check:

1. **Loss not decreasing?**
   - Verify target network is updating
   - Check learning rate isn't too small
   - Ensure replay buffer has enough samples

2. **Loss exploding?**
   - Check for NaN/Inf in observations
   - Verify reward clipping is enabled
   - Consider reducing learning rate

3. **Q-values diverging?**
   - Enable gradient clipping
   - Reduce learning rate
   - Check target network sync frequency

4. **No learning signal?**
   - Verify epsilon schedule is decaying
   - Check replay buffer sampling is working
   - Ensure environment rewards are being received

## Hyperparameter Sweep Results

*To be filled in if paper defaults require adjustment*

### Planned Sweeps (if needed)

If instability occurs on any game, run limited sweep:

```bash
# Learning rate sweep
for lr in 1e-4 2.5e-4 5e-4; do
  python train_dqn.py --cfg configs/pong.yaml --set training.optimizer.lr=$lr
done
```

Target: 2M frames per configuration (20% of full run)

### Sweep Parameters

| Parameter | Values to Test | Rationale |
|-----------|----------------|-----------|
| Learning rate | 1e-4, 2.5e-4, 5e-4 | Core stability factor |
| Target update freq | 5K, 10K, 20K | Q-target stability |
| Batch size | 16, 32, 64 | Gradient variance |

## Configuration Files

Paper-default configs:
- `experiments/dqn_atari/configs/pong.yaml`
- `experiments/dqn_atari/configs/breakout.yaml`
- `experiments/dqn_atari/configs/beam_rider.yaml`

All use identical hyperparameters, only game name differs.

## References

- Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning. arXiv:1312.5602
- Mnih et al. (2015). Human-level control through deep reinforcement learning. Nature.
- Henderson et al. (2018). Deep Reinforcement Learning that Matters. AAAI.

## Update Log

| Date | Observation | Action |
|------|-------------|--------|
| 2025-11-15 | Initial 1M frame test | Paper defaults stable, loss converges |
| | Loss: 12 -> 0.32 | No adjustment needed |
| | Returns: -20.7 (expected at 10% budget) | Will verify at 10M frames |

## Recommendations

1. **Start with paper defaults** - They are well-tuned
2. **Monitor loss curves** - Should decrease and stabilize
3. **Check Q-value magnitudes** - Should be bounded (typically <100)
4. **Use reward clipping** - Essential for games with varied score ranges
5. **Be patient** - DQN needs millions of frames to show learning

## Next Steps

- [ ] Complete 10M frame baseline runs for all games
- [ ] Document any game-specific adjustments needed
- [ ] Record final loss values and Q-value ranges
- [ ] Note any unexpected behaviors or failures
