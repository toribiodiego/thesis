# SPR 2021 Notes

> **Status**: REFERENCE | Hyperparameters and design decisions extracted for Atari-100K replication.

- **Citation:** Schwarzer et al., "Data-Efficient Reinforcement Learning with Self-Predictive Representations," ICLR 2021.
- **Objective for thesis:** Extract SPR's auxiliary loss design and hyperparameters. Identify what must change when isolating SPR on vanilla DQN instead of Rainbow.

## Key Contributions

- Self-Predictive Representations (SPR): an auxiliary self-supervised loss that trains the encoder to predict its own future latent states.
- Achieves median human-normalized score of 0.415 on Atari-100K (26 games), a large improvement over prior methods.
- Core mechanism: transition model predicts future encoder outputs; EMA target encoder provides stable prediction targets; cosine similarity loss aligns predictions with targets.
- Multi-step prediction: predict K steps ahead (K=5 default) to encourage temporally extended representations.

## Critical Finding for This Thesis

**SPR's base algorithm is a modified Rainbow, NOT vanilla DQN.**

The paper's "DQN" refers to a stripped-down Rainbow that still includes:
- Distributional Q-learning (C51-style value distribution)
- Dueling network architecture
- Noisy linear layers (for exploration, replacing epsilon-greedy)
- n-step returns (n=10)

This means the published SPR results **cannot be directly compared** to our vanilla DQN + SPR experiments. The whole point of our thesis is isolating SPR's contribution on a minimal base agent, which the original paper does not do.

## Hyperparameters (Table 3, page 14)

These are for the paper's modified Rainbow base, not vanilla DQN.

| Parameter | Value |
|-----------|-------|
| Training steps | 100K environment steps |
| Discount factor | 0.99 |
| Batch size | 32 |
| Optimizer | Adam (LR 0.0001, beta1=0.9, beta2=0.999) |
| Min replay size for sampling | 2,000 |
| Replay period | Every 1 step |
| Updates per step | 2 |
| Multi-step return length (n) | 10 |
| Target network update period | 1 (continuous soft update) |
| Max frames per episode | 108,000 |
| Evaluation episodes | 100 trajectories |
| Distributional Q | Yes (51 atoms) |
| Dueling architecture | Yes |
| Noisy nets | Yes (replaces epsilon-greedy) |

### Q-Network Architecture

```text
Input: 4 x 84 x 84 tensor
Conv1: 32 filters, 8x8 kernel, stride 4
Conv2: 64 filters, 4x4 kernel, stride 2
Conv3: 64 filters, 3x3 kernel, stride 1
FC hidden: 256 units
```

Note: hidden layer is 256 units vs Nature DQN's 512. This is likely due to the dueling architecture splitting into value and advantage streams.

### SPR-Specific Components

- **Projection head:** MLP that maps encoder output to prediction space.
- **Prediction head:** MLP on top of projected representations for the online network.
- **Transition model:** Deterministic MLP that predicts next latent state given current latent state and action.
- **Target encoder:** EMA copy of the online encoder (momentum parameter tau).
- **Loss:** Cosine similarity between predicted future representations and EMA-encoded actual future observations.
- **Prediction horizon (K):** 5 steps ahead.

### Augmentation Variants

The paper tests two configurations:

**With augmentation (SPR + aug):**
- Random shifts: +/- 4 pixels
- Intensity perturbation: scale = 0.05
- Target encoder momentum (tau) = 0 (reset each step)
- No dropout

**Without augmentation (SPR only):**
- No data augmentation applied
- Dropout = 0.5
- Target encoder momentum (tau) = 0.99

This is relevant: SPR without augmentation uses dropout as a regularizer instead, and uses a much slower EMA update rate.

## Wall-Clock Times (Table 8, page 18)

Measured on a single NVIDIA P100 GPU:

| Agent | Time Range |
|-------|------------|
| Rainbow (controlled) | 1.4 -- 2.1 hours |
| SPR | 3.0 -- 4.6 hours |

SPR roughly doubles training time due to the auxiliary loss computation and multi-step predictions.

## Adapting for Vanilla DQN (Open Questions)

To isolate SPR on our vanilla DQN baseline, we need to resolve:

1. **Optimizer:** SPR uses Adam (LR 0.0001). Our DQN uses RMSProp (LR 0.00025). Which do we use for the DQN+SPR condition? Switching optimizers changes two variables at once.
2. **Min replay size:** SPR uses 2,000 vs DQN's 50,000. For Atari-100K (only 100K steps total), 50K warmup wastes half the budget. We likely need to reduce this regardless.
3. **Updates per step:** SPR does 2 updates per step vs DQN's 1 update every 4 steps. This is an 8x difference in gradient updates per environment step.
4. **n-step returns:** SPR uses 10-step returns. Vanilla DQN uses 1-step TD. Do we add n-step returns to our DQN, or keep 1-step and only add the SPR loss?
5. **Target network:** SPR uses tau=1 soft update every step. DQN uses hard copy every 10K steps. The continuous soft update is tied to the distributional setup.
6. **Hidden layer size:** 256 vs 512. May not matter much but worth noting.
7. **Exploration:** SPR uses noisy nets (no epsilon schedule). Our DQN uses epsilon-greedy. The epsilon decay schedule (1M frames in original DQN) needs compression for 100K steps.

### Recommended Approach

Look at the SPR GitHub repo (https://github.com/mila-iqia/spr) for any vanilla DQN config or ablation that strips out Rainbow components. If none exists, we define our own Atari-100K DQN hyperparameters by:
- Keeping Nature DQN architecture and epsilon-greedy exploration
- Reducing min replay to 1K-5K (necessary for 100K budget)
- Compressing epsilon decay to fit within 100K steps
- Using the same evaluation protocol (100 episodes, greedy)
- Keeping 1-step TD (adding n-step would confound attribution)

## Results Reference (Table 4, page 15)

Selected games relevant to our experiments (mean over 10 seeds):

| Game | Random | Human | Rainbow (controlled) | SPR |
|------|--------|-------|---------------------|-----|
| Pong | -20.7 | 14.6 | -16.1 | -5.4 |
| Breakout | 1.7 | 30.5 | 6.5 | 17.1 |
| Seaquest | 68.4 | 42054.7 | 354.1 | 583.1 |

Note: Even SPR's modified Rainbow scores poorly on Pong under Atari-100K. This is expected -- Pong is hard with only 100K steps because the reward signal is very sparse (points only at end of rallies).
