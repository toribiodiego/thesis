# DQN 2013 Notes

> **Status**: REFERENCE | Completed implementation notes from the foundational DQN paper.

- **Citation:** Mnih et al., 2013 (NIPS Deep Learning Workshop) / 2015 (Nature version with appendix).
- **Objective for thesis:** Reproduce baseline DQN scores as groundwork for sample-efficient extensions.

## Environment & Preprocessing

- Atari Learning Environment (ALE) via Gym/Gymnasium; deterministic frameskip `k=4`.
- Convert RGB frames to grayscale using luminance transform, resize to `84×84` with bilinear interpolation, center-crop.
- Stack the last 4 preprocessed frames per observation to capture temporal dynamics.
- Clip rewards to `{-1, 0, +1}` to stabilize learning.
- Action repeat: execute chosen action for 4 frames (`frame_skip=4`), accumulate clipped rewards.
- Episode start: perform a random number (0–30) of `NOOP` actions before policy control.
- Treat loss of life as terminal during training but not during evaluation rollouts (Nature DQN convention).

## Network Architecture (Nature-CNN)

```text
Input: 4 × 84 × 84 tensor  (frames stacked on channel dimension)
Conv1: 32 filters, 8×8 kernel, stride 4, ReLU
Conv2: 64 filters, 4×4 kernel, stride 2, ReLU
Conv3: 64 filters, 3×3 kernel, stride 1, ReLU
Flatten
FC1: 512 units, ReLU
FC2: |A| units (one per discrete action)
```

- Weights initialized from a uniform distribution in `[-0.05, 0.05]`.
- Biases initialized to `0.0`.

## Core Hyperparameters

| Parameter | Value |
|-----------|-------|
| Replay buffer capacity | 1,000,000 transitions |
| Replay start size | 50,000 (no SGD updates before buffer holds 50k transitions) |
| Mini-batch size | 32 |
| Discount factor `γ` | 0.99 |
| Optimizer | RMSProp (no momentum) with learning rate `2.5e-4` |
| RMSProp parameters | Decay `α=0.95`, epsilon `1e-2` |
| Gradient clipping | Not used in original paper (keep max-norm option configurable) |
| Target network update | Every 10,000 parameter updates (hard copy) |
| Update frequency | One SGD update every 4 environment steps |
| Exploration schedule | ε-greedy from `ε=1.0` → `0.1` over 1,000,000 frames; fixed `ε=0.1` thereafter; evaluation `ε=0.05` |
| Max frames per run | 200,000,000 frames (50M agent steps × 4) |
| Episode termination | On true game over; during training treat life loss as terminal |
| Reward clipping | Apply per-step clip to `[-1, 1]` |
| Replay sampling | Uniform (no prioritization in baseline DQN) |
| Gradient updates per frame | 1/4 (due to update frequency) |

## Evaluation Protocol

- Every fixed interval (e.g., 250k training frames), run evaluation episodes with `ε=0.05` and no reward clipping.
- Report average score over 30 independent episodes per game.
- Track additional metrics: cumulative reward vs. frames, Q-value statistics, loss curves.

## Implementation Nuances & Risks

- **Determinism:** Seed ALE, NumPy, PyTorch; document any non-deterministic GPU ops.
- **Replay memory:** Store raw uint8 frames to disk/memory and lazily preprocess to conserve memory (compression step optional).
- **Target network lag:** Ensure parameter copy happens exactly every 10,000 gradient updates, not environment steps.
- **Life-loss termination:** Implement wrapper that returns `done=True` on life loss for training transitions only.
- **Frame stacking:** Maintain deque of length 4; reset to zero frames when starting a new episode after true terminal.
- **Logging:** Capture per-episode return, loss, Q-values, ε value, buffer fill ratio.

## Open Items for WP1

1. Confirm RMSProp variant: original Torch7 code used gradient momentum 0.95; replicate using PyTorch’s RMSProp settings.
2. Verify evaluation frequency (Nature appendix used every 1M frames); decide consistent cadence.
3. Gather hardware expectations (original runs used GPU; note approximate runtime on modern GPUs/CPUs).
4. Document ROM procurement process and licensing considerations.

These notes should guide the design decisions for `WP1 – Deep-Dive & Planning` and will inform configuration defaults once implementation begins.
