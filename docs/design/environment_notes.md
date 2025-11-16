# Environment and Toolchain Notes

This document records environment/toolchain differences affecting comparability between our DQN reproduction and the original 2013 paper.

## Software Versions

### Core Dependencies

| Package | Our Version | Paper (2013) | Impact |
|---------|-------------|--------------|--------|
| Python | 3.11+ | N/A (Lua/Torch) | Different framework |
| PyTorch | 2.0+ | Torch7 | Modern autograd |
| Gymnasium | 0.29.1 | ALE (C++) | Python wrapper |
| ALE-py | 0.11.2 | ALE 0.4.x | ROM handling differs |
| NumPy | 1.24+ | N/A | Numerical backend |

### Environment Details

```bash
# Verify versions
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')"
python -c "import ale_py; print(f'ALE: {ale_py.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
```

---

## Atari Environment Configuration

### Frame Skip

**Paper**: Fixed 4-frame skip (action repeated for 4 frames)
**Ours**: NoFrameskip-v4 with manual MaxAndSkip wrapper

Difference: We apply max-pooling over last 2 raw frames, then skip. Paper may have used last frame only.

### Action Set

**Paper**: Minimal action set (game-specific)
**Ours**: Full 18-action set

Impact: Larger action space may slow learning slightly due to more Q-values to estimate.

### Terminal Signals

**Paper**: Episode ends on life loss (for training)
**Ours**: Configurable via `episode_life: true`

Impact: More frequent episode boundaries can accelerate learning in games with multiple lives.

### No-Op Starts

**Paper**: Up to 30 random no-ops at episode start
**Ours**: Configurable via `noop_max: 30`

Purpose: Provides diverse initial states for better generalization.

---

## Preprocessing Pipeline

### Grayscale Conversion

**Paper**: "Single channel luminance"
```lua
-- Paper method (Lua/Torch)
0.299 * R + 0.587 * G + 0.114 * B
```

**Ours**: OpenCV grayscale
```python
# Our method
cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
# Uses: 0.299 * R + 0.587 * G + 0.114 * B
```

Impact: Identical formula, negligible difference.

### Frame Resizing

**Paper**: 84x84 with bilinear interpolation
**Ours**: 84x84 with area interpolation (cv2.INTER_AREA)

Impact: Area interpolation may preserve features better for downsampling.

### Frame Stacking

**Paper**: 4 consecutive frames
**Ours**: 4 frames via FrameStack wrapper

Implementation: Identical in concept. Order: oldest to newest.

### Reward Clipping

**Paper**: Clip rewards to {-1, 0, +1}
**Ours**: Configurable via `clip_rewards: true`

Purpose: Normalizes gradient magnitudes across different games.

---

## Network Architecture

### Convolutional Layers

**Paper**:
```
Conv1: 32 filters, 8x8, stride 4
Conv2: 64 filters, 4x4, stride 2
Conv3: 64 filters, 3x3, stride 1
FC: 512 units
Output: num_actions Q-values
```

**Ours**: Identical architecture (see `src/models/dqn_model.py`)

### Activation Functions

**Paper**: ReLU
**Ours**: ReLU

Impact: None.

### Weight Initialization

**Paper**: Not specified in detail
**Ours**: PyTorch defaults (Kaiming uniform for Conv, Xavier for Linear)

Impact: May affect early learning dynamics, but convergence should be similar.

---

## Optimization

### Optimizer

**Paper**: RMSprop
```
learning_rate = 0.00025
momentum = 0.95
epsilon = 0.01
```

**Ours**: Adam
```
learning_rate = 0.00025
beta1 = 0.9
beta2 = 0.999
epsilon = 0.0001
```

**Impact**: Adam is generally more stable. May converge faster but with different dynamics.

### Gradient Clipping

**Paper**: Clip error term to [-1, 1]
**Ours**: Huber loss (delta=1.0) provides equivalent clipping

Impact: Identical in effect - bounds gradients for stability.

### Batch Size

**Paper**: 32
**Ours**: 32

Impact: None.

---

## Replay Buffer

### Capacity

**Paper**: 1,000,000 frames
**Ours**: 1,000,000 frames

### Sampling

**Paper**: Uniform random sampling
**Ours**: Uniform random sampling

Impact: None.

### Storage

**Paper**: Store raw pixels (uint8)
**Ours**: Store uint8 observations

Impact: Memory efficient, identical approach.

---

## Training Schedule

### Epsilon Decay

**Paper**: Linear decay from 1.0 to 0.1 over 1M frames
**Ours**: Same schedule (configurable)

### Target Network

**Paper**: Update every 10,000 steps
**Ours**: Same frequency

### Warmup

**Paper**: 50,000 frames before training
**Ours**: 50,000 frames (replay buffer population)

---

## Hardware Differences

### Original Paper (2013)

- GPU: NVIDIA GTX 580 (3GB)
- Training time: ~2 weeks per game
- Framework: Torch7 (Lua)

### Our Reproduction

- GPU: Varies (NVIDIA RTX series or Apple Silicon)
- Training time: ~30-60 hours (CPU), ~5-15 hours (GPU)
- Framework: PyTorch (Python)

### Performance Implications

1. **Modern GPUs**: 10-100x faster computation
2. **Python overhead**: Slightly slower than Lua/C++
3. **Memory bandwidth**: Improved in modern hardware

---

## Known Discrepancies

### 1. ROM Versions

Different ALE releases may use different ROM dumps:
- Pong: "Video Olympics" ROM
- Breakout: "Super Breakout" variant possible
- Beam Rider: Activision version

**Mitigation**: Use AutoROM for consistent ROM installation.

### 2. Random Seed Handling

Paper does not specify exact seeding procedure. Our approach:
- Set Python random seed
- Set NumPy random seed
- Set PyTorch seed (CPU and CUDA)
- Set ALE seed

### 3. Floating Point Precision

Paper may have used float32 throughout. We use:
- Observations: uint8
- Network: float32
- Loss computation: float32

### 4. Action Repeat Stochasticity

Some Gymnasium versions introduce stochastic frame skipping (sticky actions). We disable this for determinism.

---

## Verification Commands

### Check Environment Setup

```bash
# Verify determinism
python -c "
import gymnasium
env = gymnasium.make('PongNoFrameskip-v4')
env.reset(seed=42)
actions = [env.action_space.sample() for _ in range(10)]
print('Actions:', actions)
"

# Check preprocessing
python -c "
from src.utils.atari_wrappers import make_atari_env
env = make_atari_env('PongNoFrameskip-v4', seed=42)
obs, _ = env.reset()
print('Observation shape:', obs.shape)
print('Dtype:', obs.dtype)
print('Range:', obs.min(), '-', obs.max())
"
```

### Validate Network Architecture

```bash
python -c "
from src.models.dqn_model import DQNModel
model = DQNModel((4, 84, 84), 6)
params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {params:,}')
# Expected: ~1.68M parameters
"
```

---

## Recommendations

1. **Document all differences**: Keep this file updated with each run
2. **Use consistent ROMs**: Install via AutoROM with fixed versions
3. **Seed everything**: Ensure reproducibility within our framework
4. **Monitor for divergence**: Watch for Q-value explosions or loss spikes
5. **Report honestly**: Note where we deviate from paper methodology

---

## References

- DQN 2013 Paper: arXiv:1312.5602
- Nature DQN 2015: doi:10.1038/nature14236
- Gymnasium Documentation: https://gymnasium.farama.org/
- ALE Documentation: https://github.com/Farama-Foundation/Arcade-Learning-Environment
