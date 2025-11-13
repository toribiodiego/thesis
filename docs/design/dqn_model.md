# DQN Model Architecture

## Overview

This document describes the DQN (Deep Q-Network) CNN architecture implementation following Mnih et al. 2013. The model maps stacked Atari frames to Q-values for action selection.

**Key characteristics:**
- Input: `(batch, 4, 84, 84)` - 4 stacked grayscale frames, channels-first format
- Output: Dictionary with `q_values` (batch, num_actions) and `features` (batch, 256)
- Parameters: float32 dtype throughout
- Initialization: Kaiming normal (fan_out mode) for ReLU activations

## Architecture Details

### Layer-by-Layer Tensor Shapes

```
Input:  (B, 4, 84, 84)   - 4 stacked frames, uint8 → float32 [0,1]
   ↓
Conv1:  in=4, out=16, kernel=8×8, stride=4, padding=0
   → ReLU
Output: (B, 16, 20, 20)  - Formula: (84 - 8) / 4 + 1 = 20
   ↓
Conv2:  in=16, out=32, kernel=4×4, stride=2, padding=0
   → ReLU
Output: (B, 32, 9, 9)    - Formula: (20 - 4) / 2 + 1 = 9
   ↓
Flatten: (B, 32×9×9) = (B, 2592)
   ↓
FC:     in=2592, out=256
   → ReLU
Output: (B, 256)         - Feature vector
   ↓
Q-head: in=256, out=num_actions
   → No activation
Output: (B, num_actions) - Q-values for each action
```

### Parameter Count

For a typical Atari game (e.g., Pong with 6 actions):

- Conv1: 4×16×8×8 + 16 = 4,112 parameters
- Conv2: 16×32×4×4 + 32 = 8,224 parameters
- FC: 2592×256 + 256 = 663,808 parameters
- Q-head: 256×num_actions + num_actions

**Total: ~676K parameters** (varies slightly with num_actions)

## Weight Initialization

### Kaiming Normal (He Initialization)

All convolutional and linear layers use Kaiming normal initialization with `fan_out` mode, suitable for ReLU activations:

```python
nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
```

**Rationale:**
- Prevents vanishing/exploding gradients in deep networks
- `fan_out` mode: variance preserved in forward pass
- Designed for ReLU: accounts for half the neurons being zeroed

### Bias Initialization

All biases are initialized to zero:

```python
nn.init.zeros_(module.bias)
```

This is standard practice and allows the network to learn appropriate biases during training.

## Dtype and Device Expectations

### Float32 Throughout

All parameters and computations use `float32`:
- **Parameters:** Stored as float32 for numerical stability
- **Inputs:** Converted from uint8 [0,255] to float32 [0,1] before forward pass
- **Activations:** All intermediate tensors are float32

### Device Handling

The model provides a `to(device)` method that ensures float32 dtype is maintained when moving between devices:

```python
model = DQN(num_actions=6)
model = model.to('cuda')  # Moves to GPU and ensures float32
```

**Important:** Always use the model's `to()` method rather than PyTorch's default to guarantee dtype consistency.

## Input Format Requirements

### Channels-First (NCHW)

PyTorch expects channels-first format: `(batch, channels, height, width)`

**Correct:**
```python
x = torch.rand(2, 4, 84, 84)  # (batch=2, channels=4, H=84, W=84)
output = model(x)
```

**Incorrect:**
```python
x = torch.rand(2, 84, 84, 4)  # Channels last - will fail!
```

### Value Range [0, 1]

Inputs should be normalized to `[0, 1]` range:
```python
# From uint8 frames
frames = np.array([...], dtype=np.uint8)  # [0, 255]
x = torch.from_numpy(frames).float() / 255.0  # [0, 1]
```

## Model Summary Utility

### Usage

Generate a detailed model summary showing layer-by-layer architecture:

```python
from src.utils import model_summary, print_model_summary

model = DQN(num_actions=6)

# Get summary as dict
summary = model_summary(model, input_shape=(4, 84, 84))
print(f"Total parameters: {summary['total_params']:,}")

# Print formatted table
print_model_summary(model, input_shape=(4, 84, 84))
```

**Example output:**
```
======================================================================
Layer (type)               Output Shape         Param #
======================================================================
Conv2d-1                   [1, 16, 20, 20]      4,112
Conv2d-2                   [1, 32, 9, 9]        8,224
Linear-3                   [1, 256]             663,808
Linear-4                   [1, 6]               1,542
======================================================================
Total params: 677,686
Trainable params: 677,686
======================================================================
```

### Shape Validation

Validate output shapes match expectations:

```python
from src.utils import assert_output_shape

model = DQN(num_actions=6)

# Assert q_values have correct shape
assert_output_shape(model, input_shape=(4, 84, 84), expected_output_shape=(6,))

# Or use the model's built-in validator
model.validate_output_shape(batch_size=2)
```

## Checkpoint Save/Load

### Saving Checkpoints

Save model state with optional metadata:

```python
model = DQN(num_actions=6)

# Train model...

# Save with metadata
meta = {
    'step': 100000,
    'episode': 500,
    'epsilon': 0.1,
    'score': 18.5
}
model.save_checkpoint('checkpoints/model_100k.pt', meta=meta)
```

**Checkpoint format:**
```python
{
    'model_state_dict': OrderedDict(...),  # PyTorch state dict
    'num_actions': 6,                      # Action space size
    'meta': {                              # Optional metadata
        'step': 100000,
        'episode': 500,
        ...
    }
}
```

### Loading Checkpoints

Load a saved checkpoint with device-safe loading:

```python
# Load on CPU
model, meta = DQN.load_checkpoint('checkpoints/model_100k.pt', device='cpu')

# Load on GPU
model, meta = DQN.load_checkpoint('checkpoints/model_100k.pt', device='cuda')

# Access metadata
print(f"Loaded from step {meta['step']}, episode {meta['episode']}")
```

**Key features:**
- `device='cpu'|'cuda'`: Specify target device for model
- `strict=True`: Enforce exact key matching in state_dict (default)
- Returns tuple: `(model, metadata_dict)`
- Automatically reconstructs model with correct num_actions

### Inspecting Checkpoints

View checkpoint contents without loading the full model:

```python
import torch

checkpoint = torch.load('checkpoints/model_100k.pt', map_location='cpu')

print(f"Action space size: {checkpoint['num_actions']}")
print(f"Metadata: {checkpoint.get('meta', {})}")
print(f"State dict keys: {list(checkpoint['model_state_dict'].keys())}")
```

## Common Debugging Tips

### 1. NaN/Inf Detection

If you encounter NaN or Inf values during training:

```python
# Check model outputs
output = model(x)
assert not torch.isnan(output['q_values']).any(), "Q-values contain NaNs"
assert not torch.isinf(output['q_values']).any(), "Q-values contain Infs"

# Check gradients after backward pass
loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        if torch.isnan(param.grad).any():
            print(f"NaN gradient in {name}")
        if torch.isinf(param.grad).any():
            print(f"Inf gradient in {name}")
```

**Common causes:**
- Learning rate too high → Exploding gradients
- Input values not normalized → Extreme activations
- Numerical instability in loss function

### 2. Mismatched Action Dimensions

Error: `RuntimeError: size mismatch` in Q-head layer

**Cause:** Model was created/loaded with different num_actions than current environment

**Solution:**
```python
# Always verify action space matches
env = gym.make('PongNoFrameskip-v4')
assert model.num_actions == env.action_space.n, \
    f"Model expects {model.num_actions} actions, env has {env.action_space.n}"

# Or use from_env() to ensure match
model = DQN.from_env(env)
```

### 3. Wrong Input Shape

Error: `RuntimeError: Expected 4D tensor, got 3D tensor`

**Cause:** Missing batch dimension or channels-last format

**Solution:**
```python
# Add batch dimension if single sample
x = torch.rand(4, 84, 84)          # Missing batch dim
x = x.unsqueeze(0)                 # → (1, 4, 84, 84) ✓

# Fix channels-last to channels-first
x = torch.rand(2, 84, 84, 4)       # Channels last
x = x.permute(0, 3, 1, 2)          # → (2, 4, 84, 84) ✓
```

### 4. Dtype Mismatches

Error: `RuntimeError: expected Float but got Byte`

**Cause:** Forgot to convert uint8 frames to float32

**Solution:**
```python
# Convert and normalize
frames = np.array([...], dtype=np.uint8)
x = torch.from_numpy(frames).float() / 255.0  # uint8 → float32 [0,1]
```

### 5. Device Mismatches

Error: `RuntimeError: Expected all tensors on cuda:0, found cpu`

**Cause:** Model and input on different devices

**Solution:**
```python
model = model.to('cuda')
x = x.to('cuda')  # Must match model device
output = model(x)
```

## Testing and Validation

### Running Tests

Execute comprehensive test suite:

```bash
# Run all DQN model tests
pytest tests/test_dqn_model.py -v

# Run specific test
pytest tests/test_dqn_model.py::test_dqn_output_shape -v

# Run manually (no pytest required)
python tests/test_dqn_model.py
```

### Test Coverage

The test suite (`tests/test_dqn_model.py`) covers:

1. **Output shapes** - Verifies correct shapes for Breakout(4), Pong(6), BeamRider(9), etc.
2. **No NaNs/Infs** - Ensures forward pass produces valid values
3. **Gradient flow** - Validates backpropagation works through all layers
4. **MSE backward** - Tests loss computation and backward pass
5. **from_env()** - Tests environment-based constructor
6. **Channels-first** - Enforces correct input format
7. **Initialization** - Verifies Kaiming init and zero biases
8. **Dtype** - Ensures float32 throughout
9. **Device transfer** - Tests moving between devices
10. **Checkpoint save/load** - Tests persistence and restoration
11. **Forward equivalence** - Validates loaded models match originals

### Model Utilities Tests

Additional tests for utilities (`tests/test_model_utils.py`):

```bash
pytest tests/test_model_utils.py -v
python tests/test_model_utils.py
```

## Environment-Aware Constructor

### from_env() Method

Create a DQN model directly from a Gymnasium environment:

```python
import gymnasium as gym
from src.models import DQN

env = gym.make('PongNoFrameskip-v4')
model = DQN.from_env(env)

print(f"Model created with {model.num_actions} actions")  # 6 for Pong
```

**Benefits:**
- Automatically extracts `num_actions` from `env.action_space.n`
- Ensures model matches environment's action space
- Cleaner code, fewer magic numbers

## Integration with Training Loop

### Typical Usage Pattern

```python
import gymnasium as gym
from src.models import DQN

# 1. Create environment and model
env = gym.make('PongNoFrameskip-v4')
model = DQN.from_env(env).to('cuda')

# 2. Get preprocessed observation (after frame stacking)
obs = env.reset()  # Already (4, 84, 84) from wrappers
obs_tensor = torch.from_numpy(obs).unsqueeze(0).float().to('cuda') / 255.0

# 3. Forward pass
model.eval()
with torch.no_grad():
    output = model(obs_tensor)
    q_values = output['q_values']
    action = q_values.argmax(dim=1).item()

# 4. Take action
next_obs, reward, done, info = env.step(action)

# 5. Training step
model.train()
# ... (compute loss, backward, optimizer step)

# 6. Periodic checkpoint
if step % 10000 == 0:
    model.save_checkpoint(
        f'checkpoints/model_{step}.pt',
        meta={'step': step, 'epsilon': epsilon}
    )
```

## Reproducing Model Summary

Generate and save model architecture summary:

```bash
# Using Python script (create if needed)
python scripts/model_summary.py --game pong --output docs/model_arch.txt

# Or interactively
python -c "
from src.models import DQN
from src.utils import print_model_summary

model = DQN(num_actions=6)
print_model_summary(model, (4, 84, 84))
"
```

## Design Decisions and Rationale

### Why channels-first?

PyTorch's CUDA kernels are optimized for NCHW (channels-first) format. While channels-last can be more cache-friendly on CPU, channels-first is the standard for PyTorch and ensures maximum GPU performance.

### Why return dict with features?

Returning both `q_values` and `features` enables:
- **Debugging:** Inspect learned representations
- **Visualization:** t-SNE/PCA of feature embeddings
- **Analysis:** Track feature statistics during training
- **Research:** Easier to extend (e.g., dueling DQN uses features)

The dict format is more extensible than returning only Q-values.

### Why store metadata in checkpoints?

Metadata enables:
- **Resume training:** Restore epsilon, learning rate schedule state
- **Experiment tracking:** Know exact training state of saved model
- **Reproducibility:** Record step, episode, hyperparameters
- **Debugging:** Correlate checkpoint with training logs

### Why Kaiming init over Xavier?

Kaiming (He) initialization is specifically designed for ReLU activations, which zero out half the neurons. Xavier initialization assumes symmetric activations (tanh, sigmoid) and would initialize with too-small weights for ReLU networks, leading to slower convergence.

## References

- **Mnih et al. 2013** - "Playing Atari with Deep Reinforcement Learning" (original DQN paper)
- **He et al. 2015** - "Delving Deep into Rectifiers" (Kaiming initialization paper)
- **PyTorch Docs** - https://pytorch.org/docs/stable/nn.html

## Quick Reference

```python
# Create model
from src.models import DQN
model = DQN(num_actions=6)                    # Manual
model = DQN.from_env(env)                     # From environment

# Forward pass
output = model(x)                             # x: (B, 4, 84, 84) float32 [0,1]
q_values = output['q_values']                 # (B, num_actions)
features = output['features']                 # (B, 256)

# Device transfer
model = model.to('cuda')                      # Ensures float32

# Save/load
model.save_checkpoint('path.pt', meta={...})
model, meta = DQN.load_checkpoint('path.pt', device='cuda')

# Validation
model.validate_output_shape(batch_size=2)

# Summary
from src.utils import print_model_summary
print_model_summary(model, (4, 84, 84))

# Tests
pytest tests/test_dqn_model.py -v
python tests/test_dqn_model.py
```
