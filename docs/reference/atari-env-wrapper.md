# Atari Environment Wrapper Specification

Comprehensive guide to the Atari preprocessing pipeline for DQN reproduction. This document covers the wrapper chain, configuration options, expected outputs, and troubleshooting common issues.

---

**Prerequisites:**
- Completed [DQN Setup](dqn-setup.md) - ROMs installed and environment working
- Gymnasium basics - Understanding env.step() and env.reset()

**Related Docs:**
- [DQN Model](dqn-model.md) - Expected input shape (4, 84, 84)
- [Episode Handling](episode-handling.md) - Life-loss termination vs full episodes
- [Training Loop](training-loop-runtime.md) - Frame counting and training/eval modes

---

## Wrapper Chain

The environment preprocessing follows the DQN 2013 paper specification. Wrappers are applied in this order:

```
Base Environment (ALE/Game-v5)
    ↓
1. NoopResetEnv          - Random no-op starts (0-30 steps)
    ↓
2. MaxAndSkipEnv         - Action repeat (4x) + max-pooling
    ↓
3. EpisodeLifeEnv        - OPTIONAL: Life loss as terminal (only if episode_life=True)
    ↓
4. RewardClipper         - Clip rewards to {-1, 0, +1}
    ↓
5. AtariPreprocessing    - Grayscale + resize to 84×84
    ↓
6. FrameStack            - Stack last 4 frames
    ↓
Final Environment
```

**Important:** Step 3 (EpisodeLifeEnv) is OPTIONAL and only applied when `episode_life=True`.
The default behavior is full-episode termination (episode_life=False).

**Implementation:** `src/envs/atari_wrappers.py`

## Default Termination Behavior

**Key Point:** Full-episode termination is the DEFAULT. Episodes end only on game over.

### Episode Life Wrapper Policy

The `EpisodeLifeEnv` wrapper is **OPTIONAL** and controlled by the `episode_life` parameter:

- **DEFAULT (episode_life=False):**
  - Full-episode termination only
  - Life loss does NOT end the episode
  - Recommended for evaluation (always use False)
  - Can be used for training (pure full-episode learning)

- **OPTIONAL (episode_life=True):**
  - Life loss treated as episode termination
  - Optional training optimization (helps agent learn to preserve lives)
  - NOT required by the DQN paper
  - NEVER use during evaluation

### Configuration Guidelines

```yaml
# Training mode
training:
  episode_life: false  # DEFAULT: Full episodes (pure DQN paper approach)
  # OR
  episode_life: true   # OPTIONAL: Life loss as terminal (common optimization)

# Evaluation mode (ALWAYS use false)
eval:
  episode_life: false  # REQUIRED: Full episodes for true returns
```

**Why this matters:** Using `episode_life=True` during evaluation will produce incorrect episode returns because the episode will end prematurely on life loss instead of game over.

## Wrapper Specifications

### 1. NoopResetEnv

**Purpose:** Create diverse initial states by executing random no-ops on reset.

**Parameters:**
- `noop_max`: Maximum number of no-ops (default: 30)

**Behavior:**
- On reset: Execute random number of no-ops (0 to `noop_max` inclusive)
- Follows Bellemare/Mnih evaluation protocol specification
- Action 0 must be NOOP (verified via assertion)
- If episode ends during no-ops, reset again
- Set `noop_max=0` to disable no-op resets entirely

**Config:** `config.env.max_noop_start`

**Note:** The range is [0, noop_max] inclusive, meaning the agent could start immediately (0 no-ops) or after up to 30 no-ops, providing maximum diversity in initial states.

### 2. MaxAndSkipEnv

**Purpose:** Repeat actions and reduce flicker via max-pooling.

**Parameters:**
- `skip`: Number of action repeats (default: 4)

**Behavior:**
- Repeat each action K times (default: 4) and accumulate rewards
- Store last 2 **raw RGB frames** (before preprocessing) in buffer
- Return element-wise maximum of last 2 frames
- Max-pooling applied to raw frames ensures flicker reduction
- Exit early if episode terminates

**Why max-pooling over last 2 frames?** Atari games flicker sprites on alternating frames due to hardware limitations (too many sprites per scanline). Taking the element-wise maximum of the last 2 frames ensures all objects are visible in the returned frame.

**Config:** `config.env.frameskip`

### 3. EpisodeLifeEnv

**Purpose:** OPTIONAL: Treat life loss as episode end during training.

**Parameters:** None (controlled by `episode_life` parameter in `make_atari_env`)

**DEFAULT BEHAVIOR:** This wrapper is NOT applied by default. Full-episode termination is the default behavior (life loss does NOT end the episode).

**Behavior when enabled (episode_life=True):**
- Track lives via `env.unwrapped.ale.lives()`
- When lives decrease: set `terminated=True`
- Track true episode end internally via `was_real_done`
- On reset: Only reset env if true episode ended, else NOOP step

**Important:**
- **DEFAULT:** `episode_life=False` (full episodes, life loss NOT terminal)
- **OPTIONAL:** `episode_life=True` (life loss as terminal, training optimization)
- This is an optional training optimization NOT required by the DQN paper
- NEVER use during evaluation! Always set `episode_life=False` for eval

**Config:**
- `config.training.episode_life` (training mode, optional optimization)
- `config.eval.episode_life` (evaluation mode, must be False)

**Usage:**
- Training (optional): `episode_life=True` (helps agent learn to preserve lives)
- Training (default): `episode_life=False` (pure full-episode training)
- Evaluation (required): `episode_life=False` (true episode returns)

### 4. RewardClipper

**Purpose:** Normalize reward scales across different games.

**Parameters:**
- `clip_rewards`: Enable/disable clipping (default: True)

**Behavior:**
- Positive rewards → +1
- Negative rewards → -1
- Zero rewards → 0

**Default:** Reward clipping is **ON by default** as specified in the DQN 2013 paper. This normalizes reward scales across different Atari games.

**Config:** `config.training.reward_clip` (default: `true`)

**Ablation:** Set to `false` to study effect of reward clipping.

### 5. AtariPreprocessing

**Purpose:** Convert frames to DQN-compatible format.

**Parameters:**
- `frame_size`: Target size (default: 84)
- `grayscale`: Convert to grayscale (default: True)

**Behavior:**
- Convert RGB to grayscale using OpenCV luminance formula: `Y = 0.299*R + 0.587*G + 0.114*B`
- Resize to 84×84 using bilinear interpolation (no cropping)
- Full frame is resized including score bar (score information is kept)
- Return uint8 [0, 255] for memory efficiency

**Resize vs Crop:** This implementation uses **resize** (not crop). The original 210×160 frame is resized to 84×84, preserving the entire frame including the score bar. The DQN 2013 paper does not specify cropping, so the score information remains visible to the agent.

**Config:** `config.preprocess.frame_size`

### 6. FrameStack

**Purpose:** Provide temporal information via frame history.

**Parameters:**
- `num_stack`: Number of frames to stack (default: 4)
- `save_samples`: Save sample stacks as PNGs (default: False)
- `sample_dir`: Directory for samples
- `max_samples`: Maximum number of sample stacks to save (default: 5)

**Behavior:**
- Maintain deque of last K frames
- Stack frames in channels-first format: `(K, H, W)`
- Store as uint8 for memory efficiency
- Provide `to_float32()` method for conversion

**Config:** `config.preprocess.stack_size`

## Expected Tensor Shapes

### Throughout the Pipeline

| Stage | Shape | Dtype | Range |
|-------|-------|-------|-------|
| Base env output | `(210, 160, 3)` | uint8 | [0, 255] |
| After NoopResetEnv | `(210, 160, 3)` | uint8 | [0, 255] |
| After MaxAndSkipEnv | `(210, 160, 3)` | uint8 | [0, 255] |
| After EpisodeLifeEnv | `(210, 160, 3)` | uint8 | [0, 255] |
| After RewardClipper | `(210, 160, 3)` | uint8 | [0, 255] |
| After AtariPreprocessing | `(84, 84)` | uint8 | [0, 255] |
| After FrameStack | `(4, 84, 84)` | uint8 | [0, 255] |
| **Final observation** | `(4, 84, 84)` | uint8 | [0, 255] |
| For neural network | `(4, 84, 84)` | float32 | [0.0, 1.0] |

**Conversion:** Use `FrameStack.to_float32(obs)` to convert uint8 to float32.

## Configuration Reference

All settings in `experiments/dqn_atari/configs/base.yaml`:

```yaml
env:
  id: "ALE/Pong-v5"
  frameskip: 4                    # MaxAndSkipEnv (disable Gym's built-in)
  repeat_action_probability: 0.0   # Deterministic
  max_noop_start: 30              # NoopResetEnv (0-30 no-ops)

preprocess:
  frame_size: 84                   # AtariPreprocessing
  grayscale: true                  # AtariPreprocessing
  stack_size: 4                    # FrameStack

training:
  # Episode termination policy:
  # - episode_life: false (DEFAULT) -> Full episode, life loss NOT terminal
  # - episode_life: true (OPTIONAL) -> Life loss as terminal (training optimization)
  episode_life: true               # OPTIONAL: EpisodeLifeEnv (not required by DQN paper)
  reward_clip: true                # RewardClipper

eval:
  # Episode termination policy for evaluation:
  # - episode_life: false (REQUIRED) -> Full episodes for true returns
  episode_life: false              # DEFAULT: Full episodes for evaluation
```

## Artifact Locations

### Dry-Run Outputs

Generated in: `experiments/dqn_atari/runs/{experiment_name}_{seed}/`

| File | Description |
|------|-------------|
| `rollout_log.json` | Comprehensive debug log with obs shapes, reward stats, preprocessing details |
| `dry_run_report.json` | Minimal evaluation report (backward compatibility) |
| `action_list.json` | Action space size and meanings |
| `meta.json` | Git hash, config, seed, ALE settings |
| `frames/reset_*_frame_*.png` | Preprocessed frame samples (4 per stack, up to 5 stacks) |

### Subtask 2 Debug Artifacts

The wrapper chain validation (Subtask 2) produces debug artifacts that verify correct preprocessing behavior. These artifacts are stored in game-specific directories for easy inspection.

**Location:** `experiments/dqn_atari/artifacts/frames/<game>/`

Each game directory contains:
- **Frame PNGs:** Individual frames from each preprocessing stage (reset_0_frame_0.png, reset_0_frame_1.png, etc.)
- **Rollout logs:** Complete preprocessing metadata and reward statistics
- **Meta files:** Configuration and environment settings used for the run

**Purpose:**
- Verify wrapper chain applies transformations correctly
- Inspect grayscale conversion and resizing quality
- Confirm frame stacking creates proper 4-frame sequences
- Debug preprocessing issues before full training runs

**Regenerating artifacts:**

```bash
# Basic dry run (Pong, 3 episodes)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# Custom seed and episodes
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/breakout.yaml \
  --dry-run --seed 42 --dry-run-episodes 5

# Different game
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/beam_rider.yaml --dry-run
```

**When to regenerate:**
- After modifying wrapper chain code
- Before training a new game for the first time
- When debugging unexpected preprocessing behavior
- After changing frame_size, stack_size, or frameskip parameters

**Expected output:** Each dry run creates a timestamped directory containing frames and logs. The frame PNGs should show:
- Correct grayscale conversion (no color artifacts)
- Proper 84×84 resizing (no distortion or cropping errors)
- Sequential frames with visible motion (verifying frame stacking)
- Consistent preprocessing across all resets

## Troubleshooting

### Life Loss Mismatch

**Symptom:** Episode returns don't match expected values; agent seems to get penalized for losing lives.

**Diagnosis:**
- Check if `EpisodeLifeEnv` is enabled during evaluation
- Look for `episode_life: true` in eval config

**Solution:**
```python
# Training
env = create_env(config, episode_life=True)

# Evaluation
env = create_env(config, episode_life=False)
```

**Verification:**
- Check `rollout_log.json` → `preprocessing.episode_life` field
- Dry run always uses `episode_life=False`

### Flicker Artifacts

**Symptom:** Objects appear to flicker or disappear in preprocessed frames.

**Diagnosis:**
- Max-pooling disabled or misconfigured
- Frame skip set to 1

**Solution:**
- Ensure `MaxAndSkipEnv` is applied before preprocessing
- Set `config.env.frameskip=4`
- Verify wrapper order in `make_atari_env()`

**Verification:**
```bash
# Check preprocessed frames
ls experiments/dqn_atari/runs/*/frames/
# Should see smooth frames without flickering sprites
```

### Reward Clipping Issues

**Symptom:** Reward statistics don't show {-1, 0, +1} values; wider range observed.

**Diagnosis:**
- Reward clipping disabled
- Clipping applied in wrong order

**Solution:**
- Set `config.training.reward_clip=true`
- Ensure `RewardClipper` is before `FrameStack` but after `MaxAndSkipEnv`

**Verification:**
```json
// In rollout_log.json
"reward_statistics": {
  "clipped_rewards": {
    "unique_values": [-1.0, 0.0, 1.0],  // Should only be these three
    "counts": {...}
  }
}
```

### Wrong Observation Shape

**Symptom:** Neural network expects different shape; errors about tensor dimensions.

**Diagnosis:**
- Frame stack size mismatch
- Channels-first vs channels-last confusion

**Solution:**
- Verify `config.preprocess.stack_size=4`
- Ensure using channels-first format: `(4, 84, 84)` not `(84, 84, 4)`
- Check `env.observation_space.shape`

**Verification:**
```python
obs, _ = env.reset()
assert obs.shape == (4, 84, 84), f"Got {obs.shape}"
assert obs.dtype == np.uint8
```

### Action Repeat Not Working

**Symptom:** Training seems 4x slower; frame counts don't match expected.

**Diagnosis:**
- Using Gymnasium's built-in frameskip instead of wrapper
- Double frame-skipping

**Solution:**
- Set Gymnasium frameskip to 1: `gym.make(..., frameskip=1)`
- Set wrapper frameskip to 4: `MaxAndSkipEnv(env, skip=4)`
- Never use both simultaneously

**Verification:**
```json
// In rollout_log.json
"preprocessing": {
  "frame_skip": 4,
  "max_pooling": "last 2 frames"
}
```

### No-Op Randomness Issues

**Symptom:** Training episodes start identically; no diversity in initial states.

**Diagnosis:**
- `NoopResetEnv` disabled or `noop_max=0`
- Seed not changing across episodes
- Episodes always starting with same number of no-ops

**Solution:**
- Set `config.env.max_noop_start=30` (Bellemare/Mnih protocol)
- Ensure `NoopResetEnv` is first wrapper applied
- Verify range is [0, noop_max] inclusive (not [1, noop_max])
- Each reset samples uniformly from [0, 30] for maximum diversity

**Verification:**
```json
// In rollout_log.json
"preprocessing": {
  "noop_max": 30
}
```

**Expected behavior:**
- Some episodes start immediately (0 no-ops)
- Some episodes have up to 30 no-op delay
- Uniform distribution provides diverse initial states

### Memory Issues

**Symptom:** OOM errors during training; high memory usage.

**Diagnosis:**
- Storing frames as float32 instead of uint8
- Replay buffer storing uncompressed observations

**Solution:**
- Keep frames as uint8: `obs.dtype == np.uint8`
- Only convert to float32 when feeding to network
- Use `FrameStack.to_float32(obs)` for conversion

**Memory savings:**
- uint8: 4 × 84 × 84 = 28,224 bytes per obs
- float32: 4 × 84 × 84 × 4 = 112,896 bytes per obs
- **4x reduction** with uint8 storage

## Toggling Key Behaviors

### Disable Reward Clipping (Ablation)

```yaml
# In per-game config (e.g., pong.yaml)
training:
  reward_clip: false
```

### Disable Episode Life (Evaluation)

```python
# In code
env = create_env(config, episode_life=False)
```

### Change Frame Stack Size

```yaml
preprocess:
  stack_size: 8  # Use 8 frames instead of 4
```

### Adjust Action Repeat

```yaml
env:
  frameskip: 2  # Reduce to 2 (faster but less stable)
```

### Disable No-Op Starts

```yaml
env:
  max_noop_start: 0  # Deterministic starts (not recommended)
```

## Validation Checklist

Before training, verify wrapper setup:

- [ ] Observation shape is `(4, 84, 84)` in uint8
- [ ] Action space matches expected size (minimal action set)
- [ ] Rewards are clipped to {-1, 0, +1} (training mode)
- [ ] Frame samples saved successfully during dry run
- [ ] `rollout_log.json` shows correct preprocessing settings
- [ ] Episode life enabled for training, disabled for evaluation
- [ ] Max-pooling active (frame skip = 4)
- [ ] No-op starts working (1-30 random no-ops)

## Quick Reference: Wrapper Order Matters

**Correct order:**
1. NoopResetEnv (before any frame modifications)
2. MaxAndSkipEnv (on raw RGB frames for max-pooling)
3. EpisodeLifeEnv (OPTIONAL, before reward/frame processing)
4. RewardClipper (independent of frame processing)
5. AtariPreprocessing (grayscale + resize to 84×84)
6. FrameStack (last, stacks preprocessed frames)

**Why this order?**
- **Max-pooling before preprocessing:** Must operate on raw RGB frames (210×160×3) before grayscale conversion. Max-pooling on grayscale would not properly reduce flicker.
- **Reward clipping independent:** Can be applied anytime, placed after episode life for clarity.
- **Frame stacking last:** Must stack the final preprocessed 84×84 grayscale frames, not raw or intermediate frames.

## Related Documentation

- **DQN Setup:** `docs/design/dqn-setup.md`
- **Wrapper Implementation:** `src/envs/atari_wrappers.py`
- **Config Reference:** `experiments/dqn_atari/configs/base.yaml`
- **Roadmap:** `docs/roadmap.md` (Subtask 2)

## References

- Mnih et al. (2013) "Playing Atari with Deep Reinforcement Learning"
- Gymnasium ALE documentation: https://gymnasium.farama.org/environments/atari/
- OpenCV resize documentation: https://docs.opencv.org/
