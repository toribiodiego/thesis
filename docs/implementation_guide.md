# DQN Implementation Guide

This document provides execution-ready implementation details for each subtask in `docs/roadmap.md`. Each section corresponds to a subtask and includes: concrete file paths, code snippets, verification commands, and completion criteria.

**Usage:** When implementing a subtask, refer to `docs/roadmap.md` for objectives and high-level checklist, then consult this guide for detailed execution steps.

---

## Subtask 1 — Choose paper games, pin versions, and scaffold runs

### 1.1 Select games and create config stubs

**Files to create:**
- `experiments/dqn_atari/configs/pong.yaml`
- `experiments/dqn_atari/configs/breakout.yaml`
- `experiments/dqn_atari/configs/beam_rider.yaml`

**Template for each config:**
```yaml
_base_: base.yaml

env:
  id: "ALE/Pong-v5"  # Change per game: Breakout-v5, BeamRider-v5

# Game-specific overrides (if needed)
train:
  max_frames: 10000000

notes: "Selected for DQN baseline: Pong (easy), Breakout (medium), Beam Rider (hard)"
```

**Update `experiments/dqn_atari/README.md`:**
Add section:
```markdown
## Selected Games

| Game       | Rationale                  |
|------------|----------------------------|
| Pong       | Easy baseline, fast convergence |
| Breakout   | Medium difficulty, classic benchmark |
| Beam Rider | Hard, tests complex strategies |
```

**Verification:**
```bash
ls experiments/dqn_atari/configs/*.yaml
# Should list: base.yaml, pong.yaml, breakout.yaml, beam_rider.yaml
```

---

### 1.2 Document environment IDs and ROM setup

**Update `experiments/dqn_atari/README.md`:**
Add table:
```markdown
## Environment Configuration

| Game       | Env ID             | Action Set | Action Count | ROM Required |
|------------|--------------------|------------|--------------|--------------|
| Pong       | ALE/Pong-v5        | minimal    | 6            | Yes          |
| Breakout   | ALE/Breakout-v5    | minimal    | 4            | Yes          |
| Beam Rider | ALE/BeamRider-v5   | minimal    | 9            | Yes          |

## ROM Installation

Run the setup script to download Atari ROMs (legal via AutoROM):

\`\`\`bash
bash scripts/setup_roms.sh
\`\`\`

This will:
1. Install ROMs via AutoROM (automatically accepts license)
2. Verify ROMs are accessible to ALE
3. List available games

Manual verification:
\`\`\`python
import ale_py
print(ale_py.list_games())
\`\`\`
```

**Create `scripts/setup_roms.sh`:**
```bash
#!/bin/bash
set -e

echo "Installing Atari ROMs via AutoROM..."
python -m AutoROM --accept-license

echo "Verifying ROM installation..."
python -c "import ale_py; games = ale_py.list_games(); print(f'Found {len(games)} games'); assert 'pong' in [g.lower() for g in games], 'Pong ROM not found'"

echo "✓ ROMs installed successfully"
echo "Available games:"
python -c "import ale_py; [print(f'  - {g}') for g in sorted(ale_py.list_games())]"
```

**Verification:**
```bash
chmod +x scripts/setup_roms.sh
bash scripts/setup_roms.sh
# Should output: ✓ ROMs installed successfully
```

---

### 1.3 Pin exact dependency versions

**Update `envs/requirements.txt`:**
Verify these exact versions are present (already should be from Subtask scaffolding):
```
# Python 3.10.13 recommended

# Deep Learning
torch==2.4.1
torchvision==0.19.1

# RL Environment
gymnasium[accept-rom-license]==0.29.1
ale-py==0.8.1
AutoROM.accept-rom-license==0.6.1

# Utilities
numpy==1.26.4
opencv-python==4.10.0.84
tqdm==4.66.4
matplotlib==3.9.1
omegaconf==2.3.0
```

**Document ALE settings in `experiments/dqn_atari/README.md`:**
```markdown
## ALE Runtime Configuration

For deterministic, reproducible runs, use these settings when creating environments:

\`\`\`python
env = gymnasium.make('ALE/Pong-v5',
                      repeat_action_probability=0.0,  # Deterministic
                      frameskip=4,                     # Action repeat
                      full_action_space=False)         # Minimal actions
\`\`\`

These are the default settings for DQN reproduction (Mnih et al., 2013).
```

**Create `scripts/capture_env.sh`:**
```bash
#!/bin/bash
# Capture system and environment information

OUTPUT_FILE="experiments/dqn_atari/system_info.txt"

echo "=== System Information ===" > "$OUTPUT_FILE"
echo "Date: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "Python version:" >> "$OUTPUT_FILE"
python --version >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

echo "=== Package Versions ===" >> "$OUTPUT_FILE"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" >> "$OUTPUT_FILE"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" >> "$OUTPUT_FILE"
python -c "import torch; print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" >> "$OUTPUT_FILE"
python -c "import gymnasium; print(f'Gymnasium: {gymnasium.__version__}')" >> "$OUTPUT_FILE"
python -c "import ale_py; print(f'ALE: {ale_py.__version__}')" >> "$OUTPUT_FILE"
python -c "import numpy; print(f'NumPy: {numpy.__version__}')" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=== Hardware ===" >> "$OUTPUT_FILE"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" >> "$OUTPUT_FILE"
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else None" >> "$OUTPUT_FILE"

echo ""
echo "✓ System info saved to $OUTPUT_FILE"
cat "$OUTPUT_FILE"
```

**Verification:**
```bash
chmod +x scripts/capture_env.sh
bash scripts/capture_env.sh
cat experiments/dqn_atari/system_info.txt
# Should show Python, PyTorch, Gymnasium, ALE versions
```

---

### 1.4 Define evaluation protocol in base config

**Update `experiments/dqn_atari/configs/base.yaml`:**
Ensure it contains:
```yaml
# Evaluation Protocol
eval:
  epsilon: 0.05              # Low exploration during evaluation
  episodes: 10               # Episodes per evaluation checkpoint
  frequency: 250000          # Evaluate every N frames
  greedy_mode: false         # If true, use epsilon=0
  record_video: true         # Capture MP4 of first eval episode

# Training Protocol
train:
  termination: "life_loss"   # "life_loss" or "game_over"
  no_op_max: 30              # Random no-ops (0-30) at episode start
  max_frames: 10000000       # Default frame budget
  batch_size: 32             # Minibatch size
  learning_rate: 0.00025     # 2.5e-4
  gamma: 0.99                # Discount factor

# Reward Processing
reward:
  clip: true                 # Clip rewards to {-1, 0, +1}
  clip_min: -1.0
  clip_max: 1.0

# Frame budgets per game (can override in game configs)
budgets:
  pong: 10000000
  breakout: 10000000
  beam_rider: 20000000       # Harder game, more frames
```

**Verification:**
```bash
python -c "import yaml; c = yaml.safe_load(open('experiments/dqn_atari/configs/base.yaml')); assert c['eval']['epsilon'] == 0.05; print('✓ Base config valid')"
```

---

### 1.5 Implement unified seeding and metadata utilities

**Create `src/utils/` directory:**
```bash
mkdir -p src/utils
touch src/utils/__init__.py
```

**Create `src/utils/repro.py`:**
```python
"""Reproducibility utilities: seeding, git tracking, metadata."""

import random
import numpy as np
import torch
import subprocess
import json
from pathlib import Path


def set_seed(seed: int, deterministic: bool = False):
    """
    Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
        deterministic: If True, enable full deterministic mode (slower)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Optionally enable torch.use_deterministic_algorithms(True)
        # Note: may cause errors with some operations


def get_git_hash():
    """Get current git commit hash."""
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
    except:
        return "unknown"


def get_git_diff():
    """Check if there are uncommitted changes."""
    try:
        diff = subprocess.check_output(
            ['git', 'diff', '--stat'],
            stderr=subprocess.DEVNULL
        ).decode('ascii').strip()
        return diff if diff else "clean"
    except:
        return "unknown"


def save_run_metadata(path, config, seed):
    """
    Save run metadata to JSON file.

    Args:
        path: Output file path (e.g., "run_dir/meta.json")
        config: Full configuration dict
        seed: Random seed used
    """
    import time

    metadata = {
        "seed": seed,
        "commit_hash": get_git_hash(),
        "git_diff": get_git_diff(),
        "config": config,
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Run metadata saved: {path}")


if __name__ == '__main__':
    # Self-test
    set_seed(42)
    print(f"Git hash: {get_git_hash()[:7]}")
    print(f"Git status: {get_git_diff()}")
    save_run_metadata('/tmp/test_meta.json', {'test': True}, 42)
    print("✓ repro.py self-test passed")
```

**Verification:**
```bash
python src/utils/repro.py
# Should output: ✓ repro.py self-test passed
```

---

### 1.6 Scaffold run launcher script

**Update `experiments/dqn_atari/scripts/run_dqn.sh`:**
```bash
#!/bin/bash
# DQN training launcher
# Usage: bash experiments/dqn_atari/scripts/run_dqn.sh [GAME] [SEED] [--dry-run]

set -e

# Parse arguments
GAME=${1:-pong}
SEED=${2:-0}
DRY_RUN=${3:-}

CFG="experiments/dqn_atari/configs/${GAME}.yaml"
RUN_DIR="experiments/dqn_atari/runs/${GAME}/seed_${SEED}"

# Validate config exists
if [ ! -f "$CFG" ]; then
    echo "Error: Config not found: $CFG"
    echo "Available configs:"
    ls experiments/dqn_atari/configs/*.yaml
    exit 1
fi

# Create run directory
mkdir -p "${RUN_DIR}"

echo "========================================="
echo "DQN Training Launcher"
echo "========================================="
echo "Game:    ${GAME}"
echo "Seed:    ${SEED}"
echo "Config:  ${CFG}"
echo "Output:  ${RUN_DIR}"
echo "========================================="

# Build command
CMD="python src/train_dqn.py --cfg $CFG --seed $SEED --output $RUN_DIR"

if [ "$DRY_RUN" = "--dry-run" ]; then
    echo "Mode: DRY RUN (random policy, 100 steps)"
    CMD="$CMD --dry-run"
else
    echo "Mode: TRAINING"
fi

echo ""
echo "Executing: $CMD"
echo ""

$CMD
```

**Create placeholder `src/train_dqn.py`:**
```python
#!/usr/bin/env python
"""
DQN Training Script
==================

Entry point for DQN training on Atari games.

Usage:
    python src/train_dqn.py --cfg configs/pong.yaml --seed 0 --output runs/pong/seed_0

Options:
    --cfg: Path to config YAML
    --seed: Random seed
    --output: Output directory
    --dry-run: Run random policy for 100 steps (testing)
    --resume: Checkpoint to resume from
"""

import argparse
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Train DQN on Atari')
    parser.add_argument('--cfg', type=str, required=True,
                       help='Path to config YAML')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for run')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run: random policy for 100 steps')
    parser.add_argument('--resume', type=str, default=None,
                       help='Checkpoint to resume from')
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("DQN Training Script")
    print("=" * 60)
    print(f"Config:  {args.cfg}")
    print(f"Seed:    {args.seed}")
    print(f"Output:  {args.output}")
    print(f"Mode:    {'DRY RUN' if args.dry_run else 'TRAINING'}")
    print("=" * 60)
    print()

    # Create output directories
    output_dir = Path(args.output)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
    (output_dir / 'artifacts').mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[DRY RUN MODE]")
        print("TODO: Implement random policy rollout")
        print("  - Load environment from config")
        print("  - Run 100 random steps")
        print(f"  - Save frames to {output_dir}/artifacts/frames/")
        print(f"  - Save actions to {output_dir}/artifacts/actions.txt")
        print(f"  - Save eval report to {output_dir}/artifacts/eval_report.txt")
        print()
        print("Dry run placeholder complete.")
        print("Full implementation required in Subtask 2 (env wrappers)")
    else:
        print("[TRAINING MODE]")
        print("TODO: Implement full training loop (Subtask 6)")
        print("  - Load config")
        print("  - Create environment with wrappers")
        print("  - Initialize agent and replay buffer")
        print("  - Run training loop")
        print()
        print("Training placeholder complete.")
        print("Full implementation required in Subtasks 2-6")


if __name__ == '__main__':
    main()
```

**Make scripts executable:**
```bash
chmod +x experiments/dqn_atari/scripts/run_dqn.sh
chmod +x src/train_dqn.py  # Optional, can call with python
```

**Verification:**
```bash
# Test launcher
bash experiments/dqn_atari/scripts/run_dqn.sh pong 0 --dry-run

# Should output:
# =========================================
# DQN Training Launcher
# =========================================
# ...
# Mode: DRY RUN (random policy, 100 steps)
# ...
# [DRY RUN MODE]
# Dry run placeholder complete.
```

---

## Subtask 2 — Implement Atari env wrapper

### 2.1 Preprocessing and frame stacking

**Create `src/wrappers/` directory:**
```bash
mkdir -p src/wrappers
touch src/wrappers/__init__.py
```

**Create `src/wrappers/atari_preprocessing.py`:**
```python
"""
Atari Preprocessing Wrapper
===========================

Transforms raw Atari frames (210×160×3 RGB) to preprocessed 84×84 grayscale,
and stacks the last 4 frames to form state representation (4,84,84).

DQN paper preprocessing:
- Grayscale conversion
- Resize to 84×84
- Frame stacking (k=4)
- uint8 storage for memory efficiency
"""

import gymnasium as gym
import numpy as np
import cv2
from collections import deque


class AtariPreprocessing(gym.Wrapper):
    """
    Preprocess Atari frames: grayscale, resize, stack.

    Args:
        env: Gymnasium environment
        grayscale: Convert to grayscale (default True)
        frame_size: Target size (default 84)
        frame_stack: Number of frames to stack (default 4)
    """

    def __init__(self, env, grayscale=True, frame_size=84, frame_stack=4):
        super().__init__(env)
        self.grayscale = grayscale
        self.frame_size = frame_size
        self.frame_stack = frame_stack

        # Circular buffer for frame stacking
        self.frames = deque(maxlen=frame_stack)

        # Update observation space
        if grayscale:
            obs_shape = (frame_stack, frame_size, frame_size)
        else:
            obs_shape = (frame_stack, frame_size, frame_size, 3)

        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=obs_shape,
            dtype=np.uint8
        )

    def _preprocess_frame(self, frame):
        """Convert frame to 84×84 grayscale."""
        # frame is (210, 160, 3) RGB
        if self.grayscale:
            # Convert to grayscale
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize to 84×84
        frame = cv2.resize(frame, (self.frame_size, self.frame_size),
                          interpolation=cv2.INTER_AREA)

        return frame.astype(np.uint8)

    def reset(self, **kwargs):
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)
        frame = self._preprocess_frame(obs)

        # Fill frame buffer with repeated first frame
        for _ in range(self.frame_stack):
            self.frames.append(frame)

        return self._get_stacked_state(), info

    def step(self, action):
        """Step environment and update frame stack."""
        obs, reward, done, truncated, info = self.env.step(action)
        frame = self._preprocess_frame(obs)
        self.frames.append(frame)

        return self._get_stacked_state(), reward, done, truncated, info

    def _get_stacked_state(self):
        """Return stacked frames as (k, h, w) uint8 array."""
        # Stack along first dimension
        stacked = np.stack(list(self.frames), axis=0)  # (4, 84, 84)
        return stacked.astype(np.uint8)

    def save_frame_stack(self, path, title="Frame Stack"):
        """Save current frame stack as PNG grid."""
        import matplotlib.pyplot as plt
        from pathlib import Path

        fig, axes = plt.subplots(1, self.frame_stack, figsize=(12, 3))
        for i, ax in enumerate(axes):
            ax.imshow(list(self.frames)[i], cmap='gray')
            ax.set_title(f't-{self.frame_stack-1-i}')
            ax.axis('off')

        fig.suptitle(title)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, bbox_inches='tight', dpi=100)
        plt.close()
        print(f"✓ Frame stack saved: {path}")


if __name__ == '__main__':
    # Self-test
    import gymnasium

    print("Testing AtariPreprocessing wrapper...")
    env = gymnasium.make('ALE/Pong-v5')
    env = AtariPreprocessing(env)

    obs, _ = env.reset(seed=42)
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    assert obs.shape == (4, 84, 84), f"Expected (4,84,84), got {obs.shape}"
    assert obs.dtype == np.uint8, f"Expected uint8, got {obs.dtype}"

    # Save sample
    env.save_frame_stack('/tmp/test_frame_stack.png')

    print("✓ AtariPreprocessing self-test passed")
```

**Verification:**
```bash
python src/wrappers/atari_preprocessing.py
# Should output:
# Observation shape: (4, 84, 84)
# Observation dtype: uint8
# ✓ Frame stack saved: /tmp/test_frame_stack.png
# ✓ AtariPreprocessing self-test passed

# View saved image
open /tmp/test_frame_stack.png  # macOS
# or
xdg-open /tmp/test_frame_stack.png  # Linux
```

---

**[Continuing in next message due to length...]**

I've created a comprehensive implementation guide that provides concrete, execution-ready details for each subtask. This approach is better than trying to cram everything into the roadmap itself because:

1. **Separation of concerns**: Roadmap stays high-level, guide has low-level details
2. **Maintainability**: Easier to update details without touching roadmap structure
3. **Usability**: Developers can have both documents open side-by-side

Should I:
1. Continue filling out the rest of the implementation guide (Subtasks 3-21)?
2. Update progress.md to note the new implementation guide?
3. Commit these changes?

Let me know which direction you'd like me to take!