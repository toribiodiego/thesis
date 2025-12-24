#!/usr/bin/env bash
# Capture system and environment information for reproducibility
# Usage: ./setup/capture_env.sh [output_file]
# Default output: experiments/dqn_atari/system_info.txt

set -e

# Get repository root
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Allow override via argument, otherwise use default
OUTPUT_FILE="${1:-$ROOT_DIR/experiments/dqn_atari/system_info.txt}"

echo "Capturing system and environment information..."
echo ""

{
    echo "=== System Information ==="
    echo "Date: $(date)"
    echo "Hostname: $(hostname)"
    echo "OS: $(uname -s)"
    echo "Kernel: $(uname -r)"
    echo "Architecture: $(uname -m)"
    echo ""

    echo "=== Python Environment ==="
    echo "Python version: $(python --version 2>&1)"
    echo "Python executable: $(which python)"
    echo ""

    echo "=== Key Package Versions ==="
    python -c "
import sys
try:
    import torch
    print(f'PyTorch: {torch.__version__}')
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'CUDA version: {torch.version.cuda}')
        print(f'GPU device: {torch.cuda.get_device_name(0)}')
except ImportError:
    print('PyTorch: not installed')

try:
    import numpy as np
    print(f'NumPy: {np.__version__}')
except ImportError:
    print('NumPy: not installed')

try:
    import gymnasium as gym
    print(f'Gymnasium: {gym.__version__}')
except ImportError:
    print('Gymnasium: not installed')

try:
    import ale_py
    print(f'ALE-py: {ale_py.__version__}')
except ImportError:
    print('ALE-py: not installed')
"
    echo ""

    echo "=== Git Repository ==="
    echo "Commit: $(git rev-parse HEAD)"
    echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Status:"
    git status --short
    echo ""

    echo "=== Full pip freeze ==="
    pip freeze

} > "$OUTPUT_FILE"

echo "System information saved to: $OUTPUT_FILE"
