#!/usr/bin/env bash
# Colab GPU environment setup for thesis training runs.
# Run once per Colab session before training or testing.
#
# Usage:
#   cd /content/thesis && bash setup/setup_colab.sh
#
# After running, set LD_LIBRARY_PATH in each command:
#   export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Colab environment setup ==="
echo "Repo: $REPO_DIR"

# Install dependencies from pinned requirements
pip install -q -r "$REPO_DIR/setup/requirements-gpu.txt"

# Verify JAX sees the GPU
echo ""
echo "=== Verification ==="
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}

python -c "
import jax
print(f'JAX {jax.__version__}')
devs = jax.devices()
print(f'Devices: {devs}')
assert any('cuda' in str(d).lower() or 'gpu' in str(d).lower() for d in devs), 'No GPU found'
print('GPU OK')
"

python -c "
import flax, optax, dopamine, gin, tensorflow as tf
print(f'flax {flax.__version__}, optax {optax.__version__}')
print(f'dopamine {dopamine.__version__}, gin {gin.__version__}')
print(f'tensorflow {tf.__version__}')
"

echo ""
echo "=== Setup complete ==="
echo "Run commands with:"
echo "  export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH"
