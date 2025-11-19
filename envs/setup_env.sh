#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="python3.11"  # Use Python 3.11 for compatibility with Colab and stable packages
USE_GPU=false

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --gpu)
      USE_GPU=true
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 [--gpu]"
      exit 1
      ;;
  esac
done

# Select requirements file
if [ "$USE_GPU" = true ]; then
  REQUIREMENTS_FILE="${ROOT_DIR}/envs/requirements-gpu.txt"
  echo "[env] Setting up GPU environment (CUDA 12.1)"
else
  REQUIREMENTS_FILE="${ROOT_DIR}/envs/requirements.txt"
  echo "[env] Setting up CPU environment"
fi

# Create venv if needed
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[env] Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

# Activate and install
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel
pip install -r "${REQUIREMENTS_FILE}"

# Install Atari ROMs
echo "[env] Installing Atari ROMs via AutoROM"
AutoROM --accept-license || echo "ROM installation failed or already complete"
echo "[env] Importing ROMs into ale-py"
ale-import-roms "${VENV_DIR}/lib/python3.11/site-packages/AutoROM/roms" || echo "ROM import failed or already complete"

# Display completion message
cat <<MSG

Environment ready!
Mode: $([ "$USE_GPU" = true ] && echo "GPU (CUDA)" || echo "CPU")
Activate with: source ${VENV_DIR}/bin/activate

Installed packages:
  PyTorch: $(python -c "import torch; print(torch.__version__)")
  CUDA available: $(python -c "import torch; print(torch.cuda.is_available())")
  Gymnasium: $(python -c "import gymnasium; print(gymnasium.__version__)")

ROMs installed under: $(python -c "import autorom; import os; print(os.path.dirname(autorom.__file__))" 2>/dev/null || echo "See AutoROM output")

MSG
