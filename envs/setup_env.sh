#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

# Auto-detect Python version (prefer 3.11, fall back to available python3)
if command -v python3.11 &> /dev/null; then
  PYTHON_BIN="python3.11"
elif command -v python3 &> /dev/null; then
  PYTHON_BIN="python3"
else
  echo "Error: No Python 3 installation found"
  exit 1
fi

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
  # Try with pip first, fallback to without-pip for environments like Colab
  if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}" 2>/dev/null; then
    echo "[env] Standard venv failed, trying without pip..."
    "${PYTHON_BIN}" -m venv --without-pip "${VENV_DIR}"
  fi
fi

# Activate and install
source "${VENV_DIR}/bin/activate"

# Ensure pip is available
if ! python -m pip --version &> /dev/null; then
  echo "[env] Installing pip..."
  curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  python /tmp/get-pip.py
  rm /tmp/get-pip.py
fi

python -m pip install --upgrade pip wheel
pip install -r "${REQUIREMENTS_FILE}"

# Install Atari ROMs
echo "[env] Installing Atari ROMs via AutoROM"
AutoROM --accept-license || echo "ROM installation failed or already complete"
echo "[env] Importing ROMs into ale-py"
# Dynamically find the Python site-packages directory
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
ale-import-roms "${VENV_DIR}/lib/python${PYTHON_VERSION}/site-packages/AutoROM/roms" || echo "ROM import failed or already complete"

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
