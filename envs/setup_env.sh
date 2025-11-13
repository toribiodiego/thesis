#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="python3"

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "[env] Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel
pip install -r "${ROOT_DIR}/envs/requirements.txt"

echo "[env] Installing Atari ROM tooling (AutoROM)."
python -m AutoROM --accept-license || true

cat <<MSG

Environment ready!
Activate with: source ${VENV_DIR}/bin/activate
ROMs installed under: $(python -c "import autorom; import os; print(os.path.dirname(autorom.__file__))" 2>/dev/null || echo "See AutoROM output")

MSG
