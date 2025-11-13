#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../" && pwd)"
CONFIG_PATH="${1:-experiments/dqn_atari/configs/base.yaml}"
shift || true

python "${ROOT_DIR}/src/train_dqn.py" \
  --config "${ROOT_DIR}/${CONFIG_PATH}" \
  "$@"
