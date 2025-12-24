#!/usr/bin/env bash
# Launch DQN training with specified config
#
# Usage:
#   ./scripts/run_dqn.sh <config> [options]
#
# Examples:
#   # Dry run with Pong
#   ./scripts/run_dqn.sh experiments/dqn_atari/configs/pong.yaml --dry-run
#
#   # Full training with custom seed
#   ./scripts/run_dqn.sh experiments/dqn_atari/configs/pong.yaml --seed 42
#
#   # Dry run with 5 episodes
#   ./scripts/run_dqn.sh experiments/dqn_atari/configs/breakout.yaml --dry-run --dry-run-episodes 5

set -euo pipefail

# Get repository root (two levels up from scripts/)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

# First argument is config path, default to base.yaml
CONFIG_PATH="${1:-experiments/dqn_atari/configs/base.yaml}"
shift || true

# Launch training script with all remaining arguments
cd "${ROOT_DIR}"
python train_dqn.py \
  --config "${CONFIG_PATH}" \
  "$@"
