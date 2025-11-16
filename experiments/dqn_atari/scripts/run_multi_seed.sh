#!/bin/bash
# Multi-seed training launcher for DQN experiments
# Usage: ./run_multi_seed.sh <game> [seeds]
#
# Examples:
#   ./run_multi_seed.sh pong           # Run Pong with default seeds (42, 123, 456)
#   ./run_multi_seed.sh breakout       # Run Breakout with default seeds
#   ./run_multi_seed.sh beam_rider     # Run Beam Rider with default seeds
#   ./run_multi_seed.sh pong "42 100"  # Custom seeds

set -e

# Configuration
GAME="${1:-pong}"
SEEDS="${2:-42 123 456}"
CONFIG_DIR="experiments/dqn_atari/configs"
LOG_DIR="experiments/dqn_atari/logs"

# Validate game config exists
CONFIG_FILE="${CONFIG_DIR}/${GAME}.yaml"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Available configs:"
    ls -1 "$CONFIG_DIR"/*.yaml
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Multi-Seed Training Launcher"
echo "=============================================="
echo "Game: $GAME"
echo "Config: $CONFIG_FILE"
echo "Seeds: $SEEDS"
echo "=============================================="
echo ""

# Check if we're in a virtual environment
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Load environment variables (e.g., WANDB_API_KEY)
if [[ -f .env ]]; then
    source .env
fi

# Launch training for each seed
for SEED in $SEEDS; do
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="$LOG_DIR/${GAME}_seed${SEED}_${TIMESTAMP}.log"

    echo "Launching $GAME with seed $SEED..."
    echo "  Log file: $LOG_FILE"

    # Launch training in background
    python -u train_dqn.py \
        --cfg "$CONFIG_FILE" \
        --seed "$SEED" \
        --set logging.wandb.enabled=true \
        --set logging.wandb.upload_artifacts=true \
        --tags "multi-seed" "seed-$SEED" "$GAME" \
        > "$LOG_FILE" 2>&1 &

    PID=$!
    echo "  PID: $PID"
    echo "  $SEED $PID" >> "$LOG_DIR/${GAME}_pids.txt"

    # Brief delay between launches to avoid race conditions
    sleep 5
done

echo ""
echo "=============================================="
echo "All training runs launched!"
echo "=============================================="
echo "Monitor with:"
echo "  tail -f $LOG_DIR/${GAME}_seed*_*.log"
echo ""
echo "PIDs saved to: $LOG_DIR/${GAME}_pids.txt"
echo "=============================================="
