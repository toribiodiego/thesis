#!/bin/bash
# =============================================================================
# DQN Ablation Study Runner
# =============================================================================
# Launches multiple ablation experiments with consistent seeds and output paths.
#
# Usage:
#   ./experiments/dqn_atari/scripts/run_ablations.sh [OPTIONS]
#
# Options:
#   --game GAME         Game to run ablations on (default: pong)
#   --seeds "S1 S2 S3"  Seeds to use (default: "42 43 44")
#   --frames N          Total frames per run (default: 5000000)
#   --ablations "A1 A2" Ablation configs to run (default: all in ablations/)
#   --wandb             Enable W&B logging
#   --dry-run           Print commands without executing
#   --parallel N        Run N experiments in parallel (default: 1)
#   --help              Show this help message
#
# Examples:
#   # Run all ablations for Pong with 3 seeds
#   ./experiments/dqn_atari/scripts/run_ablations.sh --game pong --seeds "42 43 44"
#
#   # Run specific ablation
#   ./experiments/dqn_atari/scripts/run_ablations.sh --ablations "no_target_net"
#
#   # Dry run to see commands
#   ./experiments/dqn_atari/scripts/run_ablations.sh --dry-run
#
# Output structure:
#   experiments/dqn_atari/runs/<game>/ablations/<ablation>/seed_<n>/
# =============================================================================

set -e

# Default values
GAME="pong"
SEEDS="42 43 44"
FRAMES=5000000
ABLATIONS=""
WANDB_ENABLED=false
DRY_RUN=false
PARALLEL=1

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    head -40 "$0" | tail -35 | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --game)
            GAME="$2"
            shift 2
            ;;
        --seeds)
            SEEDS="$2"
            shift 2
            ;;
        --frames)
            FRAMES="$2"
            shift 2
            ;;
        --ablations)
            ABLATIONS="$2"
            shift 2
            ;;
        --wandb)
            WANDB_ENABLED=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ABLATIONS_DIR="$PROJECT_ROOT/experiments/dqn_atari/configs/ablations"

# Validate game config exists
GAME_CONFIG="$PROJECT_ROOT/experiments/dqn_atari/configs/${GAME}.yaml"
if [[ ! -f "$GAME_CONFIG" ]]; then
    print_error "Game config not found: $GAME_CONFIG"
    exit 1
fi

# Find ablation configs
if [[ -z "$ABLATIONS" ]]; then
    # Use all ablation configs
    ABLATION_FILES=($(find "$ABLATIONS_DIR" -name "*.yaml" -type f 2>/dev/null | sort))
    if [[ ${#ABLATION_FILES[@]} -eq 0 ]]; then
        print_error "No ablation configs found in $ABLATIONS_DIR"
        exit 1
    fi
else
    # Use specified ablations
    ABLATION_FILES=()
    for abl in $ABLATIONS; do
        abl_file="$ABLATIONS_DIR/${abl}.yaml"
        if [[ -f "$abl_file" ]]; then
            ABLATION_FILES+=("$abl_file")
        else
            print_warn "Ablation config not found: $abl_file"
        fi
    done
fi

# Print configuration
print_header "DQN Ablation Study Runner"
echo ""
print_info "Game: $GAME"
print_info "Seeds: $SEEDS"
print_info "Frames per run: $FRAMES"
print_info "W&B logging: $WANDB_ENABLED"
print_info "Parallel runs: $PARALLEL"
print_info "Dry run: $DRY_RUN"
echo ""

print_info "Ablations to run:"
for abl_file in "${ABLATION_FILES[@]}"; do
    abl_name=$(basename "$abl_file" .yaml)
    echo "  - $abl_name"
done
echo ""

# Calculate total runs
SEED_ARRAY=($SEEDS)
NUM_SEEDS=${#SEED_ARRAY[@]}
NUM_ABLATIONS=${#ABLATION_FILES[@]}
TOTAL_RUNS=$((NUM_SEEDS * NUM_ABLATIONS))

print_info "Total runs: $TOTAL_RUNS ($NUM_ABLATIONS ablations x $NUM_SEEDS seeds)"
echo ""

# Build commands
COMMANDS=()
for abl_file in "${ABLATION_FILES[@]}"; do
    abl_name=$(basename "$abl_file" .yaml)

    for seed in $SEEDS; do
        # Build output directory path
        output_dir="experiments/dqn_atari/runs/${GAME}/ablations/${abl_name}/seed_${seed}"

        # Build command
        cmd="python train_dqn.py"
        cmd="$cmd --cfg $abl_file"
        cmd="$cmd --seed $seed"
        cmd="$cmd --set training.total_frames=$FRAMES"
        cmd="$cmd --set environment.env_id=${GAME^}NoFrameskip-v4"

        if [[ "$WANDB_ENABLED" == "true" ]]; then
            cmd="$cmd --set logging.wandb.enabled=true"
            cmd="$cmd --tags ablation $abl_name ${GAME}"
        fi

        # Add run identification
        cmd="$cmd --set experiment.name=ablation_${abl_name}_${GAME}"

        COMMANDS+=("$cmd")

        if [[ "$DRY_RUN" == "true" ]]; then
            echo "Command: $cmd"
            echo ""
        fi
    done
done

if [[ "$DRY_RUN" == "true" ]]; then
    print_info "Dry run complete. Commands printed above."
    exit 0
fi

# Execute commands
print_header "Starting Ablation Runs"

RUN_INDEX=0
PIDS=()

for cmd in "${COMMANDS[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))

    # Extract ablation name and seed from command
    abl_name=$(echo "$cmd" | grep -oP '(?<=--cfg ).+?(?= )' | xargs basename | sed 's/\.yaml//')
    seed=$(echo "$cmd" | grep -oP '(?<=--seed )\d+')

    print_info "[$RUN_INDEX/$TOTAL_RUNS] Starting: $abl_name (seed $seed)"

    # Create log file
    log_dir="$PROJECT_ROOT/experiments/dqn_atari/runs/ablation_logs"
    mkdir -p "$log_dir"
    log_file="$log_dir/${abl_name}_seed${seed}_$(date +%Y%m%d_%H%M%S).log"

    # Run command
    if [[ $PARALLEL -gt 1 ]]; then
        # Run in background
        (
            cd "$PROJECT_ROOT"
            eval "$cmd" > "$log_file" 2>&1
        ) &
        PIDS+=($!)
        print_info "  PID: ${PIDS[-1]}, Log: $log_file"

        # Wait if we've reached parallel limit
        if [[ ${#PIDS[@]} -ge $PARALLEL ]]; then
            print_info "Waiting for batch to complete..."
            for pid in "${PIDS[@]}"; do
                wait "$pid"
            done
            PIDS=()
        fi
    else
        # Run sequentially
        print_info "  Log: $log_file"
        (
            cd "$PROJECT_ROOT"
            eval "$cmd" 2>&1 | tee "$log_file"
        )

        if [[ $? -eq 0 ]]; then
            print_info "  Completed successfully"
        else
            print_error "  Failed! Check log: $log_file"
        fi
    fi

    echo ""
done

# Wait for remaining parallel jobs
if [[ ${#PIDS[@]} -gt 0 ]]; then
    print_info "Waiting for final batch to complete..."
    for pid in "${PIDS[@]}"; do
        wait "$pid"
    done
fi

print_header "Ablation Study Complete"
print_info "Total runs completed: $TOTAL_RUNS"
print_info "Results saved to: experiments/dqn_atari/runs/"
print_info "Logs saved to: experiments/dqn_atari/runs/ablation_logs/"
echo ""
print_info "Next steps:"
echo "  1. Analyze results: python scripts/analyze_results.py --run-dir experiments/dqn_atari/runs/"
echo "  2. Generate plots: python scripts/plot_results.py --run-dir experiments/dqn_atari/runs/"
echo "  3. Compare to baseline: Check results/summary/ for comparison tables"
