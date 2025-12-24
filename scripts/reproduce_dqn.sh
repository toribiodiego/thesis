#!/usr/bin/env bash
# reproduce_dqn.sh - One-command DQN 2013 reproduction
#
# Usage:
#   ./scripts/reproduce_dqn.sh                    # Default: Pong, seed 42, 10M frames
#   ./scripts/reproduce_dqn.sh --game breakout    # Different game
#   ./scripts/reproduce_dqn.sh --seed 123         # Different seed
#   ./scripts/reproduce_dqn.sh --frames 1000000   # Short test run
#   ./scripts/reproduce_dqn.sh --disable-wandb    # No W&B logging
#   ./scripts/reproduce_dqn.sh --skip-setup       # Skip environment setup
#   ./scripts/reproduce_dqn.sh --help             # Show help

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# Defaults
GAME="pong"
SEED=42
FRAMES=""  # Empty means use config default
WANDB_ENABLED=true
SKIP_SETUP=false
SKIP_ROMS=false
SKIP_PLOTS=false
VERBOSE=false
DRY_RUN=false

# Project paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/.venv"
CONFIG_DIR="$PROJECT_ROOT/experiments/dqn_atari/configs"
RUNS_DIR="$PROJECT_ROOT/experiments/dqn_atari/runs"
RESULTS_DIR="$PROJECT_ROOT/results"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo ""
    echo "================================================================================"
    echo "  $1"
    echo "================================================================================"
}

print_step() {
    echo ""
    echo ">>> $1"
}

print_info() {
    echo "    $1"
}

print_error() {
    echo "ERROR: $1" >&2
}

print_warning() {
    echo "WARNING: $1"
}

show_help() {
    cat << EOF
DQN 2013 Reproduction Script
=============================

Automates the complete reproduction pipeline: environment setup, ROM installation,
training, evaluation, and result visualization.

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --game <name>       Game to train (pong, breakout, beam_rider) [default: pong]
    --seed <int>        Random seed [default: 42]
    --frames <int>      Total training frames (overrides config)
    --disable-wandb     Disable Weights & Biases logging
    --skip-setup        Skip Python environment setup
    --skip-roms         Skip ROM installation check
    --skip-plots        Skip plot generation after training
    --verbose           Show detailed output
    --dry-run           Show what would be executed without running
    --help              Show this help message

EXAMPLES:
    # Quick test run (1M frames)
    $0 --game pong --seed 42 --frames 1000000

    # Full Pong reproduction (10M frames)
    $0 --game pong --seed 42

    # Multiple seeds for Breakout
    for seed in 42 123 456; do
        $0 --game breakout --seed \$seed
    done

    # Offline mode (no W&B)
    $0 --game pong --disable-wandb

PREREQUISITES:
    - Python 3.11+
    - pip
    - Git (for version tracking)
    - CUDA (optional, for GPU acceleration)

OUTPUTS:
    - Training logs: experiments/dqn_atari/runs/<run_name>/
    - Checkpoints: experiments/dqn_atari/runs/<run_name>/checkpoints/
    - Plots: results/plots/<game>_<seed>/
    - System info: experiments/dqn_atari/runs/<run_name>/system_info.txt

EOF
    exit 0
}

# =============================================================================
# Argument Parsing
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --game)
            GAME="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --frames)
            FRAMES="$2"
            shift 2
            ;;
        --disable-wandb)
            WANDB_ENABLED=false
            shift
            ;;
        --skip-setup)
            SKIP_SETUP=true
            shift
            ;;
        --skip-roms)
            SKIP_ROMS=true
            shift
            ;;
        --skip-plots)
            SKIP_PLOTS=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# =============================================================================
# Validation
# =============================================================================

# Validate game
case $GAME in
    pong|breakout|beam_rider)
        CONFIG_FILE="$CONFIG_DIR/${GAME}.yaml"
        ;;
    *)
        print_error "Unsupported game: $GAME"
        echo "Supported games: pong, breakout, beam_rider"
        exit 1
        ;;
esac

# Check config exists
if [[ ! -f "$CONFIG_FILE" ]]; then
    print_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

# =============================================================================
# Main Execution
# =============================================================================

print_header "DQN 2013 Reproduction Pipeline"

echo "Configuration:"
echo "  Game:         $GAME"
echo "  Seed:         $SEED"
echo "  Config:       $CONFIG_FILE"
if [[ -n "$FRAMES" ]]; then
    echo "  Frames:       $FRAMES (override)"
else
    echo "  Frames:       (from config)"
fi
echo "  W&B Logging:  $WANDB_ENABLED"
echo "  Project Root: $PROJECT_ROOT"

if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo "DRY RUN MODE - No commands will be executed"
fi

# -----------------------------------------------------------------------------
# Step 1: Environment Setup
# -----------------------------------------------------------------------------

if [[ "$SKIP_SETUP" == "false" ]]; then
    print_step "Step 1: Setting up Python environment"

    if [[ ! -d "$VENV_DIR" ]]; then
        print_info "Creating virtual environment..."
        if [[ "$DRY_RUN" == "false" ]]; then
            python3 -m venv "$VENV_DIR"
        fi
    else
        print_info "Virtual environment exists: $VENV_DIR"
    fi

    print_info "Activating environment..."
    if [[ "$DRY_RUN" == "false" ]]; then
        source "$VENV_DIR/bin/activate"
    fi

    print_info "Installing dependencies..."
    if [[ "$DRY_RUN" == "false" ]]; then
        pip install -q -r "$PROJECT_ROOT/requirements.txt"
    fi
else
    print_step "Step 1: Skipping environment setup (--skip-setup)"
    if [[ -d "$VENV_DIR" ]]; then
        source "$VENV_DIR/bin/activate"
    fi
fi

# -----------------------------------------------------------------------------
# Step 2: ROM Installation
# -----------------------------------------------------------------------------

if [[ "$SKIP_ROMS" == "false" ]]; then
    print_step "Step 2: Checking Atari ROMs"

    ROM_SCRIPT="$PROJECT_ROOT/setup/setup_roms.sh"
    if [[ -f "$ROM_SCRIPT" ]]; then
        print_info "Running ROM setup script..."
        if [[ "$DRY_RUN" == "false" ]]; then
            bash "$ROM_SCRIPT"
        fi
    else
        print_info "ROM setup script not found, using AutoROM..."
        if [[ "$DRY_RUN" == "false" ]]; then
            python -m ale_py.roms.install || print_warning "AutoROM may need manual setup"
        fi
    fi
else
    print_step "Step 2: Skipping ROM check (--skip-roms)"
fi

# -----------------------------------------------------------------------------
# Step 3: Capture System Information
# -----------------------------------------------------------------------------

print_step "Step 3: Capturing system information"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_NAME="${GAME}_${SEED}_${TIMESTAMP}"
RUN_DIR="$RUNS_DIR/$RUN_NAME"

print_info "Run name: $RUN_NAME"

if [[ "$DRY_RUN" == "false" ]]; then
    mkdir -p "$RUN_DIR"

    # System info
    {
        echo "DQN Reproduction System Information"
        echo "===================================="
        echo "Timestamp: $(date -Iseconds)"
        echo ""
        echo "Hardware:"
        echo "  Platform: $(uname -s)"
        echo "  Architecture: $(uname -m)"
        echo "  Hostname: $(hostname)"
        if command -v nvidia-smi &> /dev/null; then
            echo ""
            echo "GPU:"
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        fi
        echo ""
        echo "Software:"
        echo "  Python: $(python --version 2>&1)"
        echo ""
        echo "Dependencies:"
        pip freeze
        echo ""
        echo "Configuration:"
        echo "  Game: $GAME"
        echo "  Seed: $SEED"
        echo "  Frames: ${FRAMES:-default}"
        echo "  W&B: $WANDB_ENABLED"
    } > "$RUN_DIR/system_info.txt"

    # Git info
    if command -v git &> /dev/null && [[ -d "$PROJECT_ROOT/.git" ]]; then
        {
            echo "Git Repository Information"
            echo "=========================="
            echo "Branch: $(git -C "$PROJECT_ROOT" rev-parse --abbrev-ref HEAD)"
            echo "Commit: $(git -C "$PROJECT_ROOT" rev-parse HEAD)"
            echo "Status: $(git -C "$PROJECT_ROOT" status --porcelain | wc -l) modified files"
            echo ""
            echo "Recent commits:"
            git -C "$PROJECT_ROOT" log --oneline -n 5
        } > "$RUN_DIR/git_info.txt"
    fi

    print_info "System info saved to: $RUN_DIR/system_info.txt"
fi

# -----------------------------------------------------------------------------
# Step 4: Training
# -----------------------------------------------------------------------------

print_step "Step 4: Starting DQN training"

# Build command
TRAIN_CMD="python -u $PROJECT_ROOT/train_dqn.py"
TRAIN_CMD="$TRAIN_CMD --cfg $CONFIG_FILE"
TRAIN_CMD="$TRAIN_CMD --seed $SEED"

if [[ -n "$FRAMES" ]]; then
    TRAIN_CMD="$TRAIN_CMD --set training.total_frames=$FRAMES"
fi

if [[ "$WANDB_ENABLED" == "true" ]]; then
    TRAIN_CMD="$TRAIN_CMD --set logging.wandb.enabled=true"
else
    TRAIN_CMD="$TRAIN_CMD --set logging.wandb.enabled=false"
fi

print_info "Command: $TRAIN_CMD"

if [[ "$DRY_RUN" == "false" ]]; then
    # Load environment variables if available
    if [[ -f "$PROJECT_ROOT/.env" ]]; then
        source "$PROJECT_ROOT/.env"
    fi

    # Run training
    START_TIME=$(date +%s)
    $TRAIN_CMD 2>&1 | tee "$RUN_DIR/training.log"
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
    END_TIME=$(date +%s)

    DURATION=$((END_TIME - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))

    print_info "Training completed in ${HOURS}h ${MINUTES}m ${SECONDS}s"

    if [[ $TRAIN_EXIT_CODE -ne 0 ]]; then
        print_error "Training failed with exit code: $TRAIN_EXIT_CODE"
        exit $TRAIN_EXIT_CODE
    fi
fi

# -----------------------------------------------------------------------------
# Step 5: Generate Plots
# -----------------------------------------------------------------------------

if [[ "$SKIP_PLOTS" == "false" ]]; then
    print_step "Step 5: Generating result plots"

    # Find the actual run directory (created by train_dqn.py)
    if [[ "$DRY_RUN" == "false" ]]; then
        # Get most recent run for this game/seed
        ACTUAL_RUN_DIR=$(ls -td "$RUNS_DIR"/${GAME}_${SEED}_* 2>/dev/null | head -1)

        if [[ -d "$ACTUAL_RUN_DIR" ]]; then
            EPISODES_CSV="$ACTUAL_RUN_DIR/csv/episodes.csv"
            STEPS_CSV="$ACTUAL_RUN_DIR/csv/training_steps.csv"
            PLOT_OUTPUT="$RESULTS_DIR/plots/${GAME}_${SEED}"

            if [[ -f "$EPISODES_CSV" ]]; then
                print_info "Generating plots from: $EPISODES_CSV"
                mkdir -p "$PLOT_OUTPUT"

                PLOT_CMD="python $PROJECT_ROOT/scripts/plot_results.py"
                PLOT_CMD="$PLOT_CMD --episodes $EPISODES_CSV"
                PLOT_CMD="$PLOT_CMD --output $PLOT_OUTPUT"
                PLOT_CMD="$PLOT_CMD --game-name ${GAME^}"

                # Add steps CSV if available (for loss plots)
                if [[ -f "$STEPS_CSV" ]]; then
                    PLOT_CMD="$PLOT_CMD --steps $STEPS_CSV"
                fi

                $PLOT_CMD

                print_info "Plots saved to: $PLOT_OUTPUT"
            else
                print_warning "Episodes CSV not found: $EPISODES_CSV"
            fi
        else
            print_warning "Run directory not found"
        fi
    fi
else
    print_step "Step 5: Skipping plot generation (--skip-plots)"
fi

# -----------------------------------------------------------------------------
# Step 6: Summary
# -----------------------------------------------------------------------------

print_header "Reproduction Complete"

if [[ "$DRY_RUN" == "false" ]]; then
    ACTUAL_RUN_DIR=$(ls -td "$RUNS_DIR"/${GAME}_${SEED}_* 2>/dev/null | head -1)

    if [[ -d "$ACTUAL_RUN_DIR" ]]; then
        echo "Outputs:"
        echo "  Run directory: $ACTUAL_RUN_DIR"
        echo "  Training log:  $ACTUAL_RUN_DIR/training.log"
        echo "  Checkpoints:   $ACTUAL_RUN_DIR/checkpoints/"
        echo "  CSVs:          $ACTUAL_RUN_DIR/csv/"
        echo "  Videos:        $ACTUAL_RUN_DIR/videos/"

        if [[ -d "$RESULTS_DIR/plots/${GAME}_${SEED}" ]]; then
            echo "  Plots:         $RESULTS_DIR/plots/${GAME}_${SEED}/"
        fi

        # Show final evaluation if available
        EVAL_CSV="$ACTUAL_RUN_DIR/eval/evaluations.csv"
        if [[ -f "$EVAL_CSV" ]]; then
            echo ""
            echo "Final evaluation results:"
            tail -5 "$EVAL_CSV" | while IFS=, read -r step mean_return median_return std_return min_ret max_ret mean_len episodes eval_eps; do
                if [[ "$step" != "step" ]]; then
                    printf "  Step %s: Mean=%.2f +/- %.2f (median=%.2f)\n" "$step" "$mean_return" "$std_return" "$median_return"
                fi
            done
        fi
    fi

    if [[ "$WANDB_ENABLED" == "true" ]]; then
        echo ""
        echo "W&B Dashboard: https://wandb.ai/Cooper-Union/dqn-atari"
    fi
fi

echo ""
echo "Next steps:"
echo "  1. Review training curves in plots/"
echo "  2. Compare to paper scores (see docs/reports/report-results-comparison.md)"
echo "  3. Run additional seeds for statistical significance"
echo ""
