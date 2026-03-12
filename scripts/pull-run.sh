#!/bin/bash
# Download training runs from Google Drive via rclone.
#
# Usage:
#   bash scripts/pull-run.sh <run-name>         # pull one run
#   bash scripts/pull-run.sh --all              # pull every run
#   bash scripts/pull-run.sh --group rainbow    # pull all rainbow runs
#   bash scripts/pull-run.sh --group spr        # pull all spr runs
#   bash scripts/pull-run.sh --dry-run <...>    # preview any of the above
#   bash scripts/pull-run.sh --list             # list runs on Drive
#
# Groups match against run directory names (substring match):
#   base    -- runs without aug/spr/rainbow/both in the name
#   aug     -- names containing _aug_
#   spr     -- names containing _spr_
#   both    -- names containing _both_
#   rainbow -- names containing _rainbow_
#   rainbow_spr -- names containing _rainbow_spr_
#
# Requires: rclone configured with a "gdrive" remote.
# Setup:    brew install rclone && rclone config
#
# Drive path: thesis-runs/<run-name>/
# Local path: experiments/dqn_atari/runs/<run-name>/

set -euo pipefail

DRIVE_REMOTE="gdrive"
DRIVE_BASE="thesis-runs"
LOCAL_BASE="experiments/dqn_atari/runs"

usage() {
    cat <<'EOF'
Usage: pull-run.sh [OPTIONS] [<run-name>]

Download training runs from Google Drive.

Options:
  --list              List all runs on Drive
  --all               Download every run (excludes deprecated/invalid)
  --group <pattern>   Download runs matching a group pattern
  --dry-run           Preview what would be downloaded (combine with above)
  --no-checkpoints    Skip checkpoint .pt files (faster, smaller)
  -h, --help          Show this help

Groups (substring match on run name):
  base          Vanilla DQN (no aug/spr/rainbow/both in name)
  aug           DQN + augmentation
  spr           DQN + SPR (excludes rainbow_spr)
  both          DQN + augmentation + SPR
  rainbow       Rainbow DQN (excludes rainbow_spr)
  rainbow_spr   Rainbow + SPR

Examples:
  pull-run.sh --list
  pull-run.sh atari100k_boxing_spr_42_20260312_022320
  pull-run.sh --group rainbow
  pull-run.sh --all --no-checkpoints
  pull-run.sh --dry-run --group spr
EOF
    exit 1
}

# Check rclone is installed
if ! command -v rclone &>/dev/null; then
    echo "Error: rclone not installed. Run: brew install rclone"
    exit 1
fi

# Parse args
DRY_RUN=""
LIST=""
ALL=""
GROUP=""
NO_CKPTS=""
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --list) LIST=1; shift ;;
        --all) ALL=1; shift ;;
        --group) GROUP="$2"; shift 2 ;;
        --no-checkpoints) NO_CKPTS=1; shift ;;
        --help|-h) usage ;;
        -*) echo "Unknown option: $1"; usage ;;
        *) RUN_NAME="$1"; shift ;;
    esac
done

# List mode
if [[ -n "$LIST" ]]; then
    echo "Runs on Drive ($DRIVE_REMOTE:$DRIVE_BASE/):"
    echo ""
    rclone lsd "$DRIVE_REMOTE:$DRIVE_BASE/" 2>/dev/null || echo "  (none found)"
    exit 0
fi

# Build rclone filter flags
RCLONE_FILTERS=()
if [[ -n "$NO_CKPTS" ]]; then
    RCLONE_FILTERS+=(--exclude "checkpoints/*.pt")
fi

# Pull a single run directory
pull_one() {
    local name="$1"
    local src="$DRIVE_REMOTE:$DRIVE_BASE/$name/"
    local dst="$LOCAL_BASE/$name/"

    echo "--- Pulling: $name"
    if [[ -n "$DRY_RUN" ]]; then
        echo "    (dry run)"
    fi

    rclone copy $DRY_RUN --progress "${RCLONE_FILTERS[@]+"${RCLONE_FILTERS[@]}"}" "$src" "$dst" 2>&1 | tail -1 || true

    if [[ -z "$DRY_RUN" ]]; then
        echo "    -> $dst"
    fi
    echo ""
}

# Filter run names by group pattern
filter_group() {
    local pattern="$1"
    local names=("${@:2}")
    local filtered=()

    for name in "${names[@]}"; do
        # Skip deprecated/invalid directories
        [[ "$name" == "deprecated" || "$name" == "invalid" ]] && continue

        case "$pattern" in
            base)
                # No aug, spr, rainbow, or both in the name
                if [[ "$name" != *_aug_* && "$name" != *_spr_* && "$name" != *_rainbow_* && "$name" != *_both_* ]]; then
                    filtered+=("$name")
                fi
                ;;
            aug)
                # Contains _aug_ but not _both_ (both includes aug)
                if [[ "$name" == *_aug_* && "$name" != *_both_* ]]; then
                    filtered+=("$name")
                fi
                ;;
            spr)
                # Contains _spr_ but not _rainbow_spr_ and not _both_
                if [[ "$name" == *_spr_* && "$name" != *_rainbow_spr_* && "$name" != *_both_* ]]; then
                    filtered+=("$name")
                fi
                ;;
            both)
                if [[ "$name" == *_both_* ]]; then
                    filtered+=("$name")
                fi
                ;;
            rainbow)
                # Contains _rainbow_ but not _rainbow_spr_
                if [[ "$name" == *_rainbow_* && "$name" != *_rainbow_spr_* ]]; then
                    filtered+=("$name")
                fi
                ;;
            rainbow_spr)
                if [[ "$name" == *_rainbow_spr_* ]]; then
                    filtered+=("$name")
                fi
                ;;
            *)
                # Arbitrary substring match
                if [[ "$name" == *"$pattern"* ]]; then
                    filtered+=("$name")
                fi
                ;;
        esac
    done

    printf '%s\n' "${filtered[@]}"
}

# Fetch run list from Drive
get_run_names() {
    rclone lsd "$DRIVE_REMOTE:$DRIVE_BASE/" 2>/dev/null \
        | awk '{print $NF}' \
        | grep -v '^deprecated$' \
        | grep -v '^invalid$' \
        | sort
}

# Group mode or all mode
if [[ -n "$ALL" || -n "$GROUP" ]]; then
    echo "Fetching run list from Drive..."
    mapfile -t ALL_RUNS < <(get_run_names)

    if [[ ${#ALL_RUNS[@]} -eq 0 ]]; then
        echo "No runs found on Drive."
        exit 0
    fi

    if [[ -n "$GROUP" ]]; then
        echo "Filtering for group: $GROUP"
        mapfile -t TARGETS < <(filter_group "$GROUP" "${ALL_RUNS[@]}")
    else
        TARGETS=("${ALL_RUNS[@]}")
    fi

    if [[ ${#TARGETS[@]} -eq 0 ]]; then
        echo "No runs match group '$GROUP'."
        exit 0
    fi

    echo "Found ${#TARGETS[@]} run(s):"
    printf '  %s\n' "${TARGETS[@]}"
    echo ""

    PULLED=0
    FAILED=0
    for name in "${TARGETS[@]}"; do
        if pull_one "$name"; then
            ((PULLED++)) || true
        else
            echo "    FAILED: $name"
            ((FAILED++)) || true
        fi
    done

    echo "==============================="
    echo "Done: $PULLED pulled, $FAILED failed out of ${#TARGETS[@]}"
    exit 0
fi

# Single run mode
if [[ -z "$RUN_NAME" ]]; then
    usage
fi

SRC="$DRIVE_REMOTE:$DRIVE_BASE/$RUN_NAME/"
DST="$LOCAL_BASE/$RUN_NAME/"

echo "Source: $SRC"
echo "Dest:   $DST"
echo ""

if [[ -n "$DRY_RUN" ]]; then
    echo "(dry run -- no files will be downloaded)"
    echo ""
fi

rclone copy $DRY_RUN --progress "${RCLONE_FILTERS[@]+"${RCLONE_FILTERS[@]}"}" "$SRC" "$DST"

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "Downloaded to: $DST"
fi
