#!/bin/bash
# Download training runs from Google Drive via rclone.
#
# Usage:
#   bash scripts/pull-run.sh <run-name>         # pull one run
#   bash scripts/pull-run.sh --all              # pull every run
#   bash scripts/pull-run.sh --group bbf        # pull all BBF runs
#   bash scripts/pull-run.sh --group spr        # pull all SPR runs
#   bash scripts/pull-run.sh --dry-run <...>    # preview any of the above
#   bash scripts/pull-run.sh --list             # list runs on Drive
#   bash scripts/pull-run.sh --archive --list   # list archived runs
#
# Groups match by condition prefix in the new naming convention
# (<condition>_<game>_seed<seed>):
#   derc      -- DERc runs
#   der       -- DER runs (excludes derc)
#   sprc      -- SPRc runs
#   spr       -- SPR runs (excludes sprc, sr-spr, sr-sprc)
#   sr-sprc   -- SR-SPRc runs
#   sr-spr    -- SR-SPR runs (excludes sr-sprc)
#   bbfc      -- BBFc runs
#   bbf       -- BBF runs (excludes bbfc)
#
# Requires: rclone configured with a "gdrive" remote.
# Setup:    brew install rclone && rclone config
#
# Drive path: thesis-runs/<run-name>/
# Local path: experiments/dqn_atari/runs/<run-name>/
#
# Old runs (pre-2026-04 design) are archived on Drive under
# archive/v1/ and archive/deprecated/. Use --archive to access them.

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
  --all               Download every run (excludes archive/)
  --group <condition> Download runs matching a condition
  --archive           Operate on archive/v1/ instead of top level
  --dry-run           Preview what would be downloaded (combine with above)
  --no-checkpoints    Skip checkpoint files (faster, smaller)
  -h, --help          Show this help

Condition groups (new naming convention):
  derc        DERc (control)
  der         DER (excludes derc)
  sprc        SPRc (control)
  spr         SPR (excludes sprc, sr-spr, sr-sprc)
  sr-sprc     SR-SPRc (control)
  sr-spr      SR-SPR (excludes sr-sprc)
  bbfc        BBFc (control)
  bbf         BBF (excludes bbfc)

Examples:
  pull-run.sh --list
  pull-run.sh bbf_boxing_seed13
  pull-run.sh --group bbf
  pull-run.sh --all --no-checkpoints
  pull-run.sh --dry-run --group spr
  pull-run.sh --archive --list
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
ARCHIVE=""
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --list) LIST=1; shift ;;
        --all) ALL=1; shift ;;
        --group) GROUP="$2"; shift 2 ;;
        --no-checkpoints) NO_CKPTS=1; shift ;;
        --archive) ARCHIVE=1; shift ;;
        --help|-h) usage ;;
        -*) echo "Unknown option: $1"; usage ;;
        *) RUN_NAME="$1"; shift ;;
    esac
done

# Set Drive path based on --archive flag
if [[ -n "$ARCHIVE" ]]; then
    DRIVE_PATH="$DRIVE_REMOTE:$DRIVE_BASE/archive/v1"
else
    DRIVE_PATH="$DRIVE_REMOTE:$DRIVE_BASE"
fi

# List mode
if [[ -n "$LIST" ]]; then
    echo "Runs on Drive ($DRIVE_PATH/):"
    echo ""
    rclone lsd "$DRIVE_PATH/" 2>/dev/null || echo "  (none found)"
    exit 0
fi

# Build rclone filter flags
RCLONE_FILTERS=()
if [[ -n "$NO_CKPTS" ]]; then
    # Exclude both old (.pt) and new (.msgpack) checkpoint files
    RCLONE_FILTERS+=(--exclude "checkpoints/*.pt")
    RCLONE_FILTERS+=(--exclude "checkpoints/*.msgpack")
    RCLONE_FILTERS+=(--exclude "checkpoints/*.json")
    RCLONE_FILTERS+=(--exclude "replay_buffer_*.npz")
fi

# Pull a single run directory
pull_one() {
    local name="$1"
    local src="$DRIVE_PATH/$name/"
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

# Filter run names by condition group
filter_group() {
    local pattern="$1"
    local names=("${@:2}")
    local filtered=()

    for name in "${names[@]}"; do
        # Skip non-run directories
        [[ "$name" == "deprecated" || "$name" == "invalid" || "$name" == "archive" ]] && continue

        case "$pattern" in
            derc)
                [[ "$name" == derc_* ]] && filtered+=("$name")
                ;;
            der)
                [[ "$name" == der_* && "$name" != derc_* ]] && filtered+=("$name")
                ;;
            sprc)
                [[ "$name" == sprc_* ]] && filtered+=("$name")
                ;;
            spr)
                [[ "$name" == spr_* && "$name" != sprc_* ]] && filtered+=("$name")
                ;;
            sr-sprc)
                [[ "$name" == sr-sprc_* ]] && filtered+=("$name")
                ;;
            sr-spr)
                [[ "$name" == sr-spr_* && "$name" != sr-sprc_* ]] && filtered+=("$name")
                ;;
            bbfc)
                [[ "$name" == bbfc_* ]] && filtered+=("$name")
                ;;
            bbf)
                [[ "$name" == bbf_* && "$name" != bbfc_* ]] && filtered+=("$name")
                ;;
            *)
                # Arbitrary substring match for flexibility
                [[ "$name" == *"$pattern"* ]] && filtered+=("$name")
                ;;
        esac
    done

    printf '%s\n' "${filtered[@]}"
}

# Fetch run list from Drive
get_run_names() {
    rclone lsd "$DRIVE_PATH/" 2>/dev/null \
        | awk '{print $NF}' \
        | grep -v '^deprecated$' \
        | grep -v '^invalid$' \
        | grep -v '^archive$' \
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

SRC="$DRIVE_PATH/$RUN_NAME/"
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
