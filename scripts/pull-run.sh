#!/bin/bash
# Download a training run from Google Drive via rclone.
#
# Usage:
#   bash scripts/pull-run.sh <run-name>
#   bash scripts/pull-run.sh --dry-run <run-name>
#   bash scripts/pull-run.sh --list
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
    echo "Usage: $0 [--dry-run] [--list] <run-name>"
    echo ""
    echo "Options:"
    echo "  --list      List all runs on Drive with sizes"
    echo "  --dry-run   Preview what would be downloaded"
    echo ""
    echo "Examples:"
    echo "  $0 --list"
    echo "  $0 atari100k_pong_42_20260310_032432"
    echo "  $0 --dry-run atari100k_pong_42_20260310_032432"
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
RUN_NAME=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY_RUN="--dry-run"; shift ;;
        --list) LIST=1; shift ;;
        --help|-h) usage ;;
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

# Download mode
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

rclone copy $DRY_RUN --progress "$SRC" "$DST"

if [[ -z "$DRY_RUN" ]]; then
    echo ""
    echo "Downloaded to: $DST"
fi
