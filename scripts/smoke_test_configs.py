#!/usr/bin/env python
"""
Smoke test all config types to verify features activate correctly.

Runs each config type for a small number of frames (~5K), then parses
the output CSV to verify that expected columns are populated. This
catches silent config bugs where a feature flag is defined in YAML
but never read by train_dqn.py, producing runs that look normal but
contain no meaningful feature-specific data.

Config types tested:
  base         - Vanilla DQN (no augmentation, no SPR, no Rainbow)
  aug          - DQN + random shift augmentation
  spr          - DQN + SPR auxiliary loss
  both         - DQN + SPR + augmentation
  rainbow      - Rainbow DQN (distributional, NoisyNets, PER, dueling)
  rainbow_spr  - Rainbow + SPR

Usage:
    python scripts/smoke_test_configs.py
    python scripts/smoke_test_configs.py --types base spr rainbow
    python scripts/smoke_test_configs.py --frames 2000
    python scripts/smoke_test_configs.py --game boxing
    python scripts/smoke_test_configs.py --keep-runs
"""

import argparse
import csv
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path so we can import src/ and train_dqn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from omegaconf import OmegaConf

# ---------------------------------------------------------------------------
# Expected columns per config type
# ---------------------------------------------------------------------------

# Columns that must be non-empty (at least one row with a value) for each type
CORE_COLUMNS = ["step", "epsilon", "replay_size", "loss", "td_error", "grad_norm"]

SPR_COLUMNS = ["spr_loss", "cosine_similarity", "ema_update_count"]

RAINBOW_COLUMNS = [
    "distributional_loss",
    "mean_is_weight",
    "mean_priority",
    "priority_entropy",
    "beta",
]

EXPECTED_COLUMNS: Dict[str, List[str]] = {
    "base": CORE_COLUMNS,
    "aug": CORE_COLUMNS,
    "spr": CORE_COLUMNS + SPR_COLUMNS,
    "both": CORE_COLUMNS + SPR_COLUMNS,
    "rainbow": CORE_COLUMNS + RAINBOW_COLUMNS,
    "rainbow_spr": CORE_COLUMNS + RAINBOW_COLUMNS + SPR_COLUMNS,
}

# Columns that must be ABSENT (all empty) for each type
ABSENT_COLUMNS: Dict[str, List[str]] = {
    "base": SPR_COLUMNS + RAINBOW_COLUMNS,
    "aug": SPR_COLUMNS + RAINBOW_COLUMNS,
    "spr": RAINBOW_COLUMNS,
    "both": RAINBOW_COLUMNS,
    "rainbow": SPR_COLUMNS,
    "rainbow_spr": [],
}

# Config file for each type (Boxing as canonical test game)
CONFIG_FILES: Dict[str, str] = {
    "base": "experiments/dqn_atari/configs/atari100k_boxing.yaml",
    "aug": "experiments/dqn_atari/configs/atari100k_boxing_aug.yaml",
    "spr": "experiments/dqn_atari/configs/atari100k_boxing_spr.yaml",
    "both": "experiments/dqn_atari/configs/atari100k_boxing_both.yaml",
    "rainbow": "experiments/dqn_atari/configs/atari100k_boxing_rainbow.yaml",
    "rainbow_spr": "experiments/dqn_atari/configs/atari100k_boxing_rainbow_spr.yaml",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_smoke_config(config_type: str, total_frames: int, game: str) -> dict:
    """Load and override a config for fast smoke testing."""
    from src.config.config_loader import load_config

    cfg_path = CONFIG_FILES[config_type]

    # Swap game if not boxing
    if game != "boxing":
        cfg_path = cfg_path.replace("boxing", game)

    config = load_config(cfg_path)

    # Override for fast smoke testing
    config["training"]["total_frames"] = total_frames
    config["replay"]["capacity"] = 2000
    config["replay"]["batch_size"] = 4
    config["replay"]["min_size"] = 50
    config["replay"]["warmup_steps"] = 50
    config["evaluation"]["eval_every"] = total_frames + 1  # skip eval
    config["logging"]["log_every_steps"] = 10
    config["logging"]["csv"]["enabled"] = True
    config["logging"]["tensorboard"]["enabled"] = False
    config["logging"]["wandb"]["enabled"] = False
    config["logging"]["checkpoint"]["enabled"] = False
    config["network"]["device"] = "cpu"

    return config


# ---------------------------------------------------------------------------
# Training execution
# ---------------------------------------------------------------------------


def run_smoke_training(
    config_dict: dict, run_dir: Path
) -> Path:
    """Run a short training loop and return path to CSV file."""
    from src.config.run_manager import setup_run_directory
    from train_dqn import run_training, setup_device

    # Override the base_dir so runs go to our temp directory
    config_dict["logging"]["base_dir"] = str(run_dir)
    config = OmegaConf.create(config_dict)

    paths = setup_run_directory(config_dict)
    device = setup_device(config)

    run_training(config, paths, device)

    # CSV is written by MetricsLogger at {run_dir}/csv/training_steps.csv
    csv_path = Path(paths["run_dir"]) / "csv" / "training_steps.csv"
    return csv_path


# ---------------------------------------------------------------------------
# CSV validation
# ---------------------------------------------------------------------------


def validate_csv(
    csv_path: Path,
    config_type: str,
) -> Tuple[bool, List[str], List[str]]:
    """Validate that expected columns are populated in the CSV.

    Returns:
        (passed, errors, warnings) tuple
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not csv_path.exists():
        errors.append(f"CSV file not found: {csv_path}")
        return False, errors, warnings

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        errors.append("CSV file has no data rows")
        return False, errors, warnings

    # Check expected columns are populated (at least one non-empty value)
    for col in EXPECTED_COLUMNS[config_type]:
        values = [r.get(col, "") for r in rows]
        non_empty = [v for v in values if v != ""]
        if len(non_empty) == 0:
            errors.append(f"Column '{col}' is all empty (expected populated)")

    # Check absent columns are NOT populated
    for col in ABSENT_COLUMNS[config_type]:
        values = [r.get(col, "") for r in rows]
        non_empty = [v for v in values if v != ""]
        if len(non_empty) > 0:
            errors.append(
                f"Column '{col}' has {len(non_empty)} values "
                f"(expected empty for {config_type})"
            )

    # Config-specific checks
    if config_type in ("rainbow", "rainbow_spr"):
        # NoisyNets: epsilon should be 0.0 throughout
        epsilons = [float(r["epsilon"]) for r in rows if r.get("epsilon", "")]
        non_zero = [e for e in epsilons if e > 0.001]
        if non_zero:
            errors.append(
                f"NoisyNets config has non-zero epsilon: "
                f"max={max(non_zero):.4f} (expected 0.0)"
            )

    passed = len(errors) == 0
    return passed, errors, warnings


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_smoke_tests(
    config_types: List[str],
    total_frames: int,
    game: str,
    keep_runs: bool,
) -> bool:
    """Run smoke tests for specified config types. Returns True if all pass."""
    results: Dict[str, Tuple[bool, List[str], List[str]]] = {}
    all_passed = True

    for config_type in config_types:
        print(f"\n{'='*60}")
        print(f"  Smoke test: {config_type}")
        print(f"{'='*60}")

        # Create temp directory for this run
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"smoke_{config_type}_"))

        try:
            # Load config
            config_dict = load_smoke_config(config_type, total_frames, game)
            print(f"  Config: {CONFIG_FILES[config_type]}")
            print(f"  Frames: {total_frames}")

            # Run training
            csv_path = run_smoke_training(config_dict, tmp_dir)
            print(f"  CSV: {csv_path}")

            # Validate
            passed, errors, warnings = validate_csv(csv_path, config_type)
            results[config_type] = (passed, errors, warnings)

            if passed:
                print(f"  Result: PASSED")
            else:
                print(f"  Result: FAILED")
                for err in errors:
                    print(f"    ERROR: {err}")
                all_passed = False

            for warn in warnings:
                print(f"    WARNING: {warn}")

        except Exception as e:
            print(f"  Result: CRASHED")
            print(f"    {type(e).__name__}: {e}")
            results[config_type] = (False, [str(e)], [])
            all_passed = False

        finally:
            if not keep_runs:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            else:
                print(f"  Run dir: {tmp_dir}")

    # Summary
    print(f"\n{'='*60}")
    print("  SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    for config_type in config_types:
        passed, errors, _ = results.get(config_type, (False, ["not run"], []))
        status = "PASSED" if passed else "FAILED"
        print(f"  {config_type:15s} {status}")
        if not passed:
            for err in errors:
                print(f"    - {err}")
    print(f"{'='*60}")

    if all_passed:
        print(f"\n  All {len(config_types)} config types passed.")
    else:
        failed = [t for t, (p, _, _) in results.items() if not p]
        print(f"\n  {len(failed)}/{len(config_types)} config types FAILED.")

    return all_passed


def main():
    # Ensure we run from project root for config path resolution
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)

    parser = argparse.ArgumentParser(
        description="Smoke test all config types for correct feature activation.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=list(EXPECTED_COLUMNS.keys()),
        default=list(EXPECTED_COLUMNS.keys()),
        help="Config types to test (default: all)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5000,
        help="Total frames per config (default: 5000)",
    )
    parser.add_argument(
        "--game",
        type=str,
        default="boxing",
        help="Game to use for testing (default: boxing)",
    )
    parser.add_argument(
        "--keep-runs",
        action="store_true",
        help="Keep temporary run directories after testing",
    )

    args = parser.parse_args()

    passed = run_smoke_tests(
        config_types=args.types,
        total_frames=args.frames,
        game=args.game,
        keep_runs=args.keep_runs,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
