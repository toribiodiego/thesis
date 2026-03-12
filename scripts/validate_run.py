#!/usr/bin/env python
"""
Validate a completed training run against its saved config.

Reads config.yaml from the run directory, determines which features
should be active (Rainbow, SPR, augmentation), then checks that the
CSV metrics and checkpoints match expectations. Catches bugs where a
config flag was silently ignored during training.

Checks performed:
  - CSV columns populated for active features (SPR, Rainbow)
  - CSV columns empty for inactive features
  - Epsilon=0 throughout for NoisyNets (Rainbow) configs
  - Correct number of periodic checkpoints
  - SPR loss values are reasonable (not zero, not NaN)
  - progress.json exists and shows completion

Usage:
    python scripts/validate_run.py experiments/dqn_atari/runs/atari100k_boxing_42/
    python scripts/validate_run.py experiments/dqn_atari/runs/atari100k_boxing_rainbow_42/ --json
    python scripts/validate_run.py path/to/run --expected-checkpoints 10
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Column expectations by feature
# ---------------------------------------------------------------------------

CORE_COLUMNS = ["step", "epsilon", "replay_size", "loss", "td_error", "grad_norm"]

SPR_COLUMNS = ["spr_loss", "cosine_similarity", "ema_update_count"]

RAINBOW_COLUMNS = [
    "distributional_loss",
    "mean_is_weight",
    "mean_priority",
    "priority_entropy",
    "beta",
]


# ---------------------------------------------------------------------------
# Config inspection
# ---------------------------------------------------------------------------


def detect_features(config: dict) -> Dict[str, bool]:
    """Detect which features are enabled from a saved config."""
    rainbow_cfg = config.get("rainbow", {})
    spr_cfg = config.get("spr", {})
    aug_cfg = config.get("augmentation", {})

    rainbow_enabled = rainbow_cfg.get("enabled", False)
    return {
        "rainbow": rainbow_enabled,
        "noisy_nets": rainbow_enabled and rainbow_cfg.get("noisy_nets", False),
        "spr": spr_cfg.get("enabled", False),
        "augmentation": aug_cfg.get("enabled", False),
    }


def expected_checkpoint_count(config: dict) -> int:
    """Calculate expected number of periodic checkpoints."""
    total_frames = config.get("training", {}).get("total_frames", 400000)
    save_every = (
        config.get("logging", {}).get("checkpoint", {}).get("save_every", 40000)
    )
    keep_last_n = (
        config.get("logging", {}).get("checkpoint", {}).get("keep_last_n", 3)
    )
    checkpoint_enabled = (
        config.get("logging", {}).get("checkpoint", {}).get("enabled", True)
    )

    if not checkpoint_enabled or save_every <= 0:
        return 0

    total_periodic = total_frames // save_every
    if keep_last_n > 0:
        return min(total_periodic, keep_last_n)
    return total_periodic


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def validate_run(
    run_dir: Path,
    expected_ckpts: int = None,
) -> Dict[str, Any]:
    """Validate a completed run directory.

    Args:
        run_dir: Path to the run directory.
        expected_ckpts: Override expected checkpoint count (auto-detects if None).

    Returns:
        Dict with 'passed', 'checks' (list of check results), and metadata.
    """
    checks: List[Dict[str, Any]] = []

    def add_check(name: str, passed: bool, detail: str = ""):
        checks.append({"name": name, "passed": passed, "detail": detail})

    # -- Check 1: run directory exists
    if not run_dir.exists():
        add_check("run_dir_exists", False, f"Not found: {run_dir}")
        return {"passed": False, "checks": checks, "run_dir": str(run_dir)}

    # -- Check 2: config.yaml exists and is valid
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        add_check("config_exists", False, "config.yaml not found")
        return {"passed": False, "checks": checks, "run_dir": str(run_dir)}

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        add_check("config_exists", True)
    except Exception as e:
        add_check("config_exists", False, f"Invalid YAML: {e}")
        return {"passed": False, "checks": checks, "run_dir": str(run_dir)}

    features = detect_features(config)

    # -- Check 3: progress.json shows completion
    progress_path = run_dir / "progress.json"
    if progress_path.exists():
        try:
            with open(progress_path) as f:
                progress = json.load(f)
            status = progress.get("status", "unknown")
            percent = progress.get("percent", 0)
            if status == "complete" and percent >= 99.0:
                add_check("training_complete", True, f"status={status}")
            else:
                add_check(
                    "training_complete",
                    False,
                    f"status={status}, percent={percent}%",
                )
        except Exception as e:
            add_check("training_complete", False, f"Cannot parse: {e}")
    else:
        add_check("training_complete", False, "progress.json not found")

    # -- Check 4: CSV exists and has data
    csv_path = run_dir / "csv" / "training_steps.csv"
    if not csv_path.exists():
        add_check("csv_exists", False, "csv/training_steps.csv not found")
        return {
            "passed": False,
            "checks": checks,
            "run_dir": str(run_dir),
            "features": features,
        }

    try:
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        add_check("csv_exists", False, f"Cannot parse CSV: {e}")
        return {
            "passed": False,
            "checks": checks,
            "run_dir": str(run_dir),
            "features": features,
        }

    if len(rows) == 0:
        add_check("csv_exists", False, "CSV has no data rows")
    else:
        add_check("csv_exists", True, f"{len(rows)} rows")

    # -- Check 5: Core columns populated
    for col in CORE_COLUMNS:
        values = [r.get(col, "") for r in rows]
        non_empty = [v for v in values if v != ""]
        if len(non_empty) > 0:
            add_check(f"column_{col}", True, f"{len(non_empty)}/{len(rows)} populated")
        else:
            add_check(f"column_{col}", False, "all empty")

    # -- Check 6: SPR columns
    if features["spr"]:
        for col in SPR_COLUMNS:
            values = [r.get(col, "") for r in rows]
            non_empty = [v for v in values if v != ""]
            if len(non_empty) > 0:
                add_check(
                    f"spr_{col}",
                    True,
                    f"{len(non_empty)}/{len(rows)} populated",
                )
            else:
                add_check(
                    f"spr_{col}",
                    False,
                    f"SPR enabled but {col} is all empty",
                )

        # SPR loss sanity: not all zeros, no NaN
        spr_vals = [
            float(r["spr_loss"])
            for r in rows
            if r.get("spr_loss", "") != ""
        ]
        if spr_vals:
            has_nan = any(math.isnan(v) for v in spr_vals)
            all_zero = all(v == 0.0 for v in spr_vals)
            if has_nan:
                add_check("spr_loss_valid", False, "Contains NaN values")
            elif all_zero:
                add_check("spr_loss_valid", False, "All zero (SPR not active)")
            else:
                add_check(
                    "spr_loss_valid",
                    True,
                    f"range [{min(spr_vals):.4f}, {max(spr_vals):.4f}]",
                )
    else:
        # SPR disabled: columns should be empty
        for col in SPR_COLUMNS:
            values = [r.get(col, "") for r in rows]
            non_empty = [v for v in values if v != ""]
            if len(non_empty) == 0:
                add_check(f"no_spurious_{col}", True, "correctly empty")
            else:
                add_check(
                    f"no_spurious_{col}",
                    False,
                    f"SPR disabled but {col} has {len(non_empty)} values",
                )

    # -- Check 7: Rainbow columns
    if features["rainbow"]:
        for col in RAINBOW_COLUMNS:
            values = [r.get(col, "") for r in rows]
            non_empty = [v for v in values if v != ""]
            if len(non_empty) > 0:
                add_check(
                    f"rainbow_{col}",
                    True,
                    f"{len(non_empty)}/{len(rows)} populated",
                )
            else:
                add_check(
                    f"rainbow_{col}",
                    False,
                    f"Rainbow enabled but {col} is all empty",
                )
    else:
        for col in RAINBOW_COLUMNS:
            values = [r.get(col, "") for r in rows]
            non_empty = [v for v in values if v != ""]
            if len(non_empty) == 0:
                add_check(f"no_spurious_{col}", True, "correctly empty")
            else:
                add_check(
                    f"no_spurious_{col}",
                    False,
                    f"Rainbow disabled but {col} has {len(non_empty)} values",
                )

    # -- Check 8: NoisyNets epsilon=0
    if features["noisy_nets"]:
        epsilons = [
            float(r["epsilon"])
            for r in rows
            if r.get("epsilon", "") != ""
        ]
        non_zero = [e for e in epsilons if e > 0.001]
        if non_zero:
            add_check(
                "noisy_nets_epsilon",
                False,
                f"NoisyNets enabled but epsilon non-zero: "
                f"max={max(non_zero):.4f} ({len(non_zero)}/{len(epsilons)} rows)",
            )
        elif epsilons:
            add_check(
                "noisy_nets_epsilon",
                True,
                f"epsilon=0 on all {len(epsilons)} rows",
            )
        else:
            add_check("noisy_nets_epsilon", False, "No epsilon values found")

    # -- Check 9: Checkpoints
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.exists():
        periodic = sorted(ckpt_dir.glob("checkpoint_*.pt"))
        best = ckpt_dir / "best_model.pt"

        if expected_ckpts is None:
            expected_ckpts = expected_checkpoint_count(config)

        if expected_ckpts > 0:
            if len(periodic) == expected_ckpts:
                add_check(
                    "checkpoint_count",
                    True,
                    f"{len(periodic)}/{expected_ckpts} periodic checkpoints",
                )
            else:
                add_check(
                    "checkpoint_count",
                    False,
                    f"{len(periodic)}/{expected_ckpts} periodic checkpoints",
                )
        else:
            add_check(
                "checkpoint_count",
                True,
                f"checkpointing disabled, {len(periodic)} found",
            )

        save_best = (
            config.get("logging", {}).get("checkpoint", {}).get("save_best", True)
        )
        if save_best:
            if best.exists():
                add_check("best_checkpoint", True)
            else:
                add_check("best_checkpoint", False, "best_model.pt not found")
        else:
            add_check("best_checkpoint", True, "save_best=false (skipped)")
    else:
        add_check("checkpoint_count", False, "checkpoints/ directory not found")

    all_passed = all(c["passed"] for c in checks)
    return {
        "passed": all_passed,
        "checks": checks,
        "run_dir": str(run_dir),
        "features": features,
        "num_csv_rows": len(rows) if rows else 0,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def print_results(results: Dict[str, Any], as_json: bool = False):
    """Print validation results."""
    if as_json:
        print(json.dumps(results, indent=2))
        return

    print("=" * 60)
    print("  Post-Run Validation Report")
    print("=" * 60)
    print(f"  Run: {results['run_dir']}")

    features = results.get("features", {})
    if features:
        active = [k for k, v in features.items() if v]
        print(f"  Features: {', '.join(active) if active else 'base DQN'}")

    print(f"  CSV rows: {results.get('num_csv_rows', 'N/A')}")
    print()

    passed_checks = [c for c in results["checks"] if c["passed"]]
    failed_checks = [c for c in results["checks"] if not c["passed"]]

    if failed_checks:
        print(f"  FAILED ({len(failed_checks)}):")
        for c in failed_checks:
            detail = f" -- {c['detail']}" if c["detail"] else ""
            print(f"    [FAIL] {c['name']}{detail}")
        print()

    if passed_checks:
        print(f"  PASSED ({len(passed_checks)}):")
        for c in passed_checks:
            detail = f" -- {c['detail']}" if c["detail"] else ""
            print(f"    [OK]   {c['name']}{detail}")

    print()
    print("=" * 60)
    status = "PASSED" if results["passed"] else "FAILED"
    total = len(results["checks"])
    ok = len(passed_checks)
    print(f"  Result: {status} ({ok}/{total} checks passed)")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Validate a completed training run against its config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python scripts/validate_run.py experiments/dqn_atari/runs/boxing_42/
  python scripts/validate_run.py path/to/run --expected-checkpoints 10
  python scripts/validate_run.py path/to/run --json
""",
    )
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument(
        "--expected-checkpoints",
        type=int,
        default=None,
        help="Override expected checkpoint count (auto-detects from config)",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    results = validate_run(args.run_dir, expected_ckpts=args.expected_checkpoints)
    print_results(results, as_json=args.json)

    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
