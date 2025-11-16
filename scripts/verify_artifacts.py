#!/usr/bin/env python
"""
Verify DQN training run artifacts are complete.

Checks that all required files exist in a run directory before uploading to W&B
or generating reports. Prevents uploading incomplete artifacts.

Usage:
    python scripts/verify_artifacts.py experiments/dqn_atari/runs/pong_42_20251116/
    python scripts/verify_artifacts.py experiments/dqn_atari/runs/pong_42_20251116/ --strict
    python scripts/verify_artifacts.py experiments/dqn_atari/runs/pong_42_20251116/ --json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict


def verify_artifacts(run_dir: Path, strict: bool = False) -> Dict[str, Any]:
    """
    Verify that required artifacts exist in the run directory.

    Args:
        run_dir: Path to run directory
        strict: If True, require all artifacts; if False, only core artifacts

    Returns:
        Dictionary with verification results
    """
    results = {
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
        "valid": True,
        "missing": [],
        "warnings": [],
        "found": [],
    }

    if not run_dir.exists():
        results["valid"] = False
        results["error"] = f"Run directory not found: {run_dir}"
        return results

    # Core required files (always needed)
    core_files = [
        "config.yaml",
        "meta.json",
    ]

    # CSV logging files
    csv_files = [
        "csv/training_steps.csv",
        "csv/episodes.csv",
    ]

    # Evaluation files
    eval_files = [
        "eval/evaluations.csv",
        "eval/evaluations.jsonl",
        "eval/per_episode_returns.jsonl",
    ]

    # Optional but recommended files
    optional_files = [
        "checkpoints/best_model.pt",
    ]

    # Check core files
    for file in core_files:
        path = run_dir / file
        if path.exists():
            results["found"].append(file)
        else:
            results["missing"].append(file)
            results["valid"] = False

    # Check CSV files
    for file in csv_files:
        path = run_dir / file
        if path.exists():
            results["found"].append(file)
            # Verify file is not empty
            if path.stat().st_size == 0:
                results["warnings"].append(f"{file} exists but is empty")
        else:
            results["missing"].append(file)
            results["valid"] = False

    # Check evaluation files
    eval_dir = run_dir / "eval"
    if eval_dir.exists():
        for file in eval_files:
            path = run_dir / file
            if path.exists():
                results["found"].append(file)
                if path.stat().st_size == 0:
                    results["warnings"].append(f"{file} exists but is empty")
            else:
                if strict:
                    results["missing"].append(file)
                    results["valid"] = False
                else:
                    results["warnings"].append(f"{file} not found (optional)")
    else:
        if strict:
            results["missing"].append("eval/")
            results["valid"] = False
        else:
            results["warnings"].append("eval/ directory not found (no evaluations run yet)")

    # Check optional files
    for file in optional_files:
        path = run_dir / file
        if path.exists():
            results["found"].append(file)
        else:
            results["warnings"].append(f"{file} not found (optional)")

    # Check for at least one checkpoint
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            results["found"].append(f"checkpoints/ ({len(checkpoints)} files)")
        else:
            results["warnings"].append("No checkpoint files found in checkpoints/")
    else:
        if strict:
            results["missing"].append("checkpoints/")
            results["valid"] = False
        else:
            results["warnings"].append("checkpoints/ directory not found")

    # Check for videos (optional)
    video_dir = run_dir / "videos"
    if video_dir.exists():
        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.gif"))
        if videos:
            results["found"].append(f"videos/ ({len(videos)} files)")
        else:
            results["warnings"].append("videos/ directory exists but is empty")
    else:
        results["warnings"].append("videos/ directory not found (video recording may be disabled)")

    # Validate file contents
    results["content_checks"] = {}

    # Check config.yaml is valid YAML
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path, "r") as f:
                yaml.safe_load(f)
            results["content_checks"]["config.yaml"] = "valid"
        except Exception as e:
            results["content_checks"]["config.yaml"] = f"invalid: {str(e)}"
            results["valid"] = False

    # Check meta.json is valid JSON
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                json.load(f)
            results["content_checks"]["meta.json"] = "valid"
        except Exception as e:
            results["content_checks"]["meta.json"] = f"invalid: {str(e)}"
            results["valid"] = False

    # Check CSV has header row
    for csv_file in ["csv/training_steps.csv", "csv/episodes.csv"]:
        csv_path = run_dir / csv_file
        if csv_path.exists() and csv_path.stat().st_size > 0:
            try:
                with open(csv_path, "r") as f:
                    header = f.readline().strip()
                    if "," in header:
                        results["content_checks"][csv_file] = "has header"
                    else:
                        results["content_checks"][csv_file] = "missing header"
                        results["warnings"].append(f"{csv_file} may be missing header row")
            except Exception as e:
                results["content_checks"][csv_file] = f"error: {str(e)}"

    return results


def print_results(results: Dict[str, Any], as_json: bool = False):
    """Print verification results."""
    if as_json:
        print(json.dumps(results, indent=2))
        return

    print("=" * 60)
    print("Artifact Verification Report")
    print("=" * 60)

    print(f"Run Directory: {results['run_dir']}")

    if "error" in results:
        print(f"\nERROR: {results['error']}")
        return

    status = "PASS" if results["valid"] else "FAIL"
    print(f"Overall Status: {status}")

    if results["found"]:
        print(f"\nFound ({len(results['found'])} items):")
        for item in results["found"]:
            print(f"  [OK] {item}")

    if results["missing"]:
        print(f"\nMissing ({len(results['missing'])} items):")
        for item in results["missing"]:
            print(f"  [MISSING] {item}")

    if results["warnings"]:
        print(f"\nWarnings ({len(results['warnings'])} items):")
        for warning in results["warnings"]:
            print(f"  [WARN] {warning}")

    if results.get("content_checks"):
        print("\nContent Validation:")
        for file, status in results["content_checks"].items():
            print(f"  {file}: {status}")

    print("\n" + "=" * 60)

    if results["valid"]:
        print("Artifacts are complete and ready for upload.")
    else:
        print("Artifacts are INCOMPLETE. Fix missing files before uploading.")


def main():
    parser = argparse.ArgumentParser(
        description="Verify DQN training run artifacts are complete",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/verify_artifacts.py experiments/dqn_atari/runs/pong_42_20251116/
  python scripts/verify_artifacts.py experiments/dqn_atari/runs/pong_42_20251116/ --strict
  python scripts/verify_artifacts.py experiments/dqn_atari/runs/pong_42_20251116/ --json
        """,
    )
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument(
        "--strict", action="store_true", help="Require all artifacts including evaluation files"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    results = verify_artifacts(args.run_dir, strict=args.strict)
    print_results(results, as_json=args.json)

    # Exit with error code if verification failed
    if not results["valid"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
