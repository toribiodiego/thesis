#!/usr/bin/env python3
"""Aggregate results collector.

Walks all run directories under a given path, reads per-checkpoint
analysis CSVs from each run's analysis/ directory, adds index
columns (condition, game, seed, checkpoint) from meta.json, and
writes one aggregate CSV per method to the output directory.

No GPU or JAX needed -- pure file I/O.

Usage:
    python scripts/analysis/collect_results.py \\
        --runs-dir experiments/dqn_atari/runs \\
        --output-dir output/aggregate

    # Dry run: list what would be collected
    python scripts/analysis/collect_results.py \\
        --runs-dir experiments/dqn_atari/runs \\
        --output-dir output/aggregate --dry-run
"""

import argparse
import json
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# Methods and their expected CSV filenames
METHOD_FILES = {
    "atariari_probing": "atariari_probing.csv",
    "reward_probing": "reward_probing.csv",
    "inverse_dynamics": "inverse_dynamics.csv",
    "structural_health": "structural_health.csv",
    "filter_frequency": "filter_frequency.csv",
    "q_accuracy": "q_accuracy.csv",
    "transition_eval": "transition_eval.csv",
}

INDEX_COLUMNS = ["condition", "game", "seed", "checkpoint"]


def _discover_runs(runs_dir):
    """Find all run directories that have a meta.json."""
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        meta_path = os.path.join(run_dir, "meta.json")
        if os.path.isdir(run_dir) and os.path.isfile(meta_path):
            runs.append((name, run_dir, meta_path))
    return runs


def _discover_checkpoints(run_dir):
    """Find all checkpoint analysis directories."""
    analysis_dir = os.path.join(run_dir, "analysis")
    if not os.path.isdir(analysis_dir):
        return []
    steps = []
    for name in os.listdir(analysis_dir):
        m = re.match(r"checkpoint_(\d+)$", name)
        if m and os.path.isdir(os.path.join(analysis_dir, name)):
            steps.append(int(m.group(1)))
    return sorted(steps)


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-checkpoint analysis CSVs across runs"
    )
    parser.add_argument("--runs-dir", required=True,
                        help="Directory containing run folders")
    parser.add_argument("--output-dir", required=True,
                        help="Directory for aggregate CSVs")
    parser.add_argument("--dry-run", action="store_true",
                        help="List what would be collected without writing")
    args = parser.parse_args()

    # -- Discover runs -------------------------------------------------------
    runs = _discover_runs(args.runs_dir)
    print(f"Found {len(runs)} runs in {args.runs_dir}")

    # -- Collect per-method dataframes ---------------------------------------
    method_dfs = {method: [] for method in METHOD_FILES}
    total_files = 0

    for run_name, run_dir, meta_path in runs:
        with open(meta_path) as f:
            meta = json.load(f)
        condition = meta.get("condition", "unknown")
        game = meta.get("game", "unknown")
        seed = meta.get("seed", 0)

        checkpoints = _discover_checkpoints(run_dir)
        if not checkpoints:
            continue

        for step in checkpoints:
            ckpt_dir = os.path.join(run_dir, "analysis", f"checkpoint_{step}")

            for method, filename in METHOD_FILES.items():
                csv_path = os.path.join(ckpt_dir, filename)
                if not os.path.isfile(csv_path):
                    continue

                if args.dry_run:
                    print(f"  {run_name}/checkpoint_{step}/{filename}")
                    total_files += 1
                    continue

                df = pd.read_csv(csv_path)
                df.insert(0, "condition", condition)
                df.insert(1, "game", game)
                df.insert(2, "seed", seed)
                df.insert(3, "checkpoint", step)
                method_dfs[method].append(df)
                total_files += 1

    if args.dry_run:
        print(f"\nWould collect {total_files} CSV files.")
        return

    # -- Write aggregate CSVs ------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    written = 0

    for method, dfs in method_dfs.items():
        if not dfs:
            continue
        combined = pd.concat(dfs, ignore_index=True)
        out_path = os.path.join(args.output_dir, f"{method}.csv")
        combined.to_csv(out_path, index=False)
        print(f"  {method}.csv: {len(combined)} rows "
              f"from {len(dfs)} checkpoint(s)")
        written += 1

    print(f"\nWrote {written} aggregate CSVs to {args.output_dir}")
    if written == 0:
        print("No analysis results found. Run run_all.py first.")


if __name__ == "__main__":
    main()
