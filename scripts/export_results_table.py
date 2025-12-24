#!/usr/bin/env python3
"""
Results table exporter for DQN training runs.

Generates summary tables with run metadata:
- run_id, game, mean_eval_return, frames, wall_time, seed, commit_hash
- Outputs to Markdown and CSV formats
- Optional W&B upload for provenance
"""

import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime


def load_run_metadata(run_dir: Path) -> Optional[Dict]:
    """
    Load metadata from a training run directory.

    Args:
        run_dir: Path to run directory (e.g., runs/pong_123/)

    Returns:
        dict: Run metadata including eval results, or None if not found
    """
    metadata = {"run_id": run_dir.name, "run_dir": str(run_dir)}

    # Load meta.json if exists
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        try:
            with open(meta_path) as f:
                meta = json.load(f)
                metadata["game"] = meta.get("game", "unknown")
                metadata["seed"] = meta.get("seed", -1)
                metadata["commit_hash"] = meta.get("commit_hash", "unknown")
                metadata["start_time"] = meta.get("start_time", "unknown")
        except Exception as e:
            print(f"Warning: Failed to load {meta_path}: {e}")

    # Load evaluation results
    eval_csv = run_dir / "eval" / "evaluations.csv"
    if eval_csv.exists():
        try:
            with open(eval_csv) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    # Get final evaluation
                    final_eval = rows[-1]
                    metadata["mean_eval_return"] = float(
                        final_eval.get("mean_return", 0)
                    )
                    metadata["std_eval_return"] = float(final_eval.get("std_return", 0))
                    metadata["final_step"] = int(final_eval.get("step", 0))
        except Exception as e:
            print(f"Warning: Failed to load {eval_csv}: {e}")

    # Load config.yaml for frame count
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        try:
            import yaml

            with open(config_path) as f:
                config = yaml.safe_load(f)
                training = config.get("training", {})
                metadata["total_frames"] = training.get("total_frames", 0)
        except Exception as e:
            print(f"Warning: Failed to load {config_path}: {e}")

    # Estimate wall time if possible
    if "start_time" in metadata:
        try:
            start = datetime.fromisoformat(metadata["start_time"])
            # Try to get end time from checkpoint or logs
            checkpoints = list((run_dir / "checkpoints").glob("checkpoint_*.pt"))
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
                end = datetime.fromtimestamp(latest_checkpoint.stat().st_mtime)
                wall_time_hours = (end - start).total_seconds() / 3600
                metadata["wall_time_hours"] = round(wall_time_hours, 2)
        except Exception:
            pass

    return metadata


def scan_run_directories(base_dir: Path, pattern: str = "**/meta.json") -> List[Path]:
    """
    Scan for run directories containing metadata files.

    Args:
        base_dir: Base directory to scan
        pattern: Glob pattern for finding runs

    Returns:
        list: List of run directory paths
    """
    run_dirs = []
    for meta_file in base_dir.glob(pattern):
        run_dir = meta_file.parent
        run_dirs.append(run_dir)

    return sorted(run_dirs)


def export_to_csv(runs: List[Dict], output_path: Path):
    """
    Export runs to CSV file.

    Args:
        runs: List of run metadata dicts
        output_path: Path to output CSV file
    """
    if not runs:
        print("Warning: No runs to export")
        return

    # Define column order
    fieldnames = [
        "run_id",
        "game",
        "seed",
        "mean_eval_return",
        "std_eval_return",
        "final_step",
        "total_frames",
        "wall_time_hours",
        "commit_hash",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(runs)

    print(f"Exported {len(runs)} runs to CSV: {output_path}")


def export_to_markdown(runs: List[Dict], output_path: Path):
    """
    Export runs to Markdown table.

    Args:
        runs: List of run metadata dicts
        output_path: Path to output Markdown file
    """
    if not runs:
        print("Warning: No runs to export")
        return

    with open(output_path, "w") as f:
        # Write header
        f.write("# DQN Training Results Summary\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Group by game
        games = {}
        for run in runs:
            game = run.get("game", "unknown")
            if game not in games:
                games[game] = []
            games[game].append(run)

        # Write table for each game
        for game, game_runs in sorted(games.items()):
            f.write(f"## {game.title()}\n\n")

            # Table header
            f.write(
                "| Run ID | Seed | Mean Return | Std | Frames | Wall Time (hrs) | Commit |\n"
            )
            f.write(
                "|--------|------|-------------|-----|--------|----------------|--------|\n"
            )

            # Table rows
            for run in game_runs:
                run_id = run.get("run_id", "unknown")
                seed = run.get("seed", -1)
                mean_ret = run.get("mean_eval_return", 0.0)
                std_ret = run.get("std_eval_return", 0.0)
                frames = run.get("total_frames", 0)
                wall_time = run.get("wall_time_hours", 0.0)
                commit = run.get("commit_hash", "unknown")[:7]

                f.write(
                    f"| {run_id} | {seed} | {mean_ret:.2f} | {std_ret:.2f} | "
                    f"{frames:,} | {wall_time:.2f} | {commit} |\n"
                )

            f.write("\n")

    print(f"Exported {len(runs)} runs to Markdown: {output_path}")


def upload_to_wandb(
    csv_path: Path,
    markdown_path: Path,
    wandb_project: str,
    wandb_entity: Optional[str] = None,
):
    """
    Upload results tables to W&B as artifacts.

    Args:
        csv_path: Path to CSV file
        markdown_path: Path to Markdown file
        wandb_project: W&B project name
        wandb_entity: W&B entity (optional)
    """
    try:
        import wandb
    except ImportError:
        print("Warning: wandb not installed. Skipping upload.")
        return

    try:
        # Initialize W&B
        run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            job_type="results_summary",
            name=f"results_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

        # Create artifact
        artifact = wandb.Artifact(
            name="results_summary",
            type="results_table",
            description="Training results summary table",
        )

        # Add files
        if csv_path.exists():
            artifact.add_file(str(csv_path))
        if markdown_path.exists():
            artifact.add_file(str(markdown_path))

        # Log artifact
        run.log_artifact(artifact)
        run.finish()

        print(f"Uploaded results to W&B project: {wandb_project}")

    except Exception as e:
        print(f"Warning: Failed to upload to W&B: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Export DQN training results to summary tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all runs from experiments directory
  python scripts/export_results_table.py \\
    --runs-dir experiments/dqn_atari/runs \\
    --output output/summary

  # Export and upload to W&B
  python scripts/export_results_table.py \\
    --runs-dir experiments/dqn_atari/runs \\
    --output output/summary \\
    --upload-wandb \\
    --wandb-project dqn-atari
        """,
    )

    parser.add_argument(
        "--runs-dir",
        type=Path,
        required=True,
        help="Base directory containing run subdirectories",
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Output directory for summary tables"
    )
    parser.add_argument(
        "--upload-wandb", action="store_true", help="Upload tables to W&B as artifacts"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        help="W&B project name (required with --upload-wandb)",
    )
    parser.add_argument("--wandb-entity", type=str, help="W&B entity (optional)")

    args = parser.parse_args()

    # Validate
    if not args.runs_dir.exists():
        parser.error(f"Runs directory does not exist: {args.runs_dir}")

    if args.upload_wandb and not args.wandb_project:
        parser.error("--wandb-project required with --upload-wandb")

    # Scan for runs
    print(f"Scanning for runs in: {args.runs_dir}")
    run_dirs = scan_run_directories(args.runs_dir)
    print(f"Found {len(run_dirs)} run directories")

    if not run_dirs:
        print("No runs found. Exiting.")
        return

    # Load metadata from each run
    runs = []
    for run_dir in run_dirs:
        metadata = load_run_metadata(run_dir)
        if metadata:
            runs.append(metadata)

    print(f"Loaded metadata from {len(runs)} runs")

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Export to CSV
    csv_path = args.output / "results_summary.csv"
    export_to_csv(runs, csv_path)

    # Export to Markdown
    markdown_path = args.output / "results_summary.md"
    export_to_markdown(runs, markdown_path)

    # Upload to W&B if requested
    if args.upload_wandb:
        upload_to_wandb(csv_path, markdown_path, args.wandb_project, args.wandb_entity)

    print("\nDone!")


if __name__ == "__main__":
    main()
