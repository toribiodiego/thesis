#!/usr/bin/env python
"""
Summarize a DQN training run.

Reads CSV and JSONL artifacts from a run directory and prints a summary including:
- Mean/best evaluation returns
- Best checkpoint information
- Training progress statistics
- Artifact URLs (if W&B enabled)

Usage:
    python scripts/summarize_run.py experiments/dqn_atari/runs/pong_42_20251116_123456/
    python scripts/summarize_run.py experiments/dqn_atari/runs/pong_42_20251116_123456/ --json
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_csv(path: Path) -> List[Dict[str, Any]]:
    """Load CSV file into list of dictionaries."""
    if not path.exists():
        return []

    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            converted = {}
            for key, value in row.items():
                try:
                    if "." in value:
                        converted[key] = float(value)
                    else:
                        converted[key] = int(value)
                except (ValueError, TypeError):
                    converted[key] = value
            rows.append(converted)
    return rows


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dictionaries."""
    if not path.exists():
        return []

    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    if not path.exists():
        return {}

    with open(path, "r") as f:
        return json.load(f)


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    """Generate summary of a training run."""
    summary = {
        "run_dir": str(run_dir),
        "exists": run_dir.exists(),
    }

    if not run_dir.exists():
        summary["error"] = f"Run directory not found: {run_dir}"
        return summary

    # Load metadata
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = load_json(meta_path)
        summary["metadata"] = {
            "env_id": meta.get("config", {}).get("environment", {}).get("env_id", "unknown"),
            "seed": meta.get("seed", "unknown"),
            "total_frames": meta.get("config", {}).get("training", {}).get("total_frames", "unknown"),
            "start_time": meta.get("start_time", "unknown"),
        }

    # Load training progress
    step_csv = run_dir / "csv" / "training_steps.csv"
    if step_csv.exists():
        steps = load_csv(step_csv)
        if steps:
            last_step = steps[-1]
            summary["training_progress"] = {
                "total_steps_logged": len(steps),
                "last_frame": last_step.get("step", 0),
                "last_epsilon": last_step.get("epsilon", 0),
                "last_loss": last_step.get("loss", None),
                "last_fps": last_step.get("fps", 0),
            }

    # Load episode data
    episode_csv = run_dir / "csv" / "episodes.csv"
    if episode_csv.exists():
        episodes = load_csv(episode_csv)
        if episodes:
            returns = [ep.get("episode_return", 0) for ep in episodes]
            lengths = [ep.get("episode_length", 0) for ep in episodes]
            summary["episode_stats"] = {
                "total_episodes": len(episodes),
                "mean_return": sum(returns) / len(returns) if returns else 0,
                "max_return": max(returns) if returns else 0,
                "min_return": min(returns) if returns else 0,
                "mean_length": sum(lengths) / len(lengths) if lengths else 0,
            }

    # Load evaluation results
    eval_csv = run_dir / "eval" / "evaluations.csv"
    if eval_csv.exists():
        evals = load_csv(eval_csv)
        if evals:
            # Find best evaluation
            best_eval = max(evals, key=lambda x: x.get("mean_return", float("-inf")))
            last_eval = evals[-1]

            summary["evaluation"] = {
                "total_evaluations": len(evals),
                "best_eval": {
                    "step": best_eval.get("step", 0),
                    "mean_return": best_eval.get("mean_return", 0),
                    "std_return": best_eval.get("std_return", 0),
                },
                "last_eval": {
                    "step": last_eval.get("step", 0),
                    "mean_return": last_eval.get("mean_return", 0),
                    "std_return": last_eval.get("std_return", 0),
                },
            }

    # Load per-episode evaluation data
    per_episode_jsonl = run_dir / "eval" / "per_episode_returns.jsonl"
    if per_episode_jsonl.exists():
        per_ep_data = load_jsonl(per_episode_jsonl)
        if per_ep_data:
            summary["per_episode_data"] = {
                "evaluations_with_details": len(per_ep_data),
            }

    # Check for checkpoints
    checkpoint_dir = run_dir / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        best_checkpoint = checkpoint_dir / "best_model.pt"

        summary["checkpoints"] = {
            "count": len(checkpoints),
            "has_best": best_checkpoint.exists(),
        }

        if checkpoints:
            # Extract step numbers from filenames
            steps = []
            for cp in checkpoints:
                try:
                    step = int(cp.stem.split("_")[-1])
                    steps.append(step)
                except ValueError:
                    pass
            if steps:
                summary["checkpoints"]["latest_step"] = max(steps)
                summary["checkpoints"]["checkpoint_files"] = sorted(steps)

    # Check for videos
    video_dir = run_dir / "videos"
    if video_dir.exists():
        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.gif"))
        summary["videos"] = {
            "count": len(videos),
            "files": [v.name for v in sorted(videos)],
        }

    # Check for W&B artifacts
    wandb_dir = run_dir / "wandb"
    if wandb_dir.exists():
        summary["wandb"] = {
            "enabled": True,
            "local_dir": str(wandb_dir),
        }

    return summary


def print_summary(summary: Dict[str, Any], as_json: bool = False):
    """Print summary to stdout."""
    if as_json:
        print(json.dumps(summary, indent=2, default=str))
        return

    print("=" * 60)
    print("DQN Training Run Summary")
    print("=" * 60)

    print(f"Run Directory: {summary['run_dir']}")

    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    # Metadata
    if "metadata" in summary:
        meta = summary["metadata"]
        print(f"\nEnvironment: {meta.get('env_id', 'unknown')}")
        print(f"Seed: {meta.get('seed', 'unknown')}")
        print(f"Total Frames Target: {meta.get('total_frames', 'unknown'):,}")
        print(f"Start Time: {meta.get('start_time', 'unknown')}")

    # Training progress
    if "training_progress" in summary:
        prog = summary["training_progress"]
        print(f"\nTraining Progress:")
        print(f"  Last Frame: {prog.get('last_frame', 0):,}")
        print(f"  Last Epsilon: {prog.get('last_epsilon', 0):.4f}")
        if prog.get("last_loss") is not None:
            print(f"  Last Loss: {prog.get('last_loss'):.6f}")
        print(f"  Last FPS: {prog.get('last_fps', 0):.1f}")
        print(f"  Steps Logged: {prog.get('total_steps_logged', 0):,}")

    # Episode statistics
    if "episode_stats" in summary:
        eps = summary["episode_stats"]
        print(f"\nEpisode Statistics:")
        print(f"  Total Episodes: {eps.get('total_episodes', 0):,}")
        print(f"  Mean Return: {eps.get('mean_return', 0):.2f}")
        print(f"  Max Return: {eps.get('max_return', 0):.2f}")
        print(f"  Min Return: {eps.get('min_return', 0):.2f}")
        print(f"  Mean Length: {eps.get('mean_length', 0):.1f}")

    # Evaluation results
    if "evaluation" in summary:
        ev = summary["evaluation"]
        print(f"\nEvaluation Results:")
        print(f"  Total Evaluations: {ev.get('total_evaluations', 0)}")
        if "best_eval" in ev:
            best = ev["best_eval"]
            print(f"  Best Evaluation:")
            print(f"    Step: {best.get('step', 0):,}")
            print(f"    Mean Return: {best.get('mean_return', 0):.2f} +/- {best.get('std_return', 0):.2f}")
        if "last_eval" in ev:
            last = ev["last_eval"]
            print(f"  Last Evaluation:")
            print(f"    Step: {last.get('step', 0):,}")
            print(f"    Mean Return: {last.get('mean_return', 0):.2f} +/- {last.get('std_return', 0):.2f}")

    # Checkpoints
    if "checkpoints" in summary:
        cp = summary["checkpoints"]
        print(f"\nCheckpoints:")
        print(f"  Count: {cp.get('count', 0)}")
        print(f"  Has Best Model: {cp.get('has_best', False)}")
        if "latest_step" in cp:
            print(f"  Latest Step: {cp.get('latest_step', 0):,}")

    # Videos
    if "videos" in summary:
        vid = summary["videos"]
        print(f"\nVideos:")
        print(f"  Count: {vid.get('count', 0)}")
        if vid.get("files"):
            for f in vid["files"][:5]:  # Show first 5
                print(f"    - {f}")
            if len(vid["files"]) > 5:
                print(f"    ... and {len(vid['files']) - 5} more")

    # W&B
    if "wandb" in summary:
        print(f"\nWeights & Biases:")
        print(f"  Enabled: {summary['wandb'].get('enabled', False)}")
        print(f"  Local Dir: {summary['wandb'].get('local_dir', 'N/A')}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize a DQN training run",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/summarize_run.py experiments/dqn_atari/runs/pong_42_20251116_123456/
  python scripts/summarize_run.py experiments/dqn_atari/runs/pong_42_20251116_123456/ --json
        """,
    )
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    summary = summarize_run(args.run_dir)
    print_summary(summary, as_json=args.json)

    # Exit with error code if run not found
    if "error" in summary:
        sys.exit(1)


if __name__ == "__main__":
    main()
