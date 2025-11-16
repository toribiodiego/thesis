#!/usr/bin/env python3
"""
Analyze DQN training results and compare against paper baselines.

This script ingests evaluation CSVs from run directories, computes summary
statistics, and generates comparison tables against DQN 2013 paper scores.

Usage:
    python scripts/analyze_results.py --run-dir experiments/dqn_atari/runs/pong_42_*/
    python scripts/analyze_results.py --run-dir experiments/dqn_atari/runs/ --game pong
    python scripts/analyze_results.py --run-dir experiments/dqn_atari/runs/ --all-games
    python scripts/analyze_results.py --run-dir experiments/dqn_atari/runs/ --output results/summary/

Features:
    - Compute final evaluation statistics (last N evals or final checkpoint)
    - Multi-seed aggregation with mean and standard deviation
    - Paper baseline comparison with percentage calculation
    - Export to CSV and Markdown tables
    - JSON output for programmatic access
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# DQN 2013 Paper Reference Scores (Table 1)
# Source: Mnih et al. "Playing Atari with Deep Reinforcement Learning" (2013)
PAPER_SCORES = {
    "pong": {
        "random": -20.4,
        "sarsa": -19.0,
        "dqn": 20.0,
        "human": -3.0,
    },
    "breakout": {
        "random": 1.2,
        "sarsa": 5.2,
        "dqn": 168.0,
        "human": 31.0,
    },
    "beam_rider": {
        "random": 354.0,
        "sarsa": 929.0,
        "dqn": 4092.0,
        "human": 7456.0,
    },
    "space_invaders": {
        "random": 148.0,
        "sarsa": 250.0,
        "dqn": 581.0,
        "human": 1652.0,
    },
    "seaquest": {
        "random": 110.0,
        "sarsa": 665.0,
        "dqn": 1740.0,
        "human": 20182.0,
    },
    "enduro": {
        "random": 0.0,
        "sarsa": 129.0,
        "dqn": 470.0,
        "human": 368.0,
    },
    "qbert": {
        "random": 157.0,
        "sarsa": 614.0,
        "dqn": 1952.0,
        "human": 13455.0,
    },
}


def find_run_directories(base_path: Path, game: str = None) -> list[Path]:
    """
    Find all run directories matching criteria.

    Args:
        base_path: Base directory to search
        game: Optional game name filter

    Returns:
        List of run directory paths
    """
    run_dirs = []

    if base_path.is_file():
        # Single eval file provided
        return [base_path.parent]

    # Check if this is a run directory (has eval/ subdirectory)
    eval_dir = base_path / "eval"
    if eval_dir.exists():
        run_dirs.append(base_path)
    else:
        # Search for run directories
        for path in base_path.glob("**/eval"):
            if path.is_dir():
                run_dir = path.parent
                # Filter by game if specified
                if game:
                    if game.lower() in run_dir.name.lower():
                        run_dirs.append(run_dir)
                else:
                    run_dirs.append(run_dir)

    return sorted(run_dirs)


def load_evaluations(run_dir: Path) -> pd.DataFrame:
    """
    Load evaluation results from a run directory.

    Args:
        run_dir: Path to run directory

    Returns:
        DataFrame with evaluation results
    """
    eval_csv = run_dir / "eval" / "evaluations.csv"

    if not eval_csv.exists():
        raise FileNotFoundError(f"No evaluations.csv found in {run_dir}")

    df = pd.read_csv(eval_csv)
    df["run_dir"] = str(run_dir)
    df["run_name"] = run_dir.name

    # Extract game and seed from run name (format: game_seed_timestamp)
    parts = run_dir.name.split("_")
    if len(parts) >= 2:
        df["game"] = parts[0]
        try:
            df["seed"] = int(parts[1])
        except ValueError:
            df["seed"] = -1
    else:
        df["game"] = "unknown"
        df["seed"] = -1

    return df


def compute_run_statistics(
    df: pd.DataFrame, last_n_evals: int = 5, use_final_only: bool = False
) -> dict:
    """
    Compute summary statistics for a single run.

    Args:
        df: DataFrame with evaluation results
        last_n_evals: Number of final evaluations to average
        use_final_only: If True, only use the very last evaluation

    Returns:
        Dictionary with computed statistics
    """
    if df.empty:
        return {}

    if use_final_only:
        final_row = df.iloc[-1]
        return {
            "mean_return": final_row["mean_return"],
            "std_return": final_row.get("std_return", 0.0),
            "final_step": final_row["step"],
            "num_evals": 1,
            "game": df["game"].iloc[0],
            "seed": df["seed"].iloc[0],
            "run_name": df["run_name"].iloc[0],
        }

    # Use last N evaluations
    last_evals = df.tail(last_n_evals)

    return {
        "mean_return": last_evals["mean_return"].mean(),
        "std_return": last_evals["mean_return"].std(),
        "min_return": last_evals["mean_return"].min(),
        "max_return": last_evals["mean_return"].max(),
        "final_step": df["step"].max(),
        "num_evals": len(last_evals),
        "game": df["game"].iloc[0],
        "seed": df["seed"].iloc[0],
        "run_name": df["run_name"].iloc[0],
    }


def aggregate_multi_seed(run_stats: list[dict]) -> dict:
    """
    Aggregate statistics across multiple seeds for same game.

    Args:
        run_stats: List of per-run statistics

    Returns:
        Aggregated statistics with cross-seed mean and std
    """
    if not run_stats:
        return {}

    game = run_stats[0]["game"]
    returns = [s["mean_return"] for s in run_stats]
    seeds = [s["seed"] for s in run_stats]

    return {
        "game": game,
        "mean_return": np.mean(returns),
        "std_return": np.std(returns),
        "min_return": np.min(returns),
        "max_return": np.max(returns),
        "num_seeds": len(run_stats),
        "seeds": seeds,
        "final_steps": [s["final_step"] for s in run_stats],
    }


def compare_to_paper(stats: dict) -> dict:
    """
    Compare results against DQN 2013 paper scores.

    Args:
        stats: Statistics dictionary with 'game' and 'mean_return'

    Returns:
        Comparison metrics including percentage of paper score
    """
    game = stats["game"].lower()

    if game not in PAPER_SCORES:
        return {
            "paper_score": None,
            "percentage": None,
            "status": "unknown_game",
        }

    paper = PAPER_SCORES[game]
    our_score = stats["mean_return"]
    paper_score = paper["dqn"]
    random_score = paper["random"]

    # Calculate percentage of paper DQN score
    # For games where random baseline is negative (like Pong),
    # we need to account for the range
    if paper_score == random_score:
        percentage = 100.0
    else:
        # Normalize: 0% = random, 100% = paper DQN
        percentage = ((our_score - random_score) / (paper_score - random_score)) * 100

    # Determine status
    if percentage >= 100:
        status = "MATCH/EXCEED"
    elif percentage >= 80:
        status = "CLOSE"
    elif percentage >= 50:
        status = "PARTIAL"
    elif percentage > 0:
        status = "LEARNING"
    else:
        status = "RANDOM"

    return {
        "paper_score": paper_score,
        "random_score": random_score,
        "human_score": paper["human"],
        "percentage": percentage,
        "status": status,
    }


def generate_summary_table(results: list[dict], output_format: str = "markdown") -> str:
    """
    Generate a summary table of results.

    Args:
        results: List of result dictionaries
        output_format: 'markdown' or 'csv'

    Returns:
        Formatted table string
    """
    if not results:
        return "No results to display."

    # Build table data
    rows = []
    for r in results:
        row = {
            "Game": r.get("game", "unknown").capitalize(),
            "Score": f"{r.get('mean_return', 0):.2f}",
            "Std": f"{r.get('std_return', 0):.2f}",
            "Seeds": r.get("num_seeds", 1),
            "Paper": r.get("paper_score", "N/A"),
            "% of Paper": f"{r.get('percentage', 0):.1f}%",
            "Status": r.get("status", "unknown"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if output_format == "csv":
        return df.to_csv(index=False)
    else:
        return df.to_markdown(index=False)


def save_results(
    results: list[dict], output_dir: Path, prefix: str = "analysis"
) -> None:
    """
    Save results to multiple formats.

    Args:
        results: List of result dictionaries
        output_dir: Directory to save outputs
        prefix: Filename prefix
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save JSON (full data)
    json_path = output_dir / f"{prefix}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved JSON: {json_path}")

    # Save CSV table
    csv_path = output_dir / f"{prefix}.csv"
    csv_content = generate_summary_table(results, "csv")
    with open(csv_path, "w") as f:
        f.write(csv_content)
    print(f"Saved CSV: {csv_path}")

    # Save Markdown table
    md_path = output_dir / f"{prefix}.md"
    md_content = "# DQN Results Analysis\n\n"
    md_content += f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    md_content += "## Summary Table\n\n"
    md_content += generate_summary_table(results, "markdown")
    md_content += "\n\n## Interpretation\n\n"
    md_content += "- **MATCH/EXCEED**: Score >= 100% of paper DQN score\n"
    md_content += "- **CLOSE**: Score >= 80% of paper DQN score\n"
    md_content += "- **PARTIAL**: Score >= 50% of paper DQN score\n"
    md_content += "- **LEARNING**: Score > random baseline\n"
    md_content += "- **RANDOM**: Score at or below random baseline\n"
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Saved Markdown: {md_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze DQN training results and compare to paper baselines"
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path to run directory or parent directory containing runs",
    )
    parser.add_argument(
        "--game", type=str, default=None, help="Filter runs by game name"
    )
    parser.add_argument(
        "--all-games",
        action="store_true",
        help="Analyze all games found in run directory",
    )
    parser.add_argument(
        "--last-n-evals",
        type=int,
        default=5,
        help="Number of final evaluations to average (default: 5)",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Use only the final evaluation (not average)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for results (default: print to stdout)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed information"
    )

    args = parser.parse_args()

    # Find run directories
    run_dirs = find_run_directories(args.run_dir, args.game)

    if not run_dirs:
        print(f"No run directories found in {args.run_dir}")
        sys.exit(1)

    if args.verbose:
        print(f"Found {len(run_dirs)} run directories:")
        for d in run_dirs:
            print(f"  - {d.name}")
        print()

    # Load and analyze each run
    all_stats = []
    for run_dir in run_dirs:
        try:
            df = load_evaluations(run_dir)
            stats = compute_run_statistics(df, args.last_n_evals, args.final_only)
            all_stats.append(stats)

            if args.verbose:
                print(f"Run: {stats['run_name']}")
                print(f"  Game: {stats['game']}, Seed: {stats['seed']}")
                print(f"  Mean Return: {stats['mean_return']:.2f}")
                print(f"  Final Step: {stats['final_step']}")
                print()
        except Exception as e:
            print(f"Warning: Failed to load {run_dir}: {e}")

    if not all_stats:
        print("No valid runs found.")
        sys.exit(1)

    # Group by game and aggregate
    games = {}
    for stats in all_stats:
        game = stats["game"]
        if game not in games:
            games[game] = []
        games[game].append(stats)

    # Compute final results
    results = []
    for game, game_stats in sorted(games.items()):
        if len(game_stats) == 1:
            # Single run
            result = game_stats[0].copy()
        else:
            # Multi-seed aggregation
            result = aggregate_multi_seed(game_stats)

        # Compare to paper
        comparison = compare_to_paper(result)
        result.update(comparison)

        results.append(result)

    # Output results
    if args.output:
        save_results(results, args.output)
    else:
        # Print to stdout
        print("=" * 70)
        print("DQN Results Analysis")
        print("=" * 70)
        print()
        print(generate_summary_table(results, "markdown"))
        print()

        # Print detailed comparison
        for r in results:
            game = r.get("game", "unknown").capitalize()
            score = r.get("mean_return", 0)
            paper = r.get("paper_score", "N/A")
            pct = r.get("percentage", 0)
            status = r.get("status", "unknown")

            print(f"{game}:")
            print(f"  Our Score: {score:.2f}")
            print(f"  Paper DQN: {paper}")
            print(f"  Percentage: {pct:.1f}%")
            print(f"  Status: {status}")
            print()


if __name__ == "__main__":
    main()
