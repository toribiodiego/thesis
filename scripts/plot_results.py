#!/usr/bin/env python3
"""
Plotting script for DQN training results.

Generates publication-quality figures from training logs:
- Reward vs frames (episode returns over training)
- Loss vs updates (training loss progression)
- Evaluation score vs frames (periodic eval performance)
- Epsilon schedule (exploration rate decay)

Supports:
- Local CSV input
- W&B artifact downloads
- Multiple output formats (PNG, PDF, SVG)
- Configurable smoothing and styling
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set publication-quality defaults
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9
mpl.rcParams['figure.titlesize'] = 13


# ============================================================================
# Data Loading
# ============================================================================

def load_csv_data(csv_path: Path) -> Dict[str, np.ndarray]:
    """
    Load data from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        dict: Column name -> numpy array
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    data = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)

        # Read all rows
        rows = list(reader)

        if not rows:
            raise ValueError(f"CSV file is empty: {csv_path}")

        # Convert to arrays
        for key in rows[0].keys():
            values = []
            for row in rows:
                val = row[key]
                # Handle None/empty values
                if val == '' or val == 'None':
                    values.append(np.nan)
                else:
                    try:
                        values.append(float(val))
                    except ValueError:
                        values.append(np.nan)
            data[key] = np.array(values)

    return data


def download_wandb_artifact(
    project: str,
    run_id: str,
    artifact_name: str,
    output_dir: Path
) -> Path:
    """
    Download W&B artifact to local directory.

    Args:
        project: W&B project name
        run_id: W&B run ID
        artifact_name: Name of artifact to download
        output_dir: Local directory to save artifact

    Returns:
        Path to downloaded artifact directory
    """
    try:
        import wandb
    except ImportError:
        raise ImportError(
            "wandb not installed. Install with: pip install wandb"
        )

    # Initialize W&B API
    api = wandb.Api()

    # Download artifact
    artifact_path = f"{project}/{artifact_name}"
    artifact = api.artifact(artifact_path)

    download_path = artifact.download(root=str(output_dir))

    return Path(download_path)


# ============================================================================
# Smoothing Functions
# ============================================================================

def smooth_curve(
    x: np.ndarray,
    y: np.ndarray,
    window: int = 100,
    method: str = 'moving_average'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Smooth a curve using specified method.

    Args:
        x: X-axis values
        y: Y-axis values
        window: Smoothing window size
        method: Smoothing method ('moving_average', 'exponential')

    Returns:
        Tuple of (smoothed_x, smoothed_y)
    """
    if len(y) < window:
        return x, y

    if method == 'moving_average':
        # Simple moving average
        smoothed = np.convolve(y, np.ones(window)/window, mode='valid')
        # Adjust x to match smoothed length
        smoothed_x = x[window-1:]
        return smoothed_x, smoothed

    elif method == 'exponential':
        # Exponential moving average
        alpha = 2.0 / (window + 1)
        smoothed = np.zeros_like(y)
        smoothed[0] = y[0]
        for i in range(1, len(y)):
            smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i-1]
        return x, smoothed

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


# ============================================================================
# Plotting Functions
# ============================================================================

def plot_episode_returns(
    data: Dict[str, np.ndarray],
    output_path: Path,
    smoothing_window: int = 100,
    title: str = "Episode Returns vs Training Steps",
    formats: List[str] = ['png']
):
    """
    Plot episode returns over training.

    Args:
        data: Dict with 'step' and 'return' arrays
        output_path: Base path for output files (without extension)
        smoothing_window: Window size for smoothing
        title: Plot title
        formats: List of output formats ('png', 'pdf', 'svg')
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = data['step']
    y = data['return']

    # Remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        print("Warning: No valid data for episode returns")
        return

    # Plot raw data with transparency
    ax.plot(x, y, alpha=0.3, linewidth=0.5, color='#1f77b4', label='Raw')

    # Plot smoothed curve
    if len(y) >= smoothing_window:
        x_smooth, y_smooth = smooth_curve(x, y, window=smoothing_window)
        ax.plot(x_smooth, y_smooth, linewidth=2, color='#1f77b4',
                label=f'Smoothed (window={smoothing_window})')

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save in all requested formats
    for fmt in formats:
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, bbox_inches='tight', format=fmt)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_training_loss(
    data: Dict[str, np.ndarray],
    output_path: Path,
    smoothing_window: int = 100,
    title: str = "Training Loss vs Steps",
    formats: List[str] = ['png']
):
    """
    Plot training loss over steps.

    Args:
        data: Dict with 'step' and 'loss' arrays
        output_path: Base path for output files
        smoothing_window: Window size for smoothing
        title: Plot title
        formats: List of output formats
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = data['step']
    y = data['loss']

    # Remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        print("Warning: No valid data for training loss")
        return

    # Plot raw data
    ax.plot(x, y, alpha=0.3, linewidth=0.5, color='#ff7f0e', label='Raw')

    # Plot smoothed curve
    if len(y) >= smoothing_window:
        x_smooth, y_smooth = smooth_curve(x, y, window=smoothing_window)
        ax.plot(x_smooth, y_smooth, linewidth=2, color='#ff7f0e',
                label=f'Smoothed (window={smoothing_window})')

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save in all requested formats
    for fmt in formats:
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, bbox_inches='tight', format=fmt)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_evaluation_scores(
    data: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Evaluation Score vs Training Steps",
    formats: List[str] = ['png']
):
    """
    Plot evaluation scores over training.

    Args:
        data: Dict with 'step' and 'mean_return' arrays
        output_path: Base path for output files
        title: Plot title
        formats: List of output formats
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = data['step']
    y = data['mean_return']

    # Remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        print("Warning: No valid data for evaluation scores")
        return

    # Plot evaluation scores with markers
    ax.plot(x, y, marker='o', linewidth=2, markersize=5,
            color='#2ca02c', label='Mean Eval Return')

    # Add error bars if std is available
    if 'std_return' in data:
        y_std = data['std_return'][mask]
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.2, color='#2ca02c',
                        label='±1 Std Dev')

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Evaluation Return')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save in all requested formats
    for fmt in formats:
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, bbox_inches='tight', format=fmt)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_epsilon_schedule(
    data: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Epsilon Schedule",
    formats: List[str] = ['png']
):
    """
    Plot epsilon (exploration rate) over training.

    Args:
        data: Dict with 'step' and 'epsilon' arrays
        output_path: Base path for output files
        title: Plot title
        formats: List of output formats
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = data['step']
    y = data['epsilon']

    # Remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        print("Warning: No valid data for epsilon schedule")
        return

    # Plot epsilon schedule
    ax.plot(x, y, linewidth=2, color='#d62728', label='Epsilon (ε)')

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Epsilon (ε)')
    ax.set_title(title)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save in all requested formats
    for fmt in formats:
        save_path = output_path.with_suffix(f'.{fmt}')
        fig.savefig(save_path, bbox_inches='tight', format=fmt)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_all_metrics(
    episodes_data: Optional[Dict[str, np.ndarray]],
    steps_data: Optional[Dict[str, np.ndarray]],
    eval_data: Optional[Dict[str, np.ndarray]],
    output_dir: Path,
    game_name: str = "game",
    smoothing_window: int = 100,
    formats: List[str] = ['png']
):
    """
    Generate all available plots from training data.

    Args:
        episodes_data: Episodes CSV data (return, length, etc.)
        steps_data: Training steps CSV data (loss, epsilon, etc.)
        eval_data: Evaluation CSV data (mean_return, std_return, etc.)
        output_dir: Directory to save plots
        game_name: Name of game for plot titles
        smoothing_window: Smoothing window size
        formats: Output formats
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot episode returns
    if episodes_data is not None and 'return' in episodes_data:
        plot_episode_returns(
            episodes_data,
            output_dir / f"{game_name}_episode_returns",
            smoothing_window=smoothing_window,
            title=f"{game_name.title()} - Episode Returns",
            formats=formats
        )

    # Plot training loss
    if steps_data is not None and 'loss' in steps_data:
        plot_training_loss(
            steps_data,
            output_dir / f"{game_name}_training_loss",
            smoothing_window=smoothing_window,
            title=f"{game_name.title()} - Training Loss",
            formats=formats
        )

    # Plot evaluation scores
    if eval_data is not None and 'mean_return' in eval_data:
        plot_evaluation_scores(
            eval_data,
            output_dir / f"{game_name}_evaluation_scores",
            title=f"{game_name.title()} - Evaluation Scores",
            formats=formats
        )

    # Plot epsilon schedule
    if steps_data is not None and 'epsilon' in steps_data:
        plot_epsilon_schedule(
            steps_data,
            output_dir / f"{game_name}_epsilon_schedule",
            title=f"{game_name.title()} - Epsilon Schedule",
            formats=formats
        )


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from DQN training logs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot from local CSV files
  python scripts/plot_results.py \\
    --episodes runs/pong_123/logs/csv/episodes.csv \\
    --steps runs/pong_123/logs/csv/training_steps.csv \\
    --output plots/pong

  # Plot with custom smoothing and formats
  python scripts/plot_results.py \\
    --episodes runs/pong_123/logs/csv/episodes.csv \\
    --smoothing 200 \\
    --formats png pdf svg \\
    --output plots/pong

  # Plot from W&B artifact
  python scripts/plot_results.py \\
    --wandb-project dqn-atari \\
    --wandb-run abc123 \\
    --wandb-artifact training_logs_step_1000000:latest \\
    --output plots/pong
        """
    )

    # Input sources
    input_group = parser.add_argument_group('Input Sources')
    input_group.add_argument(
        '--episodes',
        type=Path,
        help='Path to episodes CSV file'
    )
    input_group.add_argument(
        '--steps',
        type=Path,
        help='Path to training steps CSV file'
    )
    input_group.add_argument(
        '--eval',
        type=Path,
        help='Path to evaluation CSV file'
    )
    input_group.add_argument(
        '--wandb-project',
        type=str,
        help='W&B project name'
    )
    input_group.add_argument(
        '--wandb-run',
        type=str,
        help='W&B run ID'
    )
    input_group.add_argument(
        '--wandb-artifact',
        type=str,
        help='W&B artifact name (e.g., training_logs_step_1000000:latest)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output directory for plots'
    )
    output_group.add_argument(
        '--formats',
        nargs='+',
        choices=['png', 'pdf', 'svg'],
        default=['png'],
        help='Output formats (default: png)'
    )
    output_group.add_argument(
        '--game-name',
        type=str,
        default='game',
        help='Game name for plot titles (default: game)'
    )

    # Plot options
    plot_group = parser.add_argument_group('Plot Options')
    plot_group.add_argument(
        '--smoothing',
        type=int,
        default=100,
        help='Smoothing window size (default: 100)'
    )
    plot_group.add_argument(
        '--no-smoothing',
        action='store_true',
        help='Disable smoothing (plot raw data only)'
    )

    args = parser.parse_args()

    # Validate inputs
    use_wandb = args.wandb_project and args.wandb_run and args.wandb_artifact
    use_local = args.episodes or args.steps or args.eval

    if not use_wandb and not use_local:
        parser.error(
            "Must specify either local CSV files (--episodes/--steps/--eval) "
            "or W&B artifact (--wandb-project/--wandb-run/--wandb-artifact)"
        )

    # Load data
    episodes_data = None
    steps_data = None
    eval_data = None

    if use_wandb:
        print(f"Downloading W&B artifact: {args.wandb_artifact}")
        artifact_dir = download_wandb_artifact(
            args.wandb_project,
            args.wandb_run,
            args.wandb_artifact,
            args.output / 'wandb_artifacts'
        )

        # Try to load CSV files from artifact
        episodes_csv = artifact_dir / 'episodes.csv'
        steps_csv = artifact_dir / 'training_steps.csv'

        if episodes_csv.exists():
            episodes_data = load_csv_data(episodes_csv)
            print(f"Loaded episodes data: {episodes_csv}")

        if steps_csv.exists():
            steps_data = load_csv_data(steps_csv)
            print(f"Loaded steps data: {steps_csv}")

    else:
        # Load from local CSV files
        if args.episodes:
            episodes_data = load_csv_data(args.episodes)
            print(f"Loaded episodes data: {args.episodes}")

        if args.steps:
            steps_data = load_csv_data(args.steps)
            print(f"Loaded steps data: {args.steps}")

        if args.eval:
            eval_data = load_csv_data(args.eval)
            print(f"Loaded evaluation data: {args.eval}")

    # Set smoothing window
    smoothing_window = 1 if args.no_smoothing else args.smoothing

    # Generate plots
    print(f"\nGenerating plots...")
    plot_all_metrics(
        episodes_data=episodes_data,
        steps_data=steps_data,
        eval_data=eval_data,
        output_dir=args.output,
        game_name=args.game_name,
        smoothing_window=smoothing_window,
        formats=args.formats
    )

    print(f"\nDone! Plots saved to: {args.output}")


if __name__ == '__main__':
    main()
