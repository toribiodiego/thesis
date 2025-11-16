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
- Metadata embedding (smoothing, commit hash)
- W&B artifact uploads
"""

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Set publication-quality defaults
mpl.rcParams["figure.dpi"] = 100
mpl.rcParams["savefig.dpi"] = 300
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.labelsize"] = 11
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["xtick.labelsize"] = 9
mpl.rcParams["ytick.labelsize"] = 9
mpl.rcParams["legend.fontsize"] = 9
mpl.rcParams["figure.titlesize"] = 13


# ============================================================================
# Utility Functions
# ============================================================================


def get_git_commit_hash() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def save_plot_metadata(output_dir: Path, game_name: str, metadata: Dict) -> Path:
    """
    Save plot generation metadata to JSON file.

    Args:
        output_dir: Directory containing plots
        game_name: Name of game
        metadata: Metadata dict to save

    Returns:
        Path to saved metadata file
    """
    metadata_path = output_dir / f"{game_name}_plot_metadata.json"

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata_path


def upload_plots_to_wandb(
    wandb_project: str,
    wandb_run_id: str,
    plot_files: List[Path],
    metadata_file: Optional[Path] = None,
    artifact_name: Optional[str] = None,
):
    """
    Upload plot files and metadata to W&B as artifact.

    Args:
        wandb_project: W&B project name
        wandb_run_id: W&B run ID
        plot_files: List of plot file paths
        metadata_file: Optional metadata JSON file
        artifact_name: Optional custom artifact name
    """
    try:
        import wandb
    except ImportError:
        print("Warning: wandb not installed. Skipping artifact upload.")
        return

    # Initialize API
    api = wandb.Api()

    # Get run
    run_path = f"{wandb_project}/{wandb_run_id}"
    try:
        run = api.run(run_path)
    except Exception as e:
        print(f"Warning: Could not access W&B run {run_path}: {e}")
        return

    # Create artifact name
    if artifact_name is None:
        artifact_name = f"plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create artifact
    artifact = wandb.Artifact(
        name=artifact_name,
        type="plots",
        metadata={"generated_at": datetime.now().isoformat()},
    )

    # Add plot files
    for plot_file in plot_files:
        if plot_file.exists():
            artifact.add_file(str(plot_file))

    # Add metadata file
    if metadata_file and metadata_file.exists():
        artifact.add_file(str(metadata_file))

    # Log artifact
    try:
        run.log_artifact(artifact)
        print(f"Uploaded {len(plot_files)} plots to W&B artifact: {artifact_name}")
    except Exception as e:
        print(f"Warning: Failed to upload artifact: {e}")


# ============================================================================
# Data Loading
# ============================================================================


def downsample_data(
    data: Dict[str, np.ndarray], max_points: int = 10000
) -> Dict[str, np.ndarray]:
    """
    Downsample data to reduce memory usage and plotting time.

    Uses uniform sampling to preserve data distribution while
    reducing number of points.

    Args:
        data: Dictionary of column name -> array
        max_points: Maximum number of points to keep

    Returns:
        Downsampled data dictionary
    """
    # Check if downsampling needed
    num_points = len(next(iter(data.values())))
    if num_points <= max_points:
        return data

    # Compute sampling indices
    indices = np.linspace(0, num_points - 1, max_points, dtype=int)

    # Downsample all arrays
    downsampled = {}
    for key, arr in data.items():
        downsampled[key] = arr[indices]

    print(f"Downsampled {num_points} points to {max_points} points")
    return downsampled


def load_csv_data(csv_path: Path, warn_size_mb: float = 50.0) -> Dict[str, np.ndarray]:
    """
    Load data from CSV file with optional size warning.

    Args:
        csv_path: Path to CSV file
        warn_size_mb: Warn if file exceeds this size (MB)

    Returns:
        dict: Column name -> numpy array
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Check file size and warn if large
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    if file_size_mb > warn_size_mb:
        print(
            f"Warning: Large CSV file ({file_size_mb:.1f} MB). "
            f"Consider using --downsample for faster plotting."
        )

    data = {}
    with open(csv_path, "r") as f:
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
                if val == "" or val == "None":
                    values.append(np.nan)
                else:
                    try:
                        values.append(float(val))
                    except ValueError:
                        values.append(np.nan)
            data[key] = np.array(values)

    return data


def download_wandb_artifact(
    project: str, run_id: str, artifact_name: str, output_dir: Path
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
        raise ImportError("wandb not installed. Install with: pip install wandb")

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
    x: np.ndarray, y: np.ndarray, window: int = 100, method: str = "moving_average"
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

    if method == "moving_average":
        # Simple moving average
        smoothed = np.convolve(y, np.ones(window) / window, mode="valid")
        # Adjust x to match smoothed length
        smoothed_x = x[window - 1 :]
        return smoothed_x, smoothed

    elif method == "exponential":
        # Exponential moving average
        alpha = 2.0 / (window + 1)
        smoothed = np.zeros_like(y)
        smoothed[0] = y[0]
        for i in range(1, len(y)):
            smoothed[i] = alpha * y[i] + (1 - alpha) * smoothed[i - 1]
        return x, smoothed

    else:
        raise ValueError(f"Unknown smoothing method: {method}")


# ============================================================================
# Video Overlay Functions
# ============================================================================


def extract_video_frame(video_path: Path, frame_index: int = 0) -> Optional[np.ndarray]:
    """
    Extract a single frame from a video file.

    Args:
        video_path: Path to video file (MP4, etc.)
        frame_index: Which frame to extract (0 = first frame)

    Returns:
        Frame as numpy array (RGB) or None if extraction fails
    """
    try:
        import cv2
    except ImportError:
        print("Warning: cv2 not installed. Cannot extract video frames.")
        return None

    if not video_path.exists():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None

    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb


def load_video_thumbnails(
    video_dir: Path, steps: List[int], thumbnail_size: Tuple[int, int] = (160, 120)
) -> Dict[int, np.ndarray]:
    """
    Load video thumbnails for specific training steps.

    Args:
        video_dir: Directory containing evaluation videos
        steps: List of training steps to load videos for
        thumbnail_size: Size to resize thumbnails (width, height)

    Returns:
        Dictionary mapping step -> thumbnail array
    """
    try:
        import cv2
    except ImportError:
        print("Warning: cv2 not installed. Cannot load video thumbnails.")
        return {}

    thumbnails = {}

    # Look for videos matching step numbers
    for step in steps:
        # Try common naming patterns
        patterns = [
            f"eval_{step}.mp4",
            f"eval_step_{step}.mp4",
            f"evaluation_{step}.mp4",
            f"*{step}*.mp4",
        ]

        video_found = None
        for pattern in patterns:
            matches = list(video_dir.glob(pattern))
            if matches:
                video_found = matches[0]
                break

        if video_found:
            frame = extract_video_frame(video_found, frame_index=0)
            if frame is not None:
                # Resize to thumbnail size
                thumbnail = cv2.resize(
                    frame, thumbnail_size, interpolation=cv2.INTER_AREA
                )
                thumbnails[step] = thumbnail

    return thumbnails


def plot_evaluation_with_video_overlay(
    eval_data: Dict[str, np.ndarray],
    video_dir: Path,
    output_path: Path,
    title: str = "Evaluation Score with Video Snapshots",
    formats: List[str] = ["png"],
    max_thumbnails: int = 6,
    thumbnail_size: Tuple[int, int] = (160, 120),
):
    """
    Plot evaluation scores with video frame thumbnails overlaid.

    Creates a learning curve with video snapshots showing agent behavior
    at key evaluation points.

    Args:
        eval_data: Dict with 'step' and 'mean_return' arrays
        video_dir: Directory containing evaluation videos
        output_path: Base path for output files
        title: Plot title
        formats: Output formats
        max_thumbnails: Maximum number of video thumbnails to overlay
        thumbnail_size: Size of thumbnails (width, height)
    """
    x = eval_data["step"]
    y = eval_data["mean_return"]

    # Remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        print("Warning: No valid evaluation data for video overlay plot")
        return

    # Select steps for video thumbnails (evenly spaced)
    if len(x) <= max_thumbnails:
        selected_indices = list(range(len(x)))
    else:
        selected_indices = np.linspace(0, len(x) - 1, max_thumbnails, dtype=int).tolist()

    selected_steps = [int(x[i]) for i in selected_indices]

    # Load video thumbnails
    print(f"Loading video thumbnails from: {video_dir}")
    thumbnails = load_video_thumbnails(video_dir, selected_steps, thumbnail_size)

    if not thumbnails:
        print("Warning: No video thumbnails found. Falling back to standard plot.")
        plot_evaluation_scores(eval_data, output_path, title, formats)
        return

    print(f"Found {len(thumbnails)} video thumbnails")

    # Create figure with extra space for thumbnails
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot evaluation scores
    ax.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        markersize=6,
        color="#2ca02c",
        label="Mean Eval Return",
        zorder=2,
    )

    # Add error bars if std is available
    if "std_return" in eval_data:
        y_std = eval_data["std_return"][mask]
        ax.fill_between(
            x, y - y_std, y + y_std, alpha=0.2, color="#2ca02c", label="std", zorder=1
        )

    # Add video thumbnails
    y_range = y.max() - y.min()
    y_offset_step = y_range * 0.15  # Offset between thumbnails vertically

    for i, (idx, step) in enumerate(zip(selected_indices, selected_steps)):
        if step in thumbnails:
            thumbnail = thumbnails[step]

            # Calculate position for thumbnail
            data_x = x[idx]
            data_y = y[idx]

            # Alternate thumbnails above and below the curve
            if i % 2 == 0:
                y_offset = y_offset_step * 1.5
            else:
                y_offset = -y_offset_step * 2.5

            # Create thumbnail image
            imagebox = OffsetImage(thumbnail, zoom=0.4)
            imagebox.image.axes = ax

            # Add annotation box
            ab = AnnotationBbox(
                imagebox,
                (data_x, data_y),
                xybox=(0, y_offset * 100),  # Offset in points
                xycoords="data",
                boxcoords="offset points",
                pad=0.3,
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1", color="gray"),
                bboxprops=dict(
                    boxstyle="round,pad=0.1", facecolor="white", edgecolor="gray", alpha=0.9
                ),
            )
            ax.add_artist(ab)

            # Add step label below thumbnail
            ax.annotate(
                f"Step {step:,}",
                xy=(data_x, data_y),
                xytext=(0, y_offset * 100 - 50),
                xycoords="data",
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=7,
                color="gray",
            )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Evaluation Return")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3, zorder=0)

    # Adjust y-axis limits to accommodate thumbnails
    current_ylim = ax.get_ylim()
    y_margin = y_range * 0.5
    ax.set_ylim(current_ylim[0] - y_margin, current_ylim[1] + y_margin)

    # Save in all requested formats
    for fmt in formats:
        save_path = output_path.with_suffix(f".{fmt}")
        fig.savefig(save_path, bbox_inches="tight", format=fmt, dpi=150)
        print(f"Saved: {save_path}")

    plt.close(fig)


# ============================================================================
# Plotting Functions
# ============================================================================


def plot_episode_returns(
    data: Dict[str, np.ndarray],
    output_path: Path,
    smoothing_window: int = 100,
    title: str = "Episode Returns vs Training Steps",
    formats: List[str] = ["png"],
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

    x = data["step"]
    y = data["return"]

    # Remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        print("Warning: No valid data for episode returns")
        return

    # Plot raw data with transparency
    ax.plot(x, y, alpha=0.3, linewidth=0.5, color="#1f77b4", label="Raw")

    # Plot smoothed curve
    if len(y) >= smoothing_window:
        x_smooth, y_smooth = smooth_curve(x, y, window=smoothing_window)
        ax.plot(
            x_smooth,
            y_smooth,
            linewidth=2,
            color="#1f77b4",
            label=f"Smoothed (window={smoothing_window})",
        )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save in all requested formats
    for fmt in formats:
        save_path = output_path.with_suffix(f".{fmt}")
        fig.savefig(save_path, bbox_inches="tight", format=fmt)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_training_loss(
    data: Dict[str, np.ndarray],
    output_path: Path,
    smoothing_window: int = 100,
    title: str = "Training Loss vs Steps",
    formats: List[str] = ["png"],
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

    x = data["step"]
    y = data["loss"]

    # Remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        print("Warning: No valid data for training loss")
        return

    # Plot raw data
    ax.plot(x, y, alpha=0.3, linewidth=0.5, color="#ff7f0e", label="Raw")

    # Plot smoothed curve
    if len(y) >= smoothing_window:
        x_smooth, y_smooth = smooth_curve(x, y, window=smoothing_window)
        ax.plot(
            x_smooth,
            y_smooth,
            linewidth=2,
            color="#ff7f0e",
            label=f"Smoothed (window={smoothing_window})",
        )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save in all requested formats
    for fmt in formats:
        save_path = output_path.with_suffix(f".{fmt}")
        fig.savefig(save_path, bbox_inches="tight", format=fmt)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_evaluation_scores(
    data: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Evaluation Score vs Training Steps",
    formats: List[str] = ["png"],
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

    x = data["step"]
    y = data["mean_return"]

    # Remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        print("Warning: No valid data for evaluation scores")
        return

    # Plot evaluation scores with markers
    ax.plot(
        x,
        y,
        marker="o",
        linewidth=2,
        markersize=5,
        color="#2ca02c",
        label="Mean Eval Return",
    )

    # Add error bars if std is available
    if "std_return" in data:
        y_std = data["std_return"][mask]
        ax.fill_between(
            x, y - y_std, y + y_std, alpha=0.2, color="#2ca02c", label="±1 Std Dev"
        )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Evaluation Return")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save in all requested formats
    for fmt in formats:
        save_path = output_path.with_suffix(f".{fmt}")
        fig.savefig(save_path, bbox_inches="tight", format=fmt)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_epsilon_schedule(
    data: Dict[str, np.ndarray],
    output_path: Path,
    title: str = "Epsilon Schedule",
    formats: List[str] = ["png"],
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

    x = data["step"]
    y = data["epsilon"]

    # Remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    if len(x) == 0:
        print("Warning: No valid data for epsilon schedule")
        return

    # Plot epsilon schedule
    ax.plot(x, y, linewidth=2, color="#d62728", label="Epsilon (ε)")

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Epsilon (ε)")
    ax.set_title(title)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save in all requested formats
    for fmt in formats:
        save_path = output_path.with_suffix(f".{fmt}")
        fig.savefig(save_path, bbox_inches="tight", format=fmt)
        print(f"Saved: {save_path}")

    plt.close(fig)


def plot_all_metrics(
    episodes_data: Optional[Dict[str, np.ndarray]],
    steps_data: Optional[Dict[str, np.ndarray]],
    eval_data: Optional[Dict[str, np.ndarray]],
    output_dir: Path,
    game_name: str = "game",
    smoothing_window: int = 100,
    formats: List[str] = ["png"],
    save_metadata: bool = True,
    upload_to_wandb: bool = False,
    wandb_project: Optional[str] = None,
    wandb_run_id: Optional[str] = None,
) -> Tuple[List[Path], Optional[Path]]:
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
        save_metadata: Whether to save plot metadata JSON
        upload_to_wandb: Whether to upload plots to W&B
        wandb_project: W&B project (required if upload_to_wandb=True)
        wandb_run_id: W&B run ID (required if upload_to_wandb=True)

    Returns:
        Tuple of (plot_files, metadata_file)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_files = []

    # Plot episode returns
    if episodes_data is not None and "return" in episodes_data:
        plot_episode_returns(
            episodes_data,
            output_dir / f"{game_name}_episode_returns",
            smoothing_window=smoothing_window,
            title=f"{game_name.title()} - Episode Returns",
            formats=formats,
        )
        for fmt in formats:
            plot_files.append(output_dir / f"{game_name}_episode_returns.{fmt}")

    # Plot training loss
    if steps_data is not None and "loss" in steps_data:
        plot_training_loss(
            steps_data,
            output_dir / f"{game_name}_training_loss",
            smoothing_window=smoothing_window,
            title=f"{game_name.title()} - Training Loss",
            formats=formats,
        )
        for fmt in formats:
            plot_files.append(output_dir / f"{game_name}_training_loss.{fmt}")

    # Plot evaluation scores
    if eval_data is not None and "mean_return" in eval_data:
        plot_evaluation_scores(
            eval_data,
            output_dir / f"{game_name}_evaluation_scores",
            title=f"{game_name.title()} - Evaluation Scores",
            formats=formats,
        )
        for fmt in formats:
            plot_files.append(output_dir / f"{game_name}_evaluation_scores.{fmt}")

    # Plot epsilon schedule
    if steps_data is not None and "epsilon" in steps_data:
        plot_epsilon_schedule(
            steps_data,
            output_dir / f"{game_name}_epsilon_schedule",
            title=f"{game_name.title()} - Epsilon Schedule",
            formats=formats,
        )
        for fmt in formats:
            plot_files.append(output_dir / f"{game_name}_epsilon_schedule.{fmt}")

    # Save metadata
    metadata_file = None
    if save_metadata:
        metadata = {
            "game_name": game_name,
            "smoothing_window": smoothing_window,
            "commit_hash": get_git_commit_hash(),
            "generated_at": datetime.now().isoformat(),
            "formats": formats,
            "plots_generated": [p.name for p in plot_files if p.exists()],
        }
        metadata_file = save_plot_metadata(output_dir, game_name, metadata)
        print(f"Saved metadata: {metadata_file}")

    # Upload to W&B if requested
    if upload_to_wandb:
        if wandb_project and wandb_run_id:
            upload_plots_to_wandb(
                wandb_project,
                wandb_run_id,
                plot_files,
                metadata_file,
                artifact_name=f"{game_name}_plots",
            )
        else:
            print("Warning: W&B upload requested but project/run ID not provided")

    return plot_files, metadata_file


# ============================================================================
# Multi-Seed Aggregation
# ============================================================================


def aggregate_multi_seed_data(
    csv_files: List[Path], align_column: str = "step"
) -> Dict[str, np.ndarray]:
    """
    Aggregate data from multiple seed runs.

    Args:
        csv_files: List of CSV file paths from different seeds
        align_column: Column to align data on (default: 'step')

    Returns:
        dict: Aggregated data with mean, std, min, max, and individual runs
    """
    # Load all CSV files
    all_data = []
    for csv_file in csv_files:
        try:
            data = load_csv_data(csv_file)
            all_data.append(data)
        except Exception as e:
            print(f"Warning: Failed to load {csv_file}: {e}")
            continue

    if not all_data:
        raise ValueError("No valid CSV files loaded for aggregation")

    # Get all unique steps across all runs
    all_steps = set()
    for data in all_data:
        if align_column in data:
            all_steps.update(data[align_column])

    all_steps = np.array(sorted(all_steps))

    # Get column names (excluding align column)
    column_names = set()
    for data in all_data:
        column_names.update(data.keys())
    column_names.discard(align_column)

    # Aggregate each column
    aggregated = {align_column: all_steps}

    for col in column_names:
        # Interpolate each run to common step grid
        interpolated_runs = []
        for data in all_data:
            if col in data and align_column in data:
                # Remove NaN values for interpolation
                mask = ~np.isnan(data[col])
                if mask.sum() > 0:
                    interp_values = np.interp(
                        all_steps,
                        data[align_column][mask],
                        data[col][mask],
                        left=np.nan,
                        right=np.nan,
                    )
                    interpolated_runs.append(interp_values)

        if interpolated_runs:
            # Stack runs
            runs_array = np.stack(interpolated_runs, axis=0)

            # Compute statistics
            aggregated[f"{col}_mean"] = np.nanmean(runs_array, axis=0)
            aggregated[f"{col}_std"] = np.nanstd(runs_array, axis=0)
            aggregated[f"{col}_min"] = np.nanmin(runs_array, axis=0)
            aggregated[f"{col}_max"] = np.nanmax(runs_array, axis=0)

            # 95% confidence interval (assumes normal distribution)
            n_runs = runs_array.shape[0]
            sem = aggregated[f"{col}_std"] / np.sqrt(n_runs)
            aggregated[f"{col}_ci95"] = 1.96 * sem

            # Store individual runs for plotting
            for i, run in enumerate(interpolated_runs):
                aggregated[f"{col}_run{i}"] = run

    return aggregated


def plot_multi_seed_aggregation(
    aggregated_data: Dict[str, np.ndarray],
    output_path: Path,
    metric_name: str,
    title: str = "Multi-Seed Aggregation",
    formats: List[str] = ["png"],
    plot_individual_runs: bool = False,
):
    """
    Plot multi-seed aggregated data with confidence intervals.

    Args:
        aggregated_data: Aggregated data from aggregate_multi_seed_data()
        output_path: Base path for output files
        metric_name: Name of metric to plot (e.g., 'return', 'loss')
        title: Plot title
        formats: Output formats
        plot_individual_runs: Whether to plot individual seed runs
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x = aggregated_data["step"]
    y_mean = aggregated_data[f"{metric_name}_mean"]
    y_ci95 = aggregated_data[f"{metric_name}_ci95"]

    # Plot individual runs if requested
    if plot_individual_runs:
        run_idx = 0
        while f"{metric_name}_run{run_idx}" in aggregated_data:
            y_run = aggregated_data[f"{metric_name}_run{run_idx}"]
            ax.plot(x, y_run, alpha=0.2, linewidth=0.5, color="gray")
            run_idx += 1

    # Plot mean
    ax.plot(x, y_mean, linewidth=2, color="#1f77b4", label="Mean")

    # Plot 95% CI shading
    ax.fill_between(
        x, y_mean - y_ci95, y_mean + y_ci95, alpha=0.3, color="#1f77b4", label="95% CI"
    )

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Save in all requested formats
    for fmt in formats:
        save_path = output_path.with_suffix(f".{fmt}")
        fig.savefig(save_path, bbox_inches="tight", format=fmt)
        print(f"Saved: {save_path}")

    plt.close(fig)


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

  # Plot with video overlay (shows agent behavior at evaluation points)
  python scripts/plot_results.py \\
    --eval runs/pong_123/eval/evaluations.csv \\
    --video-dir runs/pong_123/videos \\
    --game-name pong \\
    --output plots/pong
        """,
    )

    # Input sources
    input_group = parser.add_argument_group("Input Sources")
    input_group.add_argument("--episodes", type=Path, help="Path to episodes CSV file")
    input_group.add_argument(
        "--steps", type=Path, help="Path to training steps CSV file"
    )
    input_group.add_argument("--eval", type=Path, help="Path to evaluation CSV file")
    input_group.add_argument(
        "--multi-seed",
        nargs="+",
        type=Path,
        help="Paths to episode CSV files from multiple seeds for aggregation",
    )
    input_group.add_argument("--wandb-project", type=str, help="W&B project name")
    input_group.add_argument("--wandb-run", type=str, help="W&B run ID")
    input_group.add_argument(
        "--wandb-artifact",
        type=str,
        help="W&B artifact name (e.g., training_logs_step_1000000:latest)",
    )

    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument(
        "--output", type=Path, required=True, help="Output directory for plots"
    )
    output_group.add_argument(
        "--formats",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png"],
        help="Output formats (default: png)",
    )
    output_group.add_argument(
        "--game-name",
        type=str,
        default="game",
        help="Game name for plot titles (default: game)",
    )
    output_group.add_argument(
        "--no-metadata", action="store_true", help="Disable saving plot metadata JSON"
    )
    output_group.add_argument(
        "--upload-wandb", action="store_true", help="Upload plots to W&B as artifacts"
    )
    output_group.add_argument(
        "--wandb-upload-run",
        type=str,
        help="W&B run ID to upload plots to (required with --upload-wandb)",
    )

    # Plot options
    plot_group = parser.add_argument_group("Plot Options")
    plot_group.add_argument(
        "--smoothing",
        type=int,
        default=100,
        help="Smoothing window size (default: 100)",
    )
    plot_group.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable smoothing (plot raw data only)",
    )
    plot_group.add_argument(
        "--video-dir",
        type=Path,
        help="Directory containing evaluation videos for overlay plot",
    )
    plot_group.add_argument(
        "--max-thumbnails",
        type=int,
        default=6,
        help="Maximum number of video thumbnails in overlay (default: 6)",
    )

    # Performance options
    perf_group = parser.add_argument_group("Performance Options")
    perf_group.add_argument(
        "--downsample",
        type=int,
        metavar="MAX_POINTS",
        help="Downsample data to MAX_POINTS for faster plotting (e.g., 10000)",
    )
    perf_group.add_argument(
        "--warn-size-mb",
        type=float,
        default=50.0,
        help="Warn if CSV file exceeds this size in MB (default: 50.0)",
    )

    args = parser.parse_args()

    # Validate inputs
    use_wandb = args.wandb_project and args.wandb_run and args.wandb_artifact
    use_local = args.episodes or args.steps or args.eval
    use_multi_seed = args.multi_seed is not None

    if not use_wandb and not use_local and not use_multi_seed:
        parser.error(
            "Must specify either local CSV files (--episodes/--steps/--eval), "
            "multi-seed files (--multi-seed), "
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
            args.output / "wandb_artifacts",
        )

        # Try to load CSV files from artifact
        episodes_csv = artifact_dir / "episodes.csv"
        steps_csv = artifact_dir / "training_steps.csv"

        if episodes_csv.exists():
            episodes_data = load_csv_data(episodes_csv, warn_size_mb=args.warn_size_mb)
            print(f"Loaded episodes data: {episodes_csv}")

        if steps_csv.exists():
            steps_data = load_csv_data(steps_csv, warn_size_mb=args.warn_size_mb)
            print(f"Loaded steps data: {steps_csv}")

    else:
        # Load from local CSV files
        if args.episodes:
            episodes_data = load_csv_data(args.episodes, warn_size_mb=args.warn_size_mb)
            print(f"Loaded episodes data: {args.episodes}")

        if args.steps:
            steps_data = load_csv_data(args.steps, warn_size_mb=args.warn_size_mb)
            print(f"Loaded steps data: {args.steps}")

        if args.eval:
            eval_data = load_csv_data(args.eval, warn_size_mb=args.warn_size_mb)
            print(f"Loaded evaluation data: {args.eval}")

    # Apply downsampling if requested
    if args.downsample:
        if episodes_data:
            episodes_data = downsample_data(episodes_data, max_points=args.downsample)
        if steps_data:
            steps_data = downsample_data(steps_data, max_points=args.downsample)
        if eval_data:
            eval_data = downsample_data(eval_data, max_points=args.downsample)

    # Set smoothing window
    smoothing_window = 1 if args.no_smoothing else args.smoothing

    # Handle multi-seed aggregation
    if use_multi_seed:
        print(f"\nAggregating {len(args.multi_seed)} seed runs...")
        aggregated = aggregate_multi_seed_data(args.multi_seed, align_column="step")

        # Save aggregated data to CSV
        agg_csv_path = args.output / f"{args.game_name}_aggregated.csv"
        args.output.mkdir(parents=True, exist_ok=True)

        with open(agg_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=aggregated.keys())
            writer.writeheader()
            for i in range(len(aggregated["step"])):
                row = {k: v[i] if i < len(v) else np.nan for k, v in aggregated.items()}
                writer.writerow(row)
        print(f"Saved aggregated data: {agg_csv_path}")

        # Plot multi-seed aggregation for episode returns
        if "return_mean" in aggregated:
            print("Plotting multi-seed episode returns...")
            plot_multi_seed_aggregation(
                aggregated,
                args.output / f"{args.game_name}_multi_seed_returns",
                metric_name="return",
                title=f"{args.game_name.title()} - Multi-Seed Episode Returns (n={len(args.multi_seed)})",
                formats=args.formats,
                plot_individual_runs=True,
            )

        print("\nDone! Multi-seed aggregation complete")
        print(f"Plots saved to: {args.output}")
        return

    # Generate plots
    print("\nGenerating plots...")
    plot_files, metadata_file = plot_all_metrics(
        episodes_data=episodes_data,
        steps_data=steps_data,
        eval_data=eval_data,
        output_dir=args.output,
        game_name=args.game_name,
        smoothing_window=smoothing_window,
        formats=args.formats,
        save_metadata=not args.no_metadata,
        upload_to_wandb=args.upload_wandb,
        wandb_project=args.wandb_project if args.upload_wandb else None,
        wandb_run_id=args.wandb_upload_run if args.upload_wandb else None,
    )

    # Generate video overlay plot if requested
    if args.video_dir and eval_data is not None:
        print("\nGenerating video overlay plot...")
        if args.video_dir.exists():
            plot_evaluation_with_video_overlay(
                eval_data,
                args.video_dir,
                args.output / f"{args.game_name}_eval_with_videos",
                title=f"{args.game_name.title()} - Evaluation with Video Snapshots",
                formats=args.formats,
                max_thumbnails=args.max_thumbnails,
            )
            for fmt in args.formats:
                plot_files.append(args.output / f"{args.game_name}_eval_with_videos.{fmt}")
        else:
            print(f"Warning: Video directory not found: {args.video_dir}")

    print(f"\nDone! Generated {len([p for p in plot_files if p.exists()])} plots")
    print(f"Plots saved to: {args.output}")


if __name__ == "__main__":
    main()
