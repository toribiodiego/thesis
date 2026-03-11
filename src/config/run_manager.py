"""Run directory management and metadata persistence.

Handles creation of run directories with consistent structure and
automatic saving of configuration snapshots and metadata.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def create_run_dir(
    base_dir: str,
    experiment_name: str,
    seed: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> Path:
    """
    Create run directory with consistent naming.

    Creates directory structure:
        {base_dir}/{experiment_name}_{seed}_{timestamp}/

    Args:
        base_dir: Base directory for runs (e.g., "experiments/dqn_atari/runs")
        experiment_name: Name of experiment (e.g., "pong")
        seed: Random seed (optional, will use "noseed" if not provided)
        timestamp: Timestamp string (optional, will generate if not provided)

    Returns:
        Path object for created run directory

    Example:
        >>> run_dir = create_run_dir("runs", "pong", seed=42)
        >>> # Creates: runs/pong_42_20250113_143022/
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate seed string
    seed_str = str(seed) if seed is not None else "noseed"

    # Create run directory name
    run_name = f"{experiment_name}_{seed_str}_{timestamp}"
    run_dir = Path(base_dir) / run_name

    # Create directory
    run_dir.mkdir(parents=True, exist_ok=True)

    return run_dir


def create_run_subdirs(run_dir: Path) -> Dict[str, Path]:
    """
    Create standard subdirectories within run directory.

    Creates:
        - logs/          : Training logs (CSV, TensorBoard)
        - checkpoints/   : Model checkpoints
        - artifacts/     : Debug artifacts (plots, videos, etc.)
        - eval/          : Evaluation results
        - videos/        : Video recordings from evaluation

    Args:
        run_dir: Run directory path

    Returns:
        Dictionary mapping subdirectory names to Path objects

    Example:
        >>> subdirs = create_run_subdirs(run_dir)
        >>> checkpoint_dir = subdirs['checkpoints']
    """
    subdirs = {
        "logs": run_dir / "logs",
        "checkpoints": run_dir / "checkpoints",
        "artifacts": run_dir / "artifacts",
        "eval": run_dir / "eval",
        "videos": run_dir / "videos",
    }

    for subdir in subdirs.values():
        subdir.mkdir(parents=True, exist_ok=True)

    return subdirs


def save_config_snapshot(
    config: Dict[str, Any], run_dir: Path, format: str = "yaml"
) -> Path:
    """
    Save merged configuration snapshot to run directory.

    Saves the fully resolved configuration (after all merges and overrides)
    to ensure exact reproducibility.

    Args:
        config: Fully merged configuration dictionary
        run_dir: Run directory path
        format: Output format ('yaml' or 'json')

    Returns:
        Path to saved config file

    Example:
        >>> config_path = save_config_snapshot(config, run_dir)
    """
    if format == "yaml":
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    elif format == "json":
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'yaml' or 'json'.")

    return config_path


def get_git_info() -> Dict[str, Any]:
    """
    Get git repository information.

    Returns:
        Dictionary with commit hash, branch, dirty status
    """
    import subprocess

    git_info = {
        "commit_hash": "unknown",
        "commit_hash_full": "unknown",
        "branch": "unknown",
        "dirty": False,
        "available": False,
    }

    try:
        # Get short commit hash
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_info["commit_hash"] = result.stdout.strip()

        # Get full commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_info["commit_hash_full"] = result.stdout.strip()

        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_info["branch"] = result.stdout.strip()

        # Check if working tree is dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        git_info["dirty"] = len(result.stdout.strip()) > 0
        git_info["available"] = True

    except (
        subprocess.CalledProcessError,
        FileNotFoundError,
        subprocess.TimeoutExpired,
    ):
        pass

    return git_info


def save_metadata(
    config: Dict[str, Any], run_dir: Path, extra: Optional[Dict[str, Any]] = None
) -> Path:
    """
    Save run metadata to meta.json.

    Saves:
        - Timestamp
        - Git commit hash and status
        - Random seed
        - Python version
        - Key configuration parameters
        - Additional metadata

    Args:
        config: Configuration dictionary
        run_dir: Run directory path
        extra: Additional metadata to include

    Returns:
        Path to saved metadata file

    Example:
        >>> meta_path = save_metadata(config, run_dir, extra={'device': 'cuda'})
    """
    import sys

    # Build metadata dictionary
    metadata = {
        "created_at": datetime.now().isoformat(),
        "python_version": sys.version.split()[0],
        "git": get_git_info(),
        "seed": config.get("seed", {}).get("value"),
        "experiment": {
            "name": config.get("experiment", {}).get("name"),
            "notes": config.get("experiment", {}).get("notes", ""),
        },
        "environment": {"env_id": config.get("environment", {}).get("env_id")},
        "training": {
            "total_frames": config.get("training", {}).get("total_frames"),
            "optimizer_lr": config.get("training", {}).get("optimizer", {}).get("lr"),
        },
    }

    # Add PyTorch version if available
    try:
        import torch

        metadata["pytorch_version"] = torch.__version__
    except ImportError:
        metadata["pytorch_version"] = "not_installed"

    # Add CLI arguments if available
    if "cli" in config:
        metadata["cli"] = config["cli"]

    # Add extra metadata
    if extra is not None:
        metadata["extra"] = extra

    # Save to meta.json
    meta_path = run_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return meta_path


def setup_run_directory(
    config: Dict[str, Any],
    timestamp: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """
    Complete run directory setup: create dirs, save config, save metadata.

    This is the main entry point for setting up a new training run.

    Args:
        config: Fully resolved configuration dictionary
        timestamp: Optional timestamp (will generate if not provided)
        extra_metadata: Additional metadata to save

    Returns:
        Dictionary with paths:
            - 'run_dir': Main run directory
            - 'logs': Logs subdirectory
            - 'checkpoints': Checkpoints subdirectory
            - 'artifacts': Artifacts subdirectory
            - 'eval': Evaluation subdirectory
            - 'config_file': Saved config file
            - 'meta_file': Saved metadata file

    Example:
        >>> from src.config import load_config, setup_from_args
        >>> config = setup_from_args()
        >>> paths = setup_run_directory(config)
        >>> print(f"Training run: {paths['run_dir']}")
        >>> print(f"Checkpoints: {paths['checkpoints']}")
    """
    # Get run configuration
    base_dir = config.get("logging", {}).get("base_dir", "experiments/dqn_atari/runs")
    experiment_name = config.get("experiment", {}).get("name", "experiment")
    seed = config.get("seed", {}).get("value")

    # Append _aug suffix when augmentation is enabled but not already
    # reflected in the experiment name (SPR-only and Both configs already
    # have _spr / _both suffixes in their YAML experiment.name)
    aug_enabled = config.get("augmentation", {}).get("enabled", False)
    if aug_enabled and "_aug" not in experiment_name and "_both" not in experiment_name:
        experiment_name = f"{experiment_name}_aug"

    # Create run directory
    run_dir = create_run_dir(base_dir, experiment_name, seed, timestamp)

    # Create subdirectories
    subdirs = create_run_subdirs(run_dir)

    # Save config snapshot
    config_file = save_config_snapshot(config, run_dir, format="yaml")

    # Save metadata
    meta_file = save_metadata(config, run_dir, extra=extra_metadata)

    # Compile all paths
    paths = {
        "run_dir": run_dir,
        **subdirs,
        "config_file": config_file,
        "meta_file": meta_file,
    }

    return paths


def print_run_info(paths: Dict[str, Path]) -> None:
    """
    Print information about created run directory.

    Args:
        paths: Dictionary of paths from setup_run_directory()
    """
    print("\n" + "=" * 80)
    print("Run Directory Created".center(80))
    print("=" * 80)
    print(f"  Location:     {paths['run_dir']}")
    print(f"  Config:       {paths['config_file'].name}")
    print(f"  Metadata:     {paths['meta_file'].name}")
    print(f"  Logs:         {paths['logs'].relative_to(paths['run_dir'])}/")
    print(f"  Checkpoints:  {paths['checkpoints'].relative_to(paths['run_dir'])}/")
    print("=" * 80)
    print()
