"""Reproducibility utilities: seeding, metadata snapshots, and environment capture."""

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False, env: Any = None) -> None:
    """
    Set random seeds for Python, NumPy, PyTorch, and optionally environment for reproducibility.

    This is the centralized seeding function that should be called:
    - Once at the start of training with the base seed
    - On every environment reset (with episode-specific seed)
    - When resuming from a checkpoint (to restore state)

    Args:
        seed: Random seed value
        deterministic: If True, enables deterministic operations in PyTorch.
                      This may reduce performance but ensures full reproducibility.
                      Note: Some operations still may not be deterministic on GPU.
        env: Optional environment object to seed (will call env.reset(seed=seed) if provided)

    Example:
        >>> # Initial seeding
        >>> set_seed(42, deterministic=True)
        >>>
        >>> # Seed environment on reset
        >>> obs, info = set_seed(42 + episode, env=env)
        >>>
        >>> # With multiprocessing workers
        >>> set_seed(base_seed + worker_id, deterministic=True)

    Note:
        When using multiprocessing, each worker should call set_seed with a unique
        seed derived from the base seed (e.g., base_seed + worker_id).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: This may impact performance but ensures reproducibility

    # Seed environment if provided (returns reset observation)
    if env is not None:
        return env.reset(seed=seed)

    return None


def seed_env(env: Any, seed: int) -> tuple:
    """
    Seed environment and return reset observation.

    Convenience function for seeding environment during episode resets.
    This is equivalent to calling env.reset(seed=seed) but makes the
    intent clearer and pairs with set_seed() for full reproducibility.

    Args:
        env: Environment object with reset() method that accepts seed parameter
        seed: Random seed value for the environment

    Returns:
        Tuple of (observation, info) from env.reset()

    Example:
        >>> # Seed environment for episode 0
        >>> obs, info = seed_env(env, base_seed + 0)
        >>>
        >>> # Seed environment for episode 1
        >>> obs, info = seed_env(env, base_seed + 1)

    Note:
        For full determinism, call set_seed() before the first seed_env() call
        to ensure Python, NumPy, and PyTorch are also seeded.
    """
    return env.reset(seed=seed)


def get_git_info() -> Dict[str, str]:
    """
    Get current git commit hash, branch, and status.

    Returns:
        Dictionary with 'commit', 'branch', and 'dirty' keys
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()

        # Check if there are uncommitted changes
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        dirty = len(status) > 0

        return {
            "commit": commit,
            "branch": branch,
            "dirty": dirty
        }
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {
            "commit": "unknown",
            "branch": "unknown",
            "dirty": False
        }


def save_run_metadata(
    output_dir: Path,
    config: Dict[str, Any],
    seed: int,
    ale_settings: Optional[Dict[str, Any]] = None,
    extra_info: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save run metadata including git info, config, seed, and ALE settings to meta.json.

    Args:
        output_dir: Directory to save meta.json
        config: Merged configuration dictionary
        seed: Random seed used for this run
        ale_settings: ALE environment settings (frameskip, repeat_action_probability, etc.)
        extra_info: Any additional metadata to include

    Example:
        >>> save_run_metadata(
        ...     Path("experiments/dqn_atari/runs/run_001"),
        ...     config={"agent": {"gamma": 0.99}},
        ...     seed=42,
        ...     ale_settings={"frameskip": 4, "repeat_action_probability": 0.0}
        ... )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    git_info = get_git_info()

    metadata = {
        "git": git_info,
        "seed": seed,
        "config": config,
    }

    if ale_settings is not None:
        metadata["ale_settings"] = ale_settings

    if extra_info is not None:
        metadata["extra"] = extra_info

    # Save to meta.json
    meta_path = output_dir / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Run metadata saved to: {meta_path}")

    # Warn if git repo is dirty
    if git_info["dirty"]:
        print("WARNING: Git repository has uncommitted changes. Run may not be fully reproducible.")
