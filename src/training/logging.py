"""Structured logging for training steps, episodes, and checkpoints."""

import csv
import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .metrics import UpdateMetrics


class StepLogger:
    """
    Logger for per-step training metrics.

    Logs training metrics at regular intervals including loss, TD error,
    gradient norms, epsilon, learning rate, and replay buffer statistics.

    Parameters
    ----------
    log_dir : str
        Directory to save log files
    log_interval : int
        Steps between log writes (default: 1000)
    moving_avg_window : int
        Window size for moving average of loss (default: 100)

    Usage
    -----
    >>> logger = StepLogger(log_dir='runs/pong_123/logs', log_interval=1000)
    >>> logger.log_step(step=1000, epsilon=0.95, metrics=metrics, replay_size=50000)
    """

    def __init__(
        self, log_dir: str, log_interval: int = 1000, moving_avg_window: int = 100
    ):

        self.log_dir = log_dir
        self.log_interval = log_interval
        self.moving_avg_window = moving_avg_window

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # CSV file for step metrics
        self.csv_path = os.path.join(log_dir, "training_steps.csv")
        self._csv_initialized = False

        # Moving average tracking
        self.loss_history = []

    def log_step(
        self,
        step: int,
        epsilon: float,
        metrics: Optional["UpdateMetrics"] = None,
        replay_size: int = 0,
        fps: float = 0.0,
    ):
        """
        Log metrics for a training step.

        Args:
            step: Current environment step
            epsilon: Current exploration rate
            metrics: UpdateMetrics from training update (if training occurred)
            replay_size: Current replay buffer size
            fps: Frames per second
        """
        # Only log at intervals
        if step % self.log_interval != 0:
            return

        # Prepare log entry
        log_entry = {
            "step": step,
            "epsilon": epsilon,
            "replay_size": replay_size,
            "fps": fps,
        }

        # Add training metrics if available
        if metrics is not None:
            log_entry["loss"] = metrics.loss
            log_entry["td_error"] = metrics.td_error
            log_entry["td_error_std"] = metrics.td_error_std
            log_entry["grad_norm"] = metrics.grad_norm
            log_entry["learning_rate"] = metrics.learning_rate
            log_entry["update_count"] = metrics.update_count

            # Update moving average
            self.loss_history.append(metrics.loss)
            if len(self.loss_history) > self.moving_avg_window:
                self.loss_history.pop(0)
            log_entry["loss_ma"] = sum(self.loss_history) / len(self.loss_history)
        else:
            # No training this step
            log_entry["loss"] = None
            log_entry["td_error"] = None
            log_entry["td_error_std"] = None
            log_entry["grad_norm"] = None
            log_entry["learning_rate"] = None
            log_entry["update_count"] = None
            log_entry["loss_ma"] = None

        # Write to CSV
        self._write_csv(log_entry)

    def _write_csv(self, log_entry: dict):
        """Write log entry to CSV file."""

        # Initialize CSV with header
        if not self._csv_initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
            self._csv_initialized = True

        # Append log entry
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)


class EpisodeLogger:
    """
    Logger for per-episode statistics.

    Tracks episode returns, lengths, and computes rolling averages
    for monitoring training progress.

    Parameters
    ----------
    log_dir : str
        Directory to save log files
    rolling_window : int
        Window size for rolling average (default: 100 episodes)

    Usage
    -----
    >>> logger = EpisodeLogger(log_dir='runs/pong_123/logs', rolling_window=100)
    >>> logger.log_episode(step=5000, episode_return=21.0, episode_length=1200, fps=120.5)
    """

    def __init__(self, log_dir: str, rolling_window: int = 100):

        self.log_dir = log_dir
        self.rolling_window = rolling_window

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # CSV file for episode metrics
        self.csv_path = os.path.join(log_dir, "episodes.csv")
        self._csv_initialized = False

        # Episode tracking
        self.episode_count = 0
        self.episode_returns = []
        self.episode_lengths = []

    def log_episode(
        self,
        step: int,
        episode_return: float,
        episode_length: int,
        fps: float = 0.0,
        epsilon: float = None,
    ):
        """
        Log metrics for a completed episode.

        Args:
            step: Environment step when episode ended
            episode_return: Total undiscounted return
            episode_length: Number of steps in episode
            fps: Frames per second
            epsilon: Current exploration rate
        """
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)

        # Compute rolling statistics
        recent_returns = self.episode_returns[-self.rolling_window :]
        recent_lengths = self.episode_lengths[-self.rolling_window :]

        import numpy as np

        rolling_mean_return = np.mean(recent_returns)
        rolling_std_return = np.std(recent_returns)
        rolling_mean_length = np.mean(recent_lengths)

        # Prepare log entry
        log_entry = {
            "episode": self.episode_count,
            "step": step,
            "return": episode_return,
            "length": episode_length,
            "fps": fps,
            "rolling_mean_return": rolling_mean_return,
            "rolling_std_return": rolling_std_return,
            "rolling_mean_length": rolling_mean_length,
            "num_episodes_in_window": len(recent_returns),
        }

        if epsilon is not None:
            log_entry["epsilon"] = epsilon

        # Write to CSV
        self._write_csv(log_entry)

    def _write_csv(self, log_entry: dict):
        """Write log entry to CSV file."""

        # Initialize CSV with header
        if not self._csv_initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
            self._csv_initialized = True

        # Append log entry
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)

    def get_recent_stats(self, n: int = None):
        """
        Get statistics over last n episodes.

        Args:
            n: Number of recent episodes (default: rolling_window)

        Returns:
            dict: Statistics including mean, std, min, max
        """
        if n is None:
            n = self.rolling_window

        recent_returns = self.episode_returns[-n:]
        recent_lengths = self.episode_lengths[-n:]

        import numpy as np

        return {
            "mean_return": np.mean(recent_returns),
            "std_return": np.std(recent_returns),
            "min_return": np.min(recent_returns),
            "max_return": np.max(recent_returns),
            "mean_length": np.mean(recent_lengths),
            "num_episodes": len(recent_returns),
        }


# ============================================================================
# Checkpoint Management
# ============================================================================


class CheckpointManager:
    """
    Manages complete training state checkpoints with atomic saves.

    Saves complete training state including:
    - Online and target Q-network weights
    - Optimizer state
    - Step and episode counters
    - Epsilon value
    - Replay buffer write index and content (optional)
    - RNG states (torch, numpy, random, environment)
    - Metadata (schema version, timestamp, commit hash)

    Supports periodic saves, best-model tracking, and atomic writes.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to save checkpoints
    save_interval : int
        Steps between periodic checkpoints (default: 1,000,000)
    keep_last_n : int
        Number of periodic checkpoints to keep (default: 3, 0 = keep all)
    save_best : bool
        Whether to save best model based on eval score (default: True)
    save_replay_buffer : bool
        Whether to save replay buffer state (default: False, saves space)

    Usage
    -----
    >>> manager = CheckpointManager(checkpoint_dir='runs/pong_123/checkpoints')
    >>> state = {
    ...     'online_model': online_model,
    ...     'target_model': target_model,
    ...     'optimizer': optimizer,
    ...     'step': 1000000,
    ...     'episode': 5000,
    ...     'epsilon': 0.5,
    ...     'replay_buffer': replay_buffer,
    ...     'rng_states': get_rng_states(env)
    ... }
    >>> manager.save_checkpoint(**state)
    """

    # Checkpoint schema version for compatibility tracking
    SCHEMA_VERSION = "1.0.0"

    def __init__(
        self,
        checkpoint_dir: str,
        save_interval: int = 1_000_000,
        keep_last_n: int = 3,
        save_best: bool = True,
        save_replay_buffer: bool = False,
    ):

        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.save_best_enabled = save_best
        self.save_replay_buffer = save_replay_buffer

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Track periodic checkpoints
        self.periodic_checkpoints = []

        # Track best model
        self.best_eval_return = float("-inf")
        self.best_checkpoint_path = None

    def should_save(self, step: int) -> bool:
        """Check if periodic checkpoint should be saved at this step."""
        return step > 0 and step % self.save_interval == 0

    def save_checkpoint(
        self,
        step: int,
        episode: int,
        epsilon: float,
        online_model: nn.Module,
        target_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: Any = None,
        rng_states: Dict = None,
        extra_metadata: Dict = None,
    ) -> str:
        """
        Save complete checkpoint with all training state.

        Performs atomic write (tmp file -> rename) to prevent corruption.

        Args:
            step: Current environment step counter
            episode: Current episode counter
            epsilon: Current exploration rate
            online_model: Online Q-network
            target_model: Target Q-network
            optimizer: Optimizer state
            replay_buffer: Replay buffer object (optional, can be large)
            rng_states: Dict with RNG states from get_rng_states()
            extra_metadata: Additional metadata to include

        Returns:
            str: Path to saved checkpoint
        """
        import tempfile
        from datetime import datetime

        from ..training.metadata import get_git_commit_hash

        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}.pt")

        # Prepare checkpoint data
        checkpoint = {
            # Schema and metadata
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": datetime.now().isoformat(),
            "commit_hash": get_git_commit_hash(),
            # Training state
            "step": step,
            "episode": episode,
            "epsilon": epsilon,
            # Model weights
            "online_model_state_dict": online_model.state_dict(),
            "target_model_state_dict": target_model.state_dict(),
            # Optimizer state
            "optimizer_state_dict": optimizer.state_dict(),
            # RNG states (if provided)
            "rng_states": rng_states if rng_states is not None else {},
        }

        # Add replay buffer state (write index and size, optionally full content)
        if replay_buffer is not None:
            checkpoint["replay_buffer_state"] = {
                "index": replay_buffer.index,
                "size": replay_buffer.size,
                "capacity": replay_buffer.capacity,
                "obs_shape": replay_buffer.obs_shape,
            }

            # Optionally save full buffer content (large!)
            if self.save_replay_buffer:
                checkpoint["replay_buffer_state"]["data"] = {
                    "observations": replay_buffer.observations,
                    "actions": replay_buffer.actions,
                    "rewards": replay_buffer.rewards,
                    "dones": replay_buffer.dones,
                    "episode_starts": replay_buffer.episode_starts,
                }

        # Add extra metadata
        if extra_metadata is not None:
            checkpoint["metadata"] = extra_metadata

        # Atomic write: save to temp file, then rename
        # This prevents corruption if process is killed during write
        temp_fd, temp_path = tempfile.mkstemp(dir=self.checkpoint_dir, suffix=".pt.tmp")

        try:
            # Close the file descriptor (torch.save will open it)
            os.close(temp_fd)

            # Save to temporary file
            torch.save(checkpoint, temp_path)

            # Atomic rename (overwrites if exists)
            os.replace(temp_path, checkpoint_path)

        except Exception as e:
            # Clean up temp file on error
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

        # Track periodic checkpoint
        self.periodic_checkpoints.append(checkpoint_path)

        # Clean up old checkpoints if needed
        if self.keep_last_n > 0 and len(self.periodic_checkpoints) > self.keep_last_n:
            old_checkpoint = self.periodic_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)

        return checkpoint_path

    def save_best(
        self,
        step: int,
        episode: int,
        epsilon: float,
        eval_return: float,
        online_model: nn.Module,
        target_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        replay_buffer: Any = None,
        rng_states: Dict = None,
        extra_metadata: Dict = None,
    ) -> bool:
        """
        Save checkpoint if it's the best model so far.

        Uses same atomic write as save_checkpoint.

        Args:
            step: Current environment step
            episode: Current episode counter
            epsilon: Current exploration rate
            eval_return: Evaluation return to compare
            online_model: Online Q-network
            target_model: Target Q-network
            optimizer: Optimizer state
            replay_buffer: Replay buffer (optional)
            rng_states: RNG states
            extra_metadata: Additional metadata

        Returns:
            bool: True if checkpoint was saved (new best), False otherwise
        """
        import tempfile
        from datetime import datetime

        from ..training.metadata import get_git_commit_hash

        if not self.save_best_enabled:
            return False

        if eval_return <= self.best_eval_return:
            return False

        # New best model
        self.best_eval_return = eval_return
        checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")

        # Prepare checkpoint (same structure as periodic)
        checkpoint = {
            "schema_version": self.SCHEMA_VERSION,
            "timestamp": datetime.now().isoformat(),
            "commit_hash": get_git_commit_hash(),
            "step": step,
            "episode": episode,
            "epsilon": epsilon,
            "eval_return": eval_return,  # Additional field for best model
            "online_model_state_dict": online_model.state_dict(),
            "target_model_state_dict": target_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "rng_states": rng_states if rng_states is not None else {},
        }

        if replay_buffer is not None:
            checkpoint["replay_buffer_state"] = {
                "index": replay_buffer.index,
                "size": replay_buffer.size,
                "capacity": replay_buffer.capacity,
                "obs_shape": replay_buffer.obs_shape,
            }
            if self.save_replay_buffer:
                checkpoint["replay_buffer_state"]["data"] = {
                    "observations": replay_buffer.observations,
                    "actions": replay_buffer.actions,
                    "rewards": replay_buffer.rewards,
                    "dones": replay_buffer.dones,
                    "episode_starts": replay_buffer.episode_starts,
                }

        if extra_metadata is not None:
            checkpoint["metadata"] = extra_metadata

        # Atomic write
        temp_fd, temp_path = tempfile.mkstemp(dir=self.checkpoint_dir, suffix=".pt.tmp")

        try:
            os.close(temp_fd)
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, checkpoint_path)
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise e

        self.best_checkpoint_path = checkpoint_path
        return True

    def load_checkpoint(
        self,
        checkpoint_path: str,
        online_model: nn.Module,
        target_model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        replay_buffer: Any = None,
        device: str = "cpu",
        strict: bool = True,
    ) -> Dict:
        """
        Load checkpoint and restore complete training state.

        Args:
            checkpoint_path: Path to checkpoint file
            online_model: Online Q-network to load weights into
            target_model: Target Q-network to load weights into
            optimizer: Optimizer to load state into (optional)
            replay_buffer: Replay buffer to restore state into (optional)
            device: Device to map checkpoint tensors to
            strict: Whether to strictly enforce state_dict key matching

        Returns:
            dict: Loaded checkpoint data including step, episode, epsilon,
                  rng_states, and metadata
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )

        # Validate schema version
        schema_version = checkpoint.get("schema_version", "unknown")
        if schema_version != self.SCHEMA_VERSION:
            print(
                f"Warning: Checkpoint schema version {schema_version} "
                f"does not match current version {self.SCHEMA_VERSION}"
            )

        # Restore models
        online_model.load_state_dict(
            checkpoint["online_model_state_dict"], strict=strict
        )
        target_model.load_state_dict(
            checkpoint["target_model_state_dict"], strict=strict
        )

        # Restore optimizer
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Restore replay buffer state
        if replay_buffer is not None and "replay_buffer_state" in checkpoint:
            buffer_state = checkpoint["replay_buffer_state"]
            replay_buffer.index = buffer_state["index"]
            replay_buffer.size = buffer_state["size"]

            # Restore full buffer data if saved
            if "data" in buffer_state:
                data = buffer_state["data"]
                replay_buffer.observations = data["observations"]
                replay_buffer.actions = data["actions"]
                replay_buffer.rewards = data["rewards"]
                replay_buffer.dones = data["dones"]
                replay_buffer.episode_starts = data["episode_starts"]

        # Return all state for manual restoration
        return {
            "step": checkpoint.get("step", 0),
            "episode": checkpoint.get("episode", 0),
            "epsilon": checkpoint.get("epsilon", 1.0),
            "eval_return": checkpoint.get("eval_return", None),
            "rng_states": checkpoint.get("rng_states", {}),
            "metadata": checkpoint.get("metadata", {}),
            "timestamp": checkpoint.get("timestamp", "unknown"),
            "commit_hash": checkpoint.get("commit_hash", "unknown"),
            "schema_version": schema_version,
        }


# ============================================================================
# Evaluation Routine
# ============================================================================
