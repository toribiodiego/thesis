"""Structured logging for training steps, episodes, and checkpoints.
"""

import os
import csv
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List


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

    def __init__(self, log_dir: str, log_interval: int = 1000, moving_avg_window: int = 100):
        import os
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.moving_avg_window = moving_avg_window

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # CSV file for step metrics
        self.csv_path = os.path.join(log_dir, 'training_steps.csv')
        self._csv_initialized = False

        # Moving average tracking
        self.loss_history = []

    def log_step(
        self,
        step: int,
        epsilon: float,
        metrics: UpdateMetrics = None,
        replay_size: int = 0,
        fps: float = 0.0
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
            'step': step,
            'epsilon': epsilon,
            'replay_size': replay_size,
            'fps': fps
        }

        # Add training metrics if available
        if metrics is not None:
            log_entry['loss'] = metrics.loss
            log_entry['td_error'] = metrics.td_error
            log_entry['td_error_std'] = metrics.td_error_std
            log_entry['grad_norm'] = metrics.grad_norm
            log_entry['learning_rate'] = metrics.learning_rate
            log_entry['update_count'] = metrics.update_count

            # Update moving average
            self.loss_history.append(metrics.loss)
            if len(self.loss_history) > self.moving_avg_window:
                self.loss_history.pop(0)
            log_entry['loss_ma'] = sum(self.loss_history) / len(self.loss_history)
        else:
            # No training this step
            log_entry['loss'] = None
            log_entry['td_error'] = None
            log_entry['td_error_std'] = None
            log_entry['grad_norm'] = None
            log_entry['learning_rate'] = None
            log_entry['update_count'] = None
            log_entry['loss_ma'] = None

        # Write to CSV
        self._write_csv(log_entry)

    def _write_csv(self, log_entry: dict):
        """Write log entry to CSV file."""
        import csv

        # Initialize CSV with header
        if not self._csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
            self._csv_initialized = True

        # Append log entry
        with open(self.csv_path, 'a', newline='') as f:
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
        import os
        self.log_dir = log_dir
        self.rolling_window = rolling_window

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # CSV file for episode metrics
        self.csv_path = os.path.join(log_dir, 'episodes.csv')
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
        epsilon: float = None
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
        recent_returns = self.episode_returns[-self.rolling_window:]
        recent_lengths = self.episode_lengths[-self.rolling_window:]

        import numpy as np
        rolling_mean_return = np.mean(recent_returns)
        rolling_std_return = np.std(recent_returns)
        rolling_mean_length = np.mean(recent_lengths)

        # Prepare log entry
        log_entry = {
            'episode': self.episode_count,
            'step': step,
            'return': episode_return,
            'length': episode_length,
            'fps': fps,
            'rolling_mean_return': rolling_mean_return,
            'rolling_std_return': rolling_std_return,
            'rolling_mean_length': rolling_mean_length,
            'num_episodes_in_window': len(recent_returns)
        }

        if epsilon is not None:
            log_entry['epsilon'] = epsilon

        # Write to CSV
        self._write_csv(log_entry)

    def _write_csv(self, log_entry: dict):
        """Write log entry to CSV file."""
        import csv

        # Initialize CSV with header
        if not self._csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
            self._csv_initialized = True

        # Append log entry
        with open(self.csv_path, 'a', newline='') as f:
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
            'mean_return': np.mean(recent_returns),
            'std_return': np.std(recent_returns),
            'min_return': np.min(recent_returns),
            'max_return': np.max(recent_returns),
            'mean_length': np.mean(recent_lengths),
            'num_episodes': len(recent_returns)
        }


# ============================================================================
# Checkpoint Management
# ============================================================================

class CheckpointManager:
    """
    Manages model checkpoints with periodic and best-model saving.

    Saves checkpoints at regular intervals and tracks the best model
    based on evaluation performance.

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

    Usage
    -----
    >>> manager = CheckpointManager(checkpoint_dir='runs/pong_123/checkpoints')
    >>> manager.save_checkpoint(step=1000000, model=model, optimizer=optimizer, metadata={...})
    >>> manager.save_best(step=1000000, eval_return=25.0, model=model, optimizer=optimizer)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_interval: int = 1_000_000,
        keep_last_n: int = 3,
        save_best: bool = True
    ):
        import os
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.save_best_enabled = save_best

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Track periodic checkpoints
        self.periodic_checkpoints = []

        # Track best model
        self.best_eval_return = float('-inf')
        self.best_checkpoint_path = None

    def should_save(self, step: int) -> bool:
        """Check if periodic checkpoint should be saved at this step."""
        return step > 0 and step % self.save_interval == 0

    def save_checkpoint(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: dict = None
    ) -> str:
        """
        Save periodic checkpoint.

        Args:
            step: Current environment step
            model: Q-network to save
            optimizer: Optimizer state to save
            metadata: Additional metadata (epsilon, replay stats, etc.)

        Returns:
            str: Path to saved checkpoint
        """
        import os

        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.pt')

        # Prepare checkpoint
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        if metadata is not None:
            checkpoint['metadata'] = metadata

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

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
        eval_return: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: dict = None
    ) -> bool:
        """
        Save checkpoint if it's the best model so far.

        Args:
            step: Current environment step
            eval_return: Evaluation return to compare
            model: Q-network to save
            optimizer: Optimizer state to save
            metadata: Additional metadata

        Returns:
            bool: True if checkpoint was saved (new best), False otherwise
        """
        import os

        if not self.save_best_enabled:
            return False

        if eval_return <= self.best_eval_return:
            return False

        # New best model
        self.best_eval_return = eval_return

        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')

        # Prepare checkpoint
        checkpoint = {
            'step': step,
            'eval_return': eval_return,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        if metadata is not None:
            checkpoint['metadata'] = metadata

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        self.best_checkpoint_path = checkpoint_path

        return True

    def load_checkpoint(self, checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
        """
        Load checkpoint and restore model/optimizer state.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Q-network to load weights into
            optimizer: Optimizer to load state into (optional)

        Returns:
            dict: Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return {
            'step': checkpoint.get('step', 0),
            'eval_return': checkpoint.get('eval_return', None),
            'metadata': checkpoint.get('metadata', {})
        }


# ============================================================================
# Evaluation Routine
# ============================================================================
