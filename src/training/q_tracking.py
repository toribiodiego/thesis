"""Reference state Q-value tracking for monitoring learning progress.
"""

import os
import csv
import torch
import torch.nn as nn
from typing import Optional, List


class ReferenceStateQTracker:
    """
    Tracks average max-Q values over a fixed reference batch.

    Monitors learning progress by computing Q-values on a fixed set of
    reference states. This provides a stable signal even when episode
    rewards are noisy, as described in the DQN paper.

    Parameters
    ----------
    reference_states : torch.Tensor or np.ndarray
        Fixed batch of states to track Q-values on (B, C, H, W)
    log_interval : int
        Steps between Q-value logging (default: 10,000)
    device : str
        Device for Q-value computation (default: 'cpu')

    Usage
    -----
    >>> # Create reference batch from first N states in replay buffer
    >>> ref_states = replay_buffer.sample(batch_size=100)['states']
    >>> tracker = ReferenceStateQTracker(reference_states=ref_states)
    >>>
    >>> # Periodically log Q-values
    >>> if tracker.should_log(current_step):
    ...     tracker.log_q_values(step=current_step, model=online_net)
    """

    def __init__(
        self,
        reference_states: torch.Tensor = None,
        log_interval: int = 10_000,
        device: str = 'cpu'
    ):
        self.log_interval = log_interval
        self.device = device
        self.last_log_step = 0

        # Reference states
        if reference_states is not None:
            self.set_reference_states(reference_states)
        else:
            self.reference_states = None

        # Q-value history
        self.log_steps = []
        self.avg_max_q = []
        self.max_q = []
        self.min_q = []

    def set_reference_states(self, states):
        """
        Set or update reference states.

        Args:
            states: Tensor or array of states (B, C, H, W)
        """
        import numpy as np

        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()

        # Normalize if needed (assumes uint8 input if max > 1)
        if states.max() > 1.0:
            states = states / 255.0

        self.reference_states = states.to(self.device)

    def should_log(self, step: int) -> bool:
        """
        Check if Q-values should be logged at this step.

        Args:
            step: Current environment step

        Returns:
            True if should log, False otherwise
        """
        if self.reference_states is None:
            return False

        if step == 0:
            return False

        # Log at intervals
        if step % self.log_interval == 0:
            return step != self.last_log_step

        return False

    def compute_q_values(self, model: torch.nn.Module) -> dict:
        """
        Compute Q-values for reference states.

        Args:
            model: Q-network

        Returns:
            dict: Q-value statistics (avg_max_q, max_q, min_q)
        """
        if self.reference_states is None:
            raise ValueError("Reference states not set. Call set_reference_states() first.")

        model.eval()
        with torch.no_grad():
            output = model(self.reference_states)
            q_values = output['q_values']  # (B, num_actions)

            # Compute max Q-value per state
            max_q_per_state = q_values.max(dim=1)[0]  # (B,)

            stats = {
                'avg_max_q': max_q_per_state.mean().item(),
                'max_q': max_q_per_state.max().item(),
                'min_q': max_q_per_state.min().item()
            }

        model.train()
        return stats

    def log_q_values(self, step: int, model: torch.nn.Module):
        """
        Compute and record Q-values at current step.

        Args:
            step: Current environment step
            model: Q-network
        """
        stats = self.compute_q_values(model)

        self.last_log_step = step
        self.log_steps.append(step)
        self.avg_max_q.append(stats['avg_max_q'])
        self.max_q.append(stats['max_q'])
        self.min_q.append(stats['min_q'])

    def get_history(self) -> dict:
        """
        Get complete Q-value history.

        Returns:
            dict: History of steps and Q-values
        """
        return {
            'steps': self.log_steps.copy(),
            'avg_max_q': self.avg_max_q.copy(),
            'max_q': self.max_q.copy(),
            'min_q': self.min_q.copy()
        }

    def to_dict(self) -> dict:
        """Serialize tracker state to dictionary."""
        return {
            'log_interval': self.log_interval,
            'device': self.device,
            'last_log_step': self.last_log_step,
            'log_steps': self.log_steps.copy(),
            'avg_max_q': self.avg_max_q.copy(),
            'max_q': self.max_q.copy(),
            'min_q': self.min_q.copy()
        }


class ReferenceQLogger:
    """
    Logger for reference-state Q-values.

    Saves Q-value statistics to CSV for monitoring learning progress.

    Parameters
    ----------
    log_dir : str
        Directory to save Q-value logs

    Usage
    -----
    >>> logger = ReferenceQLogger(log_dir='runs/pong_123/logs')
    >>> q_stats = tracker.compute_q_values(model)
    >>> logger.log(step=10000, q_stats=q_stats)
    """

    def __init__(self, log_dir: str):
        import os
        self.log_dir = log_dir

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # CSV file
        self.csv_path = os.path.join(log_dir, 'reference_q_values.csv')
        self._csv_initialized = False

    def log(self, step: int, q_stats: dict):
        """
        Log Q-value statistics to CSV.

        Args:
            step: Current environment step
            q_stats: Dictionary with avg_max_q, max_q, min_q
        """
        import csv

        log_entry = {
            'step': step,
            'avg_max_q': q_stats['avg_max_q'],
            'max_q': q_stats['max_q'],
            'min_q': q_stats['min_q']
        }

        # Write CSV
        if not self._csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
            self._csv_initialized = True

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)

    def get_all_logs(self) -> list:
        """
        Load all Q-value logs from CSV.

        Returns:
            List of dictionaries with Q-value statistics
        """
        import csv
        import os

        if not os.path.exists(self.csv_path):
            return []

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)


# ============================================================================
# Reproducibility Metadata
# ============================================================================
