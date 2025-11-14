"""Evaluation system for periodic performance assessment.
"""

import os
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List


def evaluate(
    env,
    model: torch.nn.Module,
    num_episodes: int = 10,
    eval_epsilon: float = 0.05,
    num_actions: int = None,
    device: str = 'cpu',
    seed: int = None
) -> dict:
    """
    Evaluate agent over multiple episodes with low/greedy epsilon.

    Runs the agent in evaluation mode (no learning) and computes
    performance statistics over multiple episodes.

    Parameters
    ----------
    env : gym.Env
        Evaluation environment (should NOT have EpisodicLifeEnv wrapper)
    model : torch.nn.Module
        Q-network to evaluate
    num_episodes : int
        Number of episodes to run (default: 10)
    eval_epsilon : float
        Exploration rate during evaluation (default: 0.05, use 0.0 for greedy)
    num_actions : int
        Number of available actions (if None, inferred from env)
    device : str
        Device for model inference (default: 'cpu')
    seed : int
        Random seed for reproducibility (optional)

    Returns
    -------
    dict
        Evaluation results containing:
        - mean_return: Average episode return
        - median_return: Median episode return
        - std_return: Standard deviation of returns
        - min_return: Minimum episode return
        - max_return: Maximum episode return
        - mean_length: Average episode length
        - episode_returns: List of individual episode returns
        - episode_lengths: List of individual episode lengths
        - num_episodes: Number of episodes evaluated

    Example
    -------
    >>> results = evaluate(eval_env, model, num_episodes=10, eval_epsilon=0.05)
    >>> print(f"Mean return: {results['mean_return']:.2f}")
    """
    import numpy as np

    # Set model to eval mode
    model.eval()

    # Infer num_actions if not provided
    if num_actions is None:
        num_actions = env.action_space.n

    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            # Convert observation to tensor
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float().to(device)
                # Normalize if needed
                if obs_tensor.max() > 1.0:
                    obs_tensor = obs_tensor / 255.0
            else:
                obs_tensor = obs.float().to(device)

            # Select action with eval epsilon
            with torch.no_grad():
                if torch.rand(1).item() < eval_epsilon:
                    # Random action
                    action = env.action_space.sample()
                else:
                    # Greedy action
                    if obs_tensor.dim() == 3:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    output = model(obs_tensor)
                    q_values = output['q_values']
                    action = q_values.argmax(dim=1).item()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1
            done = terminated or truncated

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    # Compute statistics
    results = {
        'mean_return': np.mean(episode_returns),
        'median_return': np.median(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'num_episodes': num_episodes
    }

    # Set model back to train mode
    model.train()

    return results


class EvaluationScheduler:
    """
    Scheduler for periodic evaluation during training.

    Triggers evaluation at regular intervals and tracks evaluation history.

    Parameters
    ----------
    eval_interval : int
        Steps between evaluations (default: 250,000)
    num_episodes : int
        Number of episodes per evaluation (default: 10)
    eval_epsilon : float
        Exploration rate during evaluation (default: 0.05)

    Usage
    -----
    >>> scheduler = EvaluationScheduler(eval_interval=250000, num_episodes=10)
    >>> if scheduler.should_evaluate(current_step):
    ...     results = evaluate(eval_env, model, num_episodes=scheduler.num_episodes,
    ...                        eval_epsilon=scheduler.eval_epsilon)
    ...     scheduler.record_evaluation(current_step, results)
    """

    def __init__(
        self,
        eval_interval: int = 250_000,
        num_episodes: int = 10,
        eval_epsilon: float = 0.05
    ):
        self.eval_interval = eval_interval
        self.num_episodes = num_episodes
        self.eval_epsilon = eval_epsilon

        # Track evaluation history
        self.eval_steps = []
        self.eval_returns = []
        self.last_eval_step = 0

    def should_evaluate(self, step: int) -> bool:
        """
        Check if evaluation should be performed at this step.

        Args:
            step: Current environment step

        Returns:
            True if should evaluate, False otherwise
        """
        if step == 0:
            return False

        # Evaluate at intervals
        if step >= self.eval_interval and step % self.eval_interval == 0:
            # Avoid duplicate evaluations
            return step != self.last_eval_step

        return False

    def record_evaluation(self, step: int, results: dict):
        """
        Record evaluation results.

        Args:
            step: Environment step when evaluation occurred
            results: Dictionary returned by evaluate()
        """
        self.last_eval_step = step
        self.eval_steps.append(step)
        self.eval_returns.append(results['mean_return'])

    def get_best_return(self) -> float:
        """Get best mean return across all evaluations."""
        if not self.eval_returns:
            return float('-inf')
        return max(self.eval_returns)

    def get_recent_trend(self, n: int = 3) -> str:
        """
        Get recent performance trend.

        Args:
            n: Number of recent evaluations to consider

        Returns:
            'improving', 'declining', or 'stable'
        """
        if len(self.eval_returns) < n:
            return 'insufficient_data'

        recent = self.eval_returns[-n:]
        if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
            return 'improving'
        elif all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
            return 'declining'
        else:
            return 'stable'


class EvaluationLogger:
    """
    Logger for evaluation results.

    Saves evaluation statistics to CSV and JSON files.

    Parameters
    ----------
    log_dir : str
        Directory to save evaluation logs

    Usage
    -----
    >>> logger = EvaluationLogger(log_dir='runs/pong_123/eval')
    >>> logger.log_evaluation(step=250000, results=eval_results)
    """

    def __init__(self, log_dir: str):
        import os
        self.log_dir = log_dir

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # CSV file for evaluation summary
        self.csv_path = os.path.join(log_dir, 'evaluations.csv')
        self._csv_initialized = False

        # JSON directory for detailed per-eval results
        self.json_dir = os.path.join(log_dir, 'detailed')
        os.makedirs(self.json_dir, exist_ok=True)

    def log_evaluation(self, step: int, results: dict, epsilon: float = None):
        """
        Log evaluation results to CSV and JSON.

        Args:
            step: Environment step when evaluation occurred
            results: Dictionary returned by evaluate()
            epsilon: Current training epsilon (optional)
        """
        import csv
        import json
        import os

        # Prepare CSV entry (summary statistics)
        csv_entry = {
            'step': step,
            'mean_return': results['mean_return'],
            'median_return': results['median_return'],
            'std_return': results['std_return'],
            'min_return': results['min_return'],
            'max_return': results['max_return'],
            'mean_length': results['mean_length'],
            'num_episodes': results['num_episodes']
        }

        if epsilon is not None:
            csv_entry['training_epsilon'] = epsilon

        # Write CSV
        if not self._csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_entry.keys())
                writer.writeheader()
            self._csv_initialized = True

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_entry.keys())
            writer.writerow(csv_entry)

        # Save detailed results to JSON
        json_path = os.path.join(self.json_dir, f'eval_step_{step}.json')
        detailed_results = {
            'step': step,
            'statistics': {
                'mean_return': float(results['mean_return']),
                'median_return': float(results['median_return']),
                'std_return': float(results['std_return']),
                'min_return': float(results['min_return']),
                'max_return': float(results['max_return']),
                'mean_length': float(results['mean_length'])
            },
            'episode_returns': [float(r) for r in results['episode_returns']],
            'episode_lengths': [int(l) for l in results['episode_lengths']],
            'num_episodes': results['num_episodes']
        }

        if epsilon is not None:
            detailed_results['training_epsilon'] = epsilon

        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)

    def get_all_results(self) -> list:
        """
        Load all evaluation results from CSV.

        Returns:
            List of dictionaries with evaluation statistics
        """
        import csv
        import os

        if not os.path.exists(self.csv_path):
            return []

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)


# ============================================================================
# Reference-State Q Tracking
# ============================================================================
