"""Training metrics collection and logging utilities.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict

from .loss import compute_td_loss_components, compute_dqn_loss
from .optimization import clip_gradients
from .stability import detect_nan_inf
from .target_network import hard_update_target


class UpdateMetrics:
    """
    Container for training update metrics.

    Stores metrics from a single training update for logging and monitoring:
    - loss: Training loss value
    - td_error: Mean absolute TD error
    - td_error_std: Standard deviation of TD errors
    - grad_norm: Gradient norm before clipping
    - learning_rate: Current optimizer learning rate
    - update_count: Total number of updates performed

    Attributes:
        loss: Training loss (float)
        td_error: Mean |TD error| (float)
        td_error_std: Standard deviation of TD errors (float)
        grad_norm: Gradient norm before clipping (float)
        learning_rate: Current learning rate (float)
        update_count: Number of updates performed (int)

    Example:
        >>> metrics = UpdateMetrics(
        ...     loss=0.5,
        ...     td_error=0.3,
        ...     td_error_std=0.2,
        ...     grad_norm=2.5,
        ...     learning_rate=0.00025,
        ...     update_count=1000
        ... )
        >>> print(f"Loss: {metrics.loss:.4f}")
        Loss: 0.5000
    """

    def __init__(
        self,
        loss: float,
        td_error: float,
        td_error_std: float,
        grad_norm: float,
        learning_rate: float,
        update_count: int
    ):
        """
        Initialize update metrics.

        Args:
            loss: Training loss value
            td_error: Mean absolute TD error
            td_error_std: Standard deviation of TD errors
            grad_norm: Gradient norm before clipping
            learning_rate: Current optimizer learning rate
            update_count: Total number of updates performed
        """
        self.loss = loss
        self.td_error = td_error
        self.td_error_std = td_error_std
        self.grad_norm = grad_norm
        self.learning_rate = learning_rate
        self.update_count = update_count

    def to_dict(self) -> Dict[str, float]:
        """
        Convert metrics to dictionary for logging.

        Returns:
            Dictionary with all metrics

        Example:
            >>> metrics = UpdateMetrics(0.5, 0.3, 0.2, 2.5, 0.00025, 1000)
            >>> metrics_dict = metrics.to_dict()
            >>> assert 'loss' in metrics_dict
            >>> assert 'td_error' in metrics_dict
        """
        return {
            'loss': self.loss,
            'td_error': self.td_error,
            'td_error_std': self.td_error_std,
            'grad_norm': self.grad_norm,
            'learning_rate': self.learning_rate,
            'update_count': self.update_count
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"UpdateMetrics(loss={self.loss:.4f}, td_error={self.td_error:.4f}, "
            f"grad_norm={self.grad_norm:.4f}, lr={self.learning_rate:.6f}, "
            f"updates={self.update_count})"
        )


def perform_update_step(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    gamma: float = 0.99,
    loss_type: str = 'mse',
    max_grad_norm: float = 10.0,
    update_count: int = 0
) -> UpdateMetrics:
    """
    Perform a single training update and return metrics.

    Executes the full DQN update step:
    1. Compute TD targets using target network
    2. Select Q-values for actions from online network
    3. Compute loss (MSE or Huber)
    4. Backpropagate gradients
    5. Clip gradients by global norm
    6. Update network parameters
    7. Collect and return metrics

    Args:
        online_net: Online Q-network to train
        target_net: Target Q-network for TD targets
        optimizer: Optimizer for online network
        batch: Dictionary with keys 'states', 'actions', 'rewards', 'next_states', 'dones'
        gamma: Discount factor (default: 0.99)
        loss_type: 'mse' or 'huber' (default: 'mse')
        max_grad_norm: Maximum gradient norm for clipping (default: 10.0)
        update_count: Current update count for metrics (default: 0)

    Returns:
        UpdateMetrics object with loss, TD error, grad norm, learning rate, and count

    Example:
        >>> from src.replay import ReplayBuffer
        >>> online_net = DQN(num_actions=6)
        >>> target_net = init_target_network(online_net, num_actions=6)
        >>> optimizer = configure_optimizer(online_net)
        >>> buffer = ReplayBuffer(capacity=10000, batch_size=32)
        >>> # ... fill buffer ...
        >>> batch = buffer.sample()
        >>> metrics = perform_update_step(
        ...     online_net, target_net, optimizer, batch,
        ...     update_count=1
        ... )
        >>> print(f"Loss: {metrics.loss:.4f}")
    """
    # Set online network to training mode
    online_net.train()

    # Extract batch data
    states = batch['states']
    actions = batch['actions']
    rewards = batch['rewards']
    next_states = batch['next_states']
    dones = batch['dones']

    # Compute TD targets (no gradient)
    td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma)

    # Select Q-values for taken actions (with gradient)
    q_selected = select_q_values(online_net, states, actions)

    # Compute loss and TD error stats
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type=loss_type)
    loss = loss_dict['loss']
    td_error = loss_dict['td_error'].item()
    td_error_std = loss_dict['td_error_std'].item()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients and get norm before clipping
    grad_norm = clip_gradients(online_net, max_norm=max_grad_norm)

    # Update parameters
    optimizer.step()

    # Get current learning rate
    learning_rate = optimizer.param_groups[0]['lr']

    # Create and return metrics
    metrics = UpdateMetrics(
        loss=loss.item(),
        td_error=td_error,
        td_error_std=td_error_std,
        grad_norm=grad_norm,
        learning_rate=learning_rate,
        update_count=update_count
    )

    return metrics


# ============================================================================
# Epsilon-Greedy Exploration
# ============================================================================

class EpsilonScheduler:
    """
    Epsilon-greedy exploration scheduler with linear decay.

    Supports linear decay from epsilon_start to epsilon_end over a specified
    number of frames, with separate epsilon for training and evaluation.

    Parameters
    ----------
    epsilon_start : float
        Initial epsilon value (default: 1.0, fully random exploration)
    epsilon_end : float
        Final epsilon value after decay (default: 0.1)
    decay_frames : int
        Number of frames over which to decay epsilon (default: 1,000,000)
    eval_epsilon : float
        Fixed epsilon for evaluation mode (default: 0.05)

    Usage
    -----
    >>> scheduler = EpsilonScheduler(epsilon_start=1.0, epsilon_end=0.1, decay_frames=1000000)
    >>> epsilon = scheduler.get_epsilon(current_frame=500000)  # Returns 0.55
    >>> eval_eps = scheduler.get_eval_epsilon()  # Returns 0.05
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        decay_frames: int = 1_000_000,
        eval_epsilon: float = 0.05
    ):
        assert 0.0 <= epsilon_start <= 1.0, f"epsilon_start must be in [0,1], got {epsilon_start}"
        assert 0.0 <= epsilon_end <= 1.0, f"epsilon_end must be in [0,1], got {epsilon_end}"
        assert epsilon_start >= epsilon_end, f"epsilon_start must be >= epsilon_end"
        assert decay_frames > 0, f"decay_frames must be positive, got {decay_frames}"
        assert 0.0 <= eval_epsilon <= 1.0, f"eval_epsilon must be in [0,1], got {eval_epsilon}"

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_frames = decay_frames
        self.eval_epsilon = eval_epsilon

        # Precompute slope for efficiency
        self.slope = (epsilon_end - epsilon_start) / decay_frames

    def get_epsilon(self, current_frame: int) -> float:
        """
        Get epsilon for current training frame with linear decay.

        Parameters
        ----------
        current_frame : int
            Current environment frame count (not update count)

        Returns
        -------
        float
            Epsilon value in [epsilon_end, epsilon_start]

        Examples
        --------
        >>> scheduler = EpsilonScheduler(1.0, 0.1, 1000000)
        >>> scheduler.get_epsilon(0)
        1.0
        >>> scheduler.get_epsilon(500000)
        0.55
        >>> scheduler.get_epsilon(1000000)
        0.1
        >>> scheduler.get_epsilon(2000000)  # Clamps to epsilon_end
        0.1
        """
        if current_frame >= self.decay_frames:
            return self.epsilon_end

        # Linear interpolation: start + slope * frames
        epsilon = self.epsilon_start + self.slope * current_frame

        # Clamp to [epsilon_end, epsilon_start] for numerical stability
        return max(self.epsilon_end, min(self.epsilon_start, epsilon))

    def get_eval_epsilon(self) -> float:
        """
        Get fixed epsilon for evaluation mode.

        Returns
        -------
        float
            Fixed epsilon for evaluation (no decay)
        """
        return self.eval_epsilon

    def to_dict(self) -> dict:
        """Export scheduler configuration as dictionary."""
        return {
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'decay_frames': self.decay_frames,
            'eval_epsilon': self.eval_epsilon
        }


