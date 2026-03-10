"""DQN loss computation utilities.

Provides functions for:
- TD target computation using target network
- Q-value selection from online network
- DQN loss computation (MSE or Huber)
- Combined TD + SPR loss for joint optimization
- TD error statistics for monitoring
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_td_targets(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    target_net: nn.Module,
    gamma: float = 0.99,
) -> torch.Tensor:
    """
    Compute TD targets using the target network.

    Computes y = r + γ * (1 - done) * max_a' Q_target(s', a')

    The target network is used under no_grad for stability. The max Q-value
    over all actions is selected for the next state.

    Args:
        rewards: Reward tensor, shape (B,) in float32
        next_states: Next state tensor, shape (B, C, H, W) in float32
        dones: Done flags, shape (B,) in bool
        target_net: Target Q-network (frozen, eval mode)
        gamma: Discount factor (default 0.99)

    Returns:
        TD targets, shape (B,) in float32

    Notes:
        - Uses torch.no_grad() for target network forward pass
        - Masks out terminal states with (1 - done)
        - Uses max over actions for Q-learning (not double DQN)
        - Returned targets are detached (no gradient flow to target network)

    Example:
        >>> rewards = torch.tensor([1.0, 0.0, -1.0])
        >>> next_states = torch.randn(3, 4, 84, 84)
        >>> dones = torch.tensor([False, False, True])
        >>> targets = compute_td_targets(rewards, next_states, dones, target_net)
        >>> # targets[2] should be -1.0 (terminal state, no future value)
    """
    with torch.no_grad():
        # Forward pass through target network
        target_output = target_net(next_states)
        target_q_values = target_output["q_values"]  # (B, num_actions)

        # Get max Q-value over actions
        max_target_q, _ = target_q_values.max(dim=1)  # (B,)

        # Compute TD targets: r + γ * (1 - done) * max_a' Q(s', a')
        # Convert done from bool to float: True -> 1.0, False -> 0.0
        done_mask = dones.float()

        # TD target equation
        td_targets = rewards + gamma * (1.0 - done_mask) * max_target_q

    return td_targets


def select_q_values(
    online_net: nn.Module, states: torch.Tensor, actions: torch.Tensor
) -> torch.Tensor:
    """
    Select Q-values for specific actions from online network.

    Computes Q_online(s, a) by gathering Q-values for the actions that were taken.

    Args:
        online_net: Online Q-network being trained
        states: State tensor, shape (B, C, H, W) in float32
        actions: Action indices, shape (B,) in int64

    Returns:
        Selected Q-values, shape (B,) in float32

    Notes:
        - Forward pass through online network (gradients enabled)
        - Uses gather to select Q-values for specific actions
        - Returns shape (B,) after squeezing dimension 1
        - Gradients flow back through online network

    Example:
        >>> states = torch.randn(3, 4, 84, 84)
        >>> actions = torch.tensor([0, 2, 1])  # Actions taken
        >>> q_selected = select_q_values(online_net, states, actions)
        >>> q_selected.shape  # (3,)
    """
    # Forward pass through online network
    online_output = online_net(states)
    q_values = online_output["q_values"]  # (B, num_actions)

    # Gather Q-values for the actions that were taken
    # actions shape: (B,) -> (B, 1) for gather
    # q_values shape: (B, num_actions)
    # Result shape: (B, 1) -> squeeze to (B,)
    actions_unsqueezed = actions.unsqueeze(1)  # (B,) -> (B, 1)
    q_selected = q_values.gather(1, actions_unsqueezed)  # (B, 1)
    q_selected = q_selected.squeeze(1)  # (B,)

    return q_selected


def compute_td_loss_components(
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    online_net: nn.Module,
    target_net: nn.Module,
    gamma: float = 0.99,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Q-values and TD targets for loss computation.

    This is a convenience function that combines select_q_values and compute_td_targets.

    Args:
        states: State tensor, shape (B, C, H, W)
        actions: Action indices, shape (B,)
        rewards: Rewards, shape (B,)
        next_states: Next state tensor, shape (B, C, H, W)
        dones: Done flags, shape (B,)
        online_net: Online Q-network
        target_net: Target Q-network
        gamma: Discount factor

    Returns:
        Tuple of (q_selected, td_targets), both shape (B,)
        - q_selected: Q-values for actions taken (with gradients)
        - td_targets: TD target values (detached, no gradients)

    Example:
        >>> # After sampling a batch from replay buffer
        >>> q_selected, td_targets = compute_td_loss_components(
        ...     batch['states'], batch['actions'], batch['rewards'],
        ...     batch['next_states'], batch['dones'],
        ...     online_net, target_net, gamma=0.99
        ... )
        >>> # Now compute loss
        >>> loss = F.mse_loss(q_selected, td_targets)
    """
    # Get Q-values for actions taken (with gradients)
    q_selected = select_q_values(online_net, states, actions)

    # Get TD targets (no gradients)
    td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma)

    return q_selected, td_targets


def compute_dqn_loss(
    q_selected: torch.Tensor,
    td_targets: torch.Tensor,
    loss_type: str = "mse",
    huber_delta: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute DQN loss with configurable loss function.

    Supports MSE loss (default) or Huber loss (smooth L1).
    Returns loss and auxiliary statistics for monitoring.

    Args:
        q_selected: Selected Q-values Q(s, a), shape (B,), with gradients
        td_targets: TD target values y, shape (B,), detached
        loss_type: Loss function type, 'mse' or 'huber' (default: 'mse')
        huber_delta: Delta parameter for Huber loss (default: 1.0)

    Returns:
        Dictionary containing:
            - 'loss': Scalar loss tensor (with gradients)
            - 'td_error': Mean absolute TD error |Q(s,a) - y| (detached)
            - 'td_error_std': Standard deviation of TD errors (detached)

    Notes:
        - MSE loss: L = mean((Q(s,a) - y)^2)
        - Huber loss: L = smooth_l1_loss with delta parameter
        - TD error is computed as |Q(s,a) - y| for monitoring
        - All auxiliary stats are detached (no gradients)

    Example:
        >>> q_selected, td_targets = compute_td_loss_components(...)
        >>> loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='mse')
        >>> loss = loss_dict['loss']
        >>> loss.backward()
        >>> # Monitor TD error
        >>> print(f"Mean TD error: {loss_dict['td_error'].item():.4f}")
    """
    # Validate inputs
    assert (
        q_selected.shape == td_targets.shape
    ), f"Shape mismatch: q_selected {q_selected.shape} vs td_targets {td_targets.shape}"
    assert q_selected.requires_grad, "q_selected should have gradients"
    assert not td_targets.requires_grad, "td_targets should be detached"

    # Compute loss based on type
    if loss_type == "mse":
        # Mean Squared Error loss
        loss = F.mse_loss(q_selected, td_targets, reduction="mean")
    elif loss_type == "huber":
        # Huber loss (smooth L1)
        # PyTorch's smooth_l1_loss uses delta=1.0 by default
        # For custom delta, we use huber_loss with specified delta
        loss = F.huber_loss(q_selected, td_targets, reduction="mean", delta=huber_delta)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'mse' or 'huber'.")

    # Compute TD error statistics (for monitoring)
    with torch.no_grad():
        td_errors = torch.abs(q_selected - td_targets)  # |Q(s,a) - y|
        mean_td_error = td_errors.mean()
        std_td_error = td_errors.std()

    return {"loss": loss, "td_error": mean_td_error, "td_error_std": std_td_error}


def compute_combined_loss(
    td_loss: torch.Tensor,
    spr_loss: torch.Tensor = None,
    spr_weight: float = 2.0,
) -> Dict[str, torch.Tensor]:
    """
    Combine TD and SPR losses into a single training objective.

    Computes: total = td_loss + spr_weight * spr_loss

    When spr_loss is None, total equals td_loss (vanilla DQN path).
    Both component losses are returned detached for separate logging,
    while the total retains gradients for backpropagation.

    Args:
        td_loss: TD (temporal difference) loss, scalar with gradients.
        spr_loss: SPR auxiliary loss, scalar with gradients.
            Pass None when SPR is disabled.
        spr_weight: Multiplicative weight for SPR loss (default: 2.0,
            from Schwarzer et al. 2021, Table 3).

    Returns:
        Dictionary containing:
            - 'total_loss': Combined scalar loss (with gradients).
            - 'td_loss': TD loss value (detached, for logging).
            - 'spr_loss': SPR loss value (detached, for logging).
              Only present when spr_loss is not None.
            - 'weighted_spr_loss': spr_weight * spr_loss (detached).
              Only present when spr_loss is not None.
    """
    if spr_loss is not None:
        total_loss = td_loss + spr_weight * spr_loss
    else:
        total_loss = td_loss

    result = {
        "total_loss": total_loss,
        "td_loss": td_loss.detach(),
    }
    if spr_loss is not None:
        result["spr_loss"] = spr_loss.detach()
        result["weighted_spr_loss"] = (spr_weight * spr_loss).detach()

    return result
