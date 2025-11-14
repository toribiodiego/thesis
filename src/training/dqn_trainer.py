"""
DQN training utilities including target network management.

Provides utilities for:
- Hard target network updates
- TD target computation
- Q-learning loss computation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple


def hard_update_target(online_net: nn.Module, target_net: nn.Module) -> None:
    """
    Hard update target network with online network parameters.

    Performs a complete copy of all parameters from online network to target network.
    This is the standard DQN approach (Mnih et al. 2015) where the target network
    is updated periodically (e.g., every 10k steps) rather than continuously.

    Args:
        online_net: The online Q-network being trained
        target_net: The target Q-network used for computing TD targets

    Notes:
        - Target network gradients should be frozen (no gradient computation)
        - This is a "hard" update (full copy) vs "soft" update (polyak averaging)
        - Typically called every C steps (default 10,000 for DQN)
        - Target network is used under torch.no_grad() for stability

    Example:
        >>> online_net = DQN(num_actions=6)
        >>> target_net = DQN(num_actions=6)
        >>> hard_update_target(online_net, target_net)
        >>> # Every 10k steps:
        >>> hard_update_target(online_net, target_net)
    """
    target_net.load_state_dict(online_net.state_dict())


def init_target_network(online_net: nn.Module, num_actions: int) -> nn.Module:
    """
    Initialize target network as a copy of online network.

    Creates a new network with the same architecture, copies weights from online network,
    and freezes gradients for the target network.

    Args:
        online_net: The online Q-network
        num_actions: Number of actions in the environment

    Returns:
        Target network (copy of online, with frozen gradients)

    Example:
        >>> from src.models import DQN
        >>> online_net = DQN(num_actions=6)
        >>> target_net = init_target_network(online_net, num_actions=6)
        >>> # Verify target is a copy
        >>> assert torch.allclose(
        ...     list(online_net.parameters())[0],
        ...     list(target_net.parameters())[0]
        ... )
    """
    # Create target network with same class and parameters
    target_net = type(online_net)(num_actions=num_actions)

    # Copy weights from online to target
    hard_update_target(online_net, target_net)

    # Freeze target network (disable gradient computation)
    for param in target_net.parameters():
        param.requires_grad = False

    # Set to eval mode
    target_net.eval()

    return target_net


def compute_td_targets(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    target_net: nn.Module,
    gamma: float = 0.99
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
        target_q_values = target_output['q_values']  # (B, num_actions)

        # Get max Q-value over actions
        max_target_q, _ = target_q_values.max(dim=1)  # (B,)

        # Compute TD targets: r + γ * (1 - done) * max_a' Q(s', a')
        # Convert done from bool to float: True -> 1.0, False -> 0.0
        done_mask = dones.float()

        # TD target equation
        td_targets = rewards + gamma * (1.0 - done_mask) * max_target_q

    return td_targets


def select_q_values(
    online_net: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor
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
    q_values = online_output['q_values']  # (B, num_actions)

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
    gamma: float = 0.99
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
