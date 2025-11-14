"""
DQN training utilities including target network management.

Provides utilities for:
- Hard target network updates
- TD target computation
- Q-learning loss computation
"""

import torch
import torch.nn as nn
from typing import Optional


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
