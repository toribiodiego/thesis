"""Stability checks for detecting NaN/Inf and validating training behavior.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict

from .loss import compute_td_targets, select_q_values


def detect_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Detect NaN or Inf values in a tensor.

    Args:
        tensor: Tensor to check
        name: Name of the tensor (for warning messages)

    Returns:
        True if NaN or Inf detected, False otherwise

    Example:
        >>> loss = compute_dqn_loss(q_selected, td_targets)['loss']
        >>> if detect_nan_inf(loss, "loss"):
        ...     print("Warning: NaN/Inf detected in loss!")
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        return True
    return False


def validate_loss_decrease(
    loss_fn,
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    target_net: nn.Module,
    num_updates: int = 10,
    gamma: float = 0.99,
    loss_type: str = 'mse'
) -> Tuple[bool, Dict[str, any]]:
    """
    Validate that loss decreases over several updates on a synthetic batch.

    Performs multiple optimization steps on the same batch and checks that:
    1. Loss decreases from initial value
    2. No NaN or Inf values appear
    3. Final loss is lower than initial loss

    Args:
        loss_fn: Loss computation function (e.g., compute_dqn_loss)
        network: Online Q-network to train
        optimizer: Optimizer for network
        states: Batch of states (B, C, H, W)
        actions: Batch of actions (B,)
        rewards: Batch of rewards (B,)
        next_states: Batch of next states (B, C, H, W)
        dones: Batch of done flags (B,)
        target_net: Target Q-network for TD targets
        num_updates: Number of optimization steps to perform (default: 10)
        gamma: Discount factor (default: 0.99)
        loss_type: Type of loss ('mse' or 'huber', default: 'mse')

    Returns:
        Tuple of (success: bool, info: Dict) where info contains:
            - 'initial_loss': Initial loss value
            - 'final_loss': Final loss value
            - 'loss_history': List of all loss values
            - 'loss_decreased': Whether loss decreased
            - 'nan_inf_detected': Whether NaN/Inf was detected

    Example:
        >>> online_net = DQN(num_actions=6)
        >>> target_net = init_target_network(online_net, num_actions=6)
        >>> optimizer = configure_optimizer(online_net)
        >>> # Create synthetic batch
        >>> states = torch.randn(32, 4, 84, 84)
        >>> actions = torch.randint(0, 6, (32,))
        >>> rewards = torch.randn(32)
        >>> next_states = torch.randn(32, 4, 84, 84)
        >>> dones = torch.zeros(32)
        >>> # Validate loss decreases
        >>> success, info = validate_loss_decrease(
        ...     compute_dqn_loss, online_net, optimizer,
        ...     states, actions, rewards, next_states, dones, target_net
        ... )
        >>> assert success, f"Loss did not decrease: {info}"
    """
    loss_history = []
    nan_inf_detected = False

    network.train()

    for step in range(num_updates):
        # Compute TD targets
        td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma)

        # Select Q-values for actions
        q_selected = select_q_values(network, states, actions)

        # Compute loss
        loss_dict = loss_fn(q_selected, td_targets, loss_type=loss_type)
        loss = loss_dict['loss']

        # Check for NaN/Inf
        if detect_nan_inf(loss, "loss"):
            nan_inf_detected = True
            loss_history.append(float('nan'))
            break

        loss_history.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    initial_loss = loss_history[0] if loss_history else float('nan')
    final_loss = loss_history[-1] if loss_history else float('nan')
    loss_decreased = final_loss < initial_loss

    success = loss_decreased and not nan_inf_detected

    info = {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_history': loss_history,
        'loss_decreased': loss_decreased,
        'nan_inf_detected': nan_inf_detected
    }

    return success, info


def verify_target_sync_schedule(
    updater: 'TargetNetworkUpdater',
    online_net: nn.Module,
    target_net: nn.Module,
    max_steps: int,
    expected_interval: int
) -> Tuple[bool, Dict[str, any]]:
    """
    Verify that target network updates occur at exact multiples of update_interval.

    Simulates stepping through training and checks that:
    1. Updates occur at exact multiples of the interval
    2. No duplicate updates occur
    3. Update count matches expected count

    Args:
        updater: TargetNetworkUpdater instance
        online_net: Online Q-network
        target_net: Target Q-network
        max_steps: Maximum number of steps to simulate
        expected_interval: Expected update interval (should match updater.update_interval)

    Returns:
        Tuple of (success: bool, info: Dict) where info contains:
            - 'update_steps': List of steps where updates occurred
            - 'expected_steps': List of steps where updates should occur
            - 'schedule_correct': Whether updates occurred at correct steps
            - 'count_correct': Whether update count is correct

    Example:
        >>> online_net = DQN(num_actions=6)
        >>> target_net = init_target_network(online_net, num_actions=6)
        >>> updater = TargetNetworkUpdater(update_interval=1000)
        >>> success, info = verify_target_sync_schedule(
        ...     updater, online_net, target_net, max_steps=5000, expected_interval=1000
        ... )
        >>> assert success, f"Target sync schedule incorrect: {info}"
        >>> assert info['update_steps'] == [1000, 2000, 3000, 4000, 5000]
    """
    update_steps = []

    # Reset updater to start fresh
    updater.reset()

    # Step through training
    for step in range(1, max_steps + 1):
        update_info = updater.step(online_net, target_net, step)
        if update_info is not None:
            update_steps.append(step)

    # Calculate expected update steps
    expected_steps = list(range(expected_interval, max_steps + 1, expected_interval))

    schedule_correct = update_steps == expected_steps
    count_correct = len(update_steps) == len(expected_steps)

    success = schedule_correct and count_correct

    info = {
        'update_steps': update_steps,
        'expected_steps': expected_steps,
        'schedule_correct': schedule_correct,
        'count_correct': count_correct
    }

    return success, info

