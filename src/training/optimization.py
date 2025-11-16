"""Optimizer configuration and gradient clipping utilities."""

import torch
import torch.nn as nn


def configure_optimizer(
    network: nn.Module,
    optimizer_type: str = "rmsprop",
    learning_rate: float = 2.5e-4,
    alpha: float = 0.95,
    eps: float = 1e-2,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Configure optimizer for DQN training.

    Supports RMSProp (DQN default) or Adam. Uses paper-default hyperparameters
    for RMSProp: ρ=0.95, ε=0.01, LR=2.5e-4.

    Args:
        network: Neural network to optimize
        optimizer_type: Optimizer type, 'rmsprop' or 'adam' (default: 'rmsprop')
        learning_rate: Learning rate (default: 2.5e-4)
        alpha: RMSProp smoothing constant ρ (default: 0.95)
        eps: RMSProp epsilon for numerical stability (default: 1e-2)
        momentum: Momentum factor (default: 0.0)
        weight_decay: L2 regularization (default: 0.0)
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured optimizer

    Notes:
        - DQN paper uses RMSProp with ρ=0.95, ε=0.01, LR=2.5e-4
        - Adam is a common alternative with default β1=0.9, β2=0.999
        - Gamma (discount) is 0.99, batch size is 32 (not optimizer params)
        - Use with gradient clipping (max_norm=10.0) before optimizer.step()

    Example:
        >>> optimizer = configure_optimizer(online_net, optimizer_type='rmsprop')
        >>> # Training loop:
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> clip_gradients(online_net, max_norm=10.0)
        >>> optimizer.step()
    """
    # Get network parameters
    params = network.parameters()

    if optimizer_type.lower() == "rmsprop":
        # DQN paper defaults: alpha (ρ) = 0.95, eps = 0.01
        optimizer = torch.optim.RMSprop(
            params,
            lr=learning_rate,
            alpha=alpha,
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_type.lower() == "adam":
        # Adam defaults: beta1=0.9, beta2=0.999, eps=1e-8
        # Override eps if specified
        adam_eps = kwargs.pop("adam_eps", 1e-8)
        beta1 = kwargs.pop("beta1", 0.9)
        beta2 = kwargs.pop("beta2", 0.999)

        optimizer = torch.optim.Adam(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=adam_eps,
            weight_decay=weight_decay,
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown optimizer_type: {optimizer_type}. Use 'rmsprop' or 'adam'."
        )

    return optimizer


def clip_gradients(
    network: nn.Module, max_norm: float = 10.0, norm_type: float = 2.0
) -> float:
    """
    Clip gradients by global norm.

    Applies gradient clipping to prevent exploding gradients during training.
    Should be called after loss.backward() and before optimizer.step().

    Args:
        network: Neural network with computed gradients
        max_norm: Maximum gradient norm (default: 10.0)
        norm_type: Type of norm to use (default: 2.0 for L2 norm)

    Returns:
        Total gradient norm before clipping (for monitoring)

    Notes:
        - Clips gradients if total norm exceeds max_norm
        - Uses torch.nn.utils.clip_grad_norm_ internally
        - Returns unclipped norm for logging/monitoring
        - Common max_norm values: 10.0 (DQN), 40.0 (A3C), 0.5 (PPO)

    Example:
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> grad_norm = clip_gradients(online_net, max_norm=10.0)
        >>> print(f"Gradient norm: {grad_norm:.4f}")
        >>> optimizer.step()
    """
    # Compute and clip gradient norm
    total_norm = torch.nn.utils.clip_grad_norm_(
        network.parameters(), max_norm=max_norm, norm_type=norm_type
    )

    return total_norm.item()
