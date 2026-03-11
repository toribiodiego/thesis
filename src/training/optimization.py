"""Optimizer configuration and gradient clipping utilities."""

import torch
import torch.nn as nn


# Rainbow DQN optimizer defaults (Hessel et al. 2018, Appendix B)
RAINBOW_OPTIMIZER_DEFAULTS = {
    "optimizer_type": "adam",
    "learning_rate": 6.25e-5,
    "eps": 1.5e-4,
}


def configure_optimizer(
    network: nn.Module,
    optimizer_type: str = "rmsprop",
    learning_rate: float = 2.5e-4,
    alpha: float = 0.95,
    eps: float = None,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    **kwargs,
) -> torch.optim.Optimizer:
    """
    Configure optimizer for DQN training.

    Supports RMSProp (DQN default) or Adam. Uses paper-default
    hyperparameters for RMSProp: rho=0.95, eps=0.01, LR=2.5e-4.
    For Rainbow, use Adam with lr=6.25e-5, eps=1.5e-4 (see
    RAINBOW_OPTIMIZER_DEFAULTS).

    Args:
        network: Neural network to optimize
        optimizer_type: Optimizer type, 'rmsprop' or 'adam' (default: 'rmsprop')
        learning_rate: Learning rate (default: 2.5e-4)
        alpha: RMSProp smoothing constant rho (default: 0.95)
        eps: Epsilon for numerical stability. Default depends on optimizer:
            RMSProp=1e-2, Adam=1e-8. Pass explicitly to override.
        momentum: Momentum factor (default: 0.0)
        weight_decay: L2 regularization (default: 0.0)
        **kwargs: Additional optimizer-specific arguments (beta1, beta2)

    Returns:
        Configured optimizer

    Notes:
        - DQN paper uses RMSProp with rho=0.95, eps=0.01, LR=2.5e-4
        - Rainbow uses Adam with lr=6.25e-5, eps=1.5e-4
        - Use with gradient clipping (max_norm=10.0) before optimizer.step()

    Example:
        >>> optimizer = configure_optimizer(online_net, optimizer_type='rmsprop')
        >>> # Rainbow optimizer:
        >>> optimizer = configure_optimizer(
        ...     online_net, **RAINBOW_OPTIMIZER_DEFAULTS
        ... )
    """
    # Get network parameters
    params = network.parameters()

    if optimizer_type.lower() == "rmsprop":
        # DQN paper defaults: alpha (rho) = 0.95, eps = 0.01
        rmsprop_eps = eps if eps is not None else 1e-2
        optimizer = torch.optim.RMSprop(
            params,
            lr=learning_rate,
            alpha=alpha,
            eps=rmsprop_eps,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs,
        )
    elif optimizer_type.lower() == "adam":
        # Adam defaults: beta1=0.9, beta2=0.999, eps=1e-8
        adam_eps = eps if eps is not None else kwargs.pop("adam_eps", 1e-8)
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
