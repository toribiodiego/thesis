"""
Exponential Moving Average (EMA) target encoder for SPR.

Maintains a momentum-averaged copy of the online encoder and projection
head. Updated every training step, unlike the DQN target network which
uses hard parameter copies every 10K steps.

The EMA encoder provides stable target representations for the
self-predictive loss. When tau=0 (with augmentation), this degenerates
to a direct copy every step. When tau=0.99 (without augmentation),
parameter changes are smoothed over ~100 steps.

Reference: Schwarzer et al. 2021 (SPR), Section 2.2, Table 3
"""

import copy

import torch
import torch.nn as nn


class EMAEncoder(nn.Module):
    """
    EMA copy of an online model for SPR target representations.

    Wraps any nn.Module and maintains an exponentially averaged copy
    of its parameters. Gradients are frozen on the target side --
    only EMA updates modify the parameters.

    This is used for both the target encoder (f_m) and target
    projection head (g_m) in SPR. Each wrapped independently.

    Args:
        online_model: The online model to track (encoder or projection).
        momentum: EMA decay coefficient (tau). Higher values mean
            slower updates. Default 0.99 (without augmentation).
            Use 0.0 for direct copy (with augmentation).

    Example:
        >>> encoder = DQN(num_actions=6)
        >>> target_encoder = EMAEncoder(encoder, momentum=0.99)
        >>> # After each training step:
        >>> target_encoder.update(encoder)
        >>> # Forward pass (no gradients):
        >>> with torch.no_grad():
        ...     target_out = target_encoder(x)
    """

    def __init__(self, online_model: nn.Module, momentum: float = 0.99):
        super().__init__()
        if not 0.0 <= momentum <= 1.0:
            raise ValueError(f"Momentum must be in [0, 1], got {momentum}")
        self.momentum = momentum
        self.model = copy.deepcopy(online_model)

        # Freeze all parameters -- no gradient computation
        for param in self.model.parameters():
            param.requires_grad = False

        # Always in inference mode (BN uses running stats, not batch stats)
        self.model.train(False)

    @torch.no_grad()
    def update(self, online_model: nn.Module) -> None:
        """
        Perform one EMA update step.

        Updates parameters as: theta_m <- tau * theta_m + (1-tau) * theta_o
        Copies buffers directly (e.g., BatchNorm running stats).

        Args:
            online_model: The online model whose parameters to track.
        """
        # EMA update for trainable parameters
        for ema_param, online_param in zip(
            self.model.parameters(), online_model.parameters()
        ):
            ema_param.data.mul_(self.momentum).add_(
                online_param.data, alpha=1.0 - self.momentum
            )

        # Direct copy for buffers (BN running_mean, running_var, etc.)
        for ema_buf, online_buf in zip(
            self.model.buffers(), online_model.buffers()
        ):
            ema_buf.data.copy_(online_buf.data)

    def forward(self, *args, **kwargs):
        """
        Forward pass through the EMA model.

        Delegates to the internal copy. Should be called under
        torch.no_grad() to avoid unnecessary computation graphs.
        """
        return self.model(*args, **kwargs)
