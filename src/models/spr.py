"""
SPR (Self-Predictive Representations) model components.

Implements the transition model from Schwarzer et al. (2021) that
predicts future latent states given current state and action.
Operates on the 64x7x7 spatial output of the DQN convolutional
encoder, not the post-FC features.

Reference: https://arxiv.org/abs/2007.05929, Section 2.3
"""

import torch
import torch.nn as nn


class TransitionModel(nn.Module):
    """
    Action-conditioned transition model for SPR.

    Predicts the next latent state given the current conv output and
    an action. Actions are represented as one-hot vectors broadcast
    to every spatial position and concatenated along the channel dim.

    Architecture (from Schwarzer et al. 2021, Section 2.3):
        Conv1(64, 3x3, padding=1) -> BN -> ReLU
        Conv2(64, 3x3, padding=1) -> ReLU

    Args:
        num_actions: Number of discrete actions in the environment.
        channels: Number of channels in the conv encoder output.
            Default 64 (Nature CNN conv3 output).

    Input: conv_output (B, 64, 7, 7) and actions (B,) as integer indices
    Output: predicted next conv_output (B, 64, 7, 7)
    """

    def __init__(self, num_actions: int, channels: int = 64):
        super().__init__()
        self.num_actions = num_actions
        self.channels = channels

        # First conv: input channels = encoder channels + action one-hot
        self.conv1 = nn.Conv2d(
            in_channels=channels + num_actions,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn1 = nn.BatchNorm2d(channels)

        # Second conv: channels -> channels
        self.conv2 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(
        self, conv_output: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict next latent state from current state and action.

        Args:
            conv_output: Spatial encoder output (B, channels, H, W).
                Typically (B, 64, 7, 7) from the Nature CNN.
            action: Integer action indices (B,).

        Returns:
            Predicted next latent state (B, channels, H, W).
        """
        batch_size = conv_output.size(0)
        h, w = conv_output.shape[2], conv_output.shape[3]

        # Create one-hot action encoding and broadcast to spatial dims
        # (B,) -> (B, num_actions) -> (B, num_actions, H, W)
        action_onehot = torch.zeros(
            batch_size, self.num_actions, device=conv_output.device
        )
        action_onehot.scatter_(1, action.unsqueeze(1), 1.0)
        action_broadcast = action_onehot.unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, h, w
        )

        # Concatenate along channel dimension
        x = torch.cat([conv_output, action_broadcast], dim=1)

        # Conv1 -> BN -> ReLU
        x = torch.relu(self.bn1(self.conv1(x)))

        # Conv2 -> ReLU
        x = torch.relu(self.conv2(x))

        return x
