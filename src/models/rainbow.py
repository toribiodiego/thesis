"""
Rainbow DQN model architecture (Hessel et al., 2018).

Combines the Nature CNN encoder with a dueling, distributional,
noisy head. The convolutional encoder is identical to DQN (Mnih et
al., 2015); all modifications happen after the encoder output:

- Dueling streams (Wang et al., 2016): separate value and advantage
  paths, aggregated per atom via Q = V + A - mean(A).
- Distributional output (Bellemare et al., 2017): each action maps
  to a categorical distribution over num_atoms support atoms
  z_i in [v_min, v_max]. Probabilities via softmax.
- Noisy layers (Fortunato et al., 2018): NoisyLinear replaces
  nn.Linear in the head for parameter-space exploration.

Input:  (batch, 4, 84, 84) - 4 stacked grayscale frames
Output: dict with q_values, log_probs, conv_output
"""

import torch
import torch.nn as nn

from .noisy_linear import NoisyLinear


class RainbowDQN(nn.Module):
    """
    Rainbow DQN with dueling distributional noisy head.

    Architecture:
        Conv1(32, 8x8, stride=4) -> ReLU
        Conv2(64, 4x4, stride=2) -> ReLU
        Conv3(64, 3x3, stride=1) -> ReLU
        Flatten (3136)
        -- Dueling split --
        Value:     Linear(3136,512) -> ReLU -> Linear(512, num_atoms)
        Advantage: Linear(3136,512) -> ReLU -> Linear(512, num_atoms*A)
        Aggregation per atom: Q = V + A - mean(A)
        Softmax over atoms -> p(s,a)
        Q(s,a) = sum(z_i * p_i)

    Args:
        num_actions: Number of discrete actions.
        num_atoms: Number of support atoms for distributional RL.
        v_min: Minimum support value.
        v_max: Maximum support value.
        noisy: If True, use NoisyLinear; otherwise nn.Linear.
        dueling: If True, use dueling value/advantage streams.
        dropout: Dropout probability after hidden layers.

    Output dict keys:
        - 'q_values': (batch, num_actions) expected Q-values
        - 'log_probs': (batch, num_actions, num_atoms) log-probabilities
        - 'conv_output': (batch, 64, 7, 7) spatial features for SPR
    """

    def __init__(
        self,
        num_actions: int,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        noisy: bool = True,
        dueling: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.noisy = noisy
        self.dueling = dueling
        self.dropout = dropout

        # Support atoms z_i = v_min + i * delta_z
        support = torch.linspace(v_min, v_max, num_atoms)
        self.register_buffer("support", support)

        # ---- Convolutional encoder (Nature CNN) ----
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        conv_output_size = 64 * 7 * 7  # 3136

        # ---- Head layers ----
        Linear = NoisyLinear if noisy else nn.Linear
        self.drop = nn.Dropout(p=dropout)

        if dueling:
            # Value stream: V(s) distribution over atoms
            self.value_fc = Linear(conv_output_size, 512)
            self.value_head = Linear(512, num_atoms)
            # Advantage stream: A(s,a) distribution over atoms
            self.advantage_fc = Linear(conv_output_size, 512)
            self.advantage_head = Linear(512, num_atoms * num_actions)
        else:
            # Single stream: Q(s,a) distribution over atoms
            self.fc = Linear(conv_output_size, 512)
            self.q_head = Linear(512, num_atoms * num_actions)

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming init for conv and standard linear layers.

        NoisyLinear layers have their own initialization and are
        skipped here (they inherit from nn.Module, not nn.Linear).
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def reset_noise(self):
        """Resample noise for all NoisyLinear layers."""
        if not self.noisy:
            return
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def to(self, device):
        """Move model to device and ensure float32."""
        super().to(device)
        return self.float()

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            x: Input tensor (batch, 4, 84, 84) in float32 [0, 1].

        Returns:
            Dict with:
                - 'q_values': (batch, num_actions)
                - 'log_probs': (batch, num_actions, num_atoms)
                - 'conv_output': (batch, 64, 7, 7)
        """
        # Convolutional encoder
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        conv_output = torch.relu(self.conv3(x))

        x = conv_output.reshape(conv_output.size(0), -1)  # (B, 3136)

        if self.dueling:
            # Value stream
            v = self.drop(torch.relu(self.value_fc(x)))
            v = self.value_head(v)  # (B, num_atoms)

            # Advantage stream
            a = self.drop(torch.relu(self.advantage_fc(x)))
            a = self.advantage_head(a)  # (B, num_atoms * A)
            a = a.view(-1, self.num_actions, self.num_atoms)  # (B, A, atoms)

            # Dueling aggregation per atom (Wang et al. 2016, Eq. 9)
            v = v.unsqueeze(1)  # (B, 1, atoms)
            q_atoms = v + a - a.mean(dim=1, keepdim=True)  # (B, A, atoms)
        else:
            x = self.drop(torch.relu(self.fc(x)))
            q_atoms = self.q_head(x)  # (B, atoms * A)
            q_atoms = q_atoms.view(
                -1, self.num_actions, self.num_atoms
            )  # (B, A, atoms)

        # Softmax over atoms -> probability distribution per action
        log_probs = torch.log_softmax(q_atoms, dim=2)  # (B, A, atoms)
        probs = log_probs.exp()

        # Q-values: Q(s,a) = sum_i z_i * p_i(s,a)
        q_values = (probs * self.support).sum(dim=2)  # (B, A)

        return {
            "q_values": q_values,
            "log_probs": log_probs,
            "conv_output": conv_output,
        }
