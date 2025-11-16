"""
DQN model architecture from Mnih et al. 2013.

Implements the Nature DQN CNN architecture:
- Conv1: 16 filters, 8×8, stride 4, ReLU
- Conv2: 32 filters, 4×4, stride 2, ReLU
- FC: 256 units, ReLU
- Output: Linear layer with |A| units (number of actions)

Input: (batch, 4, 84, 84) - 4 stacked grayscale frames
Output: (batch, num_actions) - Q-values for each action
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    Deep Q-Network with convolutional architecture.

    Architecture:
        Conv1(16, 8×8, stride=4) → ReLU
        Conv2(32, 4×4, stride=2) → ReLU
        Flatten
        FC(256) → ReLU
        Linear(num_actions)

    Args:
        num_actions: Number of discrete actions in the environment

    Input shape: (batch, 4, 84, 84) - channels-first format
    Output: Dict with 'q_values' (batch, num_actions) and 'features' (batch, 256)
    """

    def __init__(self, num_actions: int):
        super().__init__()
        self.num_actions = num_actions

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=4, out_channels=16, kernel_size=8, stride=4, padding=0
        )
        self.conv2 = nn.Conv2d(
            in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=0
        )

        # Compute size after convolutions to determine FC input size
        # Input: (4, 84, 84)
        # After conv1: (16, 20, 20) -> (84 - 8) / 4 + 1 = 20
        # After conv2: (32, 9, 9) -> (20 - 4) / 2 + 1 = 9
        conv_output_size = 32 * 9 * 9

        # Fully connected layers
        self.fc = nn.Linear(conv_output_size, 256)
        self.q_head = nn.Linear(256, num_actions)

        # Initialize weights with Kaiming normal (He initialization)
        # Suitable for ReLU activations
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights using Kaiming normal initialization.

        Uses fan_out mode for conv and linear layers with ReLU activations.
        Biases are initialized to zero.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Kaiming normal initialization for layers with ReLU
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def to(self, device):
        """
        Move model to specified device and ensure float32 dtype.

        Args:
            device: torch.device or string ('cuda', 'cpu')

        Returns:
            Self for chaining
        """
        super().to(device)
        # Ensure all parameters are float32
        return self.float()

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 4, 84, 84) in float32 [0, 1]

        Returns:
            Dict containing:
                - 'q_values': Q-values for each action (batch, num_actions)
                - 'features': Feature vector before Q-head (batch, 256)
        """
        # Convolutional layers with ReLU
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))

        # Flatten spatial dimensions
        x = x.reshape(x.size(0), -1)

        # Fully connected layer with ReLU
        features = torch.relu(self.fc(x))

        # Q-value head (no activation)
        q_values = self.q_head(features)

        return {"q_values": q_values, "features": features}

    def validate_output_shape(self, batch_size: int = 2):
        """
        Validate model produces correct output shapes.

        Args:
            batch_size: Batch size for dummy input

        Raises:
            AssertionError: If output shapes are incorrect
        """
        self.eval()
        with torch.no_grad():
            x = torch.randn(batch_size, 4, 84, 84)
            output = self(x)

            # Check q_values shape
            expected_q_shape = (batch_size, self.num_actions)
            actual_q_shape = output["q_values"].shape
            assert (
                actual_q_shape == expected_q_shape
            ), f"Expected q_values shape {expected_q_shape}, got {actual_q_shape}"

            # Check features shape
            expected_feat_shape = (batch_size, 256)
            actual_feat_shape = output["features"].shape
            assert (
                actual_feat_shape == expected_feat_shape
            ), f"Expected features shape {expected_feat_shape}, got {actual_feat_shape}"

    def save_checkpoint(self, path: str, meta: dict = None):
        """
        Save model checkpoint with metadata.

        Args:
            path: Path to save checkpoint
            meta: Optional metadata dict (e.g., step, episode, config)

        Example:
            >>> model.save_checkpoint('checkpoints/model_10k.pt',
            ...                       {'step': 10000, 'episode': 100})
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_actions": self.num_actions,
        }

        # Add metadata if provided
        if meta is not None:
            checkpoint["meta"] = meta

        torch.save(checkpoint, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: str = "cpu", strict: bool = True):
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model on ('cpu', 'cuda', etc.)
            strict: Whether to strictly enforce state_dict key matching

        Returns:
            Tuple of (model, meta) where meta is the saved metadata dict

        Example:
            >>> model, meta = DQN.load_checkpoint('checkpoints/model_10k.pt')
            >>> print(f"Loaded from step {meta['step']}")
        """
        checkpoint = torch.load(path, map_location=device)

        # Create model with correct action space size
        num_actions = checkpoint["num_actions"]
        model = cls(num_actions)

        # Load state dict with strict key matching
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # Move to device and ensure float32
        model = model.to(device)

        # Extract metadata
        meta = checkpoint.get("meta", {})

        return model, meta

    @classmethod
    def from_env(cls, env):
        """
        Create DQN model from environment.

        Args:
            env: Gymnasium environment with discrete action space

        Returns:
            DQN model initialized for the environment
        """
        num_actions = env.action_space.n
        return cls(num_actions)
