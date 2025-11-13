"""
Tests for DQN model architecture.

Verifies:
- Correct output shapes for different action space sizes
- Forward pass without NaNs/Infs
- Gradient flow through the network
- Channels-first input format
"""

import torch
import pytest
from src.models.dqn import DQN


def test_dqn_output_shape():
    """Test DQN produces correct output shapes."""
    # Test different action space sizes
    action_spaces = [4, 6, 9, 18]  # Breakout, Pong, BeamRider, typical sizes

    for num_actions in action_spaces:
        model = DQN(num_actions=num_actions)
        batch_size = 2

        # Create dummy input (batch=2, channels=4, height=84, width=84)
        x = torch.randn(batch_size, 4, 84, 84)

        # Forward pass
        output = model(x)

        # Check output is a dict
        assert isinstance(output, dict)
        assert 'q_values' in output
        assert 'features' in output

        # Check q_values shape
        assert output['q_values'].shape == (batch_size, num_actions), \
            f"Expected q_values shape ({batch_size}, {num_actions}), got {output['q_values'].shape}"

        # Check features shape
        assert output['features'].shape == (batch_size, 256), \
            f"Expected features shape ({batch_size}, 256), got {output['features'].shape}"


def test_dqn_no_nans():
    """Test DQN forward pass produces no NaNs or Infs."""
    model = DQN(num_actions=6)

    # Random input normalized to [0, 1]
    x = torch.rand(2, 4, 84, 84)

    output = model(x)

    # Check for NaNs and Infs
    assert not torch.isnan(output['q_values']).any(), "Q-values contain NaNs"
    assert not torch.isinf(output['q_values']).any(), "Q-values contain Infs"
    assert not torch.isnan(output['features']).any(), "Features contain NaNs"
    assert not torch.isinf(output['features']).any(), "Features contain Infs"


def test_dqn_gradient_flow():
    """Test gradients flow through the network."""
    model = DQN(num_actions=4)

    # Create input and target
    x = torch.rand(2, 4, 84, 84, requires_grad=True)
    output = model(x)

    # Create dummy loss and backprop
    loss = output['q_values'].mean()
    loss.backward()

    # Check gradients exist for all parameters
    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
        assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"


def test_dqn_from_env():
    """Test DQN can be created from environment."""
    # Mock environment object
    class MockEnv:
        class ActionSpace:
            n = 6
        action_space = ActionSpace()

    env = MockEnv()
    model = DQN.from_env(env)

    assert model.num_actions == 6

    # Verify it works
    x = torch.rand(1, 4, 84, 84)
    output = model(x)
    assert output['q_values'].shape == (1, 6)


def test_dqn_channels_first():
    """Test DQN expects channels-first format (B, C, H, W)."""
    model = DQN(num_actions=4)

    # Correct format: (batch, channels, height, width)
    x_correct = torch.rand(2, 4, 84, 84)
    output = model(x_correct)
    assert output['q_values'].shape == (2, 4)

    # Wrong format should produce wrong output
    x_wrong = torch.rand(2, 84, 84, 4)  # channels last
    with pytest.raises(RuntimeError):
        # This should fail because conv expects channels=4, not channels=84
        model(x_wrong)


if __name__ == "__main__":
    # Run tests manually
    print("Running DQN model tests...")

    test_dqn_output_shape()
    print("✓ Output shape test passed")

    test_dqn_no_nans()
    print("✓ No NaNs/Infs test passed")

    test_dqn_gradient_flow()
    print("✓ Gradient flow test passed")

    test_dqn_from_env()
    print("✓ from_env test passed")

    test_dqn_channels_first()
    print("✓ Channels-first test passed")

    print("\nAll tests passed! ✓")
