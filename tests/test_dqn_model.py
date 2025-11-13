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


def test_dqn_mse_backward():
    """Test backward pass with MSE loss across multiple action sizes."""
    # Test game-specific action space sizes
    action_configs = [
        ('Breakout', 4),
        ('Pong', 6),
        ('BeamRider', 9)
    ]

    for game_name, num_actions in action_configs:
        model = DQN(num_actions=num_actions)
        batch_size = 2

        # Create input
        x = torch.rand(batch_size, 4, 84, 84)

        # Forward pass
        output = model(x)
        q_values = output['q_values']

        # Create target Q-values
        target_q = torch.rand(batch_size, num_actions)

        # MSE loss
        loss = torch.nn.functional.mse_loss(q_values, target_q)

        # Backward pass
        loss.backward()

        # Check loss is finite
        assert torch.isfinite(loss), f"{game_name}: Loss is not finite"

        # Check all gradients exist and are finite
        for name, param in model.named_parameters():
            assert param.grad is not None, f"{game_name}: No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"{game_name}: Non-finite gradient in {name}"

        # Clear gradients for next iteration
        model.zero_grad()


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


def test_dqn_initialization():
    """Test DQN weights are properly initialized with Kaiming."""
    model = DQN(num_actions=6)

    # Check that weights are initialized (not all zeros or ones)
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Kaiming init should produce non-zero weights
            assert not torch.all(param == 0), f"{name} is all zeros"
            assert not torch.all(param == 1), f"{name} is all ones"
            # Check reasonable variance (Kaiming produces std around sqrt(2/fan))
            assert param.std() > 0.01, f"{name} has very low variance"
            assert param.std() < 2.0, f"{name} has very high variance"
        elif 'bias' in name:
            # Biases should be initialized to zero
            assert torch.all(param == 0), f"{name} bias not zero-initialized"


def test_dqn_dtype():
    """Test DQN parameters are float32."""
    model = DQN(num_actions=4)

    # All parameters should be float32
    for name, param in model.named_parameters():
        assert param.dtype == torch.float32, f"{name} is not float32: {param.dtype}"


def test_dqn_device_transfer():
    """Test DQN can be moved to device and maintains float32."""
    model = DQN(num_actions=6)

    # Test moving to CPU explicitly
    model_cpu = model.to('cpu')
    assert model_cpu is model  # Should return self

    # Check all params are on CPU and float32
    for param in model_cpu.parameters():
        assert param.device.type == 'cpu'
        assert param.dtype == torch.float32

    # Test with input
    x = torch.rand(1, 4, 84, 84)
    output = model_cpu(x)
    assert output['q_values'].device.type == 'cpu'


if __name__ == "__main__":
    # Run tests manually
    print("Running DQN model tests...")

    test_dqn_output_shape()
    print("✓ Output shape test passed (Breakout=4, Pong=6, BeamRider=9, etc.)")

    test_dqn_no_nans()
    print("✓ No NaNs/Infs test passed")

    test_dqn_gradient_flow()
    print("✓ Gradient flow test passed")

    test_dqn_mse_backward()
    print("✓ MSE backward test passed (Breakout, Pong, BeamRider)")

    test_dqn_from_env()
    print("✓ from_env test passed")

    test_dqn_channels_first()
    print("✓ Channels-first test passed")

    test_dqn_initialization()
    print("✓ Kaiming initialization test passed")

    test_dqn_dtype()
    print("✓ Float32 dtype test passed")

    test_dqn_device_transfer()
    print("✓ Device transfer test passed")

    print("\nAll tests passed! ✓")
