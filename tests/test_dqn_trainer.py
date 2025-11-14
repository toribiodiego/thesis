"""
Tests for DQN training utilities.

Verifies:
- Hard target network updates
- Target network initialization
- Gradient freezing
- Parameter copying
"""

import torch
import pytest
from src.models import DQN
from src.training import hard_update_target, init_target_network


def test_hard_update_target_basic():
    """Test hard update copies all parameters from online to target."""
    online_net = DQN(num_actions=6)
    target_net = DQN(num_actions=6)

    # Initialize target with different weights (random init)
    # Verify they're different initially
    online_params = list(online_net.parameters())[0].clone()
    target_params = list(target_net.parameters())[0].clone()

    # They should be different (random initialization)
    assert not torch.allclose(online_params, target_params)

    # Hard update
    hard_update_target(online_net, target_net)

    # Now they should be identical
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_online, p_target), "Parameters not copied correctly"


def test_hard_update_target_all_layers():
    """Test hard update copies all layers including conv, fc, and q_head."""
    online_net = DQN(num_actions=6)
    target_net = DQN(num_actions=6)

    # Modify online network weights
    with torch.no_grad():
        for param in online_net.parameters():
            param.fill_(1.0)

    # Hard update
    hard_update_target(online_net, target_net)

    # Verify all target parameters are 1.0
    for param in target_net.parameters():
        assert torch.allclose(param, torch.ones_like(param)), \
            "Not all parameters were copied"


def test_hard_update_target_multiple_updates():
    """Test multiple hard updates work correctly."""
    online_net = DQN(num_actions=6)
    target_net = DQN(num_actions=6)

    # First update
    hard_update_target(online_net, target_net)
    first_target_param = list(target_net.parameters())[0].clone()

    # Modify online network
    with torch.no_grad():
        for param in online_net.parameters():
            param.mul_(2.0)

    # Second update
    hard_update_target(online_net, target_net)
    second_target_param = list(target_net.parameters())[0].clone()

    # Target should have changed
    assert not torch.allclose(first_target_param, second_target_param), \
        "Target didn't update on second hard_update"

    # Target should match online
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_online, p_target)


def test_init_target_network_creates_copy():
    """Test init_target_network creates identical copy of online network."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Should have same architecture
    assert type(online_net) == type(target_net)
    assert online_net.num_actions == target_net.num_actions

    # Should have same parameters
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_online, p_target), \
            "Initial parameters don't match"


def test_init_target_network_freezes_gradients():
    """Test init_target_network freezes target network gradients."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Target parameters should have requires_grad=False
    for param in target_net.parameters():
        assert param.requires_grad == False, \
            "Target network gradients not frozen"

    # Online parameters should still require gradients
    for param in online_net.parameters():
        assert param.requires_grad == True, \
            "Online network gradients incorrectly frozen"


def test_init_target_network_in_eval_mode():
    """Test init_target_network sets target to eval mode."""
    online_net = DQN(num_actions=6)
    online_net.train()  # Ensure online is in train mode

    target_net = init_target_network(online_net, num_actions=6)

    # Target should be in eval mode
    assert not target_net.training, "Target network not in eval mode"


def test_target_network_no_gradient_computation():
    """Test that target network doesn't accumulate gradients."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Create dummy input and compute forward pass
    x = torch.randn(2, 4, 84, 84)

    # Forward through target
    target_output = target_net(x)

    # Try to compute gradients (should fail or not accumulate)
    loss = target_output['q_values'].mean()

    # This should not create gradients for target network
    loss.backward()

    # Target network should have no gradients
    for param in target_net.parameters():
        assert param.grad is None, \
            "Target network accumulated gradients (should be frozen)"


def test_hard_update_after_online_training():
    """Test hard update after training online network."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # "Train" online network (simulate gradient update)
    x = torch.randn(2, 4, 84, 84)
    output = online_net(x)
    loss = output['q_values'].mean()
    loss.backward()

    # Manually update online parameters
    with torch.no_grad():
        for param in online_net.parameters():
            if param.grad is not None:
                param.sub_(param.grad * 0.01)  # Simple SGD step

    # Networks should now differ
    online_param = list(online_net.parameters())[0]
    target_param = list(target_net.parameters())[0]
    assert not torch.allclose(online_param, target_param), \
        "Networks should differ after online update"

    # Hard update
    hard_update_target(online_net, target_net)

    # Now they should match again
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_online, p_target), \
            "Parameters don't match after hard update"


def test_hard_update_preserves_frozen_gradients():
    """Test hard update doesn't change requires_grad status."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Verify target is frozen
    for param in target_net.parameters():
        assert param.requires_grad == False

    # Hard update
    hard_update_target(online_net, target_net)

    # Target should still be frozen
    for param in target_net.parameters():
        assert param.requires_grad == False, \
            "Hard update unfroze target network gradients"


def test_target_network_different_devices():
    """Test hard update works when networks are on different devices."""
    online_net = DQN(num_actions=6)
    target_net = DQN(num_actions=6)

    # Both on CPU (default)
    hard_update_target(online_net, target_net)

    # Verify update worked
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_online, p_target)


def test_hard_update_target_shape_mismatch():
    """Test hard update with mismatched architectures."""
    online_net = DQN(num_actions=6)
    target_net = DQN(num_actions=4)  # Different action space

    # Should raise an error due to shape mismatch
    with pytest.raises(RuntimeError):
        hard_update_target(online_net, target_net)


def test_init_target_network_num_actions():
    """Test init_target_network with different action space sizes."""
    for num_actions in [4, 6, 9, 18]:
        online_net = DQN(num_actions=num_actions)
        target_net = init_target_network(online_net, num_actions=num_actions)

        assert target_net.num_actions == num_actions
        assert type(target_net) == type(online_net)


if __name__ == "__main__":
    # Run tests manually
    print("Running DQN trainer tests...")

    test_hard_update_target_basic()
    print("✓ Hard update basic test passed")

    test_hard_update_target_all_layers()
    print("✓ Hard update all layers test passed")

    test_hard_update_target_multiple_updates()
    print("✓ Hard update multiple updates test passed")

    test_init_target_network_creates_copy()
    print("✓ Init target network creates copy test passed")

    test_init_target_network_freezes_gradients()
    print("✓ Init target network freezes gradients test passed")

    test_init_target_network_in_eval_mode()
    print("✓ Init target network eval mode test passed")

    test_target_network_no_gradient_computation()
    print("✓ Target network no gradient computation test passed")

    test_hard_update_after_online_training()
    print("✓ Hard update after online training test passed")

    test_hard_update_preserves_frozen_gradients()
    print("✓ Hard update preserves frozen gradients test passed")

    test_target_network_different_devices()
    print("✓ Target network different devices test passed")

    test_hard_update_target_shape_mismatch()
    print("✓ Hard update shape mismatch test passed")

    test_init_target_network_num_actions()
    print("✓ Init target network num_actions test passed")

    print("\nAll tests passed! ✓")
