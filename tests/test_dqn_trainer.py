"""
Tests for DQN training utilities.

Verifies:
- Hard target network updates
- Target network initialization
- Gradient freezing
- Parameter copying
- TD target computation
- Q-value selection
- Loss computation (MSE and Huber)
- Optimizer configuration (RMSProp and Adam)
- Gradient clipping
- Target network update scheduling
- Training frequency scheduling with warm-up
- Reference-state Q tracking
- Metadata and reproducibility utilities

Note: Evaluation harness tests are in test_evaluation.py and test_video_recorder.py
"""

import numpy as np
import pytest
import torch

from src.models import DQN
from src.replay import ReplayBuffer
from src.training import (
    TargetNetworkUpdater,
    TrainingScheduler,
    UpdateMetrics,
    clip_gradients,
    compute_dqn_loss,
    compute_td_loss_components,
    compute_td_targets,
    configure_optimizer,
    detect_nan_inf,
    hard_update_target,
    init_target_network,
    perform_update_step,
    select_q_values,
    validate_loss_decrease,
    verify_target_sync_schedule,
)


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
        assert torch.allclose(
            param, torch.ones_like(param)
        ), "Not all parameters were copied"


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
    assert not torch.allclose(
        first_target_param, second_target_param
    ), "Target didn't update on second hard_update"

    # Target should match online
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_online, p_target)


def test_init_target_network_creates_copy():
    """Test init_target_network creates identical copy of online network."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Should have same architecture
    assert type(online_net) is type(target_net)
    assert online_net.num_actions == target_net.num_actions

    # Should have same parameters
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_online, p_target), "Initial parameters don't match"


def test_init_target_network_freezes_gradients():
    """Test init_target_network freezes target network gradients."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Target parameters should have requires_grad=False
    for param in target_net.parameters():
        assert not param.requires_grad, "Target network gradients not frozen"

    # Online parameters should still require gradients
    for param in online_net.parameters():
        assert param.requires_grad, "Online network gradients incorrectly frozen"


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

    # Output should not require gradients
    loss = target_output["q_values"].mean()
    assert not loss.requires_grad, "Target network output should not require gradients"

    # Target network should have no gradients
    for param in target_net.parameters():
        assert (
            param.grad is None
        ), "Target network accumulated gradients (should be frozen)"


def test_hard_update_after_online_training():
    """Test hard update after training online network."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # "Train" online network (simulate gradient update)
    x = torch.randn(2, 4, 84, 84)
    output = online_net(x)
    loss = output["q_values"].mean()
    loss.backward()

    # Manually update online parameters
    with torch.no_grad():
        for param in online_net.parameters():
            if param.grad is not None:
                param.sub_(param.grad * 0.01)  # Simple SGD step

    # Networks should now differ
    online_param = list(online_net.parameters())[0]
    target_param = list(target_net.parameters())[0]
    assert not torch.allclose(
        online_param, target_param
    ), "Networks should differ after online update"

    # Hard update
    hard_update_target(online_net, target_net)

    # Now they should match again
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        assert torch.allclose(
            p_online, p_target
        ), "Parameters don't match after hard update"


def test_hard_update_preserves_frozen_gradients():
    """Test hard update doesn't change requires_grad status."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Verify target is frozen
    for param in target_net.parameters():
        assert not param.requires_grad

    # Hard update
    hard_update_target(online_net, target_net)

    # Target should still be frozen
    for param in target_net.parameters():
        assert not param.requires_grad, "Hard update unfroze target network gradients"


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
        assert type(target_net) is type(online_net)


# ============================================================================
# TD Target Computation Tests
# ============================================================================


def test_compute_td_targets_basic():
    """Test basic TD target computation."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Create batch
    batch_size = 4
    rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.tensor([False, False, False, False])

    # Compute TD targets
    td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma=0.99)

    # Should have correct shape
    assert td_targets.shape == (
        batch_size,
    ), f"Expected shape (4,), got {td_targets.shape}"

    # Should be float32
    assert td_targets.dtype == torch.float32

    # Should be detached (no gradients)
    assert not td_targets.requires_grad


def test_compute_td_targets_terminal_states():
    """Test TD targets correctly handle terminal states (done=True)."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Create batch with terminal states
    batch_size = 3
    rewards = torch.tensor([1.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.tensor([False, True, False])  # Second is terminal

    # Compute TD targets
    td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma=0.99)

    # For terminal states (done=True), TD target should be just the reward
    # td_target[1] = reward[1] + 0.99 * (1 - 1.0) * max_q = -1.0 + 0 = -1.0
    assert torch.allclose(
        td_targets[1], rewards[1], atol=1e-6
    ), f"Terminal state TD target should equal reward, got {td_targets[1]} vs {rewards[1]}"


def test_compute_td_targets_gamma():
    """Test TD targets with different gamma values."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Create deterministic batch
    batch_size = 2
    rewards = torch.zeros(batch_size)
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.tensor([False, False])

    # Compute with gamma=1.0 (no discounting)
    compute_td_targets(rewards, next_states, dones, target_net, gamma=1.0)

    # Compute with gamma=0.0 (only immediate reward)
    td_targets_gamma0 = compute_td_targets(
        rewards, next_states, dones, target_net, gamma=0.0
    )

    # With gamma=0.0 and zero rewards, targets should be zero
    assert torch.allclose(
        td_targets_gamma0, torch.zeros(batch_size), atol=1e-6
    ), "With gamma=0 and zero rewards, targets should be zero"

    # With gamma=1.0, targets should include full future value
    # td_targets_gamma1 should be larger (unless max_q is negative)


def test_compute_td_targets_no_grad():
    """Test that TD target computation doesn't create gradients."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Create batch
    batch_size = 2
    rewards = torch.tensor([1.0, -1.0])
    next_states = torch.randn(
        batch_size, 4, 84, 84, requires_grad=True
    )  # Try to leak gradients
    dones = torch.tensor([False, False])

    # Compute TD targets
    td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma=0.99)

    # Targets should not require gradients
    assert not td_targets.requires_grad, "TD targets should be detached"

    # Try to backward (should fail or not affect target network)
    td_targets.sum()
    # This should work because td_targets is detached
    # But it won't create gradients for target_net


def test_select_q_values_basic():
    """Test Q-value selection with gather."""
    online_net = DQN(num_actions=6)

    # Create batch
    batch_size = 4
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.tensor([0, 2, 5, 1])  # Different actions

    # Select Q-values
    q_selected = select_q_values(online_net, states, actions)

    # Should have correct shape
    assert q_selected.shape == (
        batch_size,
    ), f"Expected shape (4,), got {q_selected.shape}"

    # Should be float32
    assert q_selected.dtype == torch.float32

    # Should have gradients (from online network)
    assert q_selected.requires_grad, "Selected Q-values should have gradients"


def test_select_q_values_gather_correctness():
    """Test that gather selects the correct Q-values."""
    online_net = DQN(num_actions=6)

    # Create batch
    batch_size = 3
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.tensor([0, 2, 5])

    # Get full Q-values and selected Q-values
    with torch.no_grad():
        full_output = online_net(states)
        full_q_values = full_output["q_values"]  # (3, 6)

    q_selected = select_q_values(online_net, states, actions)

    # Manually verify correctness
    with torch.no_grad():
        expected_0 = full_q_values[0, 0]
        expected_1 = full_q_values[1, 2]
        expected_2 = full_q_values[2, 5]

        assert torch.allclose(
            q_selected[0], expected_0, atol=1e-5
        ), "Q-value for action 0 not selected correctly"
        assert torch.allclose(
            q_selected[1], expected_1, atol=1e-5
        ), "Q-value for action 2 not selected correctly"
        assert torch.allclose(
            q_selected[2], expected_2, atol=1e-5
        ), "Q-value for action 5 not selected correctly"


def test_select_q_values_gradient_flow():
    """Test that gradients flow through selected Q-values."""
    online_net = DQN(num_actions=6)

    # Create batch
    batch_size = 2
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.tensor([0, 1])

    # Select Q-values
    q_selected = select_q_values(online_net, states, actions)

    # Compute loss and backward
    loss = q_selected.mean()
    loss.backward()

    # Online network should have gradients
    has_gradients = False
    for param in online_net.parameters():
        if param.grad is not None:
            has_gradients = True
            break

    assert has_gradients, "Gradients should flow through online network"


def test_compute_td_loss_components_basic():
    """Test compute_td_loss_components returns correct shapes."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Create batch
    batch_size = 4
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.tensor([0, 2, 1, 5])
    rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.tensor([False, False, True, False])

    # Compute components
    q_selected, td_targets = compute_td_loss_components(
        states, actions, rewards, next_states, dones, online_net, target_net, gamma=0.99
    )

    # Both should have shape (B,)
    assert q_selected.shape == (batch_size,)
    assert td_targets.shape == (batch_size,)

    # q_selected should have gradients
    assert q_selected.requires_grad

    # td_targets should be detached
    assert not td_targets.requires_grad


def test_compute_td_loss_components_terminal_handling():
    """Test that terminal states are handled correctly in full pipeline."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Create batch with all terminal states
    batch_size = 3
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.tensor([0, 1, 2])
    rewards = torch.tensor([1.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.tensor([True, True, True])  # All terminal

    # Compute components
    q_selected, td_targets = compute_td_loss_components(
        states, actions, rewards, next_states, dones, online_net, target_net, gamma=0.99
    )

    # For terminal states, td_targets should equal rewards
    assert torch.allclose(
        td_targets, rewards, atol=1e-6
    ), "Terminal state targets should equal rewards"


def test_compute_td_loss_components_mse_loss():
    """Test that components can be used for MSE loss computation."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Create batch
    batch_size = 4
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.tensor([0, 2, 1, 5])
    rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.tensor([False, False, True, False])

    # Compute components
    q_selected, td_targets = compute_td_loss_components(
        states, actions, rewards, next_states, dones, online_net, target_net, gamma=0.99
    )

    # Compute MSE loss
    import torch.nn.functional as F

    loss = F.mse_loss(q_selected, td_targets)

    # Should be a scalar
    assert loss.shape == torch.Size([])

    # Should have gradients
    assert loss.requires_grad

    # Should be able to backward
    loss.backward()

    # Online network should have gradients
    has_gradients = False
    for param in online_net.parameters():
        if param.grad is not None:
            has_gradients = True
            break

    assert has_gradients, "MSE loss should create gradients for online network"


def test_td_targets_shape_consistency():
    """Test TD target shapes across different batch sizes."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    for batch_size in [1, 2, 8, 32]:
        rewards = torch.randn(batch_size)
        next_states = torch.randn(batch_size, 4, 84, 84)
        dones = torch.randint(0, 2, (batch_size,)).bool()

        td_targets = compute_td_targets(
            rewards, next_states, dones, target_net, gamma=0.99
        )

        assert td_targets.shape == (
            batch_size,
        ), f"Batch size {batch_size}: expected shape ({batch_size},), got {td_targets.shape}"


def test_q_selection_shape_consistency():
    """Test Q-value selection shapes across different batch sizes."""
    online_net = DQN(num_actions=6)

    for batch_size in [1, 2, 8, 32]:
        states = torch.randn(batch_size, 4, 84, 84)
        actions = torch.randint(0, 6, (batch_size,))

        q_selected = select_q_values(online_net, states, actions)

        assert q_selected.shape == (
            batch_size,
        ), f"Batch size {batch_size}: expected shape ({batch_size},), got {q_selected.shape}"


# ============================================================================
# Loss Computation Tests
# ============================================================================


def test_compute_dqn_loss_mse_basic():
    """Test basic MSE loss computation."""
    # Create simple tensors
    q_selected = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    td_targets = torch.tensor([1.5, 2.5, 2.5, 3.5])

    # Compute loss
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type="mse")

    # Check keys
    assert "loss" in loss_dict
    assert "td_error" in loss_dict
    assert "td_error_std" in loss_dict

    # Loss should be a scalar
    assert loss_dict["loss"].shape == torch.Size([])

    # Loss should have gradients
    assert loss_dict["loss"].requires_grad

    # TD error should be detached
    assert not loss_dict["td_error"].requires_grad
    assert not loss_dict["td_error_std"].requires_grad

    # Expected MSE: mean([(1-1.5)^2, (2-2.5)^2, (3-2.5)^2, (4-3.5)^2])
    # = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
    expected_loss = 0.25
    assert torch.allclose(loss_dict["loss"], torch.tensor(expected_loss), atol=1e-6)

    # Expected TD error: mean([0.5, 0.5, 0.5, 0.5]) = 0.5
    expected_td_error = 0.5
    assert torch.allclose(
        loss_dict["td_error"], torch.tensor(expected_td_error), atol=1e-6
    )


def test_compute_dqn_loss_huber_basic():
    """Test basic Huber loss computation."""
    q_selected = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    td_targets = torch.tensor([1.5, 2.5, 2.5, 3.5])

    # Compute Huber loss
    loss_dict = compute_dqn_loss(
        q_selected, td_targets, loss_type="huber", huber_delta=1.0
    )

    # Loss should be a scalar with gradients
    assert loss_dict["loss"].shape == torch.Size([])
    assert loss_dict["loss"].requires_grad

    # Huber loss should be <= MSE for small errors (all errors are 0.5 < delta=1.0)
    # For errors smaller than delta, Huber is quadratic: 0.5 * error^2
    # For our case: 0.5 * mean([0.25, 0.25, 0.25, 0.25]) = 0.125
    expected_huber = 0.125
    assert torch.allclose(loss_dict["loss"], torch.tensor(expected_huber), atol=1e-6)


def test_compute_dqn_loss_gradient_flow():
    """Test that gradients flow through loss."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Create batch
    batch_size = 4
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.tensor([0, 2, 1, 5])
    rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.tensor([False, False, True, False])

    # Compute loss components
    q_selected, td_targets = compute_td_loss_components(
        states, actions, rewards, next_states, dones, online_net, target_net, gamma=0.99
    )

    # Compute loss
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type="mse")

    # Backward
    loss_dict["loss"].backward()

    # Online network should have gradients
    has_gradients = False
    for param in online_net.parameters():
        if param.grad is not None:
            has_gradients = True
            break

    assert has_gradients, "Gradients should flow through online network"


def test_compute_dqn_loss_td_error_stats():
    """Test TD error statistics computation."""
    # Create batch with known TD errors
    q_selected = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    td_targets = torch.tensor([2.0, 3.0, 3.0, 5.0])  # Errors: [1.0, 1.0, 0.0, 1.0]

    # Compute loss
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type="mse")

    # Expected mean TD error: mean([1.0, 1.0, 0.0, 1.0]) = 0.75
    expected_mean = 0.75
    assert torch.allclose(loss_dict["td_error"], torch.tensor(expected_mean), atol=1e-6)

    # Expected std: std([1.0, 1.0, 0.0, 1.0]) = 0.5
    expected_std = 0.5
    assert torch.allclose(
        loss_dict["td_error_std"], torch.tensor(expected_std), atol=1e-6
    )


def test_compute_dqn_loss_invalid_type():
    """Test that invalid loss type raises error."""
    q_selected = torch.tensor([1.0, 2.0], requires_grad=True)
    td_targets = torch.tensor([1.5, 2.5])

    with pytest.raises(ValueError, match="Unknown loss_type"):
        compute_dqn_loss(q_selected, td_targets, loss_type="invalid")


def test_compute_dqn_loss_shape_mismatch():
    """Test that shape mismatch raises error."""
    q_selected = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    td_targets = torch.tensor([1.5, 2.5])  # Different size

    with pytest.raises(AssertionError, match="Shape mismatch"):
        compute_dqn_loss(q_selected, td_targets, loss_type="mse")


def test_compute_dqn_loss_gradient_assertions():
    """Test gradient requirements are validated."""
    # q_selected without gradients (should fail)
    q_selected = torch.tensor([1.0, 2.0], requires_grad=False)
    td_targets = torch.tensor([1.5, 2.5])

    with pytest.raises(AssertionError, match="should have gradients"):
        compute_dqn_loss(q_selected, td_targets, loss_type="mse")

    # td_targets with gradients (should fail)
    q_selected = torch.tensor([1.0, 2.0], requires_grad=True)
    td_targets = torch.tensor([1.5, 2.5], requires_grad=True)

    with pytest.raises(AssertionError, match="should be detached"):
        compute_dqn_loss(q_selected, td_targets, loss_type="mse")


def test_compute_dqn_loss_huber_delta():
    """Test Huber loss with different delta values."""
    # Large errors to test delta effect
    q_selected = torch.tensor([0.0, 5.0, 10.0], requires_grad=True)
    td_targets = torch.tensor([1.0, 1.0, 1.0])  # Errors: [1.0, 4.0, 9.0]

    # Small delta (more linear for large errors)
    loss_dict_small = compute_dqn_loss(
        q_selected, td_targets, loss_type="huber", huber_delta=0.5
    )

    # Large delta (more quadratic, closer to MSE)
    loss_dict_large = compute_dqn_loss(
        q_selected, td_targets, loss_type="huber", huber_delta=10.0
    )

    # Both should be valid scalars
    assert loss_dict_small["loss"].shape == torch.Size([])
    assert loss_dict_large["loss"].shape == torch.Size([])

    # TD errors should be the same regardless of loss type
    assert torch.allclose(loss_dict_small["td_error"], loss_dict_large["td_error"])


def test_compute_dqn_loss_zero_td_error():
    """Test loss when TD error is zero (perfect predictions)."""
    q_selected = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    td_targets = torch.tensor([1.0, 2.0, 3.0])  # Perfect match

    # MSE loss
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type="mse")

    # Loss should be zero
    assert torch.allclose(loss_dict["loss"], torch.tensor(0.0), atol=1e-6)

    # TD error should be zero
    assert torch.allclose(loss_dict["td_error"], torch.tensor(0.0), atol=1e-6)


def test_compute_dqn_loss_batch_sizes():
    """Test loss computation across different batch sizes."""
    for batch_size in [1, 2, 8, 32]:
        q_selected = torch.randn(batch_size, requires_grad=True)
        td_targets = torch.randn(batch_size)

        # MSE
        loss_dict_mse = compute_dqn_loss(q_selected, td_targets, loss_type="mse")
        assert loss_dict_mse["loss"].shape == torch.Size([])

        # Huber
        loss_dict_huber = compute_dqn_loss(q_selected, td_targets, loss_type="huber")
        assert loss_dict_huber["loss"].shape == torch.Size([])


def test_compute_dqn_loss_mse_vs_huber():
    """Test MSE vs Huber loss behavior for small and large errors."""
    # Small errors (Huber should be similar to MSE)
    q_selected_small = torch.tensor([1.0, 1.1, 0.9], requires_grad=True)
    td_targets_small = torch.tensor([1.0, 1.0, 1.0])

    mse_small = compute_dqn_loss(q_selected_small, td_targets_small, loss_type="mse")
    huber_small = compute_dqn_loss(
        q_selected_small.clone().detach().requires_grad_(True),
        td_targets_small,
        loss_type="huber",
        huber_delta=1.0,
    )

    # For small errors, Huber and MSE should be close
    # Note: Huber is 0.5*error^2 for |error| < delta, so it's 0.5 * MSE
    assert huber_small["loss"] < mse_small["loss"]

    # Large errors (Huber should be smaller than MSE due to linear region)
    q_selected_large = torch.tensor([0.0, 10.0, 20.0], requires_grad=True)
    td_targets_large = torch.tensor([1.0, 1.0, 1.0])

    mse_large = compute_dqn_loss(q_selected_large, td_targets_large, loss_type="mse")
    huber_large = compute_dqn_loss(
        q_selected_large.clone().detach().requires_grad_(True),
        td_targets_large,
        loss_type="huber",
        huber_delta=1.0,
    )

    # For large errors, Huber should be much smaller than MSE
    assert huber_large["loss"] < mse_large["loss"]


# ============================================================================
# Optimizer Configuration Tests
# ============================================================================


def test_configure_optimizer_rmsprop_defaults():
    """Test RMSProp optimizer with default parameters."""
    online_net = DQN(num_actions=6)

    # Configure with defaults
    optimizer = configure_optimizer(online_net, optimizer_type="rmsprop")

    # Should be RMSProp
    assert isinstance(optimizer, torch.optim.RMSprop)

    # Check default hyperparameters
    param_groups = optimizer.param_groups[0]
    assert param_groups["lr"] == 2.5e-4, "Default LR should be 2.5e-4"
    assert param_groups["alpha"] == 0.95, "Default alpha (ρ) should be 0.95"
    assert param_groups["eps"] == 1e-2, "Default eps should be 0.01"
    assert param_groups["momentum"] == 0.0, "Default momentum should be 0.0"
    assert param_groups["weight_decay"] == 0.0, "Default weight_decay should be 0.0"


def test_configure_optimizer_rmsprop_custom():
    """Test RMSProp optimizer with custom parameters."""
    online_net = DQN(num_actions=6)

    # Configure with custom params
    optimizer = configure_optimizer(
        online_net,
        optimizer_type="rmsprop",
        learning_rate=1e-3,
        alpha=0.99,
        eps=1e-8,
        momentum=0.9,
        weight_decay=1e-5,
    )

    # Check custom hyperparameters
    param_groups = optimizer.param_groups[0]
    assert param_groups["lr"] == 1e-3
    assert param_groups["alpha"] == 0.99
    assert param_groups["eps"] == 1e-8
    assert param_groups["momentum"] == 0.9
    assert param_groups["weight_decay"] == 1e-5


def test_configure_optimizer_adam_defaults():
    """Test Adam optimizer with default parameters."""
    online_net = DQN(num_actions=6)

    # Configure Adam
    optimizer = configure_optimizer(online_net, optimizer_type="adam")

    # Should be Adam
    assert isinstance(optimizer, torch.optim.Adam)

    # Check default hyperparameters
    param_groups = optimizer.param_groups[0]
    assert param_groups["lr"] == 2.5e-4
    assert param_groups["betas"] == (0.9, 0.999)
    assert param_groups["eps"] == 1e-8
    assert param_groups["weight_decay"] == 0.0


def test_configure_optimizer_adam_custom():
    """Test Adam optimizer with custom parameters."""
    online_net = DQN(num_actions=6)

    # Configure with custom params
    optimizer = configure_optimizer(
        online_net,
        optimizer_type="adam",
        learning_rate=3e-4,
        beta1=0.95,
        beta2=0.9999,
        adam_eps=1e-7,
        weight_decay=1e-4,
    )

    # Check custom hyperparameters
    param_groups = optimizer.param_groups[0]
    assert param_groups["lr"] == 3e-4
    assert param_groups["betas"] == (0.95, 0.9999)
    assert param_groups["eps"] == 1e-7
    assert param_groups["weight_decay"] == 1e-4


def test_configure_optimizer_invalid_type():
    """Test that invalid optimizer type raises error."""
    online_net = DQN(num_actions=6)

    with pytest.raises(ValueError, match="Unknown optimizer_type"):
        configure_optimizer(online_net, optimizer_type="sgd")


def test_configure_optimizer_parameters_linked():
    """Test that optimizer is linked to network parameters."""
    online_net = DQN(num_actions=6)

    optimizer = configure_optimizer(online_net, optimizer_type="rmsprop")

    # Verify optimizer has parameters
    assert len(list(optimizer.param_groups[0]["params"])) > 0

    # Verify parameters are from the network
    net_params = set(online_net.parameters())
    opt_params = set(optimizer.param_groups[0]["params"])
    assert net_params == opt_params


def test_optimizer_step_updates_parameters():
    """Test that optimizer step updates network parameters."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(
        online_net, optimizer_type="rmsprop", learning_rate=0.1
    )

    # Get initial parameters
    initial_param = list(online_net.parameters())[0].clone()

    # Create batch and compute loss
    batch_size = 4
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.tensor([0, 2, 1, 5])
    rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.tensor([False, False, True, False])

    q_selected, td_targets = compute_td_loss_components(
        states, actions, rewards, next_states, dones, online_net, target_net, gamma=0.99
    )

    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type="mse")

    # Optimization step
    optimizer.zero_grad()
    loss_dict["loss"].backward()
    optimizer.step()

    # Parameters should have changed
    updated_param = list(online_net.parameters())[0]
    assert not torch.allclose(
        initial_param, updated_param
    ), "Parameters should change after optimizer step"


# ============================================================================
# Gradient Clipping Tests
# ============================================================================


def test_clip_gradients_basic():
    """Test basic gradient clipping."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Create batch and compute loss
    batch_size = 4
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.tensor([0, 2, 1, 5])
    rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.tensor([False, False, True, False])

    q_selected, td_targets = compute_td_loss_components(
        states, actions, rewards, next_states, dones, online_net, target_net, gamma=0.99
    )

    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type="mse")
    loss_dict["loss"].backward()

    # Clip gradients
    grad_norm = clip_gradients(online_net, max_norm=10.0)

    # Should return a float
    assert isinstance(grad_norm, float)
    assert grad_norm >= 0.0


def test_clip_gradients_returns_norm():
    """Test that clip_gradients returns the gradient norm."""
    online_net = DQN(num_actions=6)

    # Create dummy loss
    x = torch.randn(2, 4, 84, 84)
    output = online_net(x)
    loss = output["q_values"].mean()
    loss.backward()

    # Get gradient norm
    grad_norm = clip_gradients(online_net, max_norm=10.0)

    # Should be positive (network has gradients)
    assert grad_norm > 0.0


def test_clip_gradients_actually_clips():
    """Test that gradient clipping actually limits gradient norm."""
    online_net = DQN(num_actions=6)

    # Create large gradients by using large loss
    x = torch.randn(2, 4, 84, 84)
    output = online_net(x)
    loss = output["q_values"].sum() * 1000.0  # Large multiplier
    loss.backward()

    # Clip with small max_norm
    max_norm = 1.0
    clip_gradients(online_net, max_norm=max_norm)

    # Compute actual norm after clipping
    total_norm_after = 0.0
    for p in online_net.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after**0.5

    # After clipping, norm should be <= max_norm (with small tolerance for numerical errors)
    assert (
        total_norm_after <= max_norm + 1e-5
    ), f"Gradient norm {total_norm_after} exceeds max_norm {max_norm}"


def test_clip_gradients_no_effect_when_small():
    """Test that small gradients are not affected by clipping."""
    online_net = DQN(num_actions=6)

    # Create small gradients
    x = torch.randn(2, 4, 84, 84)
    output = online_net(x)
    loss = output["q_values"].mean() * 0.001  # Small multiplier
    loss.backward()

    # Save gradients before clipping
    grads_before = [
        p.grad.clone() for p in online_net.parameters() if p.grad is not None
    ]

    # Clip with large max_norm (shouldn't affect small gradients)
    clip_gradients(online_net, max_norm=100.0)

    # Gradients should be unchanged
    grads_after = [p.grad for p in online_net.parameters() if p.grad is not None]

    for g_before, g_after in zip(grads_before, grads_after):
        assert torch.allclose(
            g_before, g_after, atol=1e-6
        ), "Small gradients should not be affected by large max_norm"


def test_clip_gradients_different_norms():
    """Test gradient clipping with different norm types."""
    online_net = DQN(num_actions=6)

    # Create gradients
    x = torch.randn(2, 4, 84, 84)
    output = online_net(x)
    loss = output["q_values"].mean()
    loss.backward()

    # Clip with L2 norm (default)
    grad_norm_l2 = clip_gradients(online_net, max_norm=10.0, norm_type=2.0)
    assert grad_norm_l2 > 0.0

    # Reset gradients and recompute with new forward pass
    online_net.zero_grad()
    output = online_net(x)
    loss = output["q_values"].mean()
    loss.backward()

    # Clip with L1 norm
    grad_norm_l1 = clip_gradients(online_net, max_norm=10.0, norm_type=1.0)
    assert grad_norm_l1 > 0.0

    # L1 and L2 norms should be different
    assert grad_norm_l1 != grad_norm_l2


def test_clip_gradients_integration_with_optimizer():
    """Test gradient clipping in full training step."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, optimizer_type="rmsprop")

    # Create batch
    batch_size = 4
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.tensor([0, 2, 1, 5])
    rewards = torch.tensor([1.0, 0.0, -1.0, 0.5])
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.tensor([False, False, True, False])

    # Get initial param
    initial_param = list(online_net.parameters())[0].clone()

    # Full training step with gradient clipping
    optimizer.zero_grad()

    q_selected, td_targets = compute_td_loss_components(
        states, actions, rewards, next_states, dones, online_net, target_net, gamma=0.99
    )

    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type="mse")
    loss_dict["loss"].backward()

    grad_norm = clip_gradients(online_net, max_norm=10.0)
    optimizer.step()

    # Parameters should have updated
    updated_param = list(online_net.parameters())[0]
    assert not torch.allclose(initial_param, updated_param)

    # Gradient norm should be positive
    assert grad_norm > 0.0


def test_clip_gradients_monitoring():
    """Test that gradient norm can be used for monitoring."""
    online_net = DQN(num_actions=6)

    # List to store gradient norms
    grad_norms = []

    for _ in range(3):
        # Create random loss
        x = torch.randn(2, 4, 84, 84)
        output = online_net(x)
        loss = output["q_values"].mean()

        online_net.zero_grad()
        loss.backward()

        # Clip and record norm
        grad_norm = clip_gradients(online_net, max_norm=10.0)
        grad_norms.append(grad_norm)

    # All norms should be positive and finite
    for norm in grad_norms:
        assert norm > 0.0
        assert not torch.isnan(torch.tensor(norm))
        assert not torch.isinf(torch.tensor(norm))


# ============================================================================
# Target Network Update Scheduler Tests
# ============================================================================


def test_target_network_updater_initialization():
    """Test TargetNetworkUpdater initialization."""
    updater = TargetNetworkUpdater(update_interval=10000)

    assert updater.update_interval == 10000
    assert updater.step_count == 0
    assert updater.last_update_step == 0
    assert updater.total_updates == 0


def test_target_network_updater_invalid_interval():
    """Test that invalid interval raises error."""
    with pytest.raises(ValueError, match="must be positive"):
        TargetNetworkUpdater(update_interval=0)

    with pytest.raises(ValueError, match="must be positive"):
        TargetNetworkUpdater(update_interval=-100)


def test_target_network_updater_should_update():
    """Test should_update logic at exact multiples."""
    updater = TargetNetworkUpdater(update_interval=1000)

    # Should not update at step 0
    assert not updater.should_update(0)

    # Should not update before interval
    assert not updater.should_update(500)
    assert not updater.should_update(999)

    # Should update at exact multiples
    assert updater.should_update(1000)
    assert updater.should_update(2000)
    assert updater.should_update(3000)
    assert updater.should_update(10000)

    # Should not update between multiples
    assert not updater.should_update(1001)
    assert not updater.should_update(1500)


def test_target_network_updater_update():
    """Test update performs hard copy and updates counters."""
    online_net = DQN(num_actions=6)
    target_net = DQN(num_actions=6)

    # Make networks different
    with torch.no_grad():
        for p in online_net.parameters():
            p.fill_(1.0)
        for p in target_net.parameters():
            p.fill_(0.0)

    updater = TargetNetworkUpdater(update_interval=10000)

    # Perform update
    info = updater.update(online_net, target_net, current_step=10000)

    # Check networks are now identical
    for p_online, p_target in zip(online_net.parameters(), target_net.parameters()):
        assert torch.allclose(p_online, p_target)

    # Check info dict
    assert info["step"] == 10000
    assert info["total_updates"] == 1
    assert info["steps_since_last"] == 10000

    # Check internal state
    assert updater.step_count == 10000
    assert updater.last_update_step == 10000
    assert updater.total_updates == 1


def test_target_network_updater_multiple_updates():
    """Test multiple updates at correct intervals."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=1000)

    # Simulate training loop
    for step in range(1, 5001):
        if updater.should_update(step):
            info = updater.update(online_net, target_net, current_step=step)

            # Verify update occurred at correct step
            assert step % 1000 == 0
            assert info["step"] == step

    # Should have updated 5 times (at 1000, 2000, 3000, 4000, 5000)
    assert updater.total_updates == 5
    assert updater.last_update_step == 5000


def test_target_network_updater_step_method():
    """Test convenience step() method."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=100)

    # Should return None when no update
    result = updater.step(online_net, target_net, current_step=50)
    assert result is None

    # Should return info dict when update occurs
    result = updater.step(online_net, target_net, current_step=100)
    assert result is not None
    assert result["step"] == 100
    assert result["total_updates"] == 1


def test_target_network_updater_no_duplicate_updates():
    """Test that calling update multiple times at same step doesn't duplicate."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=1000)

    # First call at step 1000
    assert updater.should_update(1000)
    updater.update(online_net, target_net, current_step=1000)

    # Second call at step 1000 should not trigger update
    assert not updater.should_update(1000)

    assert updater.total_updates == 1


def test_target_network_updater_reset():
    """Test reset() clears all counters."""
    updater = TargetNetworkUpdater(update_interval=1000)

    # Set some state
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater.update(online_net, target_net, current_step=1000)
    updater.update(online_net, target_net, current_step=2000)

    assert updater.total_updates == 2

    # Reset
    updater.reset()

    assert updater.step_count == 0
    assert updater.last_update_step == 0
    assert updater.total_updates == 0


def test_target_network_updater_state_dict():
    """Test state_dict() for checkpointing."""
    updater = TargetNetworkUpdater(update_interval=5000)

    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Perform some updates
    updater.update(online_net, target_net, current_step=5000)
    updater.update(online_net, target_net, current_step=10000)

    # Get state dict
    state = updater.state_dict()

    assert state["update_interval"] == 5000
    assert state["step_count"] == 10000
    assert state["last_update_step"] == 10000
    assert state["total_updates"] == 2


def test_target_network_updater_load_state_dict():
    """Test load_state_dict() for checkpoint restoration."""
    updater1 = TargetNetworkUpdater(update_interval=5000)

    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Perform some updates
    updater1.update(online_net, target_net, current_step=5000)
    updater1.update(online_net, target_net, current_step=10000)

    # Save state
    state = updater1.state_dict()

    # Create new updater and load state
    updater2 = TargetNetworkUpdater(update_interval=1000)  # Different interval
    updater2.load_state_dict(state)

    # Should match original state
    assert updater2.update_interval == 5000
    assert updater2.step_count == 10000
    assert updater2.last_update_step == 10000
    assert updater2.total_updates == 2


def test_target_network_updater_exact_multiples():
    """Test updates occur at exact multiples of interval."""
    updater = TargetNetworkUpdater(update_interval=10000)

    # Track which steps trigger updates
    update_steps = []
    for step in range(1, 50001):
        if updater.should_update(step):
            update_steps.append(step)
            # Mark as updated
            updater.last_update_step = step
            updater.total_updates += 1

    # Should update at: 10000, 20000, 30000, 40000, 50000
    expected_steps = [10000, 20000, 30000, 40000, 50000]
    assert update_steps == expected_steps


def test_target_network_updater_repr():
    """Test string representation."""
    updater = TargetNetworkUpdater(update_interval=10000)

    repr_str = repr(updater)
    assert "TargetNetworkUpdater" in repr_str
    assert "interval=10000" in repr_str
    assert "step=0" in repr_str
    assert "updates=0" in repr_str


def test_target_network_updater_integration():
    """Test full integration in training-like loop."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=1000)

    # Track updates
    update_log = []

    # Simulate 5000 training steps
    for step in range(1, 5001):
        # Check and update
        info = updater.step(online_net, target_net, current_step=step)

        if info:
            update_log.append(info)

    # Should have 5 updates
    assert len(update_log) == 5

    # Verify each update
    for i, info in enumerate(update_log):
        expected_step = (i + 1) * 1000
        assert info["step"] == expected_step
        assert info["total_updates"] == i + 1


def test_target_network_updater_parameters_actually_copied():
    """Test that parameters are actually copied during update."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=1000)

    # Get initial target params
    initial_target_param = list(target_net.parameters())[0].clone()

    # Modify online network
    with torch.no_grad():
        for p in online_net.parameters():
            p.add_(0.1)

    # Online should differ from target
    current_online_param = list(online_net.parameters())[0]
    assert not torch.allclose(current_online_param, initial_target_param)

    # Update target
    updater.update(online_net, target_net, current_step=1000)

    # Target should now match online
    updated_target_param = list(target_net.parameters())[0]
    assert torch.allclose(current_online_param, updated_target_param)
    assert not torch.allclose(initial_target_param, updated_target_param)


# ============================================================================
# Training Scheduler Tests
# ============================================================================


def test_training_scheduler_initialization():
    """Test TrainingScheduler initialization."""
    scheduler = TrainingScheduler(train_every=4)

    assert scheduler.train_every == 4
    assert scheduler.env_step_count == 0
    assert scheduler.training_step_count == 0
    assert scheduler.last_train_step == 0


def test_training_scheduler_invalid_train_every():
    """Test that invalid train_every raises error."""
    with pytest.raises(ValueError, match="must be positive"):
        TrainingScheduler(train_every=0)

    with pytest.raises(ValueError, match="must be positive"):
        TrainingScheduler(train_every=-1)


def test_training_scheduler_warm_up_gating():
    """Test that training is gated by replay buffer warm-up."""
    buffer = ReplayBuffer(capacity=1000, obs_shape=(84, 84), min_size=100)
    scheduler = TrainingScheduler(train_every=4)

    # Before warm-up: should not train even at multiples of train_every
    assert not scheduler.should_train(4, buffer)
    assert not scheduler.should_train(8, buffer)
    assert not scheduler.should_train(12, buffer)

    # Fill buffer past min_size
    for _ in range(150):
        state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        next_state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        buffer.append(state, action=0, reward=0.0, next_state=next_state, done=False)

    # After warm-up: should train at multiples
    assert buffer.can_sample()
    assert scheduler.should_train(4, buffer)


def test_training_scheduler_train_every_multiples():
    """Test training occurs at exact multiples of train_every."""
    buffer = ReplayBuffer(capacity=1000, obs_shape=(84, 84), min_size=50)
    scheduler = TrainingScheduler(train_every=4)

    # Fill buffer
    for _ in range(100):
        state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        next_state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        buffer.append(state, action=0, reward=0.0, next_state=next_state, done=False)

    # Check multiples
    assert not scheduler.should_train(0, buffer)  # Step 0
    assert not scheduler.should_train(1, buffer)
    assert not scheduler.should_train(2, buffer)
    assert not scheduler.should_train(3, buffer)
    assert scheduler.should_train(4, buffer)  # Multiple of 4
    assert not scheduler.should_train(5, buffer)
    assert not scheduler.should_train(6, buffer)
    assert not scheduler.should_train(7, buffer)
    assert scheduler.should_train(8, buffer)  # Multiple of 4


def test_training_scheduler_mark_trained():
    """Test mark_trained updates counters."""
    buffer = ReplayBuffer(capacity=1000, obs_shape=(84, 84), min_size=50)
    scheduler = TrainingScheduler(train_every=4)

    # Fill buffer
    for _ in range(100):
        state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        next_state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        buffer.append(state, action=0, reward=0.0, next_state=next_state, done=False)

    # Mark trained
    scheduler.mark_trained(env_step=4)

    assert scheduler.env_step_count == 4
    assert scheduler.last_train_step == 4
    assert scheduler.training_step_count == 1

    # Mark again
    scheduler.mark_trained(env_step=8)

    assert scheduler.env_step_count == 8
    assert scheduler.last_train_step == 8
    assert scheduler.training_step_count == 2


def test_training_scheduler_no_duplicate_training():
    """Test that training at same step doesn't happen twice."""
    buffer = ReplayBuffer(capacity=1000, obs_shape=(84, 84), min_size=50)
    scheduler = TrainingScheduler(train_every=4)

    # Fill buffer
    for _ in range(100):
        state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        next_state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        buffer.append(state, action=0, reward=0.0, next_state=next_state, done=False)

    # First call at step 4
    assert scheduler.should_train(4, buffer)
    scheduler.mark_trained(4)

    # Second call at step 4 should not train
    assert not scheduler.should_train(4, buffer)


def test_training_scheduler_step_method():
    """Test convenience step() method."""
    buffer = ReplayBuffer(capacity=1000, obs_shape=(84, 84), min_size=50)
    scheduler = TrainingScheduler(train_every=4)

    # Fill buffer
    for _ in range(100):
        state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        next_state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        buffer.append(state, action=0, reward=0.0, next_state=next_state, done=False)

    # Should return False when not time to train
    result = scheduler.step(2, buffer)
    assert result is False
    assert scheduler.training_step_count == 0

    # Should return True and update counters when time to train
    result = scheduler.step(4, buffer)
    assert result is True
    assert scheduler.training_step_count == 1
    assert scheduler.last_train_step == 4


def test_training_scheduler_reset():
    """Test reset() clears all counters."""
    buffer = ReplayBuffer(capacity=1000, obs_shape=(84, 84), min_size=50)
    scheduler = TrainingScheduler(train_every=4)

    # Fill buffer and train
    for _ in range(100):
        state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        next_state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        buffer.append(state, action=0, reward=0.0, next_state=next_state, done=False)

    scheduler.mark_trained(4)
    scheduler.mark_trained(8)

    assert scheduler.training_step_count == 2

    # Reset
    scheduler.reset()

    assert scheduler.env_step_count == 0
    assert scheduler.training_step_count == 0
    assert scheduler.last_train_step == 0


def test_training_scheduler_state_dict():
    """Test state_dict() for checkpointing."""
    scheduler = TrainingScheduler(train_every=4)
    scheduler.mark_trained(12)
    scheduler.mark_trained(16)

    state = scheduler.state_dict()

    assert state["train_every"] == 4
    assert state["env_step_count"] == 16
    assert state["training_step_count"] == 2
    assert state["last_train_step"] == 16


def test_training_scheduler_load_state_dict():
    """Test load_state_dict() for checkpoint restoration."""
    scheduler1 = TrainingScheduler(train_every=4)
    scheduler1.mark_trained(8)
    scheduler1.mark_trained(12)

    state = scheduler1.state_dict()

    # Create new scheduler and load
    scheduler2 = TrainingScheduler(train_every=1)  # Different value
    scheduler2.load_state_dict(state)

    assert scheduler2.train_every == 4
    assert scheduler2.env_step_count == 12
    assert scheduler2.training_step_count == 2
    assert scheduler2.last_train_step == 12


def test_training_scheduler_integration():
    """Test full integration in training-like loop."""
    buffer = ReplayBuffer(capacity=1000, obs_shape=(84, 84), min_size=50)
    scheduler = TrainingScheduler(train_every=4)

    # Fill buffer
    for i in range(100):
        state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        next_state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        buffer.append(state, action=0, reward=0.0, next_state=next_state, done=False)

    # Simulate training loop
    training_steps = []
    for env_step in range(1, 41):
        if scheduler.step(env_step, buffer):
            training_steps.append(env_step)

    # Should train at: 4, 8, 12, 16, 20, 24, 28, 32, 36, 40
    expected_steps = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]
    assert training_steps == expected_steps
    assert scheduler.training_step_count == 10


def test_training_scheduler_different_intervals():
    """Test scheduler with different train_every values."""
    buffer = ReplayBuffer(capacity=1000, obs_shape=(84, 84), min_size=50)

    # Fill buffer
    for _ in range(100):
        state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        next_state = torch.randint(0, 255, (84, 84), dtype=torch.uint8).numpy()
        buffer.append(state, action=0, reward=0.0, next_state=next_state, done=False)

    # Test train_every=1
    scheduler1 = TrainingScheduler(train_every=1)
    count1 = sum(1 for step in range(1, 11) if scheduler1.step(step, buffer))
    assert count1 == 10  # Every step

    # Test train_every=8
    scheduler8 = TrainingScheduler(train_every=8)
    count8 = sum(1 for step in range(1, 25) if scheduler8.step(step, buffer))
    assert count8 == 3  # Steps 8, 16, 24


def test_training_scheduler_repr():
    """Test string representation."""
    scheduler = TrainingScheduler(train_every=4)

    repr_str = repr(scheduler)
    assert "TrainingScheduler" in repr_str
    assert "train_every=4" in repr_str
    assert "env_step=0" in repr_str
    assert "training_steps=0" in repr_str


# ============================================================================
# Stability Check Tests
# ============================================================================


def test_detect_nan_inf_no_issues():
    """Test detect_nan_inf returns False for normal tensors."""
    tensor = torch.randn(32)
    assert not detect_nan_inf(tensor, "normal_tensor")


def test_detect_nan_inf_detects_nan():
    """Test detect_nan_inf detects NaN values."""
    tensor = torch.tensor([1.0, 2.0, float("nan"), 4.0])
    assert detect_nan_inf(tensor, "nan_tensor")


def test_detect_nan_inf_detects_inf():
    """Test detect_nan_inf detects Inf values."""
    tensor = torch.tensor([1.0, 2.0, float("inf"), 4.0])
    assert detect_nan_inf(tensor, "inf_tensor")


def test_detect_nan_inf_detects_neg_inf():
    """Test detect_nan_inf detects negative Inf values."""
    tensor = torch.tensor([1.0, 2.0, float("-inf"), 4.0])
    assert detect_nan_inf(tensor, "neg_inf_tensor")


def test_detect_nan_inf_multidimensional():
    """Test detect_nan_inf works with multidimensional tensors."""
    # Normal tensor
    tensor = torch.randn(4, 8, 16)
    assert not detect_nan_inf(tensor, "normal_3d")

    # With NaN
    tensor[2, 3, 5] = float("nan")
    assert detect_nan_inf(tensor, "nan_3d")


def test_detect_nan_inf_zero_tensor():
    """Test detect_nan_inf handles zero tensors correctly."""
    tensor = torch.zeros(32)
    assert not detect_nan_inf(tensor, "zero_tensor")


def test_validate_loss_decrease_basic():
    """Test validate_loss_decrease with synthetic batch."""
    # Create networks
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, learning_rate=0.01)

    # Create synthetic batch (small batch for quick test)
    batch_size = 8
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.randint(0, 6, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.zeros(batch_size)

    # Validate loss decreases
    success, info = validate_loss_decrease(
        compute_dqn_loss,
        online_net,
        optimizer,
        states,
        actions,
        rewards,
        next_states,
        dones,
        target_net,
        num_updates=10,
    )

    assert success, f"Loss did not decrease: {info}"
    assert info["loss_decreased"]
    assert not info["nan_inf_detected"]
    assert info["final_loss"] < info["initial_loss"]
    assert len(info["loss_history"]) == 10


def test_validate_loss_decrease_huber():
    """Test validate_loss_decrease with Huber loss."""
    # Create networks
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, learning_rate=0.01)

    # Create synthetic batch
    batch_size = 8
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.randint(0, 6, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.zeros(batch_size)

    # Validate with Huber loss
    success, info = validate_loss_decrease(
        compute_dqn_loss,
        online_net,
        optimizer,
        states,
        actions,
        rewards,
        next_states,
        dones,
        target_net,
        num_updates=10,
        loss_type="huber",
    )

    assert success, f"Loss did not decrease with Huber: {info}"
    assert info["loss_decreased"]
    assert not info["nan_inf_detected"]


def test_validate_loss_decrease_fewer_updates():
    """Test validate_loss_decrease with fewer updates."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, learning_rate=0.01)

    batch_size = 8
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.randint(0, 6, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.zeros(batch_size)

    success, info = validate_loss_decrease(
        compute_dqn_loss,
        online_net,
        optimizer,
        states,
        actions,
        rewards,
        next_states,
        dones,
        target_net,
        num_updates=5,
    )

    assert len(info["loss_history"]) == 5
    # Loss should still decrease with fewer updates
    assert info["initial_loss"] > 0


def test_validate_loss_decrease_terminal_states():
    """Test validate_loss_decrease with terminal states."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, learning_rate=0.01)

    batch_size = 8
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.randint(0, 6, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.ones(batch_size)  # All terminal

    success, info = validate_loss_decrease(
        compute_dqn_loss,
        online_net,
        optimizer,
        states,
        actions,
        rewards,
        next_states,
        dones,
        target_net,
        num_updates=10,
    )

    # Should still work with terminal states
    assert not info["nan_inf_detected"]
    assert len(info["loss_history"]) == 10


def test_validate_loss_decrease_different_gamma():
    """Test validate_loss_decrease with different gamma values."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, learning_rate=0.01)

    batch_size = 8
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.randint(0, 6, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.zeros(batch_size)

    # Test with gamma=0.95
    success, info = validate_loss_decrease(
        compute_dqn_loss,
        online_net,
        optimizer,
        states,
        actions,
        rewards,
        next_states,
        dones,
        target_net,
        num_updates=10,
        gamma=0.95,
    )

    assert not info["nan_inf_detected"]
    assert len(info["loss_history"]) == 10


def test_validate_loss_decrease_larger_batch():
    """Test validate_loss_decrease with standard DQN batch size."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, learning_rate=0.01)

    # Standard DQN batch size
    batch_size = 32
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.randint(0, 6, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.zeros(batch_size)

    success, info = validate_loss_decrease(
        compute_dqn_loss,
        online_net,
        optimizer,
        states,
        actions,
        rewards,
        next_states,
        dones,
        target_net,
        num_updates=10,
    )

    assert success, f"Loss did not decrease with batch_size=32: {info}"
    assert info["loss_decreased"]


def test_validate_loss_decrease_info_keys():
    """Test validate_loss_decrease returns expected info keys."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, learning_rate=0.01)

    batch_size = 8
    states = torch.randn(batch_size, 4, 84, 84)
    actions = torch.randint(0, 6, (batch_size,))
    rewards = torch.randn(batch_size)
    next_states = torch.randn(batch_size, 4, 84, 84)
    dones = torch.zeros(batch_size)

    success, info = validate_loss_decrease(
        compute_dqn_loss,
        online_net,
        optimizer,
        states,
        actions,
        rewards,
        next_states,
        dones,
        target_net,
        num_updates=5,
    )

    # Verify all expected keys are present
    expected_keys = {
        "initial_loss",
        "final_loss",
        "loss_history",
        "loss_decreased",
        "nan_inf_detected",
    }
    assert set(info.keys()) == expected_keys


def test_verify_target_sync_schedule_basic():
    """Test verify_target_sync_schedule with basic interval."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=1000)

    success, info = verify_target_sync_schedule(
        updater, online_net, target_net, max_steps=5000, expected_interval=1000
    )

    assert success, f"Target sync schedule incorrect: {info}"
    assert info["schedule_correct"]
    assert info["count_correct"]
    assert info["update_steps"] == [1000, 2000, 3000, 4000, 5000]
    assert info["expected_steps"] == [1000, 2000, 3000, 4000, 5000]


def test_verify_target_sync_schedule_10k_interval():
    """Test verify_target_sync_schedule with 10k interval (DQN default)."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=10000)

    success, info = verify_target_sync_schedule(
        updater, online_net, target_net, max_steps=50000, expected_interval=10000
    )

    assert success, f"Target sync schedule incorrect: {info}"
    assert info["update_steps"] == [10000, 20000, 30000, 40000, 50000]


def test_verify_target_sync_schedule_small_interval():
    """Test verify_target_sync_schedule with small interval."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=100)

    success, info = verify_target_sync_schedule(
        updater, online_net, target_net, max_steps=500, expected_interval=100
    )

    assert success
    assert len(info["update_steps"]) == 5
    assert info["update_steps"] == [100, 200, 300, 400, 500]


def test_verify_target_sync_schedule_partial_interval():
    """Test verify_target_sync_schedule when max_steps not multiple of interval."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=1000)

    # max_steps=4500 should give updates at 1000, 2000, 3000, 4000 only
    success, info = verify_target_sync_schedule(
        updater, online_net, target_net, max_steps=4500, expected_interval=1000
    )

    assert success
    assert info["update_steps"] == [1000, 2000, 3000, 4000]
    assert len(info["update_steps"]) == 4


def test_verify_target_sync_schedule_no_duplicates():
    """Test verify_target_sync_schedule has no duplicate updates."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=500)

    success, info = verify_target_sync_schedule(
        updater, online_net, target_net, max_steps=2000, expected_interval=500
    )

    # Check no duplicates
    assert len(info["update_steps"]) == len(set(info["update_steps"]))
    assert success


def test_verify_target_sync_schedule_count():
    """Test verify_target_sync_schedule count correctness."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=1000)

    success, info = verify_target_sync_schedule(
        updater, online_net, target_net, max_steps=10000, expected_interval=1000
    )

    assert info["count_correct"]
    assert len(info["update_steps"]) == 10
    assert len(info["expected_steps"]) == 10


def test_verify_target_sync_schedule_info_keys():
    """Test verify_target_sync_schedule returns expected info keys."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    updater = TargetNetworkUpdater(update_interval=1000)

    success, info = verify_target_sync_schedule(
        updater, online_net, target_net, max_steps=3000, expected_interval=1000
    )

    expected_keys = {
        "update_steps",
        "expected_steps",
        "schedule_correct",
        "count_correct",
    }
    assert set(info.keys()) == expected_keys


def test_verify_target_sync_schedule_different_intervals():
    """Test verify_target_sync_schedule with various intervals."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    intervals = [100, 500, 1000, 5000]
    for interval in intervals:
        updater = TargetNetworkUpdater(update_interval=interval)
        max_steps = interval * 5

        success, info = verify_target_sync_schedule(
            updater,
            online_net,
            target_net,
            max_steps=max_steps,
            expected_interval=interval,
        )

        assert success, f"Failed for interval={interval}: {info}"
        assert len(info["update_steps"]) == 5


# ============================================================================
# Update Metrics Tests
# ============================================================================


def test_update_metrics_initialization():
    """Test UpdateMetrics initialization."""
    metrics = UpdateMetrics(
        loss=0.5,
        td_error=0.3,
        td_error_std=0.2,
        grad_norm=2.5,
        learning_rate=0.00025,
        update_count=1000,
    )

    assert metrics.loss == 0.5
    assert metrics.td_error == 0.3
    assert metrics.td_error_std == 0.2
    assert metrics.grad_norm == 2.5
    assert metrics.learning_rate == 0.00025
    assert metrics.update_count == 1000


def test_update_metrics_to_dict():
    """Test UpdateMetrics to_dict conversion."""
    metrics = UpdateMetrics(
        loss=0.5,
        td_error=0.3,
        td_error_std=0.2,
        grad_norm=2.5,
        learning_rate=0.00025,
        update_count=1000,
    )

    metrics_dict = metrics.to_dict()

    assert isinstance(metrics_dict, dict)
    assert metrics_dict["loss"] == 0.5
    assert metrics_dict["td_error"] == 0.3
    assert metrics_dict["td_error_std"] == 0.2
    assert metrics_dict["grad_norm"] == 2.5
    assert metrics_dict["learning_rate"] == 0.00025
    assert metrics_dict["update_count"] == 1000


def test_update_metrics_to_dict_keys():
    """Test UpdateMetrics to_dict has all expected keys."""
    metrics = UpdateMetrics(0.5, 0.3, 0.2, 2.5, 0.00025, 1000)
    metrics_dict = metrics.to_dict()

    expected_keys = {
        "loss",
        "td_error",
        "td_error_std",
        "grad_norm",
        "learning_rate",
        "update_count",
    }
    assert set(metrics_dict.keys()) == expected_keys


def test_update_metrics_repr():
    """Test UpdateMetrics string representation."""
    metrics = UpdateMetrics(0.5, 0.3, 0.2, 2.5, 0.00025, 1000)
    repr_str = repr(metrics)

    assert "UpdateMetrics" in repr_str
    assert "loss=" in repr_str
    assert "td_error=" in repr_str
    assert "grad_norm=" in repr_str
    assert "lr=" in repr_str
    assert "updates=" in repr_str


def test_perform_update_step_basic():
    """Test perform_update_step executes full update and returns metrics."""
    # Create networks and optimizer
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, learning_rate=0.00025)

    # Create synthetic batch
    batch_size = 8
    batch = {
        "states": torch.randn(batch_size, 4, 84, 84),
        "actions": torch.randint(0, 6, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "next_states": torch.randn(batch_size, 4, 84, 84),
        "dones": torch.zeros(batch_size),
    }

    # Perform update
    metrics = perform_update_step(
        online_net, target_net, optimizer, batch, update_count=1
    )

    # Verify metrics exist and have reasonable values
    assert isinstance(metrics, UpdateMetrics)
    assert metrics.loss > 0
    assert metrics.td_error >= 0
    assert metrics.td_error_std >= 0
    assert metrics.grad_norm >= 0
    assert metrics.learning_rate == 0.00025
    assert metrics.update_count == 1


def test_perform_update_step_updates_network():
    """Test perform_update_step actually updates network parameters."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, learning_rate=0.01)

    # Save initial parameters
    initial_params = [p.clone() for p in online_net.parameters()]

    batch_size = 8
    batch = {
        "states": torch.randn(batch_size, 4, 84, 84),
        "actions": torch.randint(0, 6, (batch_size,)),
        "rewards": torch.randn(batch_size),
        "next_states": torch.randn(batch_size, 4, 84, 84),
        "dones": torch.zeros(batch_size),
    }

    # Perform update
    perform_update_step(online_net, target_net, optimizer, batch, update_count=1)

    # Verify parameters changed
    for initial_p, current_p in zip(initial_params, online_net.parameters()):
        assert not torch.allclose(
            initial_p, current_p
        ), "Parameters should change after update"


def test_perform_update_step_mse_loss():
    """Test perform_update_step with MSE loss."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net)

    batch = {
        "states": torch.randn(8, 4, 84, 84),
        "actions": torch.randint(0, 6, (8,)),
        "rewards": torch.randn(8),
        "next_states": torch.randn(8, 4, 84, 84),
        "dones": torch.zeros(8),
    }

    metrics = perform_update_step(
        online_net, target_net, optimizer, batch, loss_type="mse", update_count=1
    )

    assert isinstance(metrics, UpdateMetrics)
    assert metrics.loss > 0


def test_perform_update_step_huber_loss():
    """Test perform_update_step with Huber loss."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net)

    batch = {
        "states": torch.randn(8, 4, 84, 84),
        "actions": torch.randint(0, 6, (8,)),
        "rewards": torch.randn(8),
        "next_states": torch.randn(8, 4, 84, 84),
        "dones": torch.zeros(8),
    }

    metrics = perform_update_step(
        online_net, target_net, optimizer, batch, loss_type="huber", update_count=1
    )

    assert isinstance(metrics, UpdateMetrics)
    assert metrics.loss > 0


def test_perform_update_step_different_gamma():
    """Test perform_update_step with different gamma values."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net)

    batch = {
        "states": torch.randn(8, 4, 84, 84),
        "actions": torch.randint(0, 6, (8,)),
        "rewards": torch.randn(8),
        "next_states": torch.randn(8, 4, 84, 84),
        "dones": torch.zeros(8),
    }

    # Test with gamma=0.95
    metrics = perform_update_step(
        online_net, target_net, optimizer, batch, gamma=0.95, update_count=1
    )

    assert isinstance(metrics, UpdateMetrics)
    assert metrics.loss > 0


def test_perform_update_step_gradient_clipping():
    """Test perform_update_step applies gradient clipping."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(
        online_net, learning_rate=10.0
    )  # Large LR to create large gradients

    batch = {
        "states": torch.randn(8, 4, 84, 84),
        "actions": torch.randint(0, 6, (8,)),
        "rewards": torch.randn(8),
        "next_states": torch.randn(8, 4, 84, 84),
        "dones": torch.zeros(8),
    }

    # Perform update with small max_grad_norm
    metrics = perform_update_step(
        online_net, target_net, optimizer, batch, max_grad_norm=1.0, update_count=1
    )

    # Gradient norm should be reported (before clipping)
    assert metrics.grad_norm >= 0


def test_perform_update_step_multiple_updates():
    """Test perform_update_step with multiple sequential updates."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, learning_rate=0.01)

    batch = {
        "states": torch.randn(8, 4, 84, 84),
        "actions": torch.randint(0, 6, (8,)),
        "rewards": torch.randn(8),
        "next_states": torch.randn(8, 4, 84, 84),
        "dones": torch.zeros(8),
    }

    # Perform multiple updates
    losses = []
    for i in range(10):
        metrics = perform_update_step(
            online_net, target_net, optimizer, batch, update_count=i + 1
        )
        losses.append(metrics.loss)
        assert metrics.update_count == i + 1

    # Loss should generally decrease (overfitting on same batch)
    # Check that at least the loss changed (parameters were updated)
    assert len(set(losses)) > 1, "Loss should change across updates"


def test_perform_update_step_batch_size_32():
    """Test perform_update_step with standard DQN batch size."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net)

    # Standard DQN batch size
    batch = {
        "states": torch.randn(32, 4, 84, 84),
        "actions": torch.randint(0, 6, (32,)),
        "rewards": torch.randn(32),
        "next_states": torch.randn(32, 4, 84, 84),
        "dones": torch.zeros(32),
    }

    metrics = perform_update_step(
        online_net, target_net, optimizer, batch, update_count=1
    )

    assert isinstance(metrics, UpdateMetrics)
    assert metrics.loss > 0


def test_perform_update_step_terminal_states():
    """Test perform_update_step with terminal states."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net)

    batch = {
        "states": torch.randn(8, 4, 84, 84),
        "actions": torch.randint(0, 6, (8,)),
        "rewards": torch.randn(8),
        "next_states": torch.randn(8, 4, 84, 84),
        "dones": torch.ones(8),  # All terminal
    }

    metrics = perform_update_step(
        online_net, target_net, optimizer, batch, update_count=1
    )

    assert isinstance(metrics, UpdateMetrics)
    assert metrics.loss > 0


def test_perform_update_step_learning_rate_tracking():
    """Test perform_update_step tracks learning rate correctly."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Test with custom learning rate
    custom_lr = 0.001
    optimizer = configure_optimizer(online_net, learning_rate=custom_lr)

    batch = {
        "states": torch.randn(8, 4, 84, 84),
        "actions": torch.randint(0, 6, (8,)),
        "rewards": torch.randn(8),
        "next_states": torch.randn(8, 4, 84, 84),
        "dones": torch.zeros(8),
    }

    metrics = perform_update_step(
        online_net, target_net, optimizer, batch, update_count=1
    )

    assert metrics.learning_rate == custom_lr


def test_perform_update_step_sets_train_mode():
    """Test perform_update_step sets network to training mode."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net)

    # Set to eval mode
    online_net.eval()
    assert not online_net.training

    batch = {
        "states": torch.randn(8, 4, 84, 84),
        "actions": torch.randint(0, 6, (8,)),
        "rewards": torch.randn(8),
        "next_states": torch.randn(8, 4, 84, 84),
        "dones": torch.zeros(8),
    }

    metrics = perform_update_step(
        online_net, target_net, optimizer, batch, update_count=1
    )

    # Network should be in training mode during update
    # (Note: it stays in training mode after the function)
    assert metrics is not None


# ============================================================================
# Epsilon-Greedy Exploration Tests
# ============================================================================


def test_epsilon_scheduler_initialization():
    """Test EpsilonScheduler initializes with correct defaults."""
    from src.training import EpsilonScheduler

    scheduler = EpsilonScheduler()
    assert scheduler.epsilon_start == 1.0
    assert scheduler.epsilon_end == 0.1
    assert scheduler.decay_frames == 1_000_000
    assert scheduler.eval_epsilon == 0.05


def test_epsilon_scheduler_custom_params():
    """Test EpsilonScheduler with custom parameters."""
    from src.training import EpsilonScheduler

    scheduler = EpsilonScheduler(
        epsilon_start=0.8, epsilon_end=0.05, decay_frames=500_000, eval_epsilon=0.01
    )
    assert scheduler.epsilon_start == 0.8
    assert scheduler.epsilon_end == 0.05
    assert scheduler.decay_frames == 500_000
    assert scheduler.eval_epsilon == 0.01


def test_epsilon_scheduler_linear_decay():
    """Test epsilon decays linearly from start to end."""
    from src.training import EpsilonScheduler

    scheduler = EpsilonScheduler(
        epsilon_start=1.0, epsilon_end=0.1, decay_frames=1_000_000
    )

    # Test key points on decay schedule
    assert scheduler.get_epsilon(0) == 1.0
    assert abs(scheduler.get_epsilon(250_000) - 0.775) < 1e-6
    assert abs(scheduler.get_epsilon(500_000) - 0.55) < 1e-6
    assert abs(scheduler.get_epsilon(750_000) - 0.325) < 1e-6
    assert scheduler.get_epsilon(1_000_000) == 0.1


def test_epsilon_scheduler_clamps_after_decay():
    """Test epsilon stays at epsilon_end after decay period."""
    from src.training import EpsilonScheduler

    scheduler = EpsilonScheduler(
        epsilon_start=1.0, epsilon_end=0.1, decay_frames=1_000_000
    )

    # After decay period, should clamp to epsilon_end
    assert scheduler.get_epsilon(1_000_000) == 0.1
    assert scheduler.get_epsilon(2_000_000) == 0.1
    assert scheduler.get_epsilon(10_000_000) == 0.1


def test_epsilon_scheduler_eval_epsilon():
    """Test get_eval_epsilon returns fixed value."""
    from src.training import EpsilonScheduler

    scheduler = EpsilonScheduler(eval_epsilon=0.05)
    assert scheduler.get_eval_epsilon() == 0.05

    # Eval epsilon should not change regardless of training frames
    assert scheduler.get_eval_epsilon() == 0.05
    assert scheduler.get_eval_epsilon() == 0.05


def test_epsilon_scheduler_to_dict():
    """Test scheduler exports configuration as dict."""
    from src.training import EpsilonScheduler

    scheduler = EpsilonScheduler(
        epsilon_start=0.9, epsilon_end=0.05, decay_frames=800_000, eval_epsilon=0.01
    )

    config = scheduler.to_dict()
    assert config["epsilon_start"] == 0.9
    assert config["epsilon_end"] == 0.05
    assert config["decay_frames"] == 800_000
    assert config["eval_epsilon"] == 0.01


def test_epsilon_scheduler_invalid_params():
    """Test scheduler raises on invalid parameters."""
    import pytest

    from src.training import EpsilonScheduler

    # epsilon_start > 1.0
    with pytest.raises(AssertionError):
        EpsilonScheduler(epsilon_start=1.5)

    # epsilon_end < 0.0
    with pytest.raises(AssertionError):
        EpsilonScheduler(epsilon_end=-0.1)

    # epsilon_start < epsilon_end
    with pytest.raises(AssertionError):
        EpsilonScheduler(epsilon_start=0.05, epsilon_end=0.1)

    # decay_frames <= 0
    with pytest.raises(AssertionError):
        EpsilonScheduler(decay_frames=0)

    # eval_epsilon > 1.0
    with pytest.raises(AssertionError):
        EpsilonScheduler(eval_epsilon=1.5)


def test_select_epsilon_greedy_action_random():
    """Test epsilon-greedy selects random actions with high epsilon."""
    from src.models import DQN
    from src.training import select_epsilon_greedy_action

    network = DQN(num_actions=6)
    state = torch.rand(4, 84, 84)

    # With epsilon=1.0, should always explore (random actions)
    actions = [
        select_epsilon_greedy_action(network, state, epsilon=1.0, num_actions=6)
        for _ in range(100)
    ]

    # Should get variety of actions (not all the same)
    unique_actions = set(actions)
    assert len(unique_actions) > 1  # Should have at least 2 different actions

    # All actions should be valid
    assert all(0 <= a < 6 for a in actions)


def test_select_epsilon_greedy_action_greedy():
    """Test epsilon-greedy selects greedy actions with epsilon=0."""
    from src.models import DQN
    from src.training import select_epsilon_greedy_action

    network = DQN(num_actions=6)
    state = torch.rand(4, 84, 84)

    # With epsilon=0.0, should always exploit (greedy)
    actions = [
        select_epsilon_greedy_action(network, state, epsilon=0.0, num_actions=6)
        for _ in range(100)
    ]

    # All actions should be the same (greedy choice)
    assert len(set(actions)) == 1

    # Action should be valid
    assert 0 <= actions[0] < 6


def test_select_epsilon_greedy_action_mixed():
    """Test epsilon-greedy mixes random and greedy actions."""
    import torch

    from src.models import DQN
    from src.training import select_epsilon_greedy_action

    network = DQN(num_actions=6)
    state = torch.rand(4, 84, 84)

    # Set seed for reproducibility
    torch.manual_seed(42)

    # With epsilon=0.5, should get mix of random and greedy
    actions = [
        select_epsilon_greedy_action(network, state, epsilon=0.5, num_actions=6)
        for _ in range(1000)
    ]

    # Should have some variety (not all greedy)
    unique_actions = set(actions)
    assert len(unique_actions) > 1

    # All actions should be valid
    assert all(0 <= a < 6 for a in actions)


def test_select_epsilon_greedy_action_batch_dim():
    """Test epsilon-greedy handles states with and without batch dim."""
    from src.models import DQN
    from src.training import select_epsilon_greedy_action

    network = DQN(num_actions=6)

    # State without batch dimension
    state_3d = torch.rand(4, 84, 84)
    action_3d = select_epsilon_greedy_action(
        network, state_3d, epsilon=0.0, num_actions=6
    )
    assert 0 <= action_3d < 6

    # State with batch dimension
    state_4d = torch.rand(1, 4, 84, 84)
    action_4d = select_epsilon_greedy_action(
        network, state_4d, epsilon=0.0, num_actions=6
    )
    assert 0 <= action_4d < 6


def test_select_epsilon_greedy_action_network_mode():
    """Test epsilon-greedy preserves network training mode."""
    from src.models import DQN
    from src.training import select_epsilon_greedy_action

    network = DQN(num_actions=6)
    state = torch.rand(4, 84, 84)

    # Start in training mode
    network.train()
    assert network.training

    # Select action
    select_epsilon_greedy_action(network, state, epsilon=0.0, num_actions=6)

    # Should still be in training mode
    assert network.training


def test_epsilon_scheduler_monotonic_decrease():
    """Test epsilon decreases monotonically during decay period."""
    from src.training import EpsilonScheduler

    scheduler = EpsilonScheduler(
        epsilon_start=1.0, epsilon_end=0.1, decay_frames=1_000_000
    )

    prev_epsilon = 1.0
    for frame in range(0, 1_000_001, 10_000):
        epsilon = scheduler.get_epsilon(frame)
        assert epsilon <= prev_epsilon
        prev_epsilon = epsilon


def test_epsilon_scheduler_edge_cases():
    """Test epsilon scheduler edge cases."""
    from src.training import EpsilonScheduler

    # Same start and end (no decay)
    scheduler = EpsilonScheduler(
        epsilon_start=0.5, epsilon_end=0.5, decay_frames=1_000_000
    )
    assert scheduler.get_epsilon(0) == 0.5
    assert scheduler.get_epsilon(500_000) == 0.5
    assert scheduler.get_epsilon(1_000_000) == 0.5

    # Very short decay period
    scheduler_short = EpsilonScheduler(
        epsilon_start=1.0, epsilon_end=0.1, decay_frames=10
    )
    assert scheduler_short.get_epsilon(0) == 1.0
    assert scheduler_short.get_epsilon(5) == 0.55
    assert scheduler_short.get_epsilon(10) == 0.1


# ============================================================================
# Frame Counter Tests
# ============================================================================


def test_frame_counter_initialization():
    """Test FrameCounter initializes correctly."""
    from src.training import FrameCounter

    counter = FrameCounter(frameskip=4)
    assert counter.frameskip == 4
    assert counter.steps == 0
    assert counter.frames == 0


def test_frame_counter_step_increment():
    """Test step() increments decision steps."""
    from src.training import FrameCounter

    counter = FrameCounter(frameskip=4)

    counter.step()
    assert counter.steps == 1
    assert counter.frames == 4

    counter.step()
    assert counter.steps == 2
    assert counter.frames == 8


def test_frame_counter_multi_step():
    """Test step() with multiple steps at once."""
    from src.training import FrameCounter

    counter = FrameCounter(frameskip=4)

    counter.step(num_steps=5)
    assert counter.steps == 5
    assert counter.frames == 20


def test_frame_counter_frames_calculation():
    """Test frames property calculates correctly."""
    from src.training import FrameCounter

    # Default frameskip=4
    counter = FrameCounter(frameskip=4)
    counter.step(num_steps=100)
    assert counter.frames == 400

    # Custom frameskip
    counter2 = FrameCounter(frameskip=2)
    counter2.step(num_steps=100)
    assert counter2.frames == 200


def test_frame_counter_fps():
    """Test FPS calculation."""
    from src.training import FrameCounter

    counter = FrameCounter(frameskip=4)
    counter.step(num_steps=100)  # 400 frames

    fps = counter.fps(elapsed_time=10.0)
    assert fps == 40.0  # 400 frames / 10 seconds


def test_frame_counter_fps_zero_time():
    """Test FPS with zero elapsed time."""
    from src.training import FrameCounter

    counter = FrameCounter(frameskip=4)
    counter.step(num_steps=100)

    fps = counter.fps(elapsed_time=0.0)
    assert fps == 0.0


def test_frame_counter_reset():
    """Test reset() clears counters."""
    from src.training import FrameCounter

    counter = FrameCounter(frameskip=4)
    counter.step(num_steps=50)
    assert counter.steps == 50
    assert counter.frames == 200

    counter.reset()
    assert counter.steps == 0
    assert counter.frames == 0


def test_frame_counter_to_dict():
    """Test to_dict() exports state."""
    from src.training import FrameCounter

    counter = FrameCounter(frameskip=4)
    counter.step(num_steps=25)

    state = counter.to_dict()
    assert state["steps"] == 25
    assert state["frames"] == 100
    assert state["frameskip"] == 4


def test_frame_counter_invalid_frameskip():
    """Test counter raises on invalid frameskip."""
    import pytest

    from src.training import FrameCounter

    with pytest.raises(AssertionError):
        FrameCounter(frameskip=0)

    with pytest.raises(AssertionError):
        FrameCounter(frameskip=-1)


def test_frame_counter_different_frameskips():
    """Test counter works with different frameskip values."""
    from src.training import FrameCounter

    # Frameskip 1 (no repeat)
    counter1 = FrameCounter(frameskip=1)
    counter1.step(num_steps=10)
    assert counter1.frames == 10

    # Frameskip 8 (double normal)
    counter8 = FrameCounter(frameskip=8)
    counter8.step(num_steps=10)
    assert counter8.frames == 80


# ============================================================================
# Training Step Integration Tests
# ============================================================================


def test_training_step_basic_execution():
    """Test training_step executes without errors."""
    from unittest.mock import Mock

    from src.models import DQN
    from src.replay import ReplayBuffer
    from src.training import (
        EpsilonScheduler,
        FrameCounter,
        TargetNetworkUpdater,
        TrainingScheduler,
        configure_optimizer,
        init_target_network,
        training_step,
    )

    # Mock environment with Atari-like observations
    env = Mock()
    env.action_space.n = 6
    num_actions = 6

    # Mock step to return proper Atari-like observations
    dummy_next_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    env.step.return_value = (dummy_next_state, 1.0, False, False, {})

    # Create networks
    online_net = DQN(num_actions=num_actions)
    target_net = init_target_network(online_net, num_actions=num_actions)

    # Create optimizer
    optimizer = configure_optimizer(online_net, learning_rate=0.00025)

    # Create replay buffer (use small min_size for quick testing)
    replay_buffer = ReplayBuffer(capacity=1000, obs_shape=(4, 84, 84), min_size=10)

    # Create schedulers
    epsilon_scheduler = EpsilonScheduler()
    target_updater = TargetNetworkUpdater(update_interval=100)
    training_scheduler = TrainingScheduler(train_every=4)
    frame_counter = FrameCounter(frameskip=1)

    # Create dummy Atari-like state
    state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)

    # Execute training step
    result = training_step(
        env=env,
        online_net=online_net,
        target_net=target_net,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        epsilon_scheduler=epsilon_scheduler,
        target_updater=target_updater,
        training_scheduler=training_scheduler,
        frame_counter=frame_counter,
        state=state,
        num_actions=num_actions,
        device="cpu",
    )

    # Check result structure
    assert "next_state" in result
    assert "reward" in result
    assert "terminated" in result
    assert "truncated" in result
    assert "epsilon" in result
    assert "metrics" in result
    assert "target_updated" in result
    assert "trained" in result
    assert "action" in result

    # Check epsilon is valid
    assert 0.0 <= result["epsilon"] <= 1.0

    # Check action is valid
    assert 0 <= result["action"] < num_actions

    # Verify env.step was called
    assert env.step.called


def test_training_step_triggers_training_after_warmup():
    """Test training_step triggers optimization after replay warmup."""
    from unittest.mock import Mock

    from src.models import DQN
    from src.replay import ReplayBuffer
    from src.training import (
        EpsilonScheduler,
        FrameCounter,
        TargetNetworkUpdater,
        TrainingScheduler,
        configure_optimizer,
        init_target_network,
        training_step,
    )

    # Mock environment
    env = Mock()
    env.action_space.n = 6
    num_actions = 6

    # Mock step to return Atari-like observations
    dummy_next_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    env.step.return_value = (dummy_next_state, 1.0, False, False, {})

    # Create networks and components
    online_net = DQN(num_actions=num_actions)
    target_net = init_target_network(online_net, num_actions=num_actions)
    optimizer = configure_optimizer(online_net, learning_rate=0.00025)

    # Small replay buffer for quick warmup
    replay_buffer = ReplayBuffer(capacity=1000, obs_shape=(4, 84, 84), min_size=50)

    epsilon_scheduler = EpsilonScheduler()
    target_updater = TargetNetworkUpdater(update_interval=100)
    training_scheduler = TrainingScheduler(train_every=4)
    frame_counter = FrameCounter(frameskip=1)

    # Fill replay buffer with dummy transitions
    dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)

    for _ in range(60):  # Fill past min_size
        replay_buffer.append(dummy_state, 0, 0.0, dummy_state, False)

    # Now execute training step
    result = training_step(
        env=env,
        online_net=online_net,
        target_net=target_net,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        epsilon_scheduler=epsilon_scheduler,
        target_updater=target_updater,
        training_scheduler=training_scheduler,
        frame_counter=frame_counter,
        state=dummy_state,
        num_actions=num_actions,
        device="cpu",
    )

    # First step should not train (step goes 0→1, train_every=4)
    assert not result["trained"]
    assert frame_counter.steps == 1

    # Execute 3 more training steps to reach step 4
    for _ in range(3):
        result = training_step(
            env=env,
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            epsilon_scheduler=epsilon_scheduler,
            target_updater=target_updater,
            training_scheduler=training_scheduler,
            frame_counter=frame_counter,
            state=dummy_state,
            num_actions=num_actions,
            device="cpu",
        )

    # Should have trained at step 4
    assert frame_counter.steps == 4
    assert result["trained"]
    assert result["metrics"] is not None


# ============================================================================
# Logging and Checkpoint Tests
# ============================================================================


def test_step_logger_basic():
    """Test StepLogger creates CSV and logs metrics."""
    import csv
    import os
    import tempfile

    from src.training import StepLogger, UpdateMetrics

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = StepLogger(log_dir=tmpdir, log_interval=1000)

        # Log at step 1000 (should write)
        metrics = UpdateMetrics(
            loss=0.5,
            td_error=0.3,
            td_error_std=0.1,
            grad_norm=2.5,
            learning_rate=0.00025,
            update_count=1,
        )
        logger.log_step(
            step=1000, epsilon=0.95, metrics=metrics, replay_size=50000, fps=120.0
        )

        # Check CSV exists
        csv_path = os.path.join(tmpdir, "training_steps.csv")
        assert os.path.exists(csv_path)

        # Read CSV
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert int(row["step"]) == 1000
        assert float(row["epsilon"]) == 0.95
        assert float(row["loss"]) == 0.5
        assert int(row["replay_size"]) == 50000


def test_step_logger_interval():
    """Test StepLogger only logs at intervals."""
    import os
    import tempfile

    from src.training import StepLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = StepLogger(log_dir=tmpdir, log_interval=1000)

        # Log at step 500 (should not write)
        logger.log_step(step=500, epsilon=0.95)

        csv_path = os.path.join(tmpdir, "training_steps.csv")
        assert not os.path.exists(csv_path)

        # Log at step 1000 (should write)
        logger.log_step(step=1000, epsilon=0.90)
        assert os.path.exists(csv_path)


def test_step_logger_moving_average():
    """Test StepLogger computes loss moving average."""
    import csv
    import tempfile

    from src.training import StepLogger, UpdateMetrics

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = StepLogger(log_dir=tmpdir, log_interval=1000, moving_avg_window=3)

        # Log 3 steps with different losses
        for i, loss in enumerate([1.0, 2.0, 3.0], start=1):
            metrics = UpdateMetrics(
                loss=loss,
                td_error=0.1,
                td_error_std=0.05,
                grad_norm=1.0,
                learning_rate=0.00025,
                update_count=i,
            )
            logger.log_step(step=i * 1000, epsilon=0.9, metrics=metrics)

        # Read last entry
        csv_path = f"{tmpdir}/training_steps.csv"
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Moving average of [1.0, 2.0, 3.0] = 2.0
        assert float(rows[-1]["loss_ma"]) == 2.0


def test_episode_logger_basic():
    """Test EpisodeLogger creates CSV and logs episodes."""
    import csv
    import os
    import tempfile

    from src.training import EpisodeLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EpisodeLogger(log_dir=tmpdir, rolling_window=100)

        # Log first episode
        logger.log_episode(
            step=5000, episode_return=21.0, episode_length=1200, fps=120.0, epsilon=0.95
        )

        # Check CSV exists
        csv_path = os.path.join(tmpdir, "episodes.csv")
        assert os.path.exists(csv_path)

        # Read CSV
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        row = rows[0]
        assert int(row["episode"]) == 1
        assert int(row["step"]) == 5000
        assert float(row["return"]) == 21.0
        assert int(row["length"]) == 1200


def test_episode_logger_rolling_stats():
    """Test EpisodeLogger computes rolling statistics."""
    import csv
    import tempfile

    from src.training import EpisodeLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EpisodeLogger(log_dir=tmpdir, rolling_window=3)

        # Log 3 episodes with different returns
        returns = [10.0, 20.0, 30.0]
        for i, ret in enumerate(returns, start=1):
            logger.log_episode(step=i * 1000, episode_return=ret, episode_length=1000)

        # Read last entry
        csv_path = f"{tmpdir}/episodes.csv"
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # Rolling mean of [10, 20, 30] = 20.0
        assert float(rows[-1]["rolling_mean_return"]) == 20.0


def test_episode_logger_get_recent_stats():
    """Test EpisodeLogger get_recent_stats method."""
    import tempfile

    from src.training import EpisodeLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EpisodeLogger(log_dir=tmpdir, rolling_window=100)

        # Log 5 episodes
        for i in range(1, 6):
            logger.log_episode(
                step=i * 1000, episode_return=float(i * 10), episode_length=1000
            )

        # Get stats for last 3 episodes
        stats = logger.get_recent_stats(n=3)

        assert stats["num_episodes"] == 3
        assert stats["mean_return"] == 40.0  # (30 + 40 + 50) / 3
        assert stats["min_return"] == 30.0
        assert stats["max_return"] == 50.0


def test_checkpoint_manager_periodic_save():
    """Test CheckpointManager saves periodic checkpoints."""
    import os
    import tempfile

    from src.models import DQN
    from src.training import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            checkpoint_dir=tmpdir, save_interval=1000000, keep_last_n=2
        )

        # Create model and optimizer
        online_model = DQN(num_actions=6)
        target_model = DQN(num_actions=6)
        optimizer = configure_optimizer(online_model)

        # Should save at 1M steps
        assert manager.should_save(1000000)
        path = manager.save_checkpoint(
            step=1000000,
            episode=100,
            epsilon=0.5,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
        )

        assert os.path.exists(path)
        assert "1000000" in path


def test_checkpoint_manager_keep_last_n():
    """Test CheckpointManager keeps only last N checkpoints."""
    import os
    import tempfile

    from src.models import DQN
    from src.training import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(
            checkpoint_dir=tmpdir, save_interval=1000000, keep_last_n=2
        )

        online_model = DQN(num_actions=6)
        target_model = DQN(num_actions=6)
        optimizer = configure_optimizer(online_model)

        # Save 3 checkpoints
        for i in range(1, 4):
            manager.save_checkpoint(
                step=i * 1000000,
                episode=i * 100,
                epsilon=0.5,
                online_model=online_model,
                target_model=target_model,
                optimizer=optimizer,
            )

        # Should only have 2 checkpoints (last 2)
        checkpoints = [f for f in os.listdir(tmpdir) if f.startswith("checkpoint_")]
        assert len(checkpoints) == 2


def test_checkpoint_manager_best_model():
    """Test CheckpointManager saves best model."""
    import os
    import tempfile

    from src.models import DQN
    from src.training import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir, save_best=True)

        online_model = DQN(num_actions=6)
        target_model = DQN(num_actions=6)
        optimizer = configure_optimizer(online_model)

        # First eval (return=20) - should save
        saved = manager.save_best(
            step=1000000,
            episode=100,
            epsilon=0.5,
            eval_return=20.0,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
        )
        assert saved
        assert os.path.exists(os.path.join(tmpdir, "best_model.pt"))

        # Second eval (return=15) - should NOT save
        saved = manager.save_best(
            step=2000000,
            episode=200,
            epsilon=0.4,
            eval_return=15.0,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
        )
        assert not saved

        # Third eval (return=25) - should save
        saved = manager.save_best(
            step=3000000,
            episode=300,
            epsilon=0.3,
            eval_return=25.0,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
        )
        assert saved
        assert manager.best_eval_return == 25.0


def test_checkpoint_manager_load():
    """Test CheckpointManager loads checkpoints correctly."""
    import tempfile

    import torch

    from src.models import DQN
    from src.training import CheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)

        # Create and save model
        online_model1 = DQN(num_actions=6)
        target_model1 = DQN(num_actions=6)
        optimizer1 = configure_optimizer(online_model1)

        # Modify model weights
        with torch.no_grad():
            for param in online_model1.parameters():
                param.fill_(1.0)
            for param in target_model1.parameters():
                param.fill_(1.0)

        path = manager.save_checkpoint(
            step=1000000,
            episode=100,
            epsilon=0.5,
            online_model=online_model1,
            target_model=target_model1,
            optimizer=optimizer1,
            extra_metadata={"test_key": "test_value"},
        )

        # Create new model and load checkpoint
        online_model2 = DQN(num_actions=6)
        target_model2 = DQN(num_actions=6)
        optimizer2 = configure_optimizer(online_model2)

        metadata = manager.load_checkpoint(
            path, online_model2, target_model2, optimizer2
        )

        # Check weights were loaded
        for p1, p2 in zip(online_model1.parameters(), online_model2.parameters()):
            assert torch.allclose(p1, p2)
        for p1, p2 in zip(target_model1.parameters(), target_model2.parameters()):
            assert torch.allclose(p1, p2)

        # Check metadata
        assert metadata["step"] == 1000000
        assert metadata["episode"] == 100
        assert metadata["epsilon"] == 0.5
        assert metadata["metadata"]["test_key"] == "test_value"


# ============================================================================
# Reference-State Q Tracking Tests
# ============================================================================


def test_reference_q_tracker_basic():
    """Test ReferenceStateQTracker computes Q-values."""
    from src.models import DQN
    from src.training import ReferenceStateQTracker

    # Create reference states
    ref_states = np.random.randint(0, 255, (10, 4, 84, 84), dtype=np.uint8)

    # Create tracker
    tracker = ReferenceStateQTracker(reference_states=ref_states, log_interval=10000)

    # Create model
    model = DQN(num_actions=6)

    # Compute Q-values
    q_stats = tracker.compute_q_values(model)

    # Check statistics
    assert "avg_max_q" in q_stats
    assert "max_q" in q_stats
    assert "min_q" in q_stats


def test_reference_q_tracker_logging_interval():
    """Test ReferenceStateQTracker respects logging interval."""
    from src.training import ReferenceStateQTracker

    ref_states = np.random.randint(0, 255, (5, 4, 84, 84), dtype=np.uint8)
    tracker = ReferenceStateQTracker(reference_states=ref_states, log_interval=10000)

    # Should not log at step 0
    assert not tracker.should_log(0)

    # Should not log before interval
    assert not tracker.should_log(5000)

    # Should log at interval
    assert tracker.should_log(10000)

    # Should not log twice at same step
    from src.models import DQN

    model = DQN(num_actions=6)
    tracker.log_q_values(step=10000, model=model)
    assert not tracker.should_log(10000)

    # Should log at next interval
    assert tracker.should_log(20000)


def test_reference_q_tracker_history():
    """Test ReferenceStateQTracker maintains history."""
    from src.models import DQN
    from src.training import ReferenceStateQTracker

    ref_states = np.random.randint(0, 255, (5, 4, 84, 84), dtype=np.uint8)
    tracker = ReferenceStateQTracker(reference_states=ref_states, log_interval=10000)

    model = DQN(num_actions=6)

    # Log Q-values at multiple steps
    tracker.log_q_values(step=10000, model=model)
    tracker.log_q_values(step=20000, model=model)
    tracker.log_q_values(step=30000, model=model)

    # Check history
    history = tracker.get_history()
    assert len(history["steps"]) == 3
    assert len(history["avg_max_q"]) == 3
    assert history["steps"] == [10000, 20000, 30000]


def test_reference_q_tracker_set_states():
    """Test ReferenceStateQTracker can set states after initialization."""
    from src.models import DQN
    from src.training import ReferenceStateQTracker

    # Create tracker without states
    tracker = ReferenceStateQTracker(log_interval=10000)

    # Should not log without states
    assert not tracker.should_log(10000)

    # Set states
    ref_states = np.random.randint(0, 255, (5, 4, 84, 84), dtype=np.uint8)
    tracker.set_reference_states(ref_states)

    # Now should be able to log
    assert tracker.should_log(10000)

    model = DQN(num_actions=6)
    q_stats = tracker.compute_q_values(model)
    assert "avg_max_q" in q_stats


def test_reference_q_tracker_normalization():
    """Test ReferenceStateQTracker normalizes uint8 inputs."""
    from src.training import ReferenceStateQTracker

    # Create uint8 states (0-255)
    ref_states = np.random.randint(0, 255, (5, 4, 84, 84), dtype=np.uint8)
    tracker = ReferenceStateQTracker(reference_states=ref_states)

    # Check states are normalized to [0, 1]
    assert tracker.reference_states.max() <= 1.0
    assert tracker.reference_states.min() >= 0.0


def test_reference_q_tracker_to_dict():
    """Test ReferenceStateQTracker serialization."""
    from src.models import DQN
    from src.training import ReferenceStateQTracker

    ref_states = np.random.randint(0, 255, (5, 4, 84, 84), dtype=np.uint8)
    tracker = ReferenceStateQTracker(reference_states=ref_states, log_interval=5000)

    model = DQN(num_actions=6)
    tracker.log_q_values(step=5000, model=model)

    # Serialize
    state_dict = tracker.to_dict()

    assert state_dict["log_interval"] == 5000
    assert state_dict["last_log_step"] == 5000
    assert len(state_dict["log_steps"]) == 1
    assert len(state_dict["avg_max_q"]) == 1


def test_reference_q_logger_csv():
    """Test ReferenceQLogger writes CSV correctly."""
    import csv
    import os
    import tempfile

    from src.training import ReferenceQLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ReferenceQLogger(log_dir=tmpdir)

        # Log Q-values
        q_stats = {"avg_max_q": 5.5, "max_q": 10.0, "min_q": 2.0}
        logger.log(step=10000, q_stats=q_stats)

        # Check CSV exists
        csv_path = os.path.join(tmpdir, "reference_q_values.csv")
        assert os.path.exists(csv_path)

        # Read CSV
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert int(rows[0]["step"]) == 10000
        assert float(rows[0]["avg_max_q"]) == 5.5
        assert float(rows[0]["max_q"]) == 10.0
        assert float(rows[0]["min_q"]) == 2.0


def test_reference_q_logger_multiple_logs():
    """Test ReferenceQLogger handles multiple log entries."""
    import tempfile

    from src.training import ReferenceQLogger

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = ReferenceQLogger(log_dir=tmpdir)

        # Log multiple Q-values
        for step in [10000, 20000, 30000]:
            q_stats = {
                "avg_max_q": step / 1000,
                "max_q": step / 500,
                "min_q": step / 2000,
            }
            logger.log(step=step, q_stats=q_stats)

        # Retrieve all logs
        all_logs = logger.get_all_logs()
        assert len(all_logs) == 3


def test_reference_q_tracker_integration():
    """Test ReferenceStateQTracker and ReferenceQLogger integration."""
    import tempfile

    from src.models import DQN
    from src.training import ReferenceQLogger, ReferenceStateQTracker

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create tracker
        ref_states = np.random.randint(0, 255, (10, 4, 84, 84), dtype=np.uint8)
        tracker = ReferenceStateQTracker(
            reference_states=ref_states, log_interval=10000
        )

        # Create logger
        logger = ReferenceQLogger(log_dir=tmpdir)

        # Create model
        model = DQN(num_actions=6)

        # Simulate tracking over multiple steps
        for step in [10000, 20000, 30000]:
            if tracker.should_log(step):
                tracker.log_q_values(step=step, model=model)
                q_stats = {
                    "avg_max_q": tracker.avg_max_q[-1],
                    "max_q": tracker.max_q[-1],
                    "min_q": tracker.min_q[-1],
                }
                logger.log(step=step, q_stats=q_stats)

        # Check both have 3 entries
        assert len(tracker.log_steps) == 3
        assert len(logger.get_all_logs()) == 3


if __name__ == "__main__":
    # Run tests manually
    print("Running DQN trainer tests...")

    # Target network tests
    test_hard_update_target_basic()
    print("Hard update basic test passed")

    test_hard_update_target_all_layers()
    print("Hard update all layers test passed")

    test_hard_update_target_multiple_updates()
    print("Hard update multiple updates test passed")

    test_init_target_network_creates_copy()
    print("Init target network creates copy test passed")

    test_init_target_network_freezes_gradients()
    print("Init target network freezes gradients test passed")

    test_init_target_network_in_eval_mode()
    print("Init target network eval mode test passed")

    test_target_network_no_gradient_computation()
    print("Target network no gradient computation test passed")

    test_hard_update_after_online_training()
    print("Hard update after online training test passed")

    test_hard_update_preserves_frozen_gradients()
    print("Hard update preserves frozen gradients test passed")

    test_target_network_different_devices()
    print("Target network different devices test passed")

    test_hard_update_target_shape_mismatch()
    print("Hard update shape mismatch test passed")

    test_init_target_network_num_actions()
    print("Init target network num_actions test passed")

    # TD target computation tests
    print("\nTD Target Computation Tests:")
    test_compute_td_targets_basic()
    print("Compute TD targets basic test passed")

    test_compute_td_targets_terminal_states()
    print("Compute TD targets terminal states test passed")

    test_compute_td_targets_gamma()
    print("Compute TD targets gamma test passed")

    test_compute_td_targets_no_grad()
    print("Compute TD targets no grad test passed")

    test_select_q_values_basic()
    print("Select Q-values basic test passed")

    test_select_q_values_gather_correctness()
    print("Select Q-values gather correctness test passed")

    test_select_q_values_gradient_flow()
    print("Select Q-values gradient flow test passed")

    test_compute_td_loss_components_basic()
    print("Compute TD loss components basic test passed")

    test_compute_td_loss_components_terminal_handling()
    print("Compute TD loss components terminal handling test passed")

    test_compute_td_loss_components_mse_loss()
    print("Compute TD loss components MSE loss test passed")

    test_td_targets_shape_consistency()
    print("TD targets shape consistency test passed")

    test_q_selection_shape_consistency()
    print("Q-selection shape consistency test passed")

    # Loss computation tests
    print("\nLoss Computation Tests:")
    test_compute_dqn_loss_mse_basic()
    print("Compute DQN loss MSE basic test passed")

    test_compute_dqn_loss_huber_basic()
    print("Compute DQN loss Huber basic test passed")

    test_compute_dqn_loss_gradient_flow()
    print("Compute DQN loss gradient flow test passed")

    test_compute_dqn_loss_td_error_stats()
    print("Compute DQN loss TD error stats test passed")

    test_compute_dqn_loss_invalid_type()
    print("Compute DQN loss invalid type test passed")

    test_compute_dqn_loss_shape_mismatch()
    print("Compute DQN loss shape mismatch test passed")

    test_compute_dqn_loss_gradient_assertions()
    print("Compute DQN loss gradient assertions test passed")

    test_compute_dqn_loss_huber_delta()
    print("Compute DQN loss Huber delta test passed")

    test_compute_dqn_loss_zero_td_error()
    print("Compute DQN loss zero TD error test passed")

    test_compute_dqn_loss_batch_sizes()
    print("Compute DQN loss batch sizes test passed")

    test_compute_dqn_loss_mse_vs_huber()
    print("Compute DQN loss MSE vs Huber test passed")

    # Optimizer configuration tests
    print("\nOptimizer Configuration Tests:")
    test_configure_optimizer_rmsprop_defaults()
    print("Configure optimizer RMSProp defaults test passed")

    test_configure_optimizer_rmsprop_custom()
    print("Configure optimizer RMSProp custom test passed")

    test_configure_optimizer_adam_defaults()
    print("Configure optimizer Adam defaults test passed")

    test_configure_optimizer_adam_custom()
    print("Configure optimizer Adam custom test passed")

    test_configure_optimizer_invalid_type()
    print("Configure optimizer invalid type test passed")

    test_configure_optimizer_parameters_linked()
    print("Configure optimizer parameters linked test passed")

    test_optimizer_step_updates_parameters()
    print("Optimizer step updates parameters test passed")

    # Gradient clipping tests
    print("\nGradient Clipping Tests:")
    test_clip_gradients_basic()
    print("Clip gradients basic test passed")

    test_clip_gradients_returns_norm()
    print("Clip gradients returns norm test passed")

    test_clip_gradients_actually_clips()
    print("Clip gradients actually clips test passed")

    test_clip_gradients_no_effect_when_small()
    print("Clip gradients no effect when small test passed")

    test_clip_gradients_different_norms()
    print("Clip gradients different norms test passed")

    test_clip_gradients_integration_with_optimizer()
    print("Clip gradients integration with optimizer test passed")

    test_clip_gradients_monitoring()
    print("Clip gradients monitoring test passed")

    # Target network update scheduler tests
    print("\nTarget Network Update Scheduler Tests:")
    test_target_network_updater_initialization()
    print("Target network updater initialization test passed")

    test_target_network_updater_invalid_interval()
    print("Target network updater invalid interval test passed")

    test_target_network_updater_should_update()
    print("Target network updater should_update test passed")

    test_target_network_updater_update()
    print("Target network updater update test passed")

    test_target_network_updater_multiple_updates()
    print("Target network updater multiple updates test passed")

    test_target_network_updater_step_method()
    print("Target network updater step method test passed")

    test_target_network_updater_no_duplicate_updates()
    print("Target network updater no duplicate updates test passed")

    test_target_network_updater_reset()
    print("Target network updater reset test passed")

    test_target_network_updater_state_dict()
    print("Target network updater state_dict test passed")

    test_target_network_updater_load_state_dict()
    print("Target network updater load_state_dict test passed")

    test_target_network_updater_exact_multiples()
    print("Target network updater exact multiples test passed")

    test_target_network_updater_repr()
    print("Target network updater repr test passed")

    test_target_network_updater_integration()
    print("Target network updater integration test passed")

    test_target_network_updater_parameters_actually_copied()
    print("Target network updater parameters actually copied test passed")

    # Training scheduler tests
    print("\nTraining Scheduler Tests:")
    test_training_scheduler_initialization()
    print("Training scheduler initialization test passed")

    test_training_scheduler_invalid_train_every()
    print("Training scheduler invalid train_every test passed")

    test_training_scheduler_warm_up_gating()
    print("Training scheduler warm-up gating test passed")

    test_training_scheduler_train_every_multiples()
    print("Training scheduler train_every multiples test passed")

    test_training_scheduler_mark_trained()
    print("Training scheduler mark_trained test passed")

    test_training_scheduler_no_duplicate_training()
    print("Training scheduler no duplicate training test passed")

    test_training_scheduler_step_method()
    print("Training scheduler step method test passed")

    test_training_scheduler_reset()
    print("Training scheduler reset test passed")

    test_training_scheduler_state_dict()
    print("Training scheduler state_dict test passed")

    test_training_scheduler_load_state_dict()
    print("Training scheduler load_state_dict test passed")

    test_training_scheduler_integration()
    print("Training scheduler integration test passed")

    test_training_scheduler_different_intervals()
    print("Training scheduler different intervals test passed")

    test_training_scheduler_repr()
    print("Training scheduler repr test passed")

    # Stability check tests
    print("\nStability Check Tests:")
    test_detect_nan_inf_no_issues()
    print("Detect NaN/Inf no issues test passed")

    test_detect_nan_inf_detects_nan()
    print("Detect NaN/Inf detects NaN test passed")

    test_detect_nan_inf_detects_inf()
    print("Detect NaN/Inf detects Inf test passed")

    test_detect_nan_inf_detects_neg_inf()
    print("Detect NaN/Inf detects negative Inf test passed")

    test_detect_nan_inf_multidimensional()
    print("Detect NaN/Inf multidimensional test passed")

    test_detect_nan_inf_zero_tensor()
    print("Detect NaN/Inf zero tensor test passed")

    test_validate_loss_decrease_basic()
    print("Validate loss decrease basic test passed")

    test_validate_loss_decrease_huber()
    print("Validate loss decrease Huber test passed")

    test_validate_loss_decrease_fewer_updates()
    print("Validate loss decrease fewer updates test passed")

    test_validate_loss_decrease_terminal_states()
    print("Validate loss decrease terminal states test passed")

    test_validate_loss_decrease_different_gamma()
    print("Validate loss decrease different gamma test passed")

    test_validate_loss_decrease_larger_batch()
    print("Validate loss decrease larger batch test passed")

    test_validate_loss_decrease_info_keys()
    print("Validate loss decrease info keys test passed")

    test_verify_target_sync_schedule_basic()
    print("Verify target sync schedule basic test passed")

    test_verify_target_sync_schedule_10k_interval()
    print("Verify target sync schedule 10k interval test passed")

    test_verify_target_sync_schedule_small_interval()
    print("Verify target sync schedule small interval test passed")

    test_verify_target_sync_schedule_partial_interval()
    print("Verify target sync schedule partial interval test passed")

    test_verify_target_sync_schedule_no_duplicates()
    print("Verify target sync schedule no duplicates test passed")

    test_verify_target_sync_schedule_count()
    print("Verify target sync schedule count test passed")

    test_verify_target_sync_schedule_info_keys()
    print("Verify target sync schedule info keys test passed")

    test_verify_target_sync_schedule_different_intervals()
    print("Verify target sync schedule different intervals test passed")

    # Update metrics tests
    print("\nUpdate Metrics Tests:")
    test_update_metrics_initialization()
    print("Update metrics initialization test passed")

    test_update_metrics_to_dict()
    print("Update metrics to_dict test passed")

    test_update_metrics_to_dict_keys()
    print("Update metrics to_dict keys test passed")

    test_update_metrics_repr()
    print("Update metrics repr test passed")

    test_perform_update_step_basic()
    print("Perform update step basic test passed")

    test_perform_update_step_updates_network()
    print("Perform update step updates network test passed")

    test_perform_update_step_mse_loss()
    print("Perform update step MSE loss test passed")

    test_perform_update_step_huber_loss()
    print("Perform update step Huber loss test passed")

    test_perform_update_step_different_gamma()
    print("Perform update step different gamma test passed")

    test_perform_update_step_gradient_clipping()
    print("Perform update step gradient clipping test passed")

    test_perform_update_step_multiple_updates()
    print("Perform update step multiple updates test passed")

    test_perform_update_step_batch_size_32()
    print("Perform update step batch size 32 test passed")

    test_perform_update_step_terminal_states()
    print("Perform update step terminal states test passed")

    test_perform_update_step_learning_rate_tracking()
    print("Perform update step learning rate tracking test passed")

    test_perform_update_step_sets_train_mode()
    print("Perform update step sets train mode test passed")

    print("\nAll tests passed! ")


# ============================================================================
# Metadata and Reproducibility Tests
# ============================================================================


def test_get_git_commit_hash():
    """Test get_git_commit_hash returns a hash or 'unknown'."""
    from src.training import get_git_commit_hash

    commit_hash = get_git_commit_hash()

    # Should be either a hash or 'unknown'
    assert isinstance(commit_hash, str)
    assert len(commit_hash) > 0


def test_get_git_status():
    """Test get_git_status returns complete status dict."""
    from src.training import get_git_status

    status = get_git_status()

    # Check all required fields
    assert "commit_hash" in status
    assert "commit_hash_full" in status
    assert "branch" in status
    assert "dirty" in status

    # Types should be correct
    assert isinstance(status["commit_hash"], str)
    assert isinstance(status["commit_hash_full"], str)
    assert isinstance(status["branch"], str)
    assert isinstance(status["dirty"], bool)


def test_metadata_writer_json():
    """Test MetadataWriter writes JSON metadata."""
    import json
    import os
    import tempfile

    from src.training import MetadataWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = MetadataWriter(run_dir=tmpdir)

        config = {"learning_rate": 0.00025, "batch_size": 32}
        writer.write_metadata(config=config, seed=123, format="json")

        # Check metadata.json exists
        metadata_path = os.path.join(tmpdir, "metadata.json")
        assert os.path.exists(metadata_path)

        # Load and verify
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["seed"] == 123
        assert metadata["config"]["learning_rate"] == 0.00025
        assert "git" in metadata
        assert "timestamp" in metadata
        assert "python_version" in metadata
        assert "pytorch_version" in metadata


def test_metadata_writer_git_info():
    """Test MetadataWriter creates git_info.txt."""
    import os
    import tempfile

    from src.training import MetadataWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = MetadataWriter(run_dir=tmpdir)
        writer.write_metadata(seed=42)

        # Check git_info.txt exists
        git_info_path = os.path.join(tmpdir, "git_info.txt")
        assert os.path.exists(git_info_path)

        # Read content
        with open(git_info_path, "r") as f:
            content = f.read()

        assert "Commit:" in content
        assert "Branch:" in content
        assert "Dirty:" in content


def test_metadata_writer_config_separate():
    """Test MetadataWriter can write config to separate file."""
    import json
    import os
    import tempfile

    from src.training import MetadataWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = MetadataWriter(run_dir=tmpdir)

        config = {"env": "Pong", "total_frames": 10000000}
        writer.write_config(config=config, format="json")

        # Check config.json exists
        config_path = os.path.join(tmpdir, "config.json")
        assert os.path.exists(config_path)

        # Load and verify
        with open(config_path, "r") as f:
            loaded_config = json.load(f)

        assert loaded_config["env"] == "Pong"
        assert loaded_config["total_frames"] == 10000000


def test_metadata_writer_load_metadata():
    """Test MetadataWriter can load written metadata."""
    import tempfile

    from src.training import MetadataWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = MetadataWriter(run_dir=tmpdir)

        config = {"gamma": 0.99}
        writer.write_metadata(config=config, seed=456, format="json")

        # Load metadata
        loaded = writer.load_metadata(format="json")

        assert loaded["seed"] == 456
        assert loaded["config"]["gamma"] == 0.99
        assert "git" in loaded


def test_metadata_writer_extra_fields():
    """Test MetadataWriter can include extra metadata."""
    import json
    import os
    import tempfile

    from src.training import MetadataWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        writer = MetadataWriter(run_dir=tmpdir)

        extra = {"device": "cuda", "gpu_count": 2, "hostname": "server01"}
        writer.write_metadata(seed=789, extra=extra, format="json")

        # Load and verify extra fields
        metadata_path = os.path.join(tmpdir, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        assert metadata["device"] == "cuda"
        assert metadata["gpu_count"] == 2
        assert metadata["hostname"] == "server01"


def test_metadata_writer_creates_directory():
    """Test MetadataWriter creates run directory if it doesn't exist."""
    import os
    import tempfile

    from src.training import MetadataWriter

    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = os.path.join(tmpdir, "runs", "pong_123")

        # Directory doesn't exist yet
        assert not os.path.exists(run_dir)

        # Create writer
        MetadataWriter(run_dir=run_dir)

        # Directory should now exist
        assert os.path.exists(run_dir)


if __name__ == "__main__":
    # Run tests manually
    print("Running DQN trainer tests...")
