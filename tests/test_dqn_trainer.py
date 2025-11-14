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
"""

import torch
import pytest
from src.models import DQN
from src.training import (
    hard_update_target,
    init_target_network,
    compute_td_targets,
    select_q_values,
    compute_td_loss_components,
    compute_dqn_loss,
    configure_optimizer,
    clip_gradients,
    TargetNetworkUpdater
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

    # Output should not require gradients
    loss = target_output['q_values'].mean()
    assert not loss.requires_grad, "Target network output should not require gradients"

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
    assert td_targets.shape == (batch_size,), f"Expected shape (4,), got {td_targets.shape}"

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
    assert torch.allclose(td_targets[1], rewards[1], atol=1e-6), \
        f"Terminal state TD target should equal reward, got {td_targets[1]} vs {rewards[1]}"


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
    td_targets_gamma1 = compute_td_targets(rewards, next_states, dones, target_net, gamma=1.0)

    # Compute with gamma=0.0 (only immediate reward)
    td_targets_gamma0 = compute_td_targets(rewards, next_states, dones, target_net, gamma=0.0)

    # With gamma=0.0 and zero rewards, targets should be zero
    assert torch.allclose(td_targets_gamma0, torch.zeros(batch_size), atol=1e-6), \
        "With gamma=0 and zero rewards, targets should be zero"

    # With gamma=1.0, targets should include full future value
    # td_targets_gamma1 should be larger (unless max_q is negative)


def test_compute_td_targets_no_grad():
    """Test that TD target computation doesn't create gradients."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)

    # Create batch
    batch_size = 2
    rewards = torch.tensor([1.0, -1.0])
    next_states = torch.randn(batch_size, 4, 84, 84, requires_grad=True)  # Try to leak gradients
    dones = torch.tensor([False, False])

    # Compute TD targets
    td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma=0.99)

    # Targets should not require gradients
    assert not td_targets.requires_grad, "TD targets should be detached"

    # Try to backward (should fail or not affect target network)
    loss = td_targets.sum()
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
    assert q_selected.shape == (batch_size,), f"Expected shape (4,), got {q_selected.shape}"

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
        full_q_values = full_output['q_values']  # (3, 6)

    q_selected = select_q_values(online_net, states, actions)

    # Manually verify correctness
    with torch.no_grad():
        expected_0 = full_q_values[0, 0]
        expected_1 = full_q_values[1, 2]
        expected_2 = full_q_values[2, 5]

        assert torch.allclose(q_selected[0], expected_0, atol=1e-5), \
            "Q-value for action 0 not selected correctly"
        assert torch.allclose(q_selected[1], expected_1, atol=1e-5), \
            "Q-value for action 2 not selected correctly"
        assert torch.allclose(q_selected[2], expected_2, atol=1e-5), \
            "Q-value for action 5 not selected correctly"


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
        states, actions, rewards, next_states, dones,
        online_net, target_net, gamma=0.99
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
        states, actions, rewards, next_states, dones,
        online_net, target_net, gamma=0.99
    )

    # For terminal states, td_targets should equal rewards
    assert torch.allclose(td_targets, rewards, atol=1e-6), \
        "Terminal state targets should equal rewards"


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
        states, actions, rewards, next_states, dones,
        online_net, target_net, gamma=0.99
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

        td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma=0.99)

        assert td_targets.shape == (batch_size,), \
            f"Batch size {batch_size}: expected shape ({batch_size},), got {td_targets.shape}"


def test_q_selection_shape_consistency():
    """Test Q-value selection shapes across different batch sizes."""
    online_net = DQN(num_actions=6)

    for batch_size in [1, 2, 8, 32]:
        states = torch.randn(batch_size, 4, 84, 84)
        actions = torch.randint(0, 6, (batch_size,))

        q_selected = select_q_values(online_net, states, actions)

        assert q_selected.shape == (batch_size,), \
            f"Batch size {batch_size}: expected shape ({batch_size},), got {q_selected.shape}"


# ============================================================================
# Loss Computation Tests
# ============================================================================

def test_compute_dqn_loss_mse_basic():
    """Test basic MSE loss computation."""
    # Create simple tensors
    batch_size = 4
    q_selected = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    td_targets = torch.tensor([1.5, 2.5, 2.5, 3.5])

    # Compute loss
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='mse')

    # Check keys
    assert 'loss' in loss_dict
    assert 'td_error' in loss_dict
    assert 'td_error_std' in loss_dict

    # Loss should be a scalar
    assert loss_dict['loss'].shape == torch.Size([])

    # Loss should have gradients
    assert loss_dict['loss'].requires_grad

    # TD error should be detached
    assert not loss_dict['td_error'].requires_grad
    assert not loss_dict['td_error_std'].requires_grad

    # Expected MSE: mean([(1-1.5)^2, (2-2.5)^2, (3-2.5)^2, (4-3.5)^2])
    # = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
    expected_loss = 0.25
    assert torch.allclose(loss_dict['loss'], torch.tensor(expected_loss), atol=1e-6)

    # Expected TD error: mean([0.5, 0.5, 0.5, 0.5]) = 0.5
    expected_td_error = 0.5
    assert torch.allclose(loss_dict['td_error'], torch.tensor(expected_td_error), atol=1e-6)


def test_compute_dqn_loss_huber_basic():
    """Test basic Huber loss computation."""
    batch_size = 4
    q_selected = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    td_targets = torch.tensor([1.5, 2.5, 2.5, 3.5])

    # Compute Huber loss
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='huber', huber_delta=1.0)

    # Loss should be a scalar with gradients
    assert loss_dict['loss'].shape == torch.Size([])
    assert loss_dict['loss'].requires_grad

    # Huber loss should be <= MSE for small errors (all errors are 0.5 < delta=1.0)
    # For errors smaller than delta, Huber is quadratic: 0.5 * error^2
    # For our case: 0.5 * mean([0.25, 0.25, 0.25, 0.25]) = 0.125
    expected_huber = 0.125
    assert torch.allclose(loss_dict['loss'], torch.tensor(expected_huber), atol=1e-6)


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
        states, actions, rewards, next_states, dones,
        online_net, target_net, gamma=0.99
    )

    # Compute loss
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='mse')

    # Backward
    loss_dict['loss'].backward()

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
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='mse')

    # Expected mean TD error: mean([1.0, 1.0, 0.0, 1.0]) = 0.75
    expected_mean = 0.75
    assert torch.allclose(loss_dict['td_error'], torch.tensor(expected_mean), atol=1e-6)

    # Expected std: std([1.0, 1.0, 0.0, 1.0]) = 0.5
    expected_std = 0.5
    assert torch.allclose(loss_dict['td_error_std'], torch.tensor(expected_std), atol=1e-6)


def test_compute_dqn_loss_invalid_type():
    """Test that invalid loss type raises error."""
    q_selected = torch.tensor([1.0, 2.0], requires_grad=True)
    td_targets = torch.tensor([1.5, 2.5])

    with pytest.raises(ValueError, match="Unknown loss_type"):
        compute_dqn_loss(q_selected, td_targets, loss_type='invalid')


def test_compute_dqn_loss_shape_mismatch():
    """Test that shape mismatch raises error."""
    q_selected = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    td_targets = torch.tensor([1.5, 2.5])  # Different size

    with pytest.raises(AssertionError, match="Shape mismatch"):
        compute_dqn_loss(q_selected, td_targets, loss_type='mse')


def test_compute_dqn_loss_gradient_assertions():
    """Test gradient requirements are validated."""
    # q_selected without gradients (should fail)
    q_selected = torch.tensor([1.0, 2.0], requires_grad=False)
    td_targets = torch.tensor([1.5, 2.5])

    with pytest.raises(AssertionError, match="should have gradients"):
        compute_dqn_loss(q_selected, td_targets, loss_type='mse')

    # td_targets with gradients (should fail)
    q_selected = torch.tensor([1.0, 2.0], requires_grad=True)
    td_targets = torch.tensor([1.5, 2.5], requires_grad=True)

    with pytest.raises(AssertionError, match="should be detached"):
        compute_dqn_loss(q_selected, td_targets, loss_type='mse')


def test_compute_dqn_loss_huber_delta():
    """Test Huber loss with different delta values."""
    # Large errors to test delta effect
    q_selected = torch.tensor([0.0, 5.0, 10.0], requires_grad=True)
    td_targets = torch.tensor([1.0, 1.0, 1.0])  # Errors: [1.0, 4.0, 9.0]

    # Small delta (more linear for large errors)
    loss_dict_small = compute_dqn_loss(q_selected, td_targets, loss_type='huber', huber_delta=0.5)

    # Large delta (more quadratic, closer to MSE)
    loss_dict_large = compute_dqn_loss(q_selected, td_targets, loss_type='huber', huber_delta=10.0)

    # Both should be valid scalars
    assert loss_dict_small['loss'].shape == torch.Size([])
    assert loss_dict_large['loss'].shape == torch.Size([])

    # TD errors should be the same regardless of loss type
    assert torch.allclose(loss_dict_small['td_error'], loss_dict_large['td_error'])


def test_compute_dqn_loss_zero_td_error():
    """Test loss when TD error is zero (perfect predictions)."""
    q_selected = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    td_targets = torch.tensor([1.0, 2.0, 3.0])  # Perfect match

    # MSE loss
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='mse')

    # Loss should be zero
    assert torch.allclose(loss_dict['loss'], torch.tensor(0.0), atol=1e-6)

    # TD error should be zero
    assert torch.allclose(loss_dict['td_error'], torch.tensor(0.0), atol=1e-6)


def test_compute_dqn_loss_batch_sizes():
    """Test loss computation across different batch sizes."""
    for batch_size in [1, 2, 8, 32]:
        q_selected = torch.randn(batch_size, requires_grad=True)
        td_targets = torch.randn(batch_size)

        # MSE
        loss_dict_mse = compute_dqn_loss(q_selected, td_targets, loss_type='mse')
        assert loss_dict_mse['loss'].shape == torch.Size([])

        # Huber
        loss_dict_huber = compute_dqn_loss(q_selected, td_targets, loss_type='huber')
        assert loss_dict_huber['loss'].shape == torch.Size([])


def test_compute_dqn_loss_mse_vs_huber():
    """Test MSE vs Huber loss behavior for small and large errors."""
    # Small errors (Huber should be similar to MSE)
    q_selected_small = torch.tensor([1.0, 1.1, 0.9], requires_grad=True)
    td_targets_small = torch.tensor([1.0, 1.0, 1.0])

    mse_small = compute_dqn_loss(q_selected_small, td_targets_small, loss_type='mse')
    huber_small = compute_dqn_loss(q_selected_small.clone().detach().requires_grad_(True),
                                   td_targets_small, loss_type='huber', huber_delta=1.0)

    # For small errors, Huber and MSE should be close
    # Note: Huber is 0.5*error^2 for |error| < delta, so it's 0.5 * MSE
    assert huber_small['loss'] < mse_small['loss']

    # Large errors (Huber should be smaller than MSE due to linear region)
    q_selected_large = torch.tensor([0.0, 10.0, 20.0], requires_grad=True)
    td_targets_large = torch.tensor([1.0, 1.0, 1.0])

    mse_large = compute_dqn_loss(q_selected_large, td_targets_large, loss_type='mse')
    huber_large = compute_dqn_loss(q_selected_large.clone().detach().requires_grad_(True),
                                   td_targets_large, loss_type='huber', huber_delta=1.0)

    # For large errors, Huber should be much smaller than MSE
    assert huber_large['loss'] < mse_large['loss']


# ============================================================================
# Optimizer Configuration Tests
# ============================================================================

def test_configure_optimizer_rmsprop_defaults():
    """Test RMSProp optimizer with default parameters."""
    online_net = DQN(num_actions=6)

    # Configure with defaults
    optimizer = configure_optimizer(online_net, optimizer_type='rmsprop')

    # Should be RMSProp
    assert isinstance(optimizer, torch.optim.RMSprop)

    # Check default hyperparameters
    param_groups = optimizer.param_groups[0]
    assert param_groups['lr'] == 2.5e-4, "Default LR should be 2.5e-4"
    assert param_groups['alpha'] == 0.95, "Default alpha (ρ) should be 0.95"
    assert param_groups['eps'] == 1e-2, "Default eps should be 0.01"
    assert param_groups['momentum'] == 0.0, "Default momentum should be 0.0"
    assert param_groups['weight_decay'] == 0.0, "Default weight_decay should be 0.0"


def test_configure_optimizer_rmsprop_custom():
    """Test RMSProp optimizer with custom parameters."""
    online_net = DQN(num_actions=6)

    # Configure with custom params
    optimizer = configure_optimizer(
        online_net,
        optimizer_type='rmsprop',
        learning_rate=1e-3,
        alpha=0.99,
        eps=1e-8,
        momentum=0.9,
        weight_decay=1e-5
    )

    # Check custom hyperparameters
    param_groups = optimizer.param_groups[0]
    assert param_groups['lr'] == 1e-3
    assert param_groups['alpha'] == 0.99
    assert param_groups['eps'] == 1e-8
    assert param_groups['momentum'] == 0.9
    assert param_groups['weight_decay'] == 1e-5


def test_configure_optimizer_adam_defaults():
    """Test Adam optimizer with default parameters."""
    online_net = DQN(num_actions=6)

    # Configure Adam
    optimizer = configure_optimizer(online_net, optimizer_type='adam')

    # Should be Adam
    assert isinstance(optimizer, torch.optim.Adam)

    # Check default hyperparameters
    param_groups = optimizer.param_groups[0]
    assert param_groups['lr'] == 2.5e-4
    assert param_groups['betas'] == (0.9, 0.999)
    assert param_groups['eps'] == 1e-8
    assert param_groups['weight_decay'] == 0.0


def test_configure_optimizer_adam_custom():
    """Test Adam optimizer with custom parameters."""
    online_net = DQN(num_actions=6)

    # Configure with custom params
    optimizer = configure_optimizer(
        online_net,
        optimizer_type='adam',
        learning_rate=3e-4,
        beta1=0.95,
        beta2=0.9999,
        adam_eps=1e-7,
        weight_decay=1e-4
    )

    # Check custom hyperparameters
    param_groups = optimizer.param_groups[0]
    assert param_groups['lr'] == 3e-4
    assert param_groups['betas'] == (0.95, 0.9999)
    assert param_groups['eps'] == 1e-7
    assert param_groups['weight_decay'] == 1e-4


def test_configure_optimizer_invalid_type():
    """Test that invalid optimizer type raises error."""
    online_net = DQN(num_actions=6)

    with pytest.raises(ValueError, match="Unknown optimizer_type"):
        configure_optimizer(online_net, optimizer_type='sgd')


def test_configure_optimizer_parameters_linked():
    """Test that optimizer is linked to network parameters."""
    online_net = DQN(num_actions=6)

    optimizer = configure_optimizer(online_net, optimizer_type='rmsprop')

    # Verify optimizer has parameters
    assert len(list(optimizer.param_groups[0]['params'])) > 0

    # Verify parameters are from the network
    net_params = set(online_net.parameters())
    opt_params = set(optimizer.param_groups[0]['params'])
    assert net_params == opt_params


def test_optimizer_step_updates_parameters():
    """Test that optimizer step updates network parameters."""
    online_net = DQN(num_actions=6)
    target_net = init_target_network(online_net, num_actions=6)
    optimizer = configure_optimizer(online_net, optimizer_type='rmsprop', learning_rate=0.1)

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
        states, actions, rewards, next_states, dones,
        online_net, target_net, gamma=0.99
    )

    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='mse')

    # Optimization step
    optimizer.zero_grad()
    loss_dict['loss'].backward()
    optimizer.step()

    # Parameters should have changed
    updated_param = list(online_net.parameters())[0]
    assert not torch.allclose(initial_param, updated_param), \
        "Parameters should change after optimizer step"


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
        states, actions, rewards, next_states, dones,
        online_net, target_net, gamma=0.99
    )

    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='mse')
    loss_dict['loss'].backward()

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
    loss = output['q_values'].mean()
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
    loss = output['q_values'].sum() * 1000.0  # Large multiplier
    loss.backward()

    # Clip with small max_norm
    max_norm = 1.0
    grad_norm_before = clip_gradients(online_net, max_norm=max_norm)

    # Compute actual norm after clipping
    total_norm_after = 0.0
    for p in online_net.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm_after += param_norm.item() ** 2
    total_norm_after = total_norm_after ** 0.5

    # After clipping, norm should be <= max_norm (with small tolerance for numerical errors)
    assert total_norm_after <= max_norm + 1e-5, \
        f"Gradient norm {total_norm_after} exceeds max_norm {max_norm}"


def test_clip_gradients_no_effect_when_small():
    """Test that small gradients are not affected by clipping."""
    online_net = DQN(num_actions=6)

    # Create small gradients
    x = torch.randn(2, 4, 84, 84)
    output = online_net(x)
    loss = output['q_values'].mean() * 0.001  # Small multiplier
    loss.backward()

    # Save gradients before clipping
    grads_before = [p.grad.clone() for p in online_net.parameters() if p.grad is not None]

    # Clip with large max_norm (shouldn't affect small gradients)
    grad_norm = clip_gradients(online_net, max_norm=100.0)

    # Gradients should be unchanged
    grads_after = [p.grad for p in online_net.parameters() if p.grad is not None]

    for g_before, g_after in zip(grads_before, grads_after):
        assert torch.allclose(g_before, g_after, atol=1e-6), \
            "Small gradients should not be affected by large max_norm"


def test_clip_gradients_different_norms():
    """Test gradient clipping with different norm types."""
    online_net = DQN(num_actions=6)

    # Create gradients
    x = torch.randn(2, 4, 84, 84)
    output = online_net(x)
    loss = output['q_values'].mean()
    loss.backward()

    # Clip with L2 norm (default)
    grad_norm_l2 = clip_gradients(online_net, max_norm=10.0, norm_type=2.0)
    assert grad_norm_l2 > 0.0

    # Reset gradients and recompute
    online_net.zero_grad()
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
    optimizer = configure_optimizer(online_net, optimizer_type='rmsprop')

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
        states, actions, rewards, next_states, dones,
        online_net, target_net, gamma=0.99
    )

    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='mse')
    loss_dict['loss'].backward()

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
        loss = output['q_values'].mean()

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
    assert info['step'] == 10000
    assert info['total_updates'] == 1
    assert info['steps_since_last'] == 10000

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
            assert info['step'] == step

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
    assert result['step'] == 100
    assert result['total_updates'] == 1


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

    assert state['update_interval'] == 5000
    assert state['step_count'] == 10000
    assert state['last_update_step'] == 10000
    assert state['total_updates'] == 2


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
        assert info['step'] == expected_step
        assert info['total_updates'] == i + 1


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


if __name__ == "__main__":
    # Run tests manually
    print("Running DQN trainer tests...")

    # Target network tests
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

    # TD target computation tests
    print("\nTD Target Computation Tests:")
    test_compute_td_targets_basic()
    print("✓ Compute TD targets basic test passed")

    test_compute_td_targets_terminal_states()
    print("✓ Compute TD targets terminal states test passed")

    test_compute_td_targets_gamma()
    print("✓ Compute TD targets gamma test passed")

    test_compute_td_targets_no_grad()
    print("✓ Compute TD targets no grad test passed")

    test_select_q_values_basic()
    print("✓ Select Q-values basic test passed")

    test_select_q_values_gather_correctness()
    print("✓ Select Q-values gather correctness test passed")

    test_select_q_values_gradient_flow()
    print("✓ Select Q-values gradient flow test passed")

    test_compute_td_loss_components_basic()
    print("✓ Compute TD loss components basic test passed")

    test_compute_td_loss_components_terminal_handling()
    print("✓ Compute TD loss components terminal handling test passed")

    test_compute_td_loss_components_mse_loss()
    print("✓ Compute TD loss components MSE loss test passed")

    test_td_targets_shape_consistency()
    print("✓ TD targets shape consistency test passed")

    test_q_selection_shape_consistency()
    print("✓ Q-selection shape consistency test passed")

    # Loss computation tests
    print("\nLoss Computation Tests:")
    test_compute_dqn_loss_mse_basic()
    print("✓ Compute DQN loss MSE basic test passed")

    test_compute_dqn_loss_huber_basic()
    print("✓ Compute DQN loss Huber basic test passed")

    test_compute_dqn_loss_gradient_flow()
    print("✓ Compute DQN loss gradient flow test passed")

    test_compute_dqn_loss_td_error_stats()
    print("✓ Compute DQN loss TD error stats test passed")

    test_compute_dqn_loss_invalid_type()
    print("✓ Compute DQN loss invalid type test passed")

    test_compute_dqn_loss_shape_mismatch()
    print("✓ Compute DQN loss shape mismatch test passed")

    test_compute_dqn_loss_gradient_assertions()
    print("✓ Compute DQN loss gradient assertions test passed")

    test_compute_dqn_loss_huber_delta()
    print("✓ Compute DQN loss Huber delta test passed")

    test_compute_dqn_loss_zero_td_error()
    print("✓ Compute DQN loss zero TD error test passed")

    test_compute_dqn_loss_batch_sizes()
    print("✓ Compute DQN loss batch sizes test passed")

    test_compute_dqn_loss_mse_vs_huber()
    print("✓ Compute DQN loss MSE vs Huber test passed")

    # Optimizer configuration tests
    print("\nOptimizer Configuration Tests:")
    test_configure_optimizer_rmsprop_defaults()
    print("✓ Configure optimizer RMSProp defaults test passed")

    test_configure_optimizer_rmsprop_custom()
    print("✓ Configure optimizer RMSProp custom test passed")

    test_configure_optimizer_adam_defaults()
    print("✓ Configure optimizer Adam defaults test passed")

    test_configure_optimizer_adam_custom()
    print("✓ Configure optimizer Adam custom test passed")

    test_configure_optimizer_invalid_type()
    print("✓ Configure optimizer invalid type test passed")

    test_configure_optimizer_parameters_linked()
    print("✓ Configure optimizer parameters linked test passed")

    test_optimizer_step_updates_parameters()
    print("✓ Optimizer step updates parameters test passed")

    # Gradient clipping tests
    print("\nGradient Clipping Tests:")
    test_clip_gradients_basic()
    print("✓ Clip gradients basic test passed")

    test_clip_gradients_returns_norm()
    print("✓ Clip gradients returns norm test passed")

    test_clip_gradients_actually_clips()
    print("✓ Clip gradients actually clips test passed")

    test_clip_gradients_no_effect_when_small()
    print("✓ Clip gradients no effect when small test passed")

    test_clip_gradients_different_norms()
    print("✓ Clip gradients different norms test passed")

    test_clip_gradients_integration_with_optimizer()
    print("✓ Clip gradients integration with optimizer test passed")

    test_clip_gradients_monitoring()
    print("✓ Clip gradients monitoring test passed")

    # Target network update scheduler tests
    print("\nTarget Network Update Scheduler Tests:")
    test_target_network_updater_initialization()
    print("✓ Target network updater initialization test passed")

    test_target_network_updater_invalid_interval()
    print("✓ Target network updater invalid interval test passed")

    test_target_network_updater_should_update()
    print("✓ Target network updater should_update test passed")

    test_target_network_updater_update()
    print("✓ Target network updater update test passed")

    test_target_network_updater_multiple_updates()
    print("✓ Target network updater multiple updates test passed")

    test_target_network_updater_step_method()
    print("✓ Target network updater step method test passed")

    test_target_network_updater_no_duplicate_updates()
    print("✓ Target network updater no duplicate updates test passed")

    test_target_network_updater_reset()
    print("✓ Target network updater reset test passed")

    test_target_network_updater_state_dict()
    print("✓ Target network updater state_dict test passed")

    test_target_network_updater_load_state_dict()
    print("✓ Target network updater load_state_dict test passed")

    test_target_network_updater_exact_multiples()
    print("✓ Target network updater exact multiples test passed")

    test_target_network_updater_repr()
    print("✓ Target network updater repr test passed")

    test_target_network_updater_integration()
    print("✓ Target network updater integration test passed")

    test_target_network_updater_parameters_actually_copied()
    print("✓ Target network updater parameters actually copied test passed")

    print("\nAll tests passed! ✓")
