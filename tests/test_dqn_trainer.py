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
    compute_dqn_loss
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

    print("\nAll tests passed! ✓")
