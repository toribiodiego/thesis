# DQN Training Update Flow

Comprehensive documentation for the DQN Q-learning update process, including TD targets, loss computation, optimizer configuration, target network synchronization, metrics logging, and debugging strategies.

## Table of Contents

1. [Overview](#overview)
2. [TD Target Computation](#td-target-computation)
3. [Loss Functions](#loss-functions)
4. [Optimizer Configuration](#optimizer-configuration)
5. [Gradient Clipping](#gradient-clipping)
6. [Target Network Synchronization](#target-network-synchronization)
7. [Training Frequency Scheduling](#training-frequency-scheduling)
8. [Complete Update Pipeline](#complete-update-pipeline)
9. [Metrics Logging](#metrics-logging)
10. [Debugging Unstable Training](#debugging-unstable-training)
11. [Testing and Validation](#testing-and-validation)
12. [Configuration Flags](#configuration-flags)

---

## Overview

The DQN training update implements the Q-learning algorithm with deep neural networks. The core idea is to minimize the temporal difference (TD) error between predicted Q-values and target Q-values computed from the Bellman equation.

**Key Components:**
- **Online Network**: The Q-network being trained (parameters updated every step)
- **Target Network**: Frozen copy of Q-network for stable target computation (updated every C steps)
- **Replay Buffer**: Storage for past experiences (s, a, r, s', done)
- **Optimizer**: RMSProp or Adam for parameter updates
- **Loss Function**: MSE or Huber loss on TD errors

**Training Loop High-Level:**
```
For each environment step:
  1. Select action using ε-greedy policy
  2. Execute action in environment
  3. Store transition (s, a, r, s', done) in replay buffer
  4. If buffer ready and step % train_every == 0:
     - Sample minibatch from replay buffer
     - Perform training update (see Complete Update Pipeline)
  5. If step % target_update_interval == 0:
     - Sync target network with online network
```

---

## TD Target Computation

### Bellman Equation

The TD target represents the expected cumulative future reward:

```
y = r + γ × (1 - done) × max_a' Q_target(s', a')
```

Where:
- `y`: TD target (what we want Q(s,a) to equal)
- `r`: Immediate reward
- `γ`: Discount factor (default: 0.99)
- `done`: Terminal flag (1.0 if episode ended, 0.0 otherwise)
- `s'`: Next state
- `Q_target`: Target network (frozen, updated periodically)
- `max_a'`: Maximum Q-value over all actions in next state

### Implementation

**Function:** `compute_td_targets(rewards, next_states, dones, target_net, gamma=0.99)`

**Key Details:**
- Computed under `torch.no_grad()` (no gradient tracking)
- Target network must be in eval mode
- Terminal states: `done=1` → `y = r` (no future reward)
- Non-terminal states: `done=0` → `y = r + γ × max Q_target(s', a')`
- Output shape: `(batch_size,)` after squeeze

**Example:**
```python
# Compute TD targets for minibatch
td_targets = compute_td_targets(
    rewards=batch['rewards'],        # (32,)
    next_states=batch['next_states'], # (32, 4, 84, 84)
    dones=batch['dones'],            # (32,)
    target_net=target_net,
    gamma=0.99
)
# td_targets shape: (32,)
```

**Why use target network?**
- Stabilizes training by providing consistent targets
- Without it, Q-values and targets are highly correlated (moving target problem)
- Introduced in 2015 Nature DQN paper (not in 2013 arXiv version)

---

## Loss Functions

### TD Error

The temporal difference error measures prediction error:

```
TD error = Q_online(s, a) - y
```

Where:
- `Q_online(s, a)`: Predicted Q-value from online network
- `y`: TD target (from Bellman equation)

### Mean Squared Error (MSE) Loss

**Default loss function:**

```
L_MSE = (1/N) × Σ (Q_online(s, a) - y)²
```

- Penalizes large errors quadratically
- Sensitive to outliers
- Standard in original DQN paper

**Implementation:** `compute_dqn_loss(q_selected, td_targets, loss_type='mse')`

### Huber Loss

**Alternative loss function for robustness:**

```
L_Huber(δ) = {
  0.5 × (Q - y)²           if |Q - y| ≤ δ
  δ × (|Q - y| - 0.5δ)     otherwise
}
```

- Quadratic for small errors (|error| ≤ δ)
- Linear for large errors (|error| > δ)
- Less sensitive to outliers
- Default δ = 1.0

**Implementation:** `compute_dqn_loss(q_selected, td_targets, loss_type='huber', huber_delta=1.0)`

### Loss Function Selection

**Use MSE when:**
- Following original DQN paper exactly
- Reward scale is well-controlled
- Outliers are not a concern

**Use Huber when:**
- Rewards have high variance or outliers
- Training is unstable with MSE
- Want more robust learning

---

## Optimizer Configuration

### RMSProp (DQN Paper Default)

**Hyperparameters:**
```python
optimizer = torch.optim.RMSProp(
    network.parameters(),
    lr=2.5e-4,      # Learning rate
    alpha=0.95,     # Smoothing constant (ρ in paper)
    eps=1e-2        # Numerical stability (ε in paper)
)
```

**Why RMSProp?**
- Adaptive learning rate per parameter
- Normalizes gradients by moving average of squared gradients
- Works well with non-stationary objectives (RL setting)
- Original DQN paper choice

**Implementation:** `configure_optimizer(network, optimizer_type='rmsprop', learning_rate=2.5e-4, alpha=0.95, eps=1e-2)`

### Adam (Alternative)

**Hyperparameters:**
```python
optimizer = torch.optim.Adam(
    network.parameters(),
    lr=2.5e-4,      # Learning rate
    betas=(0.9, 0.999),  # Momentum parameters
    eps=1e-8        # Numerical stability
)
```

**When to use Adam:**
- Want faster initial learning
- Have experience tuning Adam hyperparameters
- Not concerned with exact DQN reproduction

### Learning Rate

**Default:** `2.5e-4` (DQN paper value)

**Considerations:**
- Too high: Unstable training, divergence
- Too low: Slow learning, may not converge in reasonable time
- Can use learning rate schedules (not in original DQN)

---

## Gradient Clipping

### Global Norm Clipping

**Purpose:** Prevent exploding gradients that cause training instability

**Method:** Clip gradient norm to maximum value

```python
torch.nn.utils.clip_grad_norm_(
    network.parameters(),
    max_norm=10.0,  # DQN paper default
    norm_type=2.0   # L2 norm
)
```

**How it works:**
1. Compute global gradient norm: `||g|| = sqrt(Σ g_i²)`
2. If `||g|| > max_norm`: Scale gradients by `max_norm / ||g||`
3. Otherwise: Leave gradients unchanged

**Implementation:** `clip_gradients(network, max_norm=10.0, norm_type=2.0)`

**Returns:** Gradient norm BEFORE clipping (useful for monitoring)

**When gradients explode:**
- Gradient norm > 100: Likely instability
- Gradient norm > 1000: Severe instability, check learning rate and reward scale
- Consistently hitting max_norm: Consider reducing learning rate

---

## Target Network Synchronization

### Update Policy

**Hard Update (DQN Paper):**
- Copy all parameters from online network to target network
- Perform update every C environment steps
- Default C = 10,000 steps

**Why periodic updates?**
- Provides stable targets for multiple updates
- Reduces correlation between Q-values and targets
- Key stability improvement from 2015 Nature paper

### Implementation

**Class:** `TargetNetworkUpdater(update_interval=10000)`

**Methods:**
- `should_update(current_step)`: Check if update should occur
- `update(online_net, target_net, current_step)`: Perform hard sync
- `step(online_net, target_net, current_step)`: Convenience method

**Example:**
```python
updater = TargetNetworkUpdater(update_interval=10000)

for env_step in range(total_steps):
    # ... training ...

    # Check and perform target update
    update_info = updater.step(online_net, target_net, env_step)
    if update_info:
        print(f"Target network updated at step {update_info['step']}")
```

**Update Schedule:**
- Updates occur at exact multiples of update_interval
- Steps: 10000, 20000, 30000, ...
- No duplicate updates at same step

### 2013 vs 2015 DQN

**2013 NIPS version (original arXiv):**
- Single network for both Q(s,a) and targets
- No separate target network
- Simpler but less stable (moving target problem)

**2015 Nature version:**
- Separate target network updated every C steps
- update_interval=10000 (default)
- More stable training, fixed targets for multiple updates

**Configuration:**

The target network can be toggled via the `agent.target_update_interval` config key:

**Enable target network (2015 mode, recommended):**
```yaml
# In config YAML
agent:
  target_update_interval: 10000  # Update every 10K steps (standard)
```

**Disable target network (2013 purist mode):**
```yaml
# In config YAML
agent:
  target_update_interval: 0  # Disabled (or set to null)
```

**CLI override:**
```bash
# Disable target network at runtime
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --agent.target_update_interval 0

# Use different update interval
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml \
  --agent.target_update_interval 5000
```

**Implementation behavior:**
- `target_update_interval > 0`: Use separate target network, hard-copy every N steps
- `target_update_interval == 0` or `null`: Disable target network (2013 mode)

**When to disable target network:**
- Ablation studies comparing 2013 vs 2015 DQN
- Reproducing original 2013 NIPS paper exactly
- Research on target network effectiveness

**When to enable target network:**
- Standard DQN training (recommended)
- Stability is important
- Reproducing 2015 Nature DQN paper

---

## Training Frequency Scheduling

### Update Frequency

**DQN Paper:** Perform one gradient update every 4 environment steps

**Why not every step?**
- More efficient (less computation)
- Allows more environment interaction per update
- Original DQN paper configuration

### Implementation

**Class:** `TrainingScheduler(train_every=4)`

**Methods:**
- `should_train(env_step, replay_buffer)`: Check if should train
- `mark_trained(env_step)`: Mark training occurred
- `step(env_step, replay_buffer)`: Convenience method

**Warm-up Gating:**
- No training until replay buffer has enough samples
- Checks `replay_buffer.can_sample()` before allowing training
- Ensures sufficient experience for meaningful updates

**Example:**
```python
scheduler = TrainingScheduler(train_every=4)
buffer = ReplayBuffer(capacity=100000, batch_size=32)

for env_step in range(total_steps):
    # Interact with environment, store in buffer
    # ...

    # Check if should train
    if scheduler.should_train(env_step, buffer):
        batch = buffer.sample()
        # Perform training update
        # ...
        scheduler.mark_trained(env_step)
```

**Training Schedule:**
- First training at step `train_every` (step 4 with default)
- Subsequent training: 8, 12, 16, ...
- Only when `buffer.can_sample() == True`

---

## Complete Update Pipeline

### Full Training Update

**Function:** `perform_update_step(online_net, target_net, optimizer, batch, ...)`

**Step-by-Step Process:**

```python
# 1. Set online network to training mode
online_net.train()

# 2. Extract batch data
states = batch['states']          # (32, 4, 84, 84)
actions = batch['actions']        # (32,)
rewards = batch['rewards']        # (32,)
next_states = batch['next_states']# (32, 4, 84, 84)
dones = batch['dones']            # (32,)

# 3. Compute TD targets (no gradient)
td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma=0.99)
# td_targets: (32,)

# 4. Select Q-values for taken actions (with gradient)
q_selected = select_q_values(online_net, states, actions)
# q_selected: (32,)

# 5. Compute loss and TD error statistics
loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='mse')
loss = loss_dict['loss']          # Scalar
td_error = loss_dict['td_error']  # Mean |TD error|
td_error_std = loss_dict['td_error_std']  # Std of TD errors

# 6. Backward pass (compute gradients)
optimizer.zero_grad()
loss.backward()

# 7. Clip gradients by global norm
grad_norm = clip_gradients(online_net, max_norm=10.0)

# 8. Update parameters
optimizer.step()

# 9. Get learning rate for logging
learning_rate = optimizer.param_groups[0]['lr']

# 10. Return metrics
return UpdateMetrics(
    loss=loss.item(),
    td_error=td_error,
    td_error_std=td_error_std,
    grad_norm=grad_norm,
    learning_rate=learning_rate,
    update_count=current_update_count
)
```

### Q-Value Selection

**Function:** `select_q_values(online_net, states, actions)`

**Purpose:** Extract Q-values for actions that were actually taken

**Process:**
```python
# Forward pass through network
output = online_net(states)       # Returns dict
q_values = output['q_values']     # (32, num_actions)

# Gather Q-values for specific actions
actions_unsqueezed = actions.unsqueeze(1)  # (32, 1)
q_selected = q_values.gather(1, actions_unsqueezed).squeeze(1)  # (32,)
```

**Why gather?**
- We have Q-values for ALL actions: `(batch_size, num_actions)`
- We only want Q-values for actions that were taken
- `gather()` selects specific indices along dimension 1

---

## Metrics Logging

### Collected Metrics

**UpdateMetrics class stores:**

1. **loss** (float)
   - Training loss value (MSE or Huber)
   - Primary signal: should decrease over training
   - Typical range: 0.01 - 10.0 (depends on reward scale)

2. **td_error** (float)
   - Mean absolute TD error: `mean(|Q - y|)`
   - Indicates prediction accuracy
   - Lower is better
   - Typical range: 0.1 - 5.0

3. **td_error_std** (float)
   - Standard deviation of TD errors
   - Indicates prediction consistency
   - High std suggests high variance in predictions
   - Useful for detecting instability

4. **grad_norm** (float)
   - Gradient norm BEFORE clipping
   - Detects exploding gradients
   - Typical range: 0.1 - 50.0
   - Warning if > 100, critical if > 1000

5. **learning_rate** (float)
   - Current optimizer learning rate
   - Useful if using LR schedules
   - Default: 2.5e-4

6. **update_count** (int)
   - Total number of updates performed
   - Use as x-axis in plots
   - Tracks training progress

### Logging Example

```python
# Perform training update
metrics = perform_update_step(
    online_net, target_net, optimizer, batch,
    gamma=0.99,
    loss_type='mse',
    max_grad_norm=10.0,
    update_count=current_step
)

# Convert to dictionary for logging
metrics_dict = metrics.to_dict()

# Log to file/tensorboard/wandb/etc.
logger.log(metrics_dict, step=current_step)

# Print for monitoring
print(f"Step {metrics.update_count}: "
      f"Loss={metrics.loss:.4f}, "
      f"TD_error={metrics.td_error:.4f}, "
      f"Grad_norm={metrics.grad_norm:.2f}")
```

### What to Plot

**Essential plots:**
1. Loss vs. update count (should decrease)
2. TD error vs. update count (should decrease)
3. Gradient norm vs. update count (should stabilize)
4. Episode return vs. episode count (should increase)

**Advanced plots:**
5. TD error std vs. update count (stability indicator)
6. Learning rate vs. update count (if using schedules)
7. Q-value distribution (histogram)
8. Action distribution during training
9. Average max-Q on reference states (Q-value growth indicator)

### Reference-State Q Logging (Subtask 6)

**Purpose:** Track Q-value growth and stability by evaluating the network on a fixed batch of reference states.

**Why reference states?**
- Training Q-values vary due to changing state distribution
- Reference states provide consistent measurement across training
- Detects Q-value collapse, explosion, or stagnation
- Standard metric in DQN papers and implementations

**Implementation:**

**1. Capture reference batch:**
```python
# During initial exploration (after replay_min_transitions)
reference_batch = replay_buffer.sample(batch_size=100)

# Save to disk for consistency across runs
torch.save({
    'states': reference_batch['states'],
    'actions': reference_batch['actions'],
    'rewards': reference_batch['rewards'],
    'next_states': reference_batch['next_states'],
    'dones': reference_batch['dones']
}, 'reference_states.pt')
```

**2. Compute average max-Q periodically:**
```python
def compute_reference_max_q(network, reference_states):
    """Compute average max-Q on reference states."""
    network.eval()
    with torch.no_grad():
        output = network(reference_states)
        q_values = output['q_values']  # (batch_size, num_actions)
        max_q = q_values.max(dim=1)[0]  # (batch_size,)
        avg_max_q = max_q.mean().item()
    network.train()
    return avg_max_q

# Log every eval_frames (e.g., 250K frames)
if env_step % config.intervals.eval_frames == 0:
    avg_max_q = compute_reference_max_q(online_net, reference_states)
    logger.log({'reference_max_q': avg_max_q}, step=env_step)
```

**3. Interpret the metric:**

**Expected behavior:**
- Initial: avg_max_q ≈ 0.0 (random initialization)
- Early training: avg_max_q increases as agent learns
- Mid training: avg_max_q stabilizes at positive value
- Late training: avg_max_q plateaus (converged Q-values)

**Warning signs:**
- **Explosion:** avg_max_q > 1000 (Q-values diverging, check learning rate)
- **Collapse:** avg_max_q drops to near 0 after initial growth (training instability)
- **Stagnation:** avg_max_q never increases (no learning, check exploration/replay)

**Storage location:**
- Reference batch: `experiments/dqn_atari/runs/{experiment_name}_{seed}/reference_states.pt`
- Created once at step `replay_min_transitions` (e.g., 50K frames)
- Reused throughout training for consistent measurements

**When to refresh reference batch:**
- After major code changes affecting preprocessing
- If reference batch corrupted or deleted
- For ablation studies (use same batch across experiments for fair comparison)

**Configuration:**
```yaml
# In config YAML (add to logging section)
logging:
  reference_batch_size: 100  # Number of states in reference batch
  compute_reference_q: true   # Enable reference-state Q logging
```

**Example usage in training loop:**
```python
# After replay buffer has enough samples
if not reference_batch_loaded and replay_buffer.size >= config.agent.replay_min_transitions:
    reference_path = Path(config.experiment.output_dir) / 'reference_states.pt'

    if reference_path.exists():
        # Load existing reference batch
        reference_data = torch.load(reference_path)
        reference_states = reference_data['states'].to(device)
    else:
        # Create and save new reference batch
        reference_batch = replay_buffer.sample(config.logging.reference_batch_size)
        reference_states = reference_batch['states'].to(device)
        torch.save(reference_batch, reference_path)

    reference_batch_loaded = True

# Log reference max-Q every eval_frames
if reference_batch_loaded and env_step % config.intervals.eval_frames == 0:
    avg_max_q = compute_reference_max_q(online_net, reference_states)
    logger.log({'reference_max_q': avg_max_q}, step=env_step)
```

---

## Debugging Unstable Training

### Common Issues and Solutions

#### 1. Exploding TD Error

**Symptoms:**
- TD error increases rapidly
- Loss diverges to very large values
- Q-values become extremely large

**Causes:**
- Learning rate too high
- Reward scale too large
- Target network not updating
- Gradient clipping not applied

**Solutions:**
```python
# Reduce learning rate
optimizer = configure_optimizer(network, learning_rate=1e-4)  # Instead of 2.5e-4

# Ensure gradient clipping
grad_norm = clip_gradients(network, max_norm=10.0)

# Verify target network updates
updater = TargetNetworkUpdater(update_interval=10000)
# Check updater.total_updates increases

# Clip rewards to [-1, 1]
reward = np.clip(reward, -1.0, 1.0)
```

#### 2. Stale Targets

**Symptoms:**
- Learning stalls
- Q-values don't improve
- TD error plateaus at high value

**Causes:**
- Target network not updating
- Update interval too large
- Target network gradients not frozen

**Solutions:**
```python
# Verify target network updates occur
update_info = updater.step(online_net, target_net, env_step)
if update_info:
    print(f"Target updated: {update_info}")

# Reduce update interval (for debugging)
updater = TargetNetworkUpdater(update_interval=5000)  # Instead of 10000

# Verify gradients frozen
for param in target_net.parameters():
    assert not param.requires_grad, "Target network should have frozen gradients"
```

#### 3. NaN/Inf Values

**Symptoms:**
- Loss becomes NaN
- Q-values become Inf
- Training crashes

**Causes:**
- Learning rate too high
- Numerical instability in loss computation
- Exploding gradients

**Solutions:**
```python
# Detect NaN/Inf early
if detect_nan_inf(loss, "loss"):
    print("WARNING: NaN/Inf detected in loss!")
    # Save checkpoint and investigate

# Reduce learning rate aggressively
optimizer = configure_optimizer(network, learning_rate=1e-5)

# Use Huber loss instead of MSE (more robust)
loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='huber')

# Check optimizer epsilon
optimizer = torch.optim.RMSProp(..., eps=1e-2)  # Increase if needed
```

#### 4. Slow Learning

**Symptoms:**
- Q-values barely change
- Episode returns don't improve
- TD error decreases very slowly

**Causes:**
- Learning rate too low
- Insufficient exploration (ε too low)
- Training frequency too low
- Replay buffer not sampling diverse experiences

**Solutions:**
```python
# Increase learning rate
optimizer = configure_optimizer(network, learning_rate=5e-4)

# Train more frequently
scheduler = TrainingScheduler(train_every=1)  # Instead of 4

# Ensure sufficient exploration
# epsilon should start at 1.0 and decay slowly

# Check replay buffer diversity
# Ensure buffer is filled with varied experiences
```

### Debugging Checklist

**Before each training run:**
- [ ] Verify target network initialized as copy of online network
- [ ] Verify target network gradients frozen (`requires_grad=False`)
- [ ] Verify replay buffer warm-up threshold set correctly
- [ ] Verify training frequency schedule (default: every 4 steps)
- [ ] Verify target update interval (default: 10,000 steps)
- [ ] Verify gradient clipping enabled (max_norm=10.0)
- [ ] Verify optimizer hyperparameters (lr=2.5e-4, alpha=0.95, eps=1e-2)

**During training (monitor these):**
- [ ] Loss decreases over time (plot it!)
- [ ] TD error decreases over time
- [ ] Gradient norm stays reasonable (< 100)
- [ ] No NaN/Inf values in loss or Q-values
- [ ] Target network updates occur at correct intervals
- [ ] Episode returns increase over time

**If training is unstable:**
1. Check gradient norm (if > 100, reduce LR)
2. Check for NaN/Inf values
3. Verify target network updates
4. Try Huber loss instead of MSE
5. Reduce learning rate by factor of 2-10
6. Increase gradient clipping (max_norm=1.0 for very unstable)

---

## Testing and Validation

### Unit Tests

**Run all training tests:**
```bash
# Activate virtual environment
source .venv/bin/activate

# Run test file
export PYTHONPATH=.
python tests/test_dqn_trainer.py
```

**Key test categories:**
1. Target network tests (12 tests)
   - Hard update correctness
   - Gradient freezing
   - Parameter copying

2. TD target computation tests (12 tests)
   - Bellman equation correctness
   - Terminal state handling
   - Gradient isolation

3. Loss computation tests (11 tests)
   - MSE loss
   - Huber loss
   - TD error statistics

4. Optimizer configuration tests (7 tests)
   - RMSProp setup
   - Adam setup
   - Parameter linking

5. Gradient clipping tests (8 tests)
   - Norm computation
   - Clipping behavior
   - Different norm types

6. Target network update scheduling tests (14 tests)
   - Update timing
   - Interval correctness
   - No duplicate updates

7. Training frequency scheduling tests (13 tests)
   - Warm-up gating
   - Update intervals
   - Replay buffer integration

8. Stability check tests (21 tests)
   - NaN/Inf detection
   - Loss decrease validation
   - Target sync schedule verification

9. Update metrics tests (15 tests)
   - Metrics collection
   - Full update pipeline
   - Different configurations

**Total: 113 comprehensive tests**

### Validation Tests

**Toy batch loss decrease:**
```python
# Verify loss decreases on synthetic data
success, info = validate_loss_decrease(
    compute_dqn_loss, online_net, optimizer,
    states, actions, rewards, next_states, dones, target_net,
    num_updates=10
)
assert success, f"Loss should decrease: {info}"
```

**Target sync schedule verification:**
```python
# Verify target updates occur at correct times
success, info = verify_target_sync_schedule(
    updater, online_net, target_net,
    max_steps=50000, expected_interval=10000
)
assert success, f"Target sync schedule incorrect: {info}"
```

### Integration Testing

**Minimal training run:**
```python
# Test full training loop for 1000 steps
online_net = DQN(num_actions=6)
target_net = init_target_network(online_net, num_actions=6)
optimizer = configure_optimizer(online_net)
buffer = ReplayBuffer(capacity=10000, batch_size=32)
updater = TargetNetworkUpdater(update_interval=500)
scheduler = TrainingScheduler(train_every=4)

# Fill buffer with random experiences
for _ in range(1000):
    buffer.add(...)

# Run training updates
for step in range(1000):
    if scheduler.should_train(step, buffer):
        batch = buffer.sample()
        metrics = perform_update_step(
            online_net, target_net, optimizer, batch,
            update_count=step
        )
        scheduler.mark_trained(step)

    updater.step(online_net, target_net, step)

# Verify training occurred
assert scheduler.training_step_count > 0
assert updater.total_updates > 0
```

---

## Configuration Flags

### Hyperparameter Summary

**Core hyperparameters (DQN paper defaults):**

```python
config = {
    # Network
    'num_actions': 6,  # Environment-specific

    # Optimizer
    'optimizer_type': 'rmsprop',
    'learning_rate': 2.5e-4,
    'rmsprop_alpha': 0.95,
    'rmsprop_eps': 1e-2,

    # Loss
    'loss_type': 'mse',  # or 'huber'
    'huber_delta': 1.0,
    'gamma': 0.99,

    # Gradient clipping
    'max_grad_norm': 10.0,

    # Target network
    'target_update_interval': 10000,

    # Training frequency
    'train_every': 4,

    # Replay buffer
    'buffer_capacity': 100000,
    'batch_size': 32,
    'min_buffer_size': 10000,  # Warm-up threshold
}
```

### Reproducing 2013 DQN (No Target Network)

```python
config = {
    # Use same network for online and target
    'use_target_network': False,  # Custom flag

    # Or set update interval to 1 (update every step)
    'target_update_interval': 1,

    # All other hyperparameters same as above
}
```

### Reproducing 2015 DQN (Nature Paper)

```python
config = {
    # Standard configuration (see above)
    'target_update_interval': 10000,

    # Nature paper specifics
    'epsilon_start': 1.0,
    'epsilon_end': 0.1,
    'epsilon_decay_frames': 1000000,
    'total_frames': 50000000,
    'eval_epsilon': 0.05,
}
```

### Debugging Configuration

**For unstable training:**
```python
config = {
    'learning_rate': 1e-4,      # Reduce LR
    'max_grad_norm': 1.0,       # Aggressive clipping
    'loss_type': 'huber',       # More robust
    'train_every': 1,           # More frequent updates
    'target_update_interval': 5000,  # More frequent target updates
}
```

**For faster debugging:**
```python
config = {
    'buffer_capacity': 10000,   # Smaller buffer
    'min_buffer_size': 1000,    # Faster warm-up
    'batch_size': 16,           # Smaller batches
    'train_every': 1,           # More frequent updates
}
```

---

## Summary

The DQN training update flow consists of:

1. **Sample** minibatch from replay buffer
2. **Compute** TD targets using target network (Bellman equation)
3. **Select** Q-values for actions taken
4. **Calculate** loss (MSE or Huber) between Q-values and targets
5. **Backpropagate** gradients through online network
6. **Clip** gradients to prevent instability
7. **Update** network parameters with optimizer
8. **Log** metrics (loss, TD error, grad norm, LR)
9. **Sync** target network every C steps
10. **Monitor** for instability (NaN/Inf, exploding gradients, stale targets)

**Key stability mechanisms:**
- Target network (periodic hard updates)
- Gradient clipping (max norm = 10.0)
- Replay buffer (breaks temporal correlations)
- Training frequency (update every 4 steps)

**Essential monitoring:**
- Loss should decrease
- TD error should decrease
- Gradient norm should stay < 100
- Episode returns should increase

For detailed API documentation, see docstrings in `src/training/dqn_trainer.py`.

For implementation examples, see tests in `tests/test_dqn_trainer.py`.
