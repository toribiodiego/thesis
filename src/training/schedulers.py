"""Schedulers for target network updates, training frequency, and epsilon decay.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Optional


class TargetNetworkUpdater:
    """
    Scheduler for periodic target network updates.

    Tracks environment steps and triggers hard updates to the target network
    at fixed intervals (default: every 10,000 steps).

    Historical Context:
        Target networks were introduced in the 2015 Nature DQN paper as a stability
        improvement. The original 2013 arXiv paper did NOT use a separate target
        network. For purist 2013 reproduction:
        - Set update_interval=1 (updates every step, equivalent to single network)
        - Or use online_net for both Q(s,a) and target Q(s',a') computations

    Attributes:
        update_interval: Number of environment steps between target updates
        step_count: Current environment step counter
        last_update_step: Step at which last update occurred
        total_updates: Total number of updates performed

    Configuration Notes:
        - DQN 2015 (Nature): update_interval=10000 (default)
        - DQN 2013 (arXiv): update_interval=1 or same network for both
        - Smaller intervals: More stable but slower learning
        - Larger intervals: Faster learning but potentially less stable

    Example:
        >>> # 2015 Nature paper setup (default)
        >>> online_net = DQN(num_actions=6)
        >>> target_net = init_target_network(online_net, num_actions=6)
        >>> updater = TargetNetworkUpdater(update_interval=10000)
        >>>
        >>> # 2013 arXiv paper setup (purist reproduction)
        >>> updater_2013 = TargetNetworkUpdater(update_interval=1)
        >>>
        >>> # Training loop
        >>> for step in range(100000):
        ...     # ... training code ...
        ...     if updater.should_update(step):
        ...         updater.update(online_net, target_net)
        ...         print(f"Target network updated at step {step}")
    """

    def __init__(self, update_interval: int = 10000):
        """
        Initialize target network updater.

        Args:
            update_interval: Number of environment steps between updates (default: 10000)

        Notes:
            - DQN 2015 Nature paper uses C=10,000 steps (default)
            - DQN 2013 arXiv paper: use update_interval=1 for purist reproduction
            - First update occurs at step update_interval (not step 0)
            - Updates occur at exact multiples: 10000, 20000, 30000, etc.
            - Setting interval=1 makes every step update (equivalent to 2013 single network)

        Configuration Examples:
            - Modern stable training: update_interval=10000 (default)
            - Faster updates for debugging: update_interval=1000
            - 2013 paper reproduction: update_interval=1
        """
        if update_interval <= 0:
            raise ValueError(f"update_interval must be positive, got {update_interval}")

        self.update_interval = update_interval
        self.step_count = 0
        self.last_update_step = 0
        self.total_updates = 0

    def should_update(self, current_step: int) -> bool:
        """
        Check if target network should be updated at current step.

        Args:
            current_step: Current environment step count

        Returns:
            True if target network should be updated, False otherwise

        Notes:
            - Updates occur at exact multiples of update_interval
            - Returns True at steps: update_interval, 2*update_interval, etc.
            - Returns False at step 0
        """
        # Update at exact multiples of update_interval
        if current_step > 0 and current_step % self.update_interval == 0:
            # Only update if we haven't already updated at this step
            return current_step != self.last_update_step
        return False

    def update(
        self,
        online_net: nn.Module,
        target_net: nn.Module,
        current_step: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Perform hard update of target network.

        Copies all parameters from online network to target network and
        updates internal counters.

        Args:
            online_net: Online Q-network
            target_net: Target Q-network
            current_step: Current environment step (optional, for logging)

        Returns:
            Dictionary with update info:
                - 'step': Step at which update occurred
                - 'total_updates': Total number of updates so far
                - 'steps_since_last': Steps since last update

        Example:
            >>> info = updater.update(online_net, target_net, current_step=10000)
            >>> print(f"Updated at step {info['step']}, total updates: {info['total_updates']}")
        """
        # Perform hard update
        hard_update_target(online_net, target_net)

        # Update counters
        if current_step is not None:
            self.step_count = current_step
        else:
            self.step_count += self.update_interval

        steps_since_last = self.step_count - self.last_update_step
        self.last_update_step = self.step_count
        self.total_updates += 1

        return {
            'step': self.step_count,
            'total_updates': self.total_updates,
            'steps_since_last': steps_since_last
        }

    def step(
        self,
        online_net: nn.Module,
        target_net: nn.Module,
        current_step: int
    ) -> Optional[Dict[str, int]]:
        """
        Convenience method to check and update in one call.

        Args:
            online_net: Online Q-network
            target_net: Target Q-network
            current_step: Current environment step

        Returns:
            Update info dict if update occurred, None otherwise

        Example:
            >>> # In training loop
            >>> update_info = updater.step(online_net, target_net, current_step)
            >>> if update_info:
            ...     print(f"Target updated: {update_info}")
        """
        if self.should_update(current_step):
            return self.update(online_net, target_net, current_step)
        return None

    def reset(self):
        """
        Reset all counters.

        Useful for starting a new training run or after loading a checkpoint.
        """
        self.step_count = 0
        self.last_update_step = 0
        self.total_updates = 0

    def state_dict(self) -> Dict[str, int]:
        """
        Get state for checkpointing.

        Returns:
            Dictionary with all internal state

        Example:
            >>> checkpoint = {
            ...     'model': model.state_dict(),
            ...     'updater': updater.state_dict()
            ... }
        """
        return {
            'update_interval': self.update_interval,
            'step_count': self.step_count,
            'last_update_step': self.last_update_step,
            'total_updates': self.total_updates
        }

    def load_state_dict(self, state_dict: Dict[str, int]):
        """
        Load state from checkpoint.

        Args:
            state_dict: Dictionary with state (from state_dict())

        Example:
            >>> updater.load_state_dict(checkpoint['updater'])
        """
        self.update_interval = state_dict['update_interval']
        self.step_count = state_dict['step_count']
        self.last_update_step = state_dict['last_update_step']
        self.total_updates = state_dict['total_updates']

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"TargetNetworkUpdater(interval={self.update_interval}, "
            f"step={self.step_count}, updates={self.total_updates})"
        )


class TrainingScheduler:
    """
    Scheduler for training frequency with replay buffer warm-up.

    Controls when optimization steps should be performed based on:
    - Replay buffer warm-up (sufficient samples available)
    - Training frequency (perform updates every k environment steps)

    Attributes:
        train_every: Number of environment steps between training updates
        env_step_count: Current environment step counter
        training_step_count: Number of training updates performed
        last_train_step: Environment step at which last training occurred

    Example:
        >>> from src.memory import ReplayBuffer
        >>> buffer = ReplayBuffer(capacity=100000, batch_size=32)
        >>> scheduler = TrainingScheduler(train_every=4)
        >>>
        >>> # Training loop
        >>> for env_step in range(100000):
        ...     # Interact with environment, store in buffer
        ...     # ...
        ...     if scheduler.should_train(env_step, buffer):
        ...         batch = buffer.sample()
        ...         # Perform training update
        ...         scheduler.mark_trained(env_step)
    """

    def __init__(self, train_every: int = 4):
        """
        Initialize training scheduler.

        Args:
            train_every: Number of environment steps between training updates (default: 4)

        Notes:
            - DQN paper uses train_every=4 (one update per 4 environment steps)
            - Smaller values: More frequent updates, slower but more sample efficient
            - Larger values: Faster environment interaction, less sample efficient
            - Training only occurs after replay buffer warm-up (can_sample=True)

        Configuration Examples:
            - Standard DQN: train_every=4 (default)
            - More frequent updates: train_every=1 (every step)
            - Less frequent updates: train_every=8 or train_every=16
        """
        if train_every <= 0:
            raise ValueError(f"train_every must be positive, got {train_every}")

        self.train_every = train_every
        self.env_step_count = 0
        self.training_step_count = 0
        self.last_train_step = 0

    def should_train(self, env_step: int, replay_buffer) -> bool:
        """
        Check if training update should be performed.

        Args:
            env_step: Current environment step count
            replay_buffer: Replay buffer with can_sample() method

        Returns:
            True if should train, False otherwise

        Notes:
            - Returns False if replay buffer cannot sample (warm-up not complete)
            - Returns True if env_step is multiple of train_every AND buffer ready
            - First training occurs at step train_every (not step 0)

        Example:
            >>> if scheduler.should_train(env_step, buffer):
            ...     # Perform training update
            ...     batch = buffer.sample()
        """
        # Check if replay buffer has enough samples
        if not replay_buffer.can_sample():
            return False

        # Check if it's time to train (every k steps)
        if env_step > 0 and env_step % self.train_every == 0:
            # Only train if we haven't already trained at this step
            return env_step != self.last_train_step

        return False

    def mark_trained(self, env_step: int):
        """
        Mark that training occurred at this environment step.

        Updates internal counters to track training progress.

        Args:
            env_step: Environment step at which training occurred

        Example:
            >>> if scheduler.should_train(env_step, buffer):
            ...     # ... perform training ...
            ...     scheduler.mark_trained(env_step)
        """
        self.env_step_count = env_step
        self.last_train_step = env_step
        self.training_step_count += 1

    def step(self, env_step: int, replay_buffer) -> bool:
        """
        Convenience method to check and mark training in one call.

        Args:
            env_step: Current environment step
            replay_buffer: Replay buffer with can_sample() method

        Returns:
            True if should train, False otherwise

        Example:
            >>> if scheduler.step(env_step, buffer):
            ...     # Perform training update
            ...     batch = buffer.sample()
        """
        should = self.should_train(env_step, replay_buffer)
        if should:
            self.mark_trained(env_step)
        return should

    def reset(self):
        """
        Reset all counters.

        Useful for starting a new training run.
        """
        self.env_step_count = 0
        self.training_step_count = 0
        self.last_train_step = 0

    def state_dict(self) -> Dict[str, int]:
        """
        Get state for checkpointing.

        Returns:
            Dictionary with all internal state

        Example:
            >>> checkpoint = {
            ...     'model': model.state_dict(),
            ...     'scheduler': scheduler.state_dict()
            ... }
        """
        return {
            'train_every': self.train_every,
            'env_step_count': self.env_step_count,
            'training_step_count': self.training_step_count,
            'last_train_step': self.last_train_step
        }

    def load_state_dict(self, state_dict: Dict[str, int]):
        """
        Load state from checkpoint.

        Args:
            state_dict: Dictionary with state (from state_dict())

        Example:
            >>> scheduler.load_state_dict(checkpoint['scheduler'])
        """
        self.train_every = state_dict['train_every']
        self.env_step_count = state_dict['env_step_count']
        self.training_step_count = state_dict['training_step_count']
        self.last_train_step = state_dict['last_train_step']

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"TrainingScheduler(train_every={self.train_every}, "
            f"env_step={self.env_step_count}, "
            f"training_steps={self.training_step_count})"
        )


def detect_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> bool:
    """
    Detect NaN or Inf values in a tensor.

    Args:
        tensor: Tensor to check
        name: Name of the tensor (for warning messages)

    Returns:
        True if NaN or Inf detected, False otherwise

    Example:
        >>> loss = compute_dqn_loss(q_selected, td_targets)['loss']
        >>> if detect_nan_inf(loss, "loss"):
        ...     print("Warning: NaN/Inf detected in loss!")
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        return True
    return False


def validate_loss_decrease(
    loss_fn,
    network: nn.Module,
    optimizer: torch.optim.Optimizer,
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    target_net: nn.Module,
    num_updates: int = 10,
    gamma: float = 0.99,
    loss_type: str = 'mse'
) -> Tuple[bool, Dict[str, any]]:
    """
    Validate that loss decreases over several updates on a synthetic batch.

    Performs multiple optimization steps on the same batch and checks that:
    1. Loss decreases from initial value
    2. No NaN or Inf values appear
    3. Final loss is lower than initial loss

    Args:
        loss_fn: Loss computation function (e.g., compute_dqn_loss)
        network: Online Q-network to train
        optimizer: Optimizer for network
        states: Batch of states (B, C, H, W)
        actions: Batch of actions (B,)
        rewards: Batch of rewards (B,)
        next_states: Batch of next states (B, C, H, W)
        dones: Batch of done flags (B,)
        target_net: Target Q-network for TD targets
        num_updates: Number of optimization steps to perform (default: 10)
        gamma: Discount factor (default: 0.99)
        loss_type: Type of loss ('mse' or 'huber', default: 'mse')

    Returns:
        Tuple of (success: bool, info: Dict) where info contains:
            - 'initial_loss': Initial loss value
            - 'final_loss': Final loss value
            - 'loss_history': List of all loss values
            - 'loss_decreased': Whether loss decreased
            - 'nan_inf_detected': Whether NaN/Inf was detected

    Example:
        >>> online_net = DQN(num_actions=6)
        >>> target_net = init_target_network(online_net, num_actions=6)
        >>> optimizer = configure_optimizer(online_net)
        >>> # Create synthetic batch
        >>> states = torch.randn(32, 4, 84, 84)
        >>> actions = torch.randint(0, 6, (32,))
        >>> rewards = torch.randn(32)
        >>> next_states = torch.randn(32, 4, 84, 84)
        >>> dones = torch.zeros(32)
        >>> # Validate loss decreases
        >>> success, info = validate_loss_decrease(
        ...     compute_dqn_loss, online_net, optimizer,
        ...     states, actions, rewards, next_states, dones, target_net
        ... )
        >>> assert success, f"Loss did not decrease: {info}"
    """
    loss_history = []
    nan_inf_detected = False

    network.train()

    for step in range(num_updates):
        # Compute TD targets
        td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma)

        # Select Q-values for actions
        q_selected = select_q_values(network, states, actions)

        # Compute loss
        loss_dict = loss_fn(q_selected, td_targets, loss_type=loss_type)
        loss = loss_dict['loss']

        # Check for NaN/Inf
        if detect_nan_inf(loss, "loss"):
            nan_inf_detected = True
            loss_history.append(float('nan'))
            break

        loss_history.append(loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    initial_loss = loss_history[0] if loss_history else float('nan')
    final_loss = loss_history[-1] if loss_history else float('nan')
    loss_decreased = final_loss < initial_loss

    success = loss_decreased and not nan_inf_detected

    info = {
        'initial_loss': initial_loss,
        'final_loss': final_loss,
        'loss_history': loss_history,
        'loss_decreased': loss_decreased,
        'nan_inf_detected': nan_inf_detected
    }

    return success, info


def verify_target_sync_schedule(
    updater: 'TargetNetworkUpdater',
    online_net: nn.Module,
    target_net: nn.Module,
    max_steps: int,
    expected_interval: int
) -> Tuple[bool, Dict[str, any]]:
    """
    Verify that target network updates occur at exact multiples of update_interval.

    Simulates stepping through training and checks that:
    1. Updates occur at exact multiples of the interval
    2. No duplicate updates occur
    3. Update count matches expected count

    Args:
        updater: TargetNetworkUpdater instance
        online_net: Online Q-network
        target_net: Target Q-network
        max_steps: Maximum number of steps to simulate
        expected_interval: Expected update interval (should match updater.update_interval)

    Returns:
        Tuple of (success: bool, info: Dict) where info contains:
            - 'update_steps': List of steps where updates occurred
            - 'expected_steps': List of steps where updates should occur
            - 'schedule_correct': Whether updates occurred at correct steps
            - 'count_correct': Whether update count is correct

    Example:
        >>> online_net = DQN(num_actions=6)
        >>> target_net = init_target_network(online_net, num_actions=6)
        >>> updater = TargetNetworkUpdater(update_interval=1000)
        >>> success, info = verify_target_sync_schedule(
        ...     updater, online_net, target_net, max_steps=5000, expected_interval=1000
        ... )
        >>> assert success, f"Target sync schedule incorrect: {info}"
        >>> assert info['update_steps'] == [1000, 2000, 3000, 4000, 5000]
    """
    update_steps = []

    # Reset updater to start fresh
    updater.reset()

    # Step through training
    for step in range(1, max_steps + 1):
        update_info = updater.step(online_net, target_net, step)
        if update_info is not None:
            update_steps.append(step)

    # Calculate expected update steps
    expected_steps = list(range(expected_interval, max_steps + 1, expected_interval))

    schedule_correct = update_steps == expected_steps
    count_correct = len(update_steps) == len(expected_steps)

    success = schedule_correct and count_correct

    info = {
        'update_steps': update_steps,
        'expected_steps': expected_steps,
        'schedule_correct': schedule_correct,
        'count_correct': count_correct
    }

    return success, info


class UpdateMetrics:
    """
    Container for training update metrics.

    Stores metrics from a single training update for logging and monitoring:
    - loss: Training loss value
    - td_error: Mean absolute TD error
    - td_error_std: Standard deviation of TD errors
    - grad_norm: Gradient norm before clipping
    - learning_rate: Current optimizer learning rate
    - update_count: Total number of updates performed

    Attributes:
        loss: Training loss (float)
        td_error: Mean |TD error| (float)
        td_error_std: Standard deviation of TD errors (float)
        grad_norm: Gradient norm before clipping (float)
        learning_rate: Current learning rate (float)
        update_count: Number of updates performed (int)

    Example:
        >>> metrics = UpdateMetrics(
        ...     loss=0.5,
        ...     td_error=0.3,
        ...     td_error_std=0.2,
        ...     grad_norm=2.5,
        ...     learning_rate=0.00025,
        ...     update_count=1000
        ... )
        >>> print(f"Loss: {metrics.loss:.4f}")
        Loss: 0.5000
    """

    def __init__(
        self,
        loss: float,
        td_error: float,
        td_error_std: float,
        grad_norm: float,
        learning_rate: float,
        update_count: int
    ):
        """
        Initialize update metrics.

        Args:
            loss: Training loss value
            td_error: Mean absolute TD error
            td_error_std: Standard deviation of TD errors
            grad_norm: Gradient norm before clipping
            learning_rate: Current optimizer learning rate
            update_count: Total number of updates performed
        """
        self.loss = loss
        self.td_error = td_error
        self.td_error_std = td_error_std
        self.grad_norm = grad_norm
        self.learning_rate = learning_rate
        self.update_count = update_count

    def to_dict(self) -> Dict[str, float]:
        """
        Convert metrics to dictionary for logging.

        Returns:
            Dictionary with all metrics

        Example:
            >>> metrics = UpdateMetrics(0.5, 0.3, 0.2, 2.5, 0.00025, 1000)
            >>> metrics_dict = metrics.to_dict()
            >>> assert 'loss' in metrics_dict
            >>> assert 'td_error' in metrics_dict
        """
        return {
            'loss': self.loss,
            'td_error': self.td_error,
            'td_error_std': self.td_error_std,
            'grad_norm': self.grad_norm,
            'learning_rate': self.learning_rate,
            'update_count': self.update_count
        }

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"UpdateMetrics(loss={self.loss:.4f}, td_error={self.td_error:.4f}, "
            f"grad_norm={self.grad_norm:.4f}, lr={self.learning_rate:.6f}, "
            f"updates={self.update_count})"
        )


def perform_update_step(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    gamma: float = 0.99,
    loss_type: str = 'mse',
    max_grad_norm: float = 10.0,
    update_count: int = 0
) -> UpdateMetrics:
    """
    Perform a single training update and return metrics.

    Executes the full DQN update step:
    1. Compute TD targets using target network
    2. Select Q-values for actions from online network
    3. Compute loss (MSE or Huber)
    4. Backpropagate gradients
    5. Clip gradients by global norm
    6. Update network parameters
    7. Collect and return metrics

    Args:
        online_net: Online Q-network to train
        target_net: Target Q-network for TD targets
        optimizer: Optimizer for online network
        batch: Dictionary with keys 'states', 'actions', 'rewards', 'next_states', 'dones'
        gamma: Discount factor (default: 0.99)
        loss_type: 'mse' or 'huber' (default: 'mse')
        max_grad_norm: Maximum gradient norm for clipping (default: 10.0)
        update_count: Current update count for metrics (default: 0)

    Returns:
        UpdateMetrics object with loss, TD error, grad norm, learning rate, and count

    Example:
        >>> from src.replay import ReplayBuffer
        >>> online_net = DQN(num_actions=6)
        >>> target_net = init_target_network(online_net, num_actions=6)
        >>> optimizer = configure_optimizer(online_net)
        >>> buffer = ReplayBuffer(capacity=10000, batch_size=32)
        >>> # ... fill buffer ...
        >>> batch = buffer.sample()
        >>> metrics = perform_update_step(
        ...     online_net, target_net, optimizer, batch,
        ...     update_count=1
        ... )
        >>> print(f"Loss: {metrics.loss:.4f}")
    """
    # Set online network to training mode
    online_net.train()

    # Extract batch data
    states = batch['states']
    actions = batch['actions']
    rewards = batch['rewards']
    next_states = batch['next_states']
    dones = batch['dones']

    # Compute TD targets (no gradient)
    td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma)

    # Select Q-values for taken actions (with gradient)
    q_selected = select_q_values(online_net, states, actions)

    # Compute loss and TD error stats
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type=loss_type)
    loss = loss_dict['loss']
    td_error = loss_dict['td_error'].item()
    td_error_std = loss_dict['td_error_std'].item()

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Clip gradients and get norm before clipping
    grad_norm = clip_gradients(online_net, max_norm=max_grad_norm)

    # Update parameters
    optimizer.step()

    # Get current learning rate
    learning_rate = optimizer.param_groups[0]['lr']

    # Create and return metrics
    metrics = UpdateMetrics(
        loss=loss.item(),
        td_error=td_error,
        td_error_std=td_error_std,
        grad_norm=grad_norm,
        learning_rate=learning_rate,
        update_count=update_count
    )

    return metrics


# ============================================================================
# Epsilon-Greedy Exploration
# ============================================================================
