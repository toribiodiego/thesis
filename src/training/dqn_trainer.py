"""
DQN training utilities including target network management.

Provides utilities for:
- Hard target network updates
- TD target computation
- Q-learning loss computation
- MSE and Huber loss functions

Target Network Note:
    The target network is a stability improvement introduced in the 2015 Nature
    paper (Mnih et al., "Human-level control through deep reinforcement learning").
    It was NOT present in the original 2013 arXiv paper (Mnih et al., "Playing
    Atari with Deep Reinforcement Learning").

    The 2013 paper used a single Q-network with the same network for both:
    - Computing Q(s,a) for the current state-action pairs
    - Computing max Q(s',a') for the next state (in TD targets)

    The 2015 paper introduced a separate target network that is updated
    periodically (every C=10,000 steps) to stabilize training by reducing
    correlations between Q-values and targets.

    For purist reproduction of the 2013 paper:
    - Set update_interval to 1 (update every step = same as 2013)
    - Or use the same network for both online and target Q-values
    - See TargetNetworkUpdater class for configuration options
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple


def hard_update_target(online_net: nn.Module, target_net: nn.Module) -> None:
    """
    Hard update target network with online network parameters.

    Performs a complete copy of all parameters from online network to target network.
    This is the standard DQN approach (Mnih et al. 2015) where the target network
    is updated periodically (e.g., every 10k steps) rather than continuously.

    Args:
        online_net: The online Q-network being trained
        target_net: The target Q-network used for computing TD targets

    Notes:
        - Target network gradients should be frozen (no gradient computation)
        - This is a "hard" update (full copy) vs "soft" update (polyak averaging)
        - Typically called every C steps (default 10,000 for DQN)
        - Target network is used under torch.no_grad() for stability

    Example:
        >>> online_net = DQN(num_actions=6)
        >>> target_net = DQN(num_actions=6)
        >>> hard_update_target(online_net, target_net)
        >>> # Every 10k steps:
        >>> hard_update_target(online_net, target_net)
    """
    target_net.load_state_dict(online_net.state_dict())


def init_target_network(online_net: nn.Module, num_actions: int) -> nn.Module:
    """
    Initialize target network as a copy of online network.

    Creates a new network with the same architecture, copies weights from online network,
    and freezes gradients for the target network.

    Args:
        online_net: The online Q-network
        num_actions: Number of actions in the environment

    Returns:
        Target network (copy of online, with frozen gradients)

    Example:
        >>> from src.models import DQN
        >>> online_net = DQN(num_actions=6)
        >>> target_net = init_target_network(online_net, num_actions=6)
        >>> # Verify target is a copy
        >>> assert torch.allclose(
        ...     list(online_net.parameters())[0],
        ...     list(target_net.parameters())[0]
        ... )
    """
    # Create target network with same class and parameters
    target_net = type(online_net)(num_actions=num_actions)

    # Copy weights from online to target
    hard_update_target(online_net, target_net)

    # Freeze target network (disable gradient computation)
    for param in target_net.parameters():
        param.requires_grad = False

    # Set to eval mode
    target_net.eval()

    return target_net


def compute_td_targets(
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    target_net: nn.Module,
    gamma: float = 0.99
) -> torch.Tensor:
    """
    Compute TD targets using the target network.

    Computes y = r + γ * (1 - done) * max_a' Q_target(s', a')

    The target network is used under no_grad for stability. The max Q-value
    over all actions is selected for the next state.

    Args:
        rewards: Reward tensor, shape (B,) in float32
        next_states: Next state tensor, shape (B, C, H, W) in float32
        dones: Done flags, shape (B,) in bool
        target_net: Target Q-network (frozen, eval mode)
        gamma: Discount factor (default 0.99)

    Returns:
        TD targets, shape (B,) in float32

    Notes:
        - Uses torch.no_grad() for target network forward pass
        - Masks out terminal states with (1 - done)
        - Uses max over actions for Q-learning (not double DQN)
        - Returned targets are detached (no gradient flow to target network)

    Example:
        >>> rewards = torch.tensor([1.0, 0.0, -1.0])
        >>> next_states = torch.randn(3, 4, 84, 84)
        >>> dones = torch.tensor([False, False, True])
        >>> targets = compute_td_targets(rewards, next_states, dones, target_net)
        >>> # targets[2] should be -1.0 (terminal state, no future value)
    """
    with torch.no_grad():
        # Forward pass through target network
        target_output = target_net(next_states)
        target_q_values = target_output['q_values']  # (B, num_actions)

        # Get max Q-value over actions
        max_target_q, _ = target_q_values.max(dim=1)  # (B,)

        # Compute TD targets: r + γ * (1 - done) * max_a' Q(s', a')
        # Convert done from bool to float: True -> 1.0, False -> 0.0
        done_mask = dones.float()

        # TD target equation
        td_targets = rewards + gamma * (1.0 - done_mask) * max_target_q

    return td_targets


def select_q_values(
    online_net: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor
) -> torch.Tensor:
    """
    Select Q-values for specific actions from online network.

    Computes Q_online(s, a) by gathering Q-values for the actions that were taken.

    Args:
        online_net: Online Q-network being trained
        states: State tensor, shape (B, C, H, W) in float32
        actions: Action indices, shape (B,) in int64

    Returns:
        Selected Q-values, shape (B,) in float32

    Notes:
        - Forward pass through online network (gradients enabled)
        - Uses gather to select Q-values for specific actions
        - Returns shape (B,) after squeezing dimension 1
        - Gradients flow back through online network

    Example:
        >>> states = torch.randn(3, 4, 84, 84)
        >>> actions = torch.tensor([0, 2, 1])  # Actions taken
        >>> q_selected = select_q_values(online_net, states, actions)
        >>> q_selected.shape  # (3,)
    """
    # Forward pass through online network
    online_output = online_net(states)
    q_values = online_output['q_values']  # (B, num_actions)

    # Gather Q-values for the actions that were taken
    # actions shape: (B,) -> (B, 1) for gather
    # q_values shape: (B, num_actions)
    # Result shape: (B, 1) -> squeeze to (B,)
    actions_unsqueezed = actions.unsqueeze(1)  # (B,) -> (B, 1)
    q_selected = q_values.gather(1, actions_unsqueezed)  # (B, 1)
    q_selected = q_selected.squeeze(1)  # (B,)

    return q_selected


def compute_td_loss_components(
    states: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_states: torch.Tensor,
    dones: torch.Tensor,
    online_net: nn.Module,
    target_net: nn.Module,
    gamma: float = 0.99
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Q-values and TD targets for loss computation.

    This is a convenience function that combines select_q_values and compute_td_targets.

    Args:
        states: State tensor, shape (B, C, H, W)
        actions: Action indices, shape (B,)
        rewards: Rewards, shape (B,)
        next_states: Next state tensor, shape (B, C, H, W)
        dones: Done flags, shape (B,)
        online_net: Online Q-network
        target_net: Target Q-network
        gamma: Discount factor

    Returns:
        Tuple of (q_selected, td_targets), both shape (B,)
        - q_selected: Q-values for actions taken (with gradients)
        - td_targets: TD target values (detached, no gradients)

    Example:
        >>> # After sampling a batch from replay buffer
        >>> q_selected, td_targets = compute_td_loss_components(
        ...     batch['states'], batch['actions'], batch['rewards'],
        ...     batch['next_states'], batch['dones'],
        ...     online_net, target_net, gamma=0.99
        ... )
        >>> # Now compute loss
        >>> loss = F.mse_loss(q_selected, td_targets)
    """
    # Get Q-values for actions taken (with gradients)
    q_selected = select_q_values(online_net, states, actions)

    # Get TD targets (no gradients)
    td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma)

    return q_selected, td_targets


def compute_dqn_loss(
    q_selected: torch.Tensor,
    td_targets: torch.Tensor,
    loss_type: str = 'mse',
    huber_delta: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Compute DQN loss with configurable loss function.

    Supports MSE loss (default) or Huber loss (smooth L1).
    Returns loss and auxiliary statistics for monitoring.

    Args:
        q_selected: Selected Q-values Q(s, a), shape (B,), with gradients
        td_targets: TD target values y, shape (B,), detached
        loss_type: Loss function type, 'mse' or 'huber' (default: 'mse')
        huber_delta: Delta parameter for Huber loss (default: 1.0)

    Returns:
        Dictionary containing:
            - 'loss': Scalar loss tensor (with gradients)
            - 'td_error': Mean absolute TD error |Q(s,a) - y| (detached)
            - 'td_error_std': Standard deviation of TD errors (detached)

    Notes:
        - MSE loss: L = mean((Q(s,a) - y)^2)
        - Huber loss: L = smooth_l1_loss with delta parameter
        - TD error is computed as |Q(s,a) - y| for monitoring
        - All auxiliary stats are detached (no gradients)

    Example:
        >>> q_selected, td_targets = compute_td_loss_components(...)
        >>> loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type='mse')
        >>> loss = loss_dict['loss']
        >>> loss.backward()
        >>> # Monitor TD error
        >>> print(f"Mean TD error: {loss_dict['td_error'].item():.4f}")
    """
    # Validate inputs
    assert q_selected.shape == td_targets.shape, \
        f"Shape mismatch: q_selected {q_selected.shape} vs td_targets {td_targets.shape}"
    assert q_selected.requires_grad, "q_selected should have gradients"
    assert not td_targets.requires_grad, "td_targets should be detached"

    # Compute loss based on type
    if loss_type == 'mse':
        # Mean Squared Error loss
        loss = F.mse_loss(q_selected, td_targets, reduction='mean')
    elif loss_type == 'huber':
        # Huber loss (smooth L1)
        # PyTorch's smooth_l1_loss uses delta=1.0 by default
        # For custom delta, we use huber_loss with specified delta
        loss = F.huber_loss(q_selected, td_targets, reduction='mean', delta=huber_delta)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Use 'mse' or 'huber'.")

    # Compute TD error statistics (for monitoring)
    with torch.no_grad():
        td_errors = torch.abs(q_selected - td_targets)  # |Q(s,a) - y|
        mean_td_error = td_errors.mean()
        std_td_error = td_errors.std()

    return {
        'loss': loss,
        'td_error': mean_td_error,
        'td_error_std': std_td_error
    }


def configure_optimizer(
    network: nn.Module,
    optimizer_type: str = 'rmsprop',
    learning_rate: float = 2.5e-4,
    alpha: float = 0.95,
    eps: float = 1e-2,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """
    Configure optimizer for DQN training.

    Supports RMSProp (DQN default) or Adam. Uses paper-default hyperparameters
    for RMSProp: ρ=0.95, ε=0.01, LR=2.5e-4.

    Args:
        network: Neural network to optimize
        optimizer_type: Optimizer type, 'rmsprop' or 'adam' (default: 'rmsprop')
        learning_rate: Learning rate (default: 2.5e-4)
        alpha: RMSProp smoothing constant ρ (default: 0.95)
        eps: RMSProp epsilon for numerical stability (default: 1e-2)
        momentum: Momentum factor (default: 0.0)
        weight_decay: L2 regularization (default: 0.0)
        **kwargs: Additional optimizer-specific arguments

    Returns:
        Configured optimizer

    Notes:
        - DQN paper uses RMSProp with ρ=0.95, ε=0.01, LR=2.5e-4
        - Adam is a common alternative with default β1=0.9, β2=0.999
        - Gamma (discount) is 0.99, batch size is 32 (not optimizer params)
        - Use with gradient clipping (max_norm=10.0) before optimizer.step()

    Example:
        >>> optimizer = configure_optimizer(online_net, optimizer_type='rmsprop')
        >>> # Training loop:
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> clip_gradients(online_net, max_norm=10.0)
        >>> optimizer.step()
    """
    # Get network parameters
    params = network.parameters()

    if optimizer_type.lower() == 'rmsprop':
        # DQN paper defaults: alpha (ρ) = 0.95, eps = 0.01
        optimizer = torch.optim.RMSprop(
            params,
            lr=learning_rate,
            alpha=alpha,
            eps=eps,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
    elif optimizer_type.lower() == 'adam':
        # Adam defaults: beta1=0.9, beta2=0.999, eps=1e-8
        # Override eps if specified
        adam_eps = kwargs.pop('adam_eps', 1e-8)
        beta1 = kwargs.pop('beta1', 0.9)
        beta2 = kwargs.pop('beta2', 0.999)

        optimizer = torch.optim.Adam(
            params,
            lr=learning_rate,
            betas=(beta1, beta2),
            eps=adam_eps,
            weight_decay=weight_decay,
            **kwargs
        )
    else:
        raise ValueError(
            f"Unknown optimizer_type: {optimizer_type}. Use 'rmsprop' or 'adam'."
        )

    return optimizer


def clip_gradients(
    network: nn.Module,
    max_norm: float = 10.0,
    norm_type: float = 2.0
) -> float:
    """
    Clip gradients by global norm.

    Applies gradient clipping to prevent exploding gradients during training.
    Should be called after loss.backward() and before optimizer.step().

    Args:
        network: Neural network with computed gradients
        max_norm: Maximum gradient norm (default: 10.0)
        norm_type: Type of norm to use (default: 2.0 for L2 norm)

    Returns:
        Total gradient norm before clipping (for monitoring)

    Notes:
        - Clips gradients if total norm exceeds max_norm
        - Uses torch.nn.utils.clip_grad_norm_ internally
        - Returns unclipped norm for logging/monitoring
        - Common max_norm values: 10.0 (DQN), 40.0 (A3C), 0.5 (PPO)

    Example:
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> grad_norm = clip_gradients(online_net, max_norm=10.0)
        >>> print(f"Gradient norm: {grad_norm:.4f}")
        >>> optimizer.step()
    """
    # Compute and clip gradient norm
    total_norm = torch.nn.utils.clip_grad_norm_(
        network.parameters(),
        max_norm=max_norm,
        norm_type=norm_type
    )

    return total_norm.item()


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

class EpsilonScheduler:
    """
    Epsilon-greedy exploration scheduler with linear decay.

    Supports linear decay from epsilon_start to epsilon_end over a specified
    number of frames, with separate epsilon for training and evaluation.

    Parameters
    ----------
    epsilon_start : float
        Initial epsilon value (default: 1.0, fully random exploration)
    epsilon_end : float
        Final epsilon value after decay (default: 0.1)
    decay_frames : int
        Number of frames over which to decay epsilon (default: 1,000,000)
    eval_epsilon : float
        Fixed epsilon for evaluation mode (default: 0.05)

    Usage
    -----
    >>> scheduler = EpsilonScheduler(epsilon_start=1.0, epsilon_end=0.1, decay_frames=1000000)
    >>> epsilon = scheduler.get_epsilon(current_frame=500000)  # Returns 0.55
    >>> eval_eps = scheduler.get_eval_epsilon()  # Returns 0.05
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        decay_frames: int = 1_000_000,
        eval_epsilon: float = 0.05
    ):
        assert 0.0 <= epsilon_start <= 1.0, f"epsilon_start must be in [0,1], got {epsilon_start}"
        assert 0.0 <= epsilon_end <= 1.0, f"epsilon_end must be in [0,1], got {epsilon_end}"
        assert epsilon_start >= epsilon_end, f"epsilon_start must be >= epsilon_end"
        assert decay_frames > 0, f"decay_frames must be positive, got {decay_frames}"
        assert 0.0 <= eval_epsilon <= 1.0, f"eval_epsilon must be in [0,1], got {eval_epsilon}"

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_frames = decay_frames
        self.eval_epsilon = eval_epsilon

        # Precompute slope for efficiency
        self.slope = (epsilon_end - epsilon_start) / decay_frames

    def get_epsilon(self, current_frame: int) -> float:
        """
        Get epsilon for current training frame with linear decay.

        Parameters
        ----------
        current_frame : int
            Current environment frame count (not update count)

        Returns
        -------
        float
            Epsilon value in [epsilon_end, epsilon_start]

        Examples
        --------
        >>> scheduler = EpsilonScheduler(1.0, 0.1, 1000000)
        >>> scheduler.get_epsilon(0)
        1.0
        >>> scheduler.get_epsilon(500000)
        0.55
        >>> scheduler.get_epsilon(1000000)
        0.1
        >>> scheduler.get_epsilon(2000000)  # Clamps to epsilon_end
        0.1
        """
        if current_frame >= self.decay_frames:
            return self.epsilon_end

        # Linear interpolation: start + slope * frames
        epsilon = self.epsilon_start + self.slope * current_frame

        # Clamp to [epsilon_end, epsilon_start] for numerical stability
        return max(self.epsilon_end, min(self.epsilon_start, epsilon))

    def get_eval_epsilon(self) -> float:
        """
        Get fixed epsilon for evaluation mode.

        Returns
        -------
        float
            Fixed epsilon for evaluation (no decay)
        """
        return self.eval_epsilon

    def to_dict(self) -> dict:
        """Export scheduler configuration as dictionary."""
        return {
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'decay_frames': self.decay_frames,
            'eval_epsilon': self.eval_epsilon
        }


def select_epsilon_greedy_action(
    network: torch.nn.Module,
    state: torch.Tensor,
    epsilon: float,
    num_actions: int
) -> int:
    """
    Select action using epsilon-greedy policy.

    With probability epsilon, choose random action.
    With probability (1 - epsilon), choose greedy action (argmax Q-value).

    Parameters
    ----------
    network : torch.nn.Module
        Q-network for computing Q-values
    state : torch.Tensor
        Current state observation, shape (C, H, W) or (1, C, H, W)
    epsilon : float
        Exploration probability in [0, 1]
    num_actions : int
        Number of available actions

    Returns
    -------
    int
        Selected action index

    Examples
    --------
    >>> network = DQNModel(num_actions=6)
    >>> state = torch.rand(4, 84, 84)  # Single state
    >>> action = select_epsilon_greedy_action(network, state, epsilon=0.1, num_actions=6)
    >>> assert 0 <= action < 6
    """
    if torch.rand(1).item() < epsilon:
        # Explore: random action
        return torch.randint(0, num_actions, (1,)).item()
    else:
        # Exploit: greedy action
        network.eval()
        with torch.no_grad():
            # Ensure state has batch dimension
            if state.dim() == 3:
                state = state.unsqueeze(0)  # (C,H,W) -> (1,C,H,W)

            output = network(state)
            q_values = output['q_values']  # (1, num_actions)
            action = q_values.argmax(dim=1).item()
        network.train()
        return action


# ============================================================================
# Training Loop Utilities
# ============================================================================

class FrameCounter:
    """
    Track environment frames vs decision steps for action repeat.

    With frame skip k=4, each decision step corresponds to k environment frames.
    This class tracks both counts to ensure correct frame budgets and logging.

    Parameters
    ----------
    frameskip : int
        Number of frames per decision step (default: 4)

    Examples
    --------
    >>> counter = FrameCounter(frameskip=4)
    >>> counter.step()  # Increment by 1 decision step
    >>> counter.frames  # Returns 4 (1 decision * 4 frames)
    >>> counter.steps   # Returns 1
    >>> counter.fps(elapsed_time=1.0)  # Returns frames per second
    """

    def __init__(self, frameskip: int = 4):
        assert frameskip > 0, f"frameskip must be positive, got {frameskip}"
        self.frameskip = frameskip
        self._steps = 0
        self._start_time = None

    def step(self, num_steps: int = 1):
        """Increment decision step counter."""
        self._steps += num_steps

    @property
    def steps(self) -> int:
        """Total decision steps taken."""
        return self._steps

    @property
    def frames(self) -> int:
        """Total environment frames (steps * frameskip)."""
        return self._steps * self.frameskip

    def fps(self, elapsed_time: float) -> float:
        """
        Calculate frames per second.

        Parameters
        ----------
        elapsed_time : float
            Elapsed time in seconds

        Returns
        -------
        float
            Frames per second (frames / elapsed_time)
        """
        if elapsed_time <= 0:
            return 0.0
        return self.frames / elapsed_time

    def reset(self):
        """Reset counter to zero."""
        self._steps = 0

    def to_dict(self) -> dict:
        """Export counter state as dictionary."""
        return {
            'steps': self.steps,
            'frames': self.frames,
            'frameskip': self.frameskip
        }


def training_step(
    env,
    online_net: torch.nn.Module,
    target_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    replay_buffer,
    epsilon_scheduler: EpsilonScheduler,
    target_updater: TargetNetworkUpdater,
    training_scheduler: TrainingScheduler,
    frame_counter: FrameCounter,
    state: torch.Tensor,
    num_actions: int,
    gamma: float = 0.99,
    loss_type: str = 'mse',
    max_grad_norm: float = 10.0,
    batch_size: int = 32,
    device: str = 'cpu'
):
    """
    Execute one step of the DQN training loop.

    This function orchestrates the 5-step training process:
    1. Select action via ε-greedy from online Q-network
    2. Step environment with frame-skip (handled by wrapper)
    3. Append transition to replay buffer
    4. If warm-up done and step % train_every == 0: perform optimization
    5. If step % target_update_interval == 0: sync target network

    Parameters
    ----------
    env : gym.Env
        Gymnasium environment (with wrappers applied)
    online_net : torch.nn.Module
        Q-network being trained
    target_net : torch.nn.Module
        Target Q-network for TD targets
    optimizer : torch.optim.Optimizer
        Optimizer for online network
    replay_buffer : ReplayBuffer
        Experience replay buffer
    epsilon_scheduler : EpsilonScheduler
        Epsilon schedule for exploration
    target_updater : TargetNetworkUpdater
        Target network update scheduler
    training_scheduler : TrainingScheduler
        Training frequency scheduler
    frame_counter : FrameCounter
        Frame counter for tracking progress
    state : torch.Tensor
        Current state observation (C, H, W)
    num_actions : int
        Number of available actions
    gamma : float
        Discount factor (default: 0.99)
    loss_type : str
        Loss function type: 'mse' or 'huber' (default: 'mse')
    max_grad_norm : float
        Maximum gradient norm for clipping (default: 10.0)
    batch_size : int
        Batch size for sampling from replay buffer (default: 32)
    device : str
        Device for tensors (default: 'cpu')

    Returns
    -------
    dict
        Step results containing:
        - next_state: Next state observation
        - reward: Reward received
        - terminated: Whether episode terminated
        - truncated: Whether episode was truncated
        - epsilon: Current epsilon value
        - metrics: UpdateMetrics if training occurred, else None
        - target_updated: Whether target network was updated
        - trained: Whether optimization step occurred
    """
    # Step 1: Select action via ε-greedy
    current_frame = frame_counter.frames
    epsilon = epsilon_scheduler.get_epsilon(current_frame)

    # Convert state to correct device and format
    if isinstance(state, np.ndarray):
        state_tensor = torch.from_numpy(state).float().to(device) / 255.0
    else:
        state_tensor = state.float().to(device)
        if state_tensor.max() > 1.0:
            state_tensor = state_tensor / 255.0

    action = select_epsilon_greedy_action(
        online_net, state_tensor, epsilon, num_actions
    )

    # Step 2: Step environment (frame-skip handled by wrapper)
    next_state, reward, terminated, truncated, info = env.step(action)

    # Increment frame counter
    frame_counter.step()

    # Step 3: Append transition to replay buffer
    replay_buffer.append(state, action, reward, next_state, terminated or truncated)

    # Step 4: Conditional training update
    metrics = None
    trained = False

    if training_scheduler.should_train(frame_counter.steps, replay_buffer):
        # Sample batch from replay
        batch = replay_buffer.sample(batch_size)

        # Convert batch to tensors and move to device
        # (Handle both numpy arrays and torch tensors from replay buffer)
        def to_tensor(x, dtype=None):
            if isinstance(x, np.ndarray):
                t = torch.from_numpy(x)
                if dtype is not None:
                    t = t.to(dtype)
                return t.to(device)
            else:
                return x.to(device)

        batch_device = {
            'states': to_tensor(batch['states']),
            'actions': to_tensor(batch['actions']),
            'rewards': to_tensor(batch['rewards']),
            'next_states': to_tensor(batch['next_states']),
            'dones': to_tensor(batch['dones'])
        }

        # Perform optimization step
        metrics = perform_update_step(
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            batch=batch_device,
            gamma=gamma,
            loss_type=loss_type,
            max_grad_norm=max_grad_norm,
            update_count=training_scheduler.training_step_count
        )

        training_scheduler.mark_trained(frame_counter.steps)
        trained = True

    # Step 5: Conditional target network sync
    target_updated = False
    update_info = target_updater.step(online_net, target_net, frame_counter.steps)
    if update_info is not None:
        target_updated = True

    return {
        'next_state': next_state,
        'reward': reward,
        'terminated': terminated,
        'truncated': truncated,
        'epsilon': epsilon,
        'metrics': metrics,
        'target_updated': target_updated,
        'trained': trained,
        'action': action
    }


# ============================================================================
# Logging Utilities
# ============================================================================

class StepLogger:
    """
    Logger for per-step training metrics.

    Logs training metrics at regular intervals including loss, TD error,
    gradient norms, epsilon, learning rate, and replay buffer statistics.

    Parameters
    ----------
    log_dir : str
        Directory to save log files
    log_interval : int
        Steps between log writes (default: 1000)
    moving_avg_window : int
        Window size for moving average of loss (default: 100)

    Usage
    -----
    >>> logger = StepLogger(log_dir='runs/pong_123/logs', log_interval=1000)
    >>> logger.log_step(step=1000, epsilon=0.95, metrics=metrics, replay_size=50000)
    """

    def __init__(self, log_dir: str, log_interval: int = 1000, moving_avg_window: int = 100):
        import os
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.moving_avg_window = moving_avg_window

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # CSV file for step metrics
        self.csv_path = os.path.join(log_dir, 'training_steps.csv')
        self._csv_initialized = False

        # Moving average tracking
        self.loss_history = []

    def log_step(
        self,
        step: int,
        epsilon: float,
        metrics: UpdateMetrics = None,
        replay_size: int = 0,
        fps: float = 0.0
    ):
        """
        Log metrics for a training step.

        Args:
            step: Current environment step
            epsilon: Current exploration rate
            metrics: UpdateMetrics from training update (if training occurred)
            replay_size: Current replay buffer size
            fps: Frames per second
        """
        # Only log at intervals
        if step % self.log_interval != 0:
            return

        # Prepare log entry
        log_entry = {
            'step': step,
            'epsilon': epsilon,
            'replay_size': replay_size,
            'fps': fps
        }

        # Add training metrics if available
        if metrics is not None:
            log_entry['loss'] = metrics.loss
            log_entry['td_error'] = metrics.td_error
            log_entry['td_error_std'] = metrics.td_error_std
            log_entry['grad_norm'] = metrics.grad_norm
            log_entry['learning_rate'] = metrics.learning_rate
            log_entry['update_count'] = metrics.update_count

            # Update moving average
            self.loss_history.append(metrics.loss)
            if len(self.loss_history) > self.moving_avg_window:
                self.loss_history.pop(0)
            log_entry['loss_ma'] = sum(self.loss_history) / len(self.loss_history)
        else:
            # No training this step
            log_entry['loss'] = None
            log_entry['td_error'] = None
            log_entry['td_error_std'] = None
            log_entry['grad_norm'] = None
            log_entry['learning_rate'] = None
            log_entry['update_count'] = None
            log_entry['loss_ma'] = None

        # Write to CSV
        self._write_csv(log_entry)

    def _write_csv(self, log_entry: dict):
        """Write log entry to CSV file."""
        import csv

        # Initialize CSV with header
        if not self._csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
            self._csv_initialized = True

        # Append log entry
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)


class EpisodeLogger:
    """
    Logger for per-episode statistics.

    Tracks episode returns, lengths, and computes rolling averages
    for monitoring training progress.

    Parameters
    ----------
    log_dir : str
        Directory to save log files
    rolling_window : int
        Window size for rolling average (default: 100 episodes)

    Usage
    -----
    >>> logger = EpisodeLogger(log_dir='runs/pong_123/logs', rolling_window=100)
    >>> logger.log_episode(step=5000, episode_return=21.0, episode_length=1200, fps=120.5)
    """

    def __init__(self, log_dir: str, rolling_window: int = 100):
        import os
        self.log_dir = log_dir
        self.rolling_window = rolling_window

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # CSV file for episode metrics
        self.csv_path = os.path.join(log_dir, 'episodes.csv')
        self._csv_initialized = False

        # Episode tracking
        self.episode_count = 0
        self.episode_returns = []
        self.episode_lengths = []

    def log_episode(
        self,
        step: int,
        episode_return: float,
        episode_length: int,
        fps: float = 0.0,
        epsilon: float = None
    ):
        """
        Log metrics for a completed episode.

        Args:
            step: Environment step when episode ended
            episode_return: Total undiscounted return
            episode_length: Number of steps in episode
            fps: Frames per second
            epsilon: Current exploration rate
        """
        self.episode_count += 1
        self.episode_returns.append(episode_return)
        self.episode_lengths.append(episode_length)

        # Compute rolling statistics
        recent_returns = self.episode_returns[-self.rolling_window:]
        recent_lengths = self.episode_lengths[-self.rolling_window:]

        import numpy as np
        rolling_mean_return = np.mean(recent_returns)
        rolling_std_return = np.std(recent_returns)
        rolling_mean_length = np.mean(recent_lengths)

        # Prepare log entry
        log_entry = {
            'episode': self.episode_count,
            'step': step,
            'return': episode_return,
            'length': episode_length,
            'fps': fps,
            'rolling_mean_return': rolling_mean_return,
            'rolling_std_return': rolling_std_return,
            'rolling_mean_length': rolling_mean_length,
            'num_episodes_in_window': len(recent_returns)
        }

        if epsilon is not None:
            log_entry['epsilon'] = epsilon

        # Write to CSV
        self._write_csv(log_entry)

    def _write_csv(self, log_entry: dict):
        """Write log entry to CSV file."""
        import csv

        # Initialize CSV with header
        if not self._csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=log_entry.keys())
                writer.writeheader()
            self._csv_initialized = True

        # Append log entry
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)

    def get_recent_stats(self, n: int = None):
        """
        Get statistics over last n episodes.

        Args:
            n: Number of recent episodes (default: rolling_window)

        Returns:
            dict: Statistics including mean, std, min, max
        """
        if n is None:
            n = self.rolling_window

        recent_returns = self.episode_returns[-n:]
        recent_lengths = self.episode_lengths[-n:]

        import numpy as np
        return {
            'mean_return': np.mean(recent_returns),
            'std_return': np.std(recent_returns),
            'min_return': np.min(recent_returns),
            'max_return': np.max(recent_returns),
            'mean_length': np.mean(recent_lengths),
            'num_episodes': len(recent_returns)
        }


# ============================================================================
# Checkpoint Management
# ============================================================================

class CheckpointManager:
    """
    Manages model checkpoints with periodic and best-model saving.

    Saves checkpoints at regular intervals and tracks the best model
    based on evaluation performance.

    Parameters
    ----------
    checkpoint_dir : str
        Directory to save checkpoints
    save_interval : int
        Steps between periodic checkpoints (default: 1,000,000)
    keep_last_n : int
        Number of periodic checkpoints to keep (default: 3, 0 = keep all)
    save_best : bool
        Whether to save best model based on eval score (default: True)

    Usage
    -----
    >>> manager = CheckpointManager(checkpoint_dir='runs/pong_123/checkpoints')
    >>> manager.save_checkpoint(step=1000000, model=model, optimizer=optimizer, metadata={...})
    >>> manager.save_best(step=1000000, eval_return=25.0, model=model, optimizer=optimizer)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        save_interval: int = 1_000_000,
        keep_last_n: int = 3,
        save_best: bool = True
    ):
        import os
        self.checkpoint_dir = checkpoint_dir
        self.save_interval = save_interval
        self.keep_last_n = keep_last_n
        self.save_best_enabled = save_best

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Track periodic checkpoints
        self.periodic_checkpoints = []

        # Track best model
        self.best_eval_return = float('-inf')
        self.best_checkpoint_path = None

    def should_save(self, step: int) -> bool:
        """Check if periodic checkpoint should be saved at this step."""
        return step > 0 and step % self.save_interval == 0

    def save_checkpoint(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: dict = None
    ) -> str:
        """
        Save periodic checkpoint.

        Args:
            step: Current environment step
            model: Q-network to save
            optimizer: Optimizer state to save
            metadata: Additional metadata (epsilon, replay stats, etc.)

        Returns:
            str: Path to saved checkpoint
        """
        import os

        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_step_{step}.pt')

        # Prepare checkpoint
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        if metadata is not None:
            checkpoint['metadata'] = metadata

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Track periodic checkpoint
        self.periodic_checkpoints.append(checkpoint_path)

        # Clean up old checkpoints if needed
        if self.keep_last_n > 0 and len(self.periodic_checkpoints) > self.keep_last_n:
            old_checkpoint = self.periodic_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)

        return checkpoint_path

    def save_best(
        self,
        step: int,
        eval_return: float,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        metadata: dict = None
    ) -> bool:
        """
        Save checkpoint if it's the best model so far.

        Args:
            step: Current environment step
            eval_return: Evaluation return to compare
            model: Q-network to save
            optimizer: Optimizer state to save
            metadata: Additional metadata

        Returns:
            bool: True if checkpoint was saved (new best), False otherwise
        """
        import os

        if not self.save_best_enabled:
            return False

        if eval_return <= self.best_eval_return:
            return False

        # New best model
        self.best_eval_return = eval_return

        checkpoint_path = os.path.join(self.checkpoint_dir, 'best_model.pt')

        # Prepare checkpoint
        checkpoint = {
            'step': step,
            'eval_return': eval_return,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }

        if metadata is not None:
            checkpoint['metadata'] = metadata

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)
        self.best_checkpoint_path = checkpoint_path

        return True

    def load_checkpoint(self, checkpoint_path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None):
        """
        Load checkpoint and restore model/optimizer state.

        Args:
            checkpoint_path: Path to checkpoint file
            model: Q-network to load weights into
            optimizer: Optimizer to load state into (optional)

        Returns:
            dict: Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return {
            'step': checkpoint.get('step', 0),
            'eval_return': checkpoint.get('eval_return', None),
            'metadata': checkpoint.get('metadata', {})
        }


# ============================================================================
# Evaluation Routine
# ============================================================================

def evaluate(
    env,
    model: torch.nn.Module,
    num_episodes: int = 10,
    eval_epsilon: float = 0.05,
    num_actions: int = None,
    device: str = 'cpu',
    seed: int = None
) -> dict:
    """
    Evaluate agent over multiple episodes with low/greedy epsilon.

    Runs the agent in evaluation mode (no learning) and computes
    performance statistics over multiple episodes.

    Parameters
    ----------
    env : gym.Env
        Evaluation environment (should NOT have EpisodicLifeEnv wrapper)
    model : torch.nn.Module
        Q-network to evaluate
    num_episodes : int
        Number of episodes to run (default: 10)
    eval_epsilon : float
        Exploration rate during evaluation (default: 0.05, use 0.0 for greedy)
    num_actions : int
        Number of available actions (if None, inferred from env)
    device : str
        Device for model inference (default: 'cpu')
    seed : int
        Random seed for reproducibility (optional)

    Returns
    -------
    dict
        Evaluation results containing:
        - mean_return: Average episode return
        - median_return: Median episode return
        - std_return: Standard deviation of returns
        - min_return: Minimum episode return
        - max_return: Maximum episode return
        - mean_length: Average episode length
        - episode_returns: List of individual episode returns
        - episode_lengths: List of individual episode lengths
        - num_episodes: Number of episodes evaluated

    Example
    -------
    >>> results = evaluate(eval_env, model, num_episodes=10, eval_epsilon=0.05)
    >>> print(f"Mean return: {results['mean_return']:.2f}")
    """
    import numpy as np

    # Set model to eval mode
    model.eval()

    # Infer num_actions if not provided
    if num_actions is None:
        num_actions = env.action_space.n

    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False

        while not done:
            # Convert observation to tensor
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float().to(device)
                # Normalize if needed
                if obs_tensor.max() > 1.0:
                    obs_tensor = obs_tensor / 255.0
            else:
                obs_tensor = obs.float().to(device)

            # Select action with eval epsilon
            with torch.no_grad():
                if torch.rand(1).item() < eval_epsilon:
                    # Random action
                    action = env.action_space.sample()
                else:
                    # Greedy action
                    if obs_tensor.dim() == 3:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    output = model(obs_tensor)
                    q_values = output['q_values']
                    action = q_values.argmax(dim=1).item()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1
            done = terminated or truncated

        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

    # Compute statistics
    results = {
        'mean_return': np.mean(episode_returns),
        'median_return': np.median(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'num_episodes': num_episodes
    }

    # Set model back to train mode
    model.train()

    return results


class EvaluationScheduler:
    """
    Scheduler for periodic evaluation during training.

    Triggers evaluation at regular intervals and tracks evaluation history.

    Parameters
    ----------
    eval_interval : int
        Steps between evaluations (default: 250,000)
    num_episodes : int
        Number of episodes per evaluation (default: 10)
    eval_epsilon : float
        Exploration rate during evaluation (default: 0.05)

    Usage
    -----
    >>> scheduler = EvaluationScheduler(eval_interval=250000, num_episodes=10)
    >>> if scheduler.should_evaluate(current_step):
    ...     results = evaluate(eval_env, model, num_episodes=scheduler.num_episodes,
    ...                        eval_epsilon=scheduler.eval_epsilon)
    ...     scheduler.record_evaluation(current_step, results)
    """

    def __init__(
        self,
        eval_interval: int = 250_000,
        num_episodes: int = 10,
        eval_epsilon: float = 0.05
    ):
        self.eval_interval = eval_interval
        self.num_episodes = num_episodes
        self.eval_epsilon = eval_epsilon

        # Track evaluation history
        self.eval_steps = []
        self.eval_returns = []
        self.last_eval_step = 0

    def should_evaluate(self, step: int) -> bool:
        """
        Check if evaluation should be performed at this step.

        Args:
            step: Current environment step

        Returns:
            True if should evaluate, False otherwise
        """
        if step == 0:
            return False

        # Evaluate at intervals
        if step >= self.eval_interval and step % self.eval_interval == 0:
            # Avoid duplicate evaluations
            return step != self.last_eval_step

        return False

    def record_evaluation(self, step: int, results: dict):
        """
        Record evaluation results.

        Args:
            step: Environment step when evaluation occurred
            results: Dictionary returned by evaluate()
        """
        self.last_eval_step = step
        self.eval_steps.append(step)
        self.eval_returns.append(results['mean_return'])

    def get_best_return(self) -> float:
        """Get best mean return across all evaluations."""
        if not self.eval_returns:
            return float('-inf')
        return max(self.eval_returns)

    def get_recent_trend(self, n: int = 3) -> str:
        """
        Get recent performance trend.

        Args:
            n: Number of recent evaluations to consider

        Returns:
            'improving', 'declining', or 'stable'
        """
        if len(self.eval_returns) < n:
            return 'insufficient_data'

        recent = self.eval_returns[-n:]
        if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
            return 'improving'
        elif all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
            return 'declining'
        else:
            return 'stable'


class EvaluationLogger:
    """
    Logger for evaluation results.

    Saves evaluation statistics to CSV and JSON files.

    Parameters
    ----------
    log_dir : str
        Directory to save evaluation logs

    Usage
    -----
    >>> logger = EvaluationLogger(log_dir='runs/pong_123/eval')
    >>> logger.log_evaluation(step=250000, results=eval_results)
    """

    def __init__(self, log_dir: str):
        import os
        self.log_dir = log_dir

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # CSV file for evaluation summary
        self.csv_path = os.path.join(log_dir, 'evaluations.csv')
        self._csv_initialized = False

        # JSON directory for detailed per-eval results
        self.json_dir = os.path.join(log_dir, 'detailed')
        os.makedirs(self.json_dir, exist_ok=True)

    def log_evaluation(self, step: int, results: dict, epsilon: float = None):
        """
        Log evaluation results to CSV and JSON.

        Args:
            step: Environment step when evaluation occurred
            results: Dictionary returned by evaluate()
            epsilon: Current training epsilon (optional)
        """
        import csv
        import json
        import os

        # Prepare CSV entry (summary statistics)
        csv_entry = {
            'step': step,
            'mean_return': results['mean_return'],
            'median_return': results['median_return'],
            'std_return': results['std_return'],
            'min_return': results['min_return'],
            'max_return': results['max_return'],
            'mean_length': results['mean_length'],
            'num_episodes': results['num_episodes']
        }

        if epsilon is not None:
            csv_entry['training_epsilon'] = epsilon

        # Write CSV
        if not self._csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_entry.keys())
                writer.writeheader()
            self._csv_initialized = True

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_entry.keys())
            writer.writerow(csv_entry)

        # Save detailed results to JSON
        json_path = os.path.join(self.json_dir, f'eval_step_{step}.json')
        detailed_results = {
            'step': step,
            'statistics': {
                'mean_return': float(results['mean_return']),
                'median_return': float(results['median_return']),
                'std_return': float(results['std_return']),
                'min_return': float(results['min_return']),
                'max_return': float(results['max_return']),
                'mean_length': float(results['mean_length'])
            },
            'episode_returns': [float(r) for r in results['episode_returns']],
            'episode_lengths': [int(l) for l in results['episode_lengths']],
            'num_episodes': results['num_episodes']
        }

        if epsilon is not None:
            detailed_results['training_epsilon'] = epsilon

        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)

    def get_all_results(self) -> list:
        """
        Load all evaluation results from CSV.

        Returns:
            List of dictionaries with evaluation statistics
        """
        import csv
        import os

        if not os.path.exists(self.csv_path):
            return []

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
