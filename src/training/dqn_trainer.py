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
