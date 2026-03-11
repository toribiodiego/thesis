"""Main training loop components including action selection and training step orchestration."""

import numpy as np
import torch

from .metrics import EpsilonScheduler, perform_rainbow_update_step, perform_update_step
from .schedulers import TargetNetworkUpdater, TrainingScheduler


def select_epsilon_greedy_action(
    network: torch.nn.Module, state: torch.Tensor, epsilon: float, num_actions: int
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
            q_values = output["q_values"]  # (1, num_actions)
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
        return {"steps": self.steps, "frames": self.frames, "frameskip": self.frameskip}


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
    loss_type: str = "mse",
    max_grad_norm: float = 10.0,
    batch_size: int = 32,
    device: str = "cpu",
    augment_fn=None,
    spr_components=None,
    spr_weight: float = 2.0,
    spr_prediction_steps: int = 5,
    rainbow_config=None,
):
    """
    Execute one step of the DQN training loop.

    This function orchestrates the 5-step training process:
    1. Select action via epsilon-greedy from online Q-network
    2. Step environment with frame-skip (handled by wrapper)
    3. Append transition to replay buffer
    4. If warm-up done and step % train_every == 0: perform optimization
       (with optional SPR auxiliary loss)
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
        Optimizer for online network (and SPR modules if enabled)
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
    augment_fn : callable, optional
        Data augmentation function applied to observations
    spr_components : dict, optional
        SPR model components. Required keys: 'transition_model',
        'projection_head', 'prediction_head', 'target_encoder',
        'target_projection'. Pass None to disable SPR.
    spr_weight : float
        Weight for SPR loss in combined objective (default: 2.0)
    spr_prediction_steps : int
        Number of future steps K for SPR prediction (default: 5)
    rainbow_config : dict, optional
        Rainbow configuration dict with keys: 'support' (Tensor),
        'n_step' (int), 'double_dqn' (bool), 'buffer' (PrioritizedReplayBuffer).
        When provided, uses perform_rainbow_update_step instead of
        perform_update_step. Pass None for vanilla DQN.

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
            "states": to_tensor(batch["states"]),
            "actions": to_tensor(batch["actions"]),
            "rewards": to_tensor(batch["rewards"]),
            "next_states": to_tensor(batch["next_states"]),
            "dones": to_tensor(batch["dones"]),
        }

        # Include PER fields when using Rainbow
        if rainbow_config is not None and "weights" in batch:
            batch_device["weights"] = to_tensor(batch["weights"])
            batch_device["indices"] = batch["indices"]  # int array, stays on CPU

        # Apply data augmentation if enabled
        if augment_fn is not None:
            batch_device["states"] = augment_fn(batch_device["states"])
            batch_device["next_states"] = augment_fn(batch_device["next_states"])

        # Sample SPR sequence batch (if SPR enabled)
        spr_batch_device = None
        if spr_components is not None:
            seq_batch = replay_buffer.sample_sequences(
                batch_size, spr_prediction_steps
            )
            spr_batch_device = {
                "states": to_tensor(seq_batch["states"]),
                "actions": to_tensor(seq_batch["actions"]),
                "dones": to_tensor(seq_batch["dones"]),
            }
            # Apply augmentation to each state in the sequence
            if augment_fn is not None:
                s = spr_batch_device["states"]
                B, Kp1 = s.shape[0], s.shape[1]
                s_flat = augment_fn(s.reshape(B * Kp1, *s.shape[2:]))
                spr_batch_device["states"] = s_flat.reshape(
                    B, Kp1, *s_flat.shape[1:]
                )

        # Perform optimization step
        if rainbow_config is not None:
            metrics = perform_rainbow_update_step(
                online_net=online_net,
                target_net=target_net,
                optimizer=optimizer,
                batch=batch_device,
                support=rainbow_config["support"],
                gamma=gamma,
                n_step=rainbow_config["n_step"],
                max_grad_norm=max_grad_norm,
                update_count=training_scheduler.training_step_count,
                double_dqn=rainbow_config["double_dqn"],
                buffer=rainbow_config["buffer"],
                spr_components=spr_components,
                spr_batch=spr_batch_device,
                spr_weight=spr_weight,
            )
        else:
            metrics = perform_update_step(
                online_net=online_net,
                target_net=target_net,
                optimizer=optimizer,
                batch=batch_device,
                gamma=gamma,
                loss_type=loss_type,
                max_grad_norm=max_grad_norm,
                update_count=training_scheduler.training_step_count,
                spr_components=spr_components,
                spr_batch=spr_batch_device,
                spr_weight=spr_weight,
            )

        training_scheduler.mark_trained(frame_counter.steps)
        trained = True

    # Step 5: Conditional target network sync
    target_updated = False
    update_info = target_updater.step(online_net, target_net, frame_counter.steps)
    if update_info is not None:
        target_updated = True

    return {
        "next_state": next_state,
        "reward": reward,
        "terminated": terminated,
        "truncated": truncated,
        "epsilon": epsilon,
        "metrics": metrics,
        "target_updated": target_updated,
        "trained": trained,
        "action": action,
    }


# ============================================================================
# Logging Utilities
# ============================================================================
