"""Training metrics collection and logging utilities."""

from typing import Dict

import torch
import torch.nn as nn

import numpy as np

from .distributional import compute_distributional_loss
from .loss import (
    compute_combined_loss,
    compute_dqn_loss,
    compute_td_targets,
    select_next_actions,
    select_q_values,
)
from .optimization import clip_gradients


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
        update_count: int,
        spr_loss: float = None,
        cosine_similarity: float = None,
    ):
        """
        Initialize update metrics.

        Args:
            loss: Training loss value (total loss when SPR enabled)
            td_error: Mean absolute TD error
            td_error_std: Standard deviation of TD errors
            grad_norm: Gradient norm before clipping
            learning_rate: Current optimizer learning rate
            update_count: Total number of updates performed
            spr_loss: SPR auxiliary loss value (None when SPR disabled)
            cosine_similarity: Mean cosine similarity between predicted
                and target representations (None when SPR disabled)
        """
        self.loss = loss
        self.td_error = td_error
        self.td_error_std = td_error_std
        self.grad_norm = grad_norm
        self.learning_rate = learning_rate
        self.update_count = update_count
        self.spr_loss = spr_loss
        self.cosine_similarity = cosine_similarity

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
        d = {
            "loss": self.loss,
            "td_error": self.td_error,
            "td_error_std": self.td_error_std,
            "grad_norm": self.grad_norm,
            "learning_rate": self.learning_rate,
            "update_count": self.update_count,
        }
        if self.spr_loss is not None:
            d["spr_loss"] = self.spr_loss
        if self.cosine_similarity is not None:
            d["cosine_similarity"] = self.cosine_similarity
        return d

    def __repr__(self) -> str:
        """String representation for debugging."""
        base = (
            f"UpdateMetrics(loss={self.loss:.4f}, td_error={self.td_error:.4f}, "
            f"grad_norm={self.grad_norm:.4f}, lr={self.learning_rate:.6f}, "
            f"updates={self.update_count}"
        )
        if self.spr_loss is not None:
            base += f", spr_loss={self.spr_loss:.4f}"
        if self.cosine_similarity is not None:
            base += f", cos_sim={self.cosine_similarity:.4f}"
        return base + ")"


def perform_update_step(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    gamma: float = 0.99,
    loss_type: str = "mse",
    max_grad_norm: float = 10.0,
    update_count: int = 0,
    spr_components: Dict[str, nn.Module] = None,
    spr_batch: Dict[str, torch.Tensor] = None,
    spr_weight: float = 2.0,
) -> UpdateMetrics:
    """
    Perform a single training update and return metrics.

    Executes the full DQN update step:
    1. Compute TD targets using target network
    2. Select Q-values for actions from online network
    3. Compute loss (MSE or Huber)
    4. (Optional) Compute SPR loss and combine with TD loss
    5. Backpropagate gradients
    6. Clip gradients by global norm
    7. Update network parameters
    8. (Optional) Update EMA encoder and projection
    9. Collect and return metrics

    When spr_components and spr_batch are provided, the SPR auxiliary
    loss is computed alongside the TD loss. The total loss is:
        total = td_loss + spr_weight * spr_loss
    After the optimizer step, the EMA encoder and projection are
    updated to track the online encoder and projection head.

    The optimizer must include parameters from all trainable modules
    (online network, transition model, projection head, prediction
    head) when SPR is enabled.

    Args:
        online_net: Online Q-network to train
        target_net: Target Q-network for TD targets
        optimizer: Optimizer for online network (and SPR modules if enabled)
        batch: Dictionary with keys 'states', 'actions', 'rewards', 'next_states', 'dones'
        gamma: Discount factor (default: 0.99)
        loss_type: 'mse' or 'huber' (default: 'mse')
        max_grad_norm: Maximum gradient norm for clipping (default: 10.0)
        update_count: Current update count for metrics (default: 0)
        spr_components: Optional dict with SPR modules. Required keys:
            'transition_model', 'projection_head', 'prediction_head',
            'target_encoder', 'target_projection'. Pass None to disable SPR.
        spr_batch: Optional dict with sequence data for SPR. Required keys:
            'states' (B, K+1, C, H, W), 'actions' (B, K), 'dones' (B, K).
        spr_weight: Weight for SPR loss in combined objective (default: 2.0).

    Returns:
        UpdateMetrics object with loss, TD error, grad norm, learning rate,
        update count, and optionally spr_loss and cosine_similarity.
    """
    # Set online network and SPR components to training mode
    online_net.train()
    if spr_components is not None:
        spr_components["transition_model"].train()
        spr_components["projection_head"].train()
        spr_components["prediction_head"].train()

    # Extract batch data
    states = batch["states"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_states = batch["next_states"]
    dones = batch["dones"]

    # Compute TD targets (no gradient)
    td_targets = compute_td_targets(rewards, next_states, dones, target_net, gamma)

    # Select Q-values for taken actions (with gradient)
    q_selected = select_q_values(online_net, states, actions)

    # Compute TD loss and error stats
    loss_dict = compute_dqn_loss(q_selected, td_targets, loss_type=loss_type)
    td_loss = loss_dict["loss"]
    td_error = loss_dict["td_error"].item()
    td_error_std = loss_dict["td_error_std"].item()

    # Compute SPR loss (if enabled)
    spr_loss_val = None
    cos_sim_val = None
    spr_loss_tensor = None
    if spr_components is not None and spr_batch is not None:
        from .spr_loss import compute_spr_forward

        spr_result = compute_spr_forward(
            online_encoder=online_net,
            transition_model=spr_components["transition_model"],
            projection_head=spr_components["projection_head"],
            prediction_head=spr_components["prediction_head"],
            target_encoder=spr_components["target_encoder"],
            target_projection=spr_components["target_projection"],
            states=spr_batch["states"],
            actions=spr_batch["actions"],
            dones=spr_batch["dones"],
        )

        spr_loss_tensor = spr_result["loss"]
        spr_loss_val = spr_loss_tensor.item()
        cos_sim_val = spr_result["cosine_similarity"].item()

    # Combine TD + SPR into total training loss
    combined = compute_combined_loss(td_loss, spr_loss_tensor, spr_weight)
    total_loss = combined["total_loss"]

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()

    # Clip gradients across all trainable parameters
    if spr_components is not None:
        all_params = (
            list(online_net.parameters())
            + list(spr_components["transition_model"].parameters())
            + list(spr_components["projection_head"].parameters())
            + list(spr_components["prediction_head"].parameters())
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            all_params, max_norm=max_grad_norm
        ).item()
    else:
        grad_norm = clip_gradients(online_net, max_norm=max_grad_norm)

    # Update parameters
    optimizer.step()

    # Update EMA after gradient step (SPR only)
    if spr_components is not None:
        spr_components["target_encoder"].update(online_net)
        spr_components["target_projection"].update(
            spr_components["projection_head"]
        )

    # Get current learning rate
    learning_rate = optimizer.param_groups[0]["lr"]

    # Create and return metrics
    metrics = UpdateMetrics(
        loss=total_loss.item(),
        td_error=td_error,
        td_error_std=td_error_std,
        grad_norm=grad_norm,
        learning_rate=learning_rate,
        update_count=update_count,
        spr_loss=spr_loss_val,
        cosine_similarity=cos_sim_val,
    )

    return metrics


def perform_rainbow_update_step(
    online_net: nn.Module,
    target_net: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    support: torch.Tensor,
    gamma: float = 0.99,
    n_step: int = 1,
    max_grad_norm: float = 10.0,
    update_count: int = 0,
    double_dqn: bool = True,
    buffer=None,
    spr_components: Dict[str, nn.Module] = None,
    spr_batch: Dict[str, torch.Tensor] = None,
    spr_weight: float = 2.0,
) -> UpdateMetrics:
    """
    Perform a single Rainbow DQN training update.

    Combines distributional C51 loss with importance sampling weights
    from prioritized experience replay, noisy net noise resampling,
    Double DQN action selection, and optional SPR auxiliary loss.

    Steps:
    1. Reset noise on online and target networks (NoisyNet)
    2. Forward pass through online net (log_probs with gradients)
    3. Forward pass through target net (log_probs, no gradients)
    4. Select next actions (Double DQN: online selects, target evaluates)
    5. Compute distributional cross-entropy loss with C51 projection
    6. Weight per-sample loss by IS weights from prioritized replay
    7. Add SPR loss if enabled
    8. Backprop, clip gradients, optimizer step
    9. Update priorities in replay buffer
    10. Update EMA encoder/projection (SPR only)

    Args:
        online_net: Online Rainbow network (must have reset_noise()).
        target_net: Target Rainbow network (must have reset_noise()).
        optimizer: Optimizer for online net (and SPR modules if enabled).
        batch: Dict from PrioritizedReplayBuffer.sample() with keys:
            'states', 'actions', 'rewards', 'next_states', 'dones',
            'indices' (int array), 'weights' (IS weights tensor).
        support: C51 support atoms, shape (num_atoms,).
        gamma: Base discount factor (default: 0.99).
        n_step: Number of multi-step returns (default: 1).
            The effective discount is gamma^n_step.
        max_grad_norm: Maximum gradient norm for clipping (default: 10.0).
        update_count: Current update count for metrics (default: 0).
        double_dqn: Use online net for action selection (default: True).
        buffer: PrioritizedReplayBuffer for priority updates.
            Pass None to skip priority updates.
        spr_components: Optional SPR module dict (same as perform_update_step).
        spr_batch: Optional SPR sequence data dict.
        spr_weight: Weight for SPR loss (default: 2.0).

    Returns:
        UpdateMetrics with loss, td_error, grad_norm, learning_rate,
        update_count, and optionally spr_loss and cosine_similarity.
    """
    # Set training mode
    online_net.train()
    if spr_components is not None:
        spr_components["transition_model"].train()
        spr_components["projection_head"].train()
        spr_components["prediction_head"].train()

    # Reset noise before forward passes (NoisyNet exploration)
    if hasattr(online_net, "reset_noise"):
        online_net.reset_noise()
    if hasattr(target_net, "reset_noise"):
        target_net.reset_noise()

    # Extract batch data
    states = batch["states"]
    actions = batch["actions"]
    rewards = batch["rewards"]
    next_states = batch["next_states"]
    dones = batch["dones"]
    is_weights = batch["weights"]  # (B,) IS weights from PER
    indices = batch["indices"]  # buffer indices for priority update

    # Effective discount for n-step returns
    gamma_n = gamma ** n_step

    # Online forward pass (with gradients for loss)
    online_output = online_net(states)
    online_log_probs = online_output["log_probs"]  # (B, A, atoms)

    # Target forward pass (no gradients)
    with torch.no_grad():
        target_output = target_net(next_states)
        target_log_probs = target_output["log_probs"]  # (B, A, atoms)

    # Select next actions (Double DQN: online selects, target evaluates)
    next_actions = select_next_actions(
        online_net, target_net, next_states, double_dqn=double_dqn,
    )

    # Distributional cross-entropy loss with C51 projection
    dist_result = compute_distributional_loss(
        online_log_probs=online_log_probs,
        actions=actions,
        rewards=rewards,
        dones=dones,
        target_log_probs=target_log_probs,
        next_actions=next_actions,
        support=support,
        gamma=gamma_n,
    )

    per_sample_loss = dist_result["per_sample_loss"]  # (B,) with gradients

    # Apply IS weights: weighted mean of per-sample losses
    weighted_loss = (is_weights * per_sample_loss).mean()

    # TD error proxy for monitoring (mean per-sample cross-entropy)
    with torch.no_grad():
        td_error = per_sample_loss.mean().item()
        td_error_std = per_sample_loss.std().item()

    # Compute SPR loss (if enabled)
    spr_loss_val = None
    cos_sim_val = None
    spr_loss_tensor = None
    if spr_components is not None and spr_batch is not None:
        from .spr_loss import compute_spr_forward

        spr_result = compute_spr_forward(
            online_encoder=online_net,
            transition_model=spr_components["transition_model"],
            projection_head=spr_components["projection_head"],
            prediction_head=spr_components["prediction_head"],
            target_encoder=spr_components["target_encoder"],
            target_projection=spr_components["target_projection"],
            states=spr_batch["states"],
            actions=spr_batch["actions"],
            dones=spr_batch["dones"],
        )

        spr_loss_tensor = spr_result["loss"]
        spr_loss_val = spr_loss_tensor.item()
        cos_sim_val = spr_result["cosine_similarity"].item()

    # Combine distributional + SPR loss
    combined = compute_combined_loss(weighted_loss, spr_loss_tensor, spr_weight)
    total_loss = combined["total_loss"]

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()

    # Clip gradients across all trainable parameters
    if spr_components is not None:
        all_params = (
            list(online_net.parameters())
            + list(spr_components["transition_model"].parameters())
            + list(spr_components["projection_head"].parameters())
            + list(spr_components["prediction_head"].parameters())
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            all_params, max_norm=max_grad_norm
        ).item()
    else:
        grad_norm = clip_gradients(online_net, max_norm=max_grad_norm)

    # Update parameters
    optimizer.step()

    # Update priorities in replay buffer
    if buffer is not None:
        new_priorities = per_sample_loss.detach().cpu().numpy()
        buffer.update_priorities(indices, new_priorities)

    # Update EMA after gradient step (SPR only)
    if spr_components is not None:
        spr_components["target_encoder"].update(online_net)
        spr_components["target_projection"].update(
            spr_components["projection_head"]
        )

    # Get current learning rate
    learning_rate = optimizer.param_groups[0]["lr"]

    # Create and return metrics
    metrics = UpdateMetrics(
        loss=total_loss.item(),
        td_error=td_error,
        td_error_std=td_error_std,
        grad_norm=grad_norm,
        learning_rate=learning_rate,
        update_count=update_count,
        spr_loss=spr_loss_val,
        cosine_similarity=cos_sim_val,
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

    When noisy_nets=True, epsilon is always 0.0 (both training and
    evaluation). Exploration is handled entirely by NoisyLinear noise
    injection, so epsilon-greedy randomness is disabled.

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
    noisy_nets : bool
        If True, always return 0.0 for all epsilon queries. Exploration
        is provided by NoisyLinear layers instead (default: False).

    Usage
    -----
    >>> scheduler = EpsilonScheduler(epsilon_start=1.0, epsilon_end=0.1, decay_frames=1000000)
    >>> epsilon = scheduler.get_epsilon(current_frame=500000)  # Returns 0.55
    >>> eval_eps = scheduler.get_eval_epsilon()  # Returns 0.05
    >>> # With noisy nets:
    >>> scheduler = EpsilonScheduler(noisy_nets=True)
    >>> scheduler.get_epsilon(0)  # Returns 0.0
    >>> scheduler.get_eval_epsilon()  # Returns 0.0
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        decay_frames: int = 1_000_000,
        eval_epsilon: float = 0.05,
        noisy_nets: bool = False,
    ):
        assert (
            0.0 <= epsilon_start <= 1.0
        ), f"epsilon_start must be in [0,1], got {epsilon_start}"
        assert (
            0.0 <= epsilon_end <= 1.0
        ), f"epsilon_end must be in [0,1], got {epsilon_end}"
        assert epsilon_start >= epsilon_end, "epsilon_start must be >= epsilon_end"
        assert decay_frames > 0, f"decay_frames must be positive, got {decay_frames}"
        assert (
            0.0 <= eval_epsilon <= 1.0
        ), f"eval_epsilon must be in [0,1], got {eval_epsilon}"

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_frames = decay_frames
        self.eval_epsilon = eval_epsilon
        self.noisy_nets = noisy_nets

        # Precompute slope for efficiency
        self.slope = (epsilon_end - epsilon_start) / decay_frames

        # State for resume (can be set when loading checkpoint)
        self.frame_counter = 0
        self.current_epsilon = 0.0 if noisy_nets else epsilon_start

    def get_epsilon(self, current_frame: int) -> float:
        """
        Get epsilon for current training frame with linear decay.

        Returns 0.0 immediately when noisy_nets is enabled.

        Parameters
        ----------
        current_frame : int
            Current environment frame count (not update count)

        Returns
        -------
        float
            Epsilon value in [epsilon_end, epsilon_start], or 0.0 if noisy_nets

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
        if self.noisy_nets:
            return 0.0

        if current_frame >= self.decay_frames:
            return self.epsilon_end

        # Linear interpolation: start + slope * frames
        epsilon = self.epsilon_start + self.slope * current_frame

        # Clamp to [epsilon_end, epsilon_start] for numerical stability
        return max(self.epsilon_end, min(self.epsilon_start, epsilon))

    def get_eval_epsilon(self) -> float:
        """
        Get fixed epsilon for evaluation mode.

        Returns 0.0 when noisy_nets is enabled.

        Returns
        -------
        float
            Fixed epsilon for evaluation (no decay), or 0.0 if noisy_nets
        """
        if self.noisy_nets:
            return 0.0
        return self.eval_epsilon

    def to_dict(self) -> dict:
        """Export scheduler configuration as dictionary."""
        d = {
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "decay_frames": self.decay_frames,
            "eval_epsilon": self.eval_epsilon,
        }
        if self.noisy_nets:
            d["noisy_nets"] = True
        return d
