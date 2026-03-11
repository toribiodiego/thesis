"""
Distributional RL loss (Bellemare et al. 2017, C51).

Implements the categorical projected Bellman update and cross-entropy
loss for distributional reinforcement learning. Used by Rainbow DQN
where the value distribution is represented as a categorical over
fixed support atoms z_i = v_min + i * delta_z.

The projection maps the Bellman-updated target distribution onto the
fixed support, distributing probability to neighboring atoms via
linear interpolation (Algorithm 1 in the C51 paper).

Multi-step variant: the replay buffer computes n-step returns R^(n)
and stores them as the reward. The caller passes gamma^n instead of
gamma so the Bellman projection uses the correct discount.
"""

from typing import Dict

import torch


def project_distribution(
    next_probs: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    support: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """
    Project target distribution onto fixed support (C51 Algorithm 1).

    For each target atom z_j, computes the shifted atom
    Tz_j = clamp(r + gamma * (1 - done) * z_j, v_min, v_max) and
    distributes the target probability p_j to the two neighboring
    support atoms via linear interpolation.

    Args:
        next_probs: Target probabilities for the selected next action,
            shape (B, num_atoms). Should be detached.
        rewards: Reward or n-step return, shape (B,).
        dones: Terminal flags, shape (B,).
        support: Fixed support atoms, shape (num_atoms,). Assumed to be
            evenly spaced from v_min to v_max.
        gamma: Discount factor. For multi-step, pass gamma^n.

    Returns:
        Projected distribution, shape (B, num_atoms). Each row sums
        to 1 (valid probability distribution).
    """
    num_atoms = support.shape[0]
    v_min = support[0].item()
    v_max = support[-1].item()
    delta_z = (v_max - v_min) / (num_atoms - 1)

    # Tz_j = r + gamma * (1 - done) * z_j, clipped to [v_min, v_max]
    rewards_2d = rewards.unsqueeze(1)              # (B, 1)
    dones_2d = dones.float().unsqueeze(1)          # (B, 1)
    support_2d = support.unsqueeze(0)              # (1, num_atoms)

    Tz = rewards_2d + (1.0 - dones_2d) * gamma * support_2d  # (B, num_atoms)
    Tz = Tz.clamp(v_min, v_max)

    # Position on support: b_j = (Tz_j - v_min) / delta_z
    b = (Tz - v_min) / delta_z  # (B, num_atoms), values in [0, num_atoms-1]
    l = b.floor().long().clamp(0, num_atoms - 1)   # lower neighbor
    u = b.ceil().long().clamp(0, num_atoms - 1)    # upper neighbor

    # Interpolation weights
    weight_l = u.float() - b   # fraction assigned to lower neighbor
    weight_u = b - l.float()   # fraction assigned to upper neighbor

    # When b is exactly an integer, l == u and both weights are 0.
    # Fix: assign full probability to that atom.
    eq_mask = (l == u)
    weight_l[eq_mask] = 1.0

    # Scatter probabilities to neighbors
    projected = torch.zeros_like(next_probs)  # (B, num_atoms)
    projected.scatter_add_(1, l, next_probs * weight_l)
    projected.scatter_add_(1, u, next_probs * weight_u)

    return projected


def compute_distributional_loss(
    online_log_probs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    target_log_probs: torch.Tensor,
    next_actions: torch.Tensor,
    support: torch.Tensor,
    gamma: float,
) -> Dict[str, torch.Tensor]:
    """
    Compute distributional cross-entropy loss for C51/Rainbow.

    Selects the online distribution for the taken action, projects
    the target distribution for the best next action onto the fixed
    support, and computes the cross-entropy between them.

    For Double DQN, next_actions should come from the online net's
    Q-values (argmax). For standard DQN, next_actions come from the
    target net's Q-values.

    Args:
        online_log_probs: Log probabilities from online network,
            shape (B, num_actions, num_atoms). Has gradients.
        actions: Actions taken in the batch, shape (B,) int64.
        rewards: Rewards or n-step returns, shape (B,) float32.
        dones: Terminal flags, shape (B,).
        target_log_probs: Log probabilities from target network,
            shape (B, num_actions, num_atoms). Should be detached
            (computed under torch.no_grad).
        next_actions: Actions to select from the target distribution,
            shape (B,) int64. For Double DQN these come from the
            online net; for standard C51 from the target net.
        support: Support atoms z_i, shape (num_atoms,).
        gamma: Discount factor. For multi-step, pass gamma^n.

    Returns:
        Dict with:
            - 'loss': Scalar mean cross-entropy (with gradients).
            - 'per_sample_loss': (B,) per-sample cross-entropy with
              gradients, for IS weight application. Use .detach()
              for priority updates.
    """
    num_atoms = online_log_probs.shape[2]

    # Select online log-probs for the taken actions: (B, atoms)
    idx_a = actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, num_atoms)
    log_probs_a = online_log_probs.gather(1, idx_a).squeeze(1)  # (B, atoms)

    # Select target probs for best next actions: (B, atoms)
    idx_next = next_actions.unsqueeze(1).unsqueeze(2).expand(-1, 1, num_atoms)
    target_probs = target_log_probs.exp().gather(1, idx_next).squeeze(1)

    # Project target distribution onto support (detached)
    projected = project_distribution(
        target_probs, rewards, dones, support, gamma,
    ).detach()

    # Cross-entropy: L_i = -sum_j m_j * log(p_j) per sample
    per_sample_loss = -(projected * log_probs_a).sum(dim=1)  # (B,)
    loss = per_sample_loss.mean()

    return {
        "loss": loss,
        "per_sample_loss": per_sample_loss,
    }
