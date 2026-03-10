"""
SPR auxiliary loss computation.

Implements the self-predictive representation loss from Schwarzer et al.
(2021), Equation 4. The loss is the negative cosine similarity between
online predicted representations and EMA target representations, summed
over K prediction steps and averaged over the batch.

Predictions that cross episode boundaries are masked out via a
cumulative done flag, so the transition model is never penalized for
predicting across resets.

Reference: https://arxiv.org/abs/2007.05929, Section 2.2
"""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_spr_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    dones: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute SPR auxiliary loss (negative cosine similarity).

    L_SPR = -(1/N) * sum_{k,b} mask_{k,b} * cos_sim(y_hat_{k,b}, y_tilde_{k,b})

    where the mask zeroes out all steps at and after the first episode
    boundary in each sample's sequence.

    Args:
        predictions: Online predicted representations after projection
            and prediction heads, shape (K, B, dim). Gradients flow
            back through the online encoder, transition model,
            projection head, and prediction head.
        targets: Target representations from the EMA encoder and
            target projection, shape (K, B, dim). Must be detached
            (no gradients).
        dones: Episode termination flags for transitions in the
            sequence, shape (K, B). dones[k, b] = True means the
            transition producing the target at prediction step k
            crossed an episode boundary, making step k and all
            subsequent steps invalid for sample b.

    Returns:
        Dict containing:
            - 'loss': Scalar SPR loss (negative cosine similarity,
              averaged over valid predictions). Minimizing this
              maximizes alignment between predicted and target
              representations.
            - 'per_step_loss': Mean loss at each prediction step (K,),
              detached, for logging.
            - 'num_valid': Total number of valid (unmasked) predictions
              across all steps and batch samples, detached.
            - 'cosine_similarity': Mean cosine similarity across valid
              predictions, detached, for logging (higher is better).
    """
    K, B, dim = predictions.shape

    # L2-normalize along feature dimension
    pred_norm = F.normalize(predictions, dim=2)
    tgt_norm = F.normalize(targets, dim=2)

    # Cosine similarity per step per sample: (K, B)
    cos_sim = (pred_norm * tgt_norm).sum(dim=2)

    # Build valid mask from done flags.
    # Once any done is True, all subsequent steps are invalid.
    # cumprod of (1 - done) stays 1 until the first done, then 0.
    not_done = 1.0 - dones.float()
    valid_mask = torch.cumprod(not_done, dim=0)  # (K, B)

    # Negative cosine similarity, masked
    neg_cos_sim = -cos_sim * valid_mask  # (K, B)

    # Average over all valid entries
    num_valid = valid_mask.sum().clamp(min=1.0)
    loss = neg_cos_sim.sum() / num_valid

    # Per-step diagnostics (detached)
    with torch.no_grad():
        step_valid = valid_mask.sum(dim=1).clamp(min=1.0)  # (K,)
        per_step_loss = neg_cos_sim.detach().sum(dim=1) / step_valid
        mean_cos_sim = (cos_sim.detach() * valid_mask).sum() / num_valid

    return {
        "loss": loss,
        "per_step_loss": per_step_loss,
        "num_valid": num_valid.detach(),
        "cosine_similarity": mean_cos_sim,
    }


def compute_spr_forward(
    online_encoder: nn.Module,
    transition_model: nn.Module,
    projection_head: nn.Module,
    prediction_head: nn.Module,
    target_encoder: nn.Module,
    target_projection: nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
    dones: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Compute SPR loss from a sequence batch.

    Runs the full SPR forward pass:
    - Online side: encode s_0 -> transition model (K steps) -> project -> predict
    - Target side: EMA encode s_{1..K} -> EMA project (no gradients)
    - Loss: negative cosine similarity with episode boundary masking

    Gradients flow through the online encoder (shared with Q-learning),
    transition model, projection head, and prediction head. The target
    side is gradient-free (updated only via EMA).

    Args:
        online_encoder: Online DQN encoder (shared with Q-learning).
        transition_model: Action-conditioned transition model.
        projection_head: Online projection head.
        prediction_head: Online prediction head.
        target_encoder: EMA encoder wrapping the online encoder.
        target_projection: EMA projection wrapping the projection head.
        states: Sequence observations, shape (B, K+1, C, H, W).
            states[:, 0] is the initial state for the online encoder,
            states[:, k+1] is the target for prediction step k.
        actions: Actions taken at each step, shape (B, K) as int64.
        dones: Episode termination flags, shape (B, K).

    Returns:
        Dict from compute_spr_loss with 'loss', 'per_step_loss',
        'num_valid', and 'cosine_similarity'.
    """
    K = actions.shape[1]

    # Online side: encode initial state to get spatial features
    online_output = online_encoder(states[:, 0])
    z = online_output["conv_output"]  # (B, 64, 7, 7)

    # Iteratively predict K future latent states
    predictions = []
    for k in range(K):
        z = transition_model(z, actions[:, k])
        y = projection_head(z)        # (B, proj_dim)
        pred = prediction_head(y)      # (B, proj_dim)
        predictions.append(pred)

    predictions = torch.stack(predictions, dim=0)  # (K, B, proj_dim)

    # Target side: encode and project each future state (no gradients)
    targets = []
    with torch.no_grad():
        for k in range(K):
            tgt_output = target_encoder(states[:, k + 1])
            z_tilde = tgt_output["conv_output"]   # (B, 64, 7, 7)
            y_tilde = target_projection(z_tilde)   # (B, proj_dim)
            targets.append(y_tilde)

    targets = torch.stack(targets, dim=0)  # (K, B, proj_dim)

    # Transpose dones from batch-first (B, K) to step-first (K, B)
    dones_kfirst = dones.transpose(0, 1).contiguous()

    return compute_spr_loss(predictions, targets, dones_kfirst)
