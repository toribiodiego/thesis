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
