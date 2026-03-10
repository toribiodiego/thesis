"""Data augmentation transforms for Atari observations.

Implements DrQ-style random shift augmentation (Kostrikov et al., 2020).
"""

import torch
import torch.nn.functional as F


def random_shift(x: torch.Tensor, pad: int = 4) -> torch.Tensor:
    """Apply random shift augmentation via pad-and-crop.

    Pads the spatial dimensions with zeros, then randomly crops back to
    the original size. Each sample in the batch gets an independent shift.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (B, C, H, W) in [0, 1].
    pad : int
        Number of pixels to pad on each side (default: 4).

    Returns
    -------
    torch.Tensor
        Augmented tensor with the same shape as input.
    """
    b, c, h, w = x.shape
    padded = F.pad(x, (pad, pad, pad, pad), mode="constant", value=0.0)

    # Random crop offsets per sample
    crop_h = torch.randint(0, 2 * pad + 1, (b,), device=x.device)
    crop_w = torch.randint(0, 2 * pad + 1, (b,), device=x.device)

    # Gather crops using advanced indexing
    # Build index grids for each sample
    rows = torch.arange(h, device=x.device).unsqueeze(0) + crop_h.unsqueeze(1)  # (B, H)
    cols = torch.arange(w, device=x.device).unsqueeze(0) + crop_w.unsqueeze(1)  # (B, W)

    # Index into padded tensor: (B, C, H, W)
    batch_idx = torch.arange(b, device=x.device)[:, None, None, None]
    chan_idx = torch.arange(c, device=x.device)[None, :, None, None]
    row_idx = rows[:, None, :, None].expand(b, c, h, w)
    col_idx = cols[:, None, None, :].expand(b, c, h, w)

    return padded[batch_idx, chan_idx, row_idx, col_idx]
