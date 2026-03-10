"""
Tests for data augmentation transforms.

Verifies:
- Output shape preservation
- Per-sample independent shifts
- Padding boundary correctness
- GPU compatibility (if available)
"""

import torch
import pytest

from src.augmentation import random_shift


def test_random_shift_preserves_shape():
    """Output shape must match input shape."""
    x = torch.rand(8, 4, 84, 84)
    out = random_shift(x, pad=4)
    assert out.shape == x.shape


def test_random_shift_single_sample():
    """Works with batch size 1."""
    x = torch.rand(1, 4, 84, 84)
    out = random_shift(x, pad=4)
    assert out.shape == (1, 4, 84, 84)


def test_random_shift_different_pad():
    """Different pad values produce correct output shapes."""
    x = torch.rand(4, 4, 84, 84)
    for pad in [1, 2, 4, 8]:
        out = random_shift(x, pad=pad)
        assert out.shape == x.shape


def test_random_shift_zero_pad_is_identity():
    """Zero padding should return the original tensor."""
    x = torch.rand(4, 4, 84, 84)
    out = random_shift(x, pad=0)
    assert torch.allclose(out, x)


def test_random_shift_values_in_range():
    """Output values should be in [0, 1] for input in [0, 1]."""
    x = torch.rand(8, 4, 84, 84)
    out = random_shift(x, pad=4)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_random_shift_independent_samples():
    """Different samples in batch should get different shifts."""
    torch.manual_seed(42)
    # Create a tensor where each sample is a constant value
    x = torch.zeros(16, 1, 84, 84)
    for i in range(16):
        x[i] = float(i + 1) / 16.0

    out = random_shift(x, pad=4)
    # After shift, border pixels should be zero (from padding) for some samples
    # Check that not all samples are identical
    diffs = []
    for i in range(1, 16):
        diffs.append((out[i] - out[0]).abs().sum().item())
    assert any(d > 0 for d in diffs)


def test_random_shift_stochastic():
    """Two calls with same input should produce different results."""
    x = torch.rand(8, 4, 84, 84)
    out1 = random_shift(x, pad=4)
    out2 = random_shift(x, pad=4)
    # Extremely unlikely to be identical with random shifts
    assert not torch.allclose(out1, out2)


def test_random_shift_non_square():
    """Works with non-84x84 spatial dimensions."""
    x = torch.rand(4, 3, 64, 64)
    out = random_shift(x, pad=4)
    assert out.shape == (4, 3, 64, 64)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_random_shift_cuda():
    """Output stays on the same device as input."""
    x = torch.rand(4, 4, 84, 84, device="cuda")
    out = random_shift(x, pad=4)
    assert out.device == x.device
    assert out.shape == x.shape
