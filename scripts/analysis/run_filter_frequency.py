#!/usr/bin/env python3
"""Filter frequency analysis script (M12).

Computes 2D FFT power spectrum of conv filter kernels per encoder
layer. No forward pass or observations needed -- operates directly
on checkpoint weight tensors.

For each conv layer, computes the magnitude spectrum of each
(H, W) filter slice, averages across input/output channel pairs,
and reports per-layer frequency statistics (mean power at DC,
low, mid, and high spatial frequencies).

Usage:
    python scripts/analysis/run_filter_frequency.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step 10000

    python scripts/analysis/run_filter_frequency.py \\
        --run-dir experiments/dqn_atari/runs/spr_crazy_climber_seed13 \\
        --step 10000 --output output/probing/filter_freq.json
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def _collect_conv_kernels(encoder_params, prefix=""):
    """Recursively collect conv kernel arrays from the encoder param tree.

    Returns list of (layer_name, kernel_array) where kernel_array
    has shape (H, W, C_in, C_out) in Flax convention.
    """
    kernels = []
    for k in sorted(encoder_params.keys()):
        v = encoder_params[k]
        if isinstance(v, dict):
            kernels.extend(_collect_conv_kernels(v, prefix + k + "/"))
        elif k == "kernel":
            arr = np.array(v)
            if arr.ndim == 4:  # (H, W, C_in, C_out) conv kernel
                kernels.append((prefix.rstrip("/"), arr))
    return kernels


def analyze_kernel_frequency(kernel):
    """Compute frequency statistics for a conv kernel.

    Args:
        kernel: (H, W, C_in, C_out) float32 conv kernel.

    Returns:
        Dict with per-layer aggregate frequency statistics.
    """
    h, w, c_in, c_out = kernel.shape

    # Compute 2D FFT magnitude for each (c_in, c_out) filter slice
    # Reshape to (C_in * C_out, H, W) for batch FFT
    filters = kernel.transpose(2, 3, 0, 1).reshape(-1, h, w)
    n_filters = filters.shape[0]

    # 2D FFT, shift DC to center, take magnitude
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(filters), axes=(-2, -1)))

    # Power spectrum (squared magnitude), averaged across all filters
    power = (fft_mag ** 2).mean(axis=0)  # (H, W)

    # Frequency band decomposition based on distance from DC
    cy, cx = h // 2, w // 2
    y_coords, x_coords = np.mgrid[:h, :w]
    dist = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)
    max_dist = dist.max()

    # DC component (center pixel)
    dc_power = float(power[cy, cx])

    # Band boundaries: low (0-33%), mid (33-66%), high (66-100%)
    if max_dist > 0:
        low_mask = (dist > 0) & (dist <= max_dist / 3)
        mid_mask = (dist > max_dist / 3) & (dist <= 2 * max_dist / 3)
        high_mask = dist > 2 * max_dist / 3

        low_power = float(power[low_mask].mean()) if low_mask.any() else 0.0
        mid_power = float(power[mid_mask].mean()) if mid_mask.any() else 0.0
        high_power = float(power[high_mask].mean()) if high_mask.any() else 0.0
    else:
        low_power = mid_power = high_power = 0.0

    # Total power excluding DC
    non_dc_power = float(power[dist > 0].mean()) if (dist > 0).any() else 0.0

    # High-to-low ratio (higher = more high-frequency content)
    hl_ratio = high_power / low_power if low_power > 0 else float("nan")

    # DC fraction: what share of total power is at DC
    total_power = float(power.sum())
    dc_fraction = dc_power / total_power if total_power > 0 else 0.0

    return {
        "kernel_shape": [h, w],
        "c_in": c_in,
        "c_out": c_out,
        "n_filters": n_filters,
        "dc_power": round(dc_power, 6),
        "low_freq_power": round(low_power, 6),
        "mid_freq_power": round(mid_power, 6),
        "high_freq_power": round(high_power, 6),
        "non_dc_mean_power": round(non_dc_power, 6),
        "high_low_ratio": round(hl_ratio, 4) if not np.isnan(hl_ratio) else None,
        "dc_fraction": round(dc_fraction, 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter frequency analysis: 2D FFT of conv kernels (M12)"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # -- Load checkpoint (lightweight, no JAX forward pass) ------------------
    print(f"Loading checkpoint: {args.run_dir} step {args.step}")
    from src.analysis.checkpoint import load_checkpoint
    ckpt = load_checkpoint(args.run_dir, args.step)
    print(f"  encoder: {ckpt.encoder_type}, hidden_dim: {ckpt.hidden_dim}")

    # -- Extract conv kernels from encoder params ----------------------------
    encoder_params = ckpt.online_params["encoder"]
    kernels = _collect_conv_kernels(encoder_params)
    print(f"  {len(kernels)} conv layers found")

    # -- Compute frequency statistics per layer ------------------------------
    print()
    print(f"{'Layer':<35} {'Kernel':>7} {'Filters':>8} "
          f"{'DC':>10} {'NonDC':>10} {'DC%':>7} {'H/L':>7}")
    print("-" * 90)

    results = []
    for name, kernel in kernels:
        stats = analyze_kernel_frequency(kernel)
        h, w = stats["kernel_shape"]
        hl = stats["high_low_ratio"]
        hl_str = f"{hl:>7.3f}" if hl is not None else "    n/a"
        print(f"{name:<35} {h}x{w:>5} {stats['n_filters']:>8} "
              f"{stats['dc_power']:>10.4f} {stats['non_dc_mean_power']:>10.4f} "
              f"{100 * stats['dc_fraction']:>6.1f}% {hl_str}")
        results.append({"layer": name, **stats})

    print()

    # -- Save JSON -----------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output = {
            "run_dir": args.run_dir,
            "step": args.step,
            "encoder_type": ckpt.encoder_type,
            "layers": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
