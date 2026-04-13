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


def _resolve_steps(args_steps, run_dir):
    """Resolve step arguments to a sorted list of ints.

    Accepts individual step numbers or 'all' to auto-discover
    from the checkpoints directory.
    """
    from src.analysis.checkpoint import discover_checkpoints

    if args_steps == ["all"]:
        steps = discover_checkpoints(run_dir)
        if not steps:
            raise ValueError(f"No checkpoints found in {run_dir}")
        return steps
    return sorted(int(s) for s in args_steps)


def main():
    parser = argparse.ArgumentParser(
        description="Filter frequency analysis: 2D FFT of conv kernels (M12)"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", nargs="+", required=True,
                        help="Checkpoint steps (e.g., 10000 50000 100000) or 'all'")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save CSV results (e.g., run_dir/analysis/filter_frequency.csv)")
    args = parser.parse_args()

    import pandas as pd
    from src.analysis.checkpoint import load_checkpoint

    steps = _resolve_steps(args.steps, args.run_dir)
    print(f"Filter frequency analysis: {args.run_dir}")
    print(f"  checkpoints: {steps}")

    all_rows = []

    for step in steps:
        print(f"\n--- step {step} ---")
        ckpt = load_checkpoint(args.run_dir, step)

        encoder_params = ckpt.online_params["encoder"]
        kernels = _collect_conv_kernels(encoder_params)

        for name, kernel in kernels:
            stats = analyze_kernel_frequency(kernel)
            hl = stats["high_low_ratio"]
            hl_str = f"{hl:.3f}" if hl is not None else "n/a"
            print(f"  {name:<30} DC%={100*stats['dc_fraction']:.1f}%  H/L={hl_str}")
            all_rows.append({
                "step": step,
                "layer": name,
                "kernel_h": stats["kernel_shape"][0],
                "kernel_w": stats["kernel_shape"][1],
                "n_filters": stats["n_filters"],
                "dc_power": stats["dc_power"],
                "non_dc_mean_power": stats["non_dc_mean_power"],
                "dc_fraction": round(stats["dc_fraction"], 6),
                "high_low_ratio": stats["high_low_ratio"],
            })

    # -- Save CSV ------------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df = pd.DataFrame(all_rows)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output} ({len(df)} rows)")
    else:
        print(f"\n{len(all_rows)} rows computed (use --output to save)")


if __name__ == "__main__":
    main()
