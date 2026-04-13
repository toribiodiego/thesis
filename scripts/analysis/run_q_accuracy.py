#!/usr/bin/env python3
"""Q-value accuracy analysis script (M15).

Compares predicted Q-values from the network against actual
discounted returns computed from replay buffer reward sequences.
Reports Spearman rank correlation, mean signed error (Q - G),
and RMSE.

Only uses timesteps from complete episodes where the discounted
return is known. Indexes Q(s_t, a_t) by the action actually
taken in the replay buffer.

Note: representation extraction on the full replay buffer takes
~80s on CPU for IMPALA, ~20s for Nature CNN.

Usage:
    python scripts/analysis/run_q_accuracy.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step 10000

    python scripts/analysis/run_q_accuracy.py \\
        --run-dir experiments/dqn_atari/runs/spr_crazy_climber_seed13 \\
        --step 10000 --output output/probing/q_accuracy.json
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def _read_effective_gamma(run_dir, step):
    """Read effective_gamma from steps.csv at the given step."""
    csv_path = os.path.join(run_dir, "steps.csv")
    if not os.path.isfile(csv_path):
        return None
    df = pd.read_csv(csv_path)
    if "effective_gamma" not in df.columns:
        return None
    row = df.iloc[(df["step"] - step).abs().argsort()[:1]]
    return float(row["effective_gamma"].values[0])


def _stack_replay_frames(replay):
    """Stack single replay frames into 4-frame HWC observations.

    Returns (observations, valid_indices) where valid_indices maps
    each stacked observation back to the replay buffer index of the
    newest frame in the stack.
    """
    frames = replay.observations
    terms = replay.terminals
    obs_list, idx_list = [], []
    for i in range(3, len(frames)):
        if not any(terms[i - 3 : i]):
            obs_list.append(np.stack(frames[i - 3 : i + 1], axis=-1))
            idx_list.append(i)
    return np.array(obs_list, dtype=np.uint8), np.array(idx_list)


def main():
    parser = argparse.ArgumentParser(
        description="Q-value accuracy: predicted Q vs actual returns (M15)"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override gamma (default: read from steps.csv)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # -- Load checkpoint -----------------------------------------------------
    print(f"Loading checkpoint: {args.run_dir} step {args.step}")
    from src.analysis.checkpoint import load_checkpoint
    ckpt = load_checkpoint(args.run_dir, args.step)
    print(f"  encoder: {ckpt.encoder_type}, hidden_dim: {ckpt.hidden_dim}")

    # -- Get gamma -----------------------------------------------------------
    if args.gamma is not None:
        gamma = args.gamma
    else:
        gamma = _read_effective_gamma(args.run_dir, args.step)
        if gamma is None:
            print("ERROR: Could not read effective_gamma from steps.csv. "
                  "Use --gamma to specify manually.")
            sys.exit(1)
    print(f"  gamma: {gamma}")

    # -- Load replay buffer and compute returns ------------------------------
    print("Loading replay buffer...")
    from src.analysis.replay_buffer import load_replay_buffer
    replay = load_replay_buffer(args.run_dir, args.step)
    print(f"  {replay.add_count} entries, "
          f"{replay.terminals.sum()} episode boundaries")

    print("Computing discounted returns...")
    from src.analysis.returns import compute_returns
    returns = compute_returns(replay, gamma)
    n_valid_returns = int(np.isfinite(returns).sum())
    n_nan = int(np.isnan(returns).sum())
    print(f"  {n_valid_returns} valid returns, {n_nan} NaN (incomplete episode)")

    # -- Stack frames and extract Q-values -----------------------------------
    print("Stacking replay frames...")
    observations, valid_indices = _stack_replay_frames(replay)
    print(f"  {len(observations)} stacked observations")

    print(f"Extracting Q-values (batch_size={args.batch_size})...")
    print("  (this may take a few minutes on CPU)")
    t0 = time.time()
    from src.analysis.representations import extract_q_values
    q_all = extract_q_values(
        ckpt, observations, batch_size=args.batch_size, seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"  shape: {q_all.shape} ({elapsed:.1f}s)")

    # -- Index Q(s_t, a_t) by taken action -----------------------------------
    actions_at_valid = replay.actions[valid_indices]
    q_taken = q_all[np.arange(len(q_all)), actions_at_valid]

    # -- Filter to timesteps with known returns ------------------------------
    returns_at_valid = returns[valid_indices]
    mask = np.isfinite(returns_at_valid)
    q_matched = q_taken[mask]
    g_matched = returns_at_valid[mask]
    n_matched = len(q_matched)

    print(f"  {n_matched} matched Q-return pairs")

    if n_matched < 2:
        print("ERROR: Not enough matched pairs for correlation.")
        sys.exit(1)

    # -- Compute metrics -----------------------------------------------------
    spearman_r, spearman_p = stats.spearmanr(q_matched, g_matched)
    signed_error = q_matched - g_matched
    mean_signed_error = float(signed_error.mean())
    rmse = float(np.sqrt((signed_error ** 2).mean()))

    print()
    print("Q-Value Accuracy Results")
    print(f"  Spearman r:         {spearman_r:.4f} (p={spearman_p:.2e})")
    print(f"  Mean signed error:  {mean_signed_error:.4f} (Q - G)")
    print(f"  RMSE:               {rmse:.4f}")
    print(f"  Q mean:             {q_matched.mean():.4f}")
    print(f"  G mean:             {g_matched.mean():.4f}")
    print(f"  Matched pairs:      {n_matched}")
    print()

    # -- Save JSON -----------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output = {
            "run_dir": args.run_dir,
            "step": args.step,
            "gamma": gamma,
            "encoder_type": ckpt.encoder_type,
            "hidden_dim": ckpt.hidden_dim,
            "num_buffer_entries": replay.add_count,
            "num_matched_pairs": n_matched,
            "spearman_r": round(spearman_r, 6),
            "spearman_p": spearman_p,
            "mean_signed_error": round(mean_signed_error, 6),
            "rmse": round(rmse, 6),
            "q_mean": round(float(q_matched.mean()), 6),
            "g_mean": round(float(g_matched.mean()), 6),
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
