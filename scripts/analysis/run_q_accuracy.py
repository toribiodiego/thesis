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


def _resolve_steps(args_steps, run_dir):
    """Resolve step arguments to a sorted list of ints."""
    from src.analysis.checkpoint import discover_checkpoints

    if args_steps == ["all"]:
        steps = discover_checkpoints(run_dir)
        if not steps:
            raise ValueError(f"No checkpoints found in {run_dir}")
        return steps
    return sorted(int(s) for s in args_steps)


def main():
    parser = argparse.ArgumentParser(
        description="Q-value accuracy: predicted Q vs actual returns (M15)"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", nargs="+", required=True,
                        help="Checkpoint steps (e.g., 10000 50000 100000) or 'all'")
    parser.add_argument("--gamma", type=float, default=None,
                        help="Override gamma (default: read from steps.csv)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save CSV results")
    args = parser.parse_args()

    import gc
    import pandas as pd_
    from src.analysis.checkpoint import load_checkpoint
    from src.analysis.replay_buffer import load_replay_buffer
    from src.analysis.returns import compute_returns
    from src.analysis.representations import extract_q_values

    steps = _resolve_steps(args.steps, args.run_dir)
    print(f"Q-value accuracy: {args.run_dir}")
    print(f"  checkpoints: {steps}")

    # Resolve gamma once (same for all checkpoints in a run)
    if args.gamma is not None:
        gamma = args.gamma
    else:
        gamma = _read_effective_gamma(args.run_dir, steps[0])
        if gamma is None:
            print("ERROR: Could not read effective_gamma from steps.csv. "
                  "Use --gamma to specify manually.")
            sys.exit(1)
    print(f"  gamma: {gamma}")

    all_rows = []

    for step in steps:
        print(f"\n--- step {step} ---")
        t0 = time.time()

        ckpt = load_checkpoint(args.run_dir, step)
        replay = load_replay_buffer(args.run_dir, step)
        returns = compute_returns(replay, gamma)
        observations, valid_indices = _stack_replay_frames(replay)

        q_all = extract_q_values(
            ckpt, observations, batch_size=args.batch_size, seed=args.seed,
        )

        actions_at_valid = replay.actions[valid_indices]
        q_taken = q_all[np.arange(len(q_all)), actions_at_valid]
        returns_at_valid = returns[valid_indices]
        mask = np.isfinite(returns_at_valid)
        q_matched = q_taken[mask]
        g_matched = returns_at_valid[mask]
        n_matched = len(q_matched)

        if n_matched < 2:
            print(f"  SKIPPED: only {n_matched} matched pairs")
            continue

        spearman_r, spearman_p = stats.spearmanr(q_matched, g_matched)
        signed_error = q_matched - g_matched
        mean_signed_error = float(signed_error.mean())
        rmse = float(np.sqrt((signed_error ** 2).mean()))

        elapsed = time.time() - t0
        print(f"  r={spearman_r:.4f}  err={mean_signed_error:+.3f}  "
              f"Q={q_matched.mean():.3f}  G={g_matched.mean():.3f}  ({elapsed:.1f}s)")

        all_rows.append({
            "step": step,
            "spearman_r": round(spearman_r, 6),
            "spearman_p": spearman_p,
            "mean_signed_error": round(mean_signed_error, 6),
            "rmse": round(rmse, 6),
            "q_mean": round(float(q_matched.mean()), 6),
            "g_mean": round(float(g_matched.mean()), 6),
            "matched_pairs": n_matched,
        })

        del ckpt, replay, returns, observations, q_all, q_matched, g_matched
        gc.collect()

    # -- Save CSV ------------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        df = pd_.DataFrame(all_rows)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output} ({len(df)} rows)")
    else:
        print(f"\n{len(all_rows)} rows computed (use --output to save)")


if __name__ == "__main__":
    main()
