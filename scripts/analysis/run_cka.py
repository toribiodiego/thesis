#!/usr/bin/env python3
"""CKA similarity analysis script (M13).

Computes linear Centered Kernel Alignment between two sets of
encoder representations. Three modes:

- cross-checkpoint: same run, two different steps
- cross-condition: two different runs, same step
- online-target: single checkpoint, online vs target params

Uses the Frobenius norm formulation of linear CKA:
    CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
where X, Y are centered representation matrices (N, D).

Usage:
    # Cross-checkpoint (same run, steps 5000 vs 10000)
    python scripts/analysis/run_cka.py \\
        --run-dir-a experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step-a 5000 --step-b 10000 \\
        --game CrazyClimber --mode cross-checkpoint --num-steps 500

    # Cross-condition (BBF vs BBFc at same step)
    python scripts/analysis/run_cka.py \\
        --run-dir-a experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --run-dir-b experiments/dqn_atari/runs/bbfc_crazy_climber_seed13 \\
        --step-a 10000 --mode cross-condition \\
        --game CrazyClimber --num-steps 500

    # Online vs target (single checkpoint)
    python scripts/analysis/run_cka.py \\
        --run-dir-a experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step-a 10000 --mode online-target \\
        --game CrazyClimber --num-steps 500
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def linear_cka(X, Y):
    """Compute linear CKA between two representation matrices.

    Uses the Frobenius norm formulation (Kornblith et al. 2019):
        CKA(X, Y) = ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)

    Both X and Y are centered before computation.

    Args:
        X: (N, D1) float32 representation matrix.
        Y: (N, D2) float32 representation matrix.

    Returns:
        CKA similarity in [0, 1].
    """
    # Center features
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Frobenius norm formulation
    YtX = Y.T @ X
    XtX = X.T @ X
    YtY = Y.T @ Y

    numerator = np.linalg.norm(YtX, "fro") ** 2
    denominator = np.linalg.norm(XtX, "fro") * np.linalg.norm(YtY, "fro")

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _get_observations(game, num_steps, ckpt, seed, source, run_dir=None, step=None):
    """Collect observations for representation extraction."""
    if source == "replay":
        from src.analysis.replay_buffer import load_replay_buffer
        replay = load_replay_buffer(run_dir, step)
        frames, terms = replay.observations, replay.terminals
        obs_list = []
        for i in range(3, len(frames)):
            if not any(terms[i - 3:i]):
                obs_list.append(np.stack(frames[i - 3:i + 1], axis=-1))
        return np.array(obs_list, dtype=np.uint8)
    elif source == "greedy":
        from src.analysis.observations import collect_greedy
        return collect_greedy(
            ckpt, game=game, num_steps=num_steps, seed=seed, noop_max=30,
        ).observations
    else:
        from src.analysis.observations import collect_random
        return collect_random(
            game=game, num_actions=ckpt.num_actions,
            num_steps=num_steps, seed=seed, noop_max=30,
        ).observations


def main():
    parser = argparse.ArgumentParser(
        description="Linear CKA similarity between encoder representations (M13)"
    )
    parser.add_argument("--run-dir-a", required=True,
                        help="Run directory for checkpoint A")
    parser.add_argument("--run-dir-b", type=str, default=None,
                        help="Run directory for checkpoint B (cross-condition mode)")
    parser.add_argument("--steps", nargs="+", required=True,
                        help="Checkpoint steps (e.g., 10000 50000 100000) or 'all'")
    parser.add_argument("--game", type=str, default=None,
                        help="Game name (required for greedy/random source)")
    parser.add_argument("--source", choices=["greedy", "random", "replay"],
                        default="replay",
                        help="Observation source (default: replay)")
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="Steps for greedy/random collection (default: 1000)")
    parser.add_argument("--mode", required=True,
                        choices=["cross-condition", "online-target"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save CSV results")
    args = parser.parse_args()

    if args.mode == "cross-condition" and args.run_dir_b is None:
        parser.error("--run-dir-b required for cross-condition mode")

    import gc
    import pandas as pd
    from src.analysis.checkpoint import discover_checkpoints, load_checkpoint
    from src.analysis.representations import (
        extract_representations,
        extract_representations_target,
    )

    if args.steps == ["all"]:
        steps = discover_checkpoints(args.run_dir_a)
        if not steps:
            raise ValueError(f"No checkpoints found in {args.run_dir_a}")
    else:
        steps = sorted(int(s) for s in args.steps)

    print(f"CKA ({args.mode}): {args.run_dir_a}")
    if args.mode == "cross-condition":
        print(f"  vs: {args.run_dir_b}")
    print(f"  checkpoints: {steps}")

    all_rows = []

    for step in steps:
        print(f"\n--- step {step} ---")
        t0 = time.time()

        ckpt_a = load_checkpoint(args.run_dir_a, step)
        observations = _get_observations(
            args.game, args.num_steps, ckpt_a, args.seed, args.source,
            run_dir=args.run_dir_a, step=step,
        )

        reps_a = extract_representations(
            ckpt_a, observations, batch_size=args.batch_size, seed=args.seed,
        )

        if args.mode == "online-target":
            reps_b = extract_representations_target(
                ckpt_a, observations, batch_size=args.batch_size, seed=args.seed,
            )
            if reps_b is None:
                reps_b = reps_a
        else:
            ckpt_b = load_checkpoint(args.run_dir_b, step)
            reps_b = extract_representations(
                ckpt_b, observations, batch_size=args.batch_size, seed=args.seed,
            )

        cka = linear_cka(reps_a, reps_b)
        elapsed = time.time() - t0
        print(f"  CKA={cka:.6f}  ({elapsed:.1f}s)")

        all_rows.append({
            "step": step,
            "mode": args.mode,
            "cka": round(cka, 6),
            "n_observations": len(observations),
        })

        del ckpt_a, observations, reps_a, reps_b
        gc.collect()

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
