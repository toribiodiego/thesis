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


def _get_observations(game, num_steps, ckpt, seed, source):
    """Collect observations for representation extraction."""
    if source == "replay":
        from src.analysis.replay_buffer import load_replay_buffer
        replay = load_replay_buffer(ckpt._run_dir, ckpt._step)
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
    parser.add_argument("--step-a", type=int, required=True,
                        help="Checkpoint step for A")
    parser.add_argument("--step-b", type=int, default=None,
                        help="Checkpoint step for B (cross-checkpoint mode)")
    parser.add_argument("--game", type=str, required=True,
                        help="Game name for observation collection")
    parser.add_argument("--source", choices=["greedy", "random", "replay"],
                        default="random",
                        help="Observation source (default: random)")
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="Steps for observation collection (default: 1000)")
    parser.add_argument("--mode", required=True,
                        choices=["cross-checkpoint", "cross-condition", "online-target"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Validate mode-specific args
    if args.mode == "cross-checkpoint":
        if args.step_b is None:
            parser.error("--step-b required for cross-checkpoint mode")
        args.run_dir_b = args.run_dir_a
    elif args.mode == "cross-condition":
        if args.run_dir_b is None:
            parser.error("--run-dir-b required for cross-condition mode")
        if args.step_b is None:
            args.step_b = args.step_a
    elif args.mode == "online-target":
        args.run_dir_b = args.run_dir_a
        args.step_b = args.step_a

    from src.analysis.checkpoint import load_checkpoint
    from src.analysis.representations import (
        extract_representations,
        extract_representations_target,
    )

    # -- Load checkpoint(s) --------------------------------------------------
    print(f"Loading checkpoint A: {args.run_dir_a} step {args.step_a}")
    ckpt_a = load_checkpoint(args.run_dir_a, args.step_a)
    print(f"  encoder: {ckpt_a.encoder_type}, hidden_dim: {ckpt_a.hidden_dim}")

    if args.mode != "online-target":
        print(f"Loading checkpoint B: {args.run_dir_b} step {args.step_b}")
        ckpt_b = load_checkpoint(args.run_dir_b, args.step_b)
    else:
        ckpt_b = ckpt_a
        if ckpt_a.target_params is None:
            print("WARNING: No target params in checkpoint. "
                  "Online-target CKA will be 1.0 (same params).")

    # -- Collect observations (shared across both representations) -----------
    print(f"Collecting observations ({args.source}, {args.num_steps} steps)...")
    t0 = time.time()
    observations = _get_observations(
        args.game, args.num_steps, ckpt_a, args.seed, args.source,
    )
    print(f"  {len(observations)} observations ({time.time() - t0:.1f}s)")

    # -- Extract representations ---------------------------------------------
    print("Extracting representations...")
    t0 = time.time()

    reps_a = extract_representations(
        ckpt_a, observations, batch_size=args.batch_size, seed=args.seed,
    )
    print(f"  A (online): {reps_a.shape}")

    if args.mode == "online-target":
        reps_b = extract_representations_target(
            ckpt_a, observations, batch_size=args.batch_size, seed=args.seed,
        )
        if reps_b is None:
            # No target params -- use online (CKA will be 1.0)
            reps_b = reps_a
            b_label = "online (no target params)"
        else:
            b_label = "target"
        print(f"  B ({b_label}): {reps_b.shape}")
    else:
        reps_b = extract_representations(
            ckpt_b, observations, batch_size=args.batch_size, seed=args.seed,
        )
        print(f"  B (online): {reps_b.shape}")

    print(f"  ({time.time() - t0:.1f}s)")

    # -- Compute CKA ---------------------------------------------------------
    cka = linear_cka(reps_a, reps_b)

    print()
    print(f"Linear CKA: {cka:.6f}")
    print(f"  Mode: {args.mode}")
    if args.mode == "cross-checkpoint":
        print(f"  A: step {args.step_a}, B: step {args.step_b}")
    elif args.mode == "cross-condition":
        print(f"  A: {os.path.basename(args.run_dir_a)}")
        print(f"  B: {os.path.basename(args.run_dir_b)}")
    elif args.mode == "online-target":
        print(f"  Online vs target at step {args.step_a}")
    print()

    # -- Save JSON -----------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output = {
            "mode": args.mode,
            "run_dir_a": args.run_dir_a,
            "run_dir_b": args.run_dir_b,
            "step_a": args.step_a,
            "step_b": args.step_b,
            "game": args.game,
            "source": args.source,
            "num_observations": len(observations),
            "encoder_type": ckpt_a.encoder_type,
            "hidden_dim": ckpt_a.hidden_dim,
            "cka": cka,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
