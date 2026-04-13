#!/usr/bin/env python3
"""Reward probing script (M10).

Probes whether encoder representations linearly predict reward
occurrence. Uses a binarized reward target (reward > 0 vs
reward == 0) and the shared linear probe trainer.

Supports three observation sources:
- greedy: epsilon-greedy policy using loaded checkpoint
- random: uniform random policy
- replay: replay buffer transitions from checkpoint

Usage:
    # Greedy policy on CrazyClimber
    python scripts/run_reward_probing.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step 10000 --game CrazyClimber --source greedy --num-steps 5000

    # From replay buffer (no environment needed)
    python scripts/run_reward_probing.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step 10000 --source replay

    # Save results to JSON
    python scripts/run_reward_probing.py \\
        --run-dir ... --step 10000 --source replay \\
        --output output/probing/reward_results.json
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _resolve_steps(args_steps, run_dir):
    """Resolve step arguments to a sorted list of ints."""
    from src.analysis.checkpoint import discover_checkpoints

    if args_steps == ["all"]:
        steps = discover_checkpoints(run_dir)
        if not steps:
            raise ValueError(f"No checkpoints found in {run_dir}")
        return steps
    return sorted(int(s) for s in args_steps)


def _load_replay_observations(run_dir, step):
    """Load replay buffer and stack into 4-frame HWC observations.

    Returns (observations, rewards) where observations is
    (N, 84, 84, 4) uint8 and rewards is (N,) float32.
    """
    from src.analysis.replay_buffer import load_replay_buffer

    replay = load_replay_buffer(run_dir, step)
    frames = replay.observations
    terms = replay.terminals
    obs_list, rew_list = [], []
    for i in range(len(frames) - 3):
        if not any(terms[i:i + 3]):
            stack = np.stack(frames[i:i + 4], axis=-1)
            obs_list.append(stack)
            rew_list.append(replay.rewards[i + 3])
    return np.array(obs_list, dtype=np.uint8), np.array(rew_list, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Reward probing on encoder representations (M10)"
    )
    parser.add_argument("--run-dir", required=True,
                        help="Path to the training run directory")
    parser.add_argument("--steps", nargs="+", required=True,
                        help="Checkpoint steps (e.g., 10000 50000 100000) or 'all'")
    parser.add_argument("--source", choices=["replay"],
                        default="replay",
                        help="Observation source (default: replay)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for representation extraction")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save CSV results")
    args = parser.parse_args()

    import gc
    import pandas as pd
    from src.analysis.checkpoint import load_checkpoint
    from src.analysis.representations import extract_representations
    from src.analysis.probing import train_probe

    steps = _resolve_steps(args.steps, args.run_dir)
    print(f"Reward probing: {args.run_dir}")
    print(f"  checkpoints: {steps}")

    all_rows = []

    for step in steps:
        print(f"\n--- step {step} ---")
        t0 = time.time()

        ckpt = load_checkpoint(args.run_dir, step)
        observations, rewards = _load_replay_observations(args.run_dir, step)

        labels = (rewards > 0).astype(np.int32)
        n_pos = int(labels.sum())
        n = len(labels)
        print(f"  {n} obs, reward>0: {n_pos} ({100*n_pos/n:.1f}%)")

        reps = extract_representations(
            ckpt, observations, batch_size=args.batch_size, seed=args.seed,
        )

        result = train_probe(
            reps, labels, variable_name="reward_binary",
            entropy_threshold=0.0,
        )

        elapsed = time.time() - t0
        print(f"  F1_test={result.f1_test:.4f}  F1_train={result.f1_train:.4f}  ({elapsed:.1f}s)")

        all_rows.append({
            "step": step,
            "f1_test": round(result.f1_test, 6),
            "f1_train": round(result.f1_train, 6),
            "accuracy_test": round(result.accuracy_test, 6),
            "n_observations": n,
            "reward_positive_pct": round(n_pos / n, 6),
            "normalized_entropy": round(result.normalized_entropy, 6),
        })

        # Free memory before next checkpoint
        del ckpt, observations, rewards, labels, reps, result
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
