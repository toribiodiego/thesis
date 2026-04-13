#!/usr/bin/env python3
"""Inverse dynamics probing script (M14).

Probes whether encoder representations capture action-relevant
transition features by predicting the action taken between
consecutive states. Concatenates [phi(obs_t); phi(obs_{t+1})]
and trains a linear probe on the action label.

Uses replay buffer transitions to get (obs_t, action_t, obs_{t+1})
triples with terminal filtering.

Usage:
    python scripts/run_inverse_dynamics.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step 10000

    # Save results
    python scripts/run_inverse_dynamics.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step 10000 --output output/probing/inverse_dynamics.json
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _stack_transitions(replay):
    """Build 4-frame stacked observations for obs_t and obs_{t+1}.

    The replay buffer stores single (84, 84) frames. For each valid
    transition index, stack frames [i-3, i-2, i-1, i] into (84, 84, 4)
    HWC format.

    Returns:
        obs_t: (M, 84, 84, 4) uint8
        obs_next: (M, 84, 84, 4) uint8
        actions: (M,) int32
    """
    frames = replay.observations
    terms = replay.terminals
    n = len(frames)

    obs_t_list, obs_next_list, act_list = [], [], []

    for i in range(3, n - 1):
        # Need frames [i-3..i] for obs_t and [i-2..i+1] for obs_{t+1}
        # All 5 frames [i-3..i+1] must be within same episode
        if any(terms[i - 3 : i + 1]):
            continue
        obs_t = np.stack(frames[i - 3 : i + 1], axis=-1)      # (84, 84, 4)
        obs_next = np.stack(frames[i - 2 : i + 2], axis=-1)    # (84, 84, 4)
        obs_t_list.append(obs_t)
        obs_next_list.append(obs_next)
        act_list.append(replay.actions[i])

    return (
        np.array(obs_t_list, dtype=np.uint8),
        np.array(obs_next_list, dtype=np.uint8),
        np.array(act_list, dtype=np.int32),
    )


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
        description="Inverse dynamics probing on encoder representations (M14)"
    )
    parser.add_argument("--run-dir", required=True,
                        help="Path to the training run directory")
    parser.add_argument("--steps", nargs="+", required=True,
                        help="Checkpoint steps (e.g., 10000 50000 100000) or 'all'")
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
    from src.analysis.replay_buffer import load_replay_buffer
    from src.analysis.representations import extract_representations
    from src.analysis.probing import train_probe

    steps = _resolve_steps(args.steps, args.run_dir)
    print(f"Inverse dynamics: {args.run_dir}")
    print(f"  checkpoints: {steps}")

    all_rows = []

    for step in steps:
        print(f"\n--- step {step} ---")
        t0 = time.time()

        ckpt = load_checkpoint(args.run_dir, step)
        replay = load_replay_buffer(args.run_dir, step)
        obs_t, obs_next, actions = _stack_transitions(replay)
        chance = 1.0 / ckpt.num_actions
        print(f"  {len(obs_t)} transitions, {ckpt.num_actions} actions")

        reps_t = extract_representations(
            ckpt, obs_t, batch_size=args.batch_size, seed=args.seed,
        )
        reps_next = extract_representations(
            ckpt, obs_next, batch_size=args.batch_size, seed=args.seed,
        )
        features = np.concatenate([reps_t, reps_next], axis=1)

        result = train_probe(
            features, actions, variable_name="action",
            entropy_threshold=0.0,
        )

        elapsed = time.time() - t0
        print(f"  F1_test={result.f1_test:.4f}  F1_train={result.f1_train:.4f}  ({elapsed:.1f}s)")

        all_rows.append({
            "step": step,
            "f1_test": round(result.f1_test, 6),
            "f1_train": round(result.f1_train, 6),
            "accuracy_test": round(result.accuracy_test, 6),
            "chance_baseline": round(chance, 6),
            "n_classes": result.n_classes,
            "n_transitions": len(obs_t),
        })

        del ckpt, replay, obs_t, obs_next, actions, reps_t, reps_next, features, result
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
