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


def _stack_transitions(replay, transitions):
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


def main():
    parser = argparse.ArgumentParser(
        description="Inverse dynamics probing on encoder representations (M14)"
    )
    parser.add_argument("--run-dir", required=True,
                        help="Path to the training run directory")
    parser.add_argument("--step", type=int, required=True,
                        help="Checkpoint step to load")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for representation extraction")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results (optional)")
    args = parser.parse_args()

    # -- Step 1: Load checkpoint ---------------------------------------------
    print(f"Loading checkpoint: {args.run_dir} step {args.step}")
    from src.analysis.checkpoint import load_checkpoint
    ckpt = load_checkpoint(args.run_dir, args.step)
    print(f"  encoder: {ckpt.encoder_type}, hidden_dim: {ckpt.hidden_dim}, "
          f"num_actions: {ckpt.num_actions}")

    # -- Step 2: Load replay buffer transitions ------------------------------
    print("Loading replay buffer...")
    from src.analysis.replay_buffer import load_replay_buffer
    replay = load_replay_buffer(args.run_dir, args.step)
    print(f"  {replay.add_count} entries, "
          f"{replay.terminals.sum()} episode boundaries")

    print("Building stacked transition pairs...")
    t0 = time.time()
    from src.analysis.replay_buffer import get_valid_transitions
    obs_t, obs_next, actions = _stack_transitions(replay, get_valid_transitions(replay))
    elapsed = time.time() - t0
    print(f"  {len(obs_t)} valid transition pairs ({elapsed:.1f}s)")

    chance = 1.0 / ckpt.num_actions
    print(f"  chance baseline: {chance:.4f} (1/{ckpt.num_actions})")

    # -- Step 3: Extract representations for both frames ---------------------
    print(f"Extracting representations (batch_size={args.batch_size})...")
    t0 = time.time()
    from src.analysis.representations import extract_representations
    reps_t = extract_representations(
        ckpt, obs_t, batch_size=args.batch_size, seed=args.seed,
    )
    reps_next = extract_representations(
        ckpt, obs_next, batch_size=args.batch_size, seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"  phi(t): {reps_t.shape}, phi(t+1): {reps_next.shape} ({elapsed:.1f}s)")

    # -- Step 4: Concatenate and train probe ---------------------------------
    features = np.concatenate([reps_t, reps_next], axis=1)
    print(f"Concatenated features: {features.shape}")

    print("Training inverse dynamics probe...")
    t0 = time.time()
    from src.analysis.probing import train_probe
    result = train_probe(
        features, actions, variable_name="action",
        entropy_threshold=0.0,  # always train
    )
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.1f}s)")

    # -- Step 5: Print results -----------------------------------------------
    print()
    if result.skipped:
        print(f"SKIPPED: {result.skip_reason}")
    else:
        print(f"Inverse Dynamics Probe Results")
        print(f"  F1 test (macro):  {result.f1_test:.4f}")
        print(f"  F1 train (macro): {result.f1_train:.4f}")
        print(f"  Accuracy test:    {result.accuracy_test:.4f}")
        print(f"  Chance baseline:  {chance:.4f}")
        print(f"  Above chance:     {result.accuracy_test > chance}")
        print(f"  Classes:          {result.n_classes}")
    print()

    # -- Step 6: Save JSON ---------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output = {
            "run_dir": args.run_dir,
            "step": args.step,
            "num_transitions": len(obs_t),
            "num_actions": ckpt.num_actions,
            "chance_baseline": chance,
            "encoder_type": ckpt.encoder_type,
            "hidden_dim": ckpt.hidden_dim,
            "feature_dim": features.shape[1],
            "f1_test": result.f1_test,
            "f1_train": result.f1_train,
            "accuracy_test": result.accuracy_test,
            "n_classes": result.n_classes,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
