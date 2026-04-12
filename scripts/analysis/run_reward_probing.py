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

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    parser = argparse.ArgumentParser(
        description="Reward probing on encoder representations (M10)"
    )
    parser.add_argument("--run-dir", required=True,
                        help="Path to the training run directory")
    parser.add_argument("--step", type=int, required=True,
                        help="Checkpoint step to load")
    parser.add_argument("--game", type=str, default=None,
                        help="Game name (required for greedy/random source)")
    parser.add_argument("--source", choices=["greedy", "random", "replay"],
                        default="greedy",
                        help="Observation source (default: greedy)")
    parser.add_argument("--num-steps", type=int, default=10000,
                        help="Steps for greedy/random collection (default: 10000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for representation extraction")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results (optional)")
    args = parser.parse_args()

    if args.source in ("greedy", "random") and args.game is None:
        parser.error("--game is required for greedy and random sources")

    # -- Step 1: Load checkpoint ---------------------------------------------
    print(f"Loading checkpoint: {args.run_dir} step {args.step}")
    from src.analysis.checkpoint import load_checkpoint
    ckpt = load_checkpoint(args.run_dir, args.step)
    print(f"  encoder: {ckpt.encoder_type}, hidden_dim: {ckpt.hidden_dim}, "
          f"num_actions: {ckpt.num_actions}")

    # -- Step 2: Get observations and rewards --------------------------------
    print(f"Loading observations (source={args.source})...")
    t0 = time.time()

    if args.source == "greedy":
        from src.analysis.observations import collect_greedy
        data = collect_greedy(
            ckpt, game=args.game, num_steps=args.num_steps,
            seed=args.seed, noop_max=30,
        )
        observations = data.observations
        rewards = data.rewards

    elif args.source == "random":
        from src.analysis.observations import collect_random
        data = collect_random(
            game=args.game, num_actions=ckpt.num_actions,
            num_steps=args.num_steps, seed=args.seed, noop_max=30,
        )
        observations = data.observations
        rewards = data.rewards

    else:  # replay
        from src.analysis.replay_buffer import load_replay_buffer
        replay = load_replay_buffer(args.run_dir, args.step)
        # Replay stores single frames; stack 4 consecutive non-terminal
        # frames into HWC format for representation extraction
        frames = replay.observations
        terms = replay.terminals
        obs_list, rew_list = [], []
        for i in range(len(frames) - 3):
            if not any(terms[i:i + 3]):
                stack = np.stack(frames[i:i + 4], axis=-1)  # (84, 84, 4)
                obs_list.append(stack)
                rew_list.append(replay.rewards[i + 3])
        observations = np.array(obs_list, dtype=np.uint8)
        rewards = np.array(rew_list, dtype=np.float32)

    elapsed = time.time() - t0
    n = len(observations)
    print(f"  {n} observations ({elapsed:.1f}s)")

    # -- Step 3: Binarize rewards --------------------------------------------
    labels = (rewards > 0).astype(np.int32)
    n_pos = labels.sum()
    n_neg = n - n_pos
    print(f"  reward>0: {n_pos} ({100 * n_pos / n:.1f}%), "
          f"reward==0: {n_neg} ({100 * n_neg / n:.1f}%)")

    # -- Step 4: Extract representations -------------------------------------
    print(f"Extracting representations (batch_size={args.batch_size})...")
    t0 = time.time()
    from src.analysis.representations import extract_representations
    reps = extract_representations(
        ckpt, observations, batch_size=args.batch_size, seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"  shape: {reps.shape} ({elapsed:.1f}s)")

    # -- Step 5: Train probe -------------------------------------------------
    print("Training reward probe...")
    t0 = time.time()
    from src.analysis.probing import train_probe
    result = train_probe(
        reps, labels, variable_name="reward_binary",
        entropy_threshold=0.0,  # always train, even if imbalanced
    )
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.1f}s)")

    # -- Step 6: Print results -----------------------------------------------
    print()
    if result.skipped:
        print(f"SKIPPED: {result.skip_reason}")
    else:
        print(f"Reward Probe Results (binary: reward > 0)")
        print(f"  F1 test (macro):  {result.f1_test:.4f}")
        print(f"  F1 train (macro): {result.f1_train:.4f}")
        print(f"  Accuracy test:    {result.accuracy_test:.4f}")
        print(f"  Classes:          {result.n_classes}")
        print(f"  Norm entropy:     {result.normalized_entropy:.3f}")
    print()

    # -- Step 7: Save JSON ---------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output = {
            "run_dir": args.run_dir,
            "step": args.step,
            "game": args.game,
            "source": args.source,
            "num_observations": n,
            "reward_positive": int(n_pos),
            "reward_zero": int(n_neg),
            "encoder_type": ckpt.encoder_type,
            "hidden_dim": ckpt.hidden_dim,
            "f1_test": result.f1_test,
            "f1_train": result.f1_train,
            "accuracy_test": result.accuracy_test,
            "n_classes": result.n_classes,
            "normalized_entropy": result.normalized_entropy,
            "skipped": result.skipped,
            "skip_reason": result.skip_reason,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
