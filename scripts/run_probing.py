#!/usr/bin/env python3
"""End-to-end AtariARI probing script (M9).

Loads a JAX checkpoint, collects observations with AtariARI RAM
labels, extracts FeatureLayer representations, trains a linear
probe per variable, and reports macro-averaged F1 scores.

Usage:
    # Greedy policy on an annotated game
    python scripts/run_probing.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step 10000 --game Boxing --source greedy --num-steps 5000

    # Random policy baseline
    python scripts/run_probing.py \\
        --run-dir experiments/dqn_atari/runs/spr_crazy_climber_seed13 \\
        --step 10000 --game Pong --source random --num-steps 5000

    # Save results to JSON
    python scripts/run_probing.py \\
        --run-dir ... --step 10000 --game Boxing --source greedy \\
        --output output/probing/results.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description="AtariARI linear probing (M9)"
    )
    parser.add_argument("--run-dir", required=True,
                        help="Path to the training run directory")
    parser.add_argument("--step", type=int, required=True,
                        help="Checkpoint step to load")
    parser.add_argument("--game", required=True,
                        help="Game name (CamelCase, e.g., Boxing)")
    parser.add_argument("--source", choices=["greedy", "random"],
                        default="greedy",
                        help="Observation collection policy (default: greedy)")
    parser.add_argument("--num-steps", type=int, default=10000,
                        help="Number of environment steps (default: 10000)")
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

    # -- Step 2: Collect observations with labels ----------------------------
    print(f"Collecting {args.num_steps} steps on {args.game} "
          f"({args.source} policy, seed={args.seed})...")
    t0 = time.time()

    if args.source == "greedy":
        from src.analysis.observations import collect_greedy
        data = collect_greedy(
            ckpt, game=args.game, num_steps=args.num_steps,
            seed=args.seed, noop_max=30, collect_labels=True,
        )
    else:
        from src.analysis.observations import collect_random
        data = collect_random(
            game=args.game, num_actions=ckpt.num_actions,
            num_steps=args.num_steps, seed=args.seed,
            noop_max=30, collect_labels=True,
        )

    elapsed = time.time() - t0
    print(f"  collected {len(data.observations)} observations, "
          f"{len(data.episode_returns)} episodes ({elapsed:.1f}s)")
    print(f"  label variables: {sorted(data.labels.keys())}")

    # -- Step 3: Extract representations -------------------------------------
    print(f"Extracting representations (batch_size={args.batch_size})...")
    t0 = time.time()
    from src.analysis.representations import extract_representations
    reps = extract_representations(
        ckpt, data.observations, batch_size=args.batch_size, seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"  shape: {reps.shape} ({elapsed:.1f}s)")

    # -- Step 4: Train probes ------------------------------------------------
    print("Training linear probes...")
    t0 = time.time()
    from src.analysis.probing import train_probes_multi
    results = train_probes_multi(reps, data.labels)
    elapsed = time.time() - t0
    print(f"  {len(results)} variables ({elapsed:.1f}s)")

    # -- Step 5: Print results table -----------------------------------------
    print()
    print(f"{'Variable':<20} {'F1 Test':>8} {'F1 Train':>9} "
          f"{'Accuracy':>9} {'Classes':>8} {'Entropy':>8} {'Status':>10}")
    print("-" * 82)

    for r in results:
        if r.skipped:
            print(f"{r.variable:<20} {'':>8} {'':>9} {'':>9} "
                  f"{r.n_classes:>8} {r.normalized_entropy:>8.3f} "
                  f"{'SKIP':>10}")
        else:
            print(f"{r.variable:<20} {r.f1_test:>8.4f} {r.f1_train:>9.4f} "
                  f"{r.accuracy_test:>9.4f} {r.n_classes:>8} "
                  f"{r.normalized_entropy:>8.3f} {'OK':>10}")

    active = [r for r in results if not r.skipped]
    if active:
        mean_f1 = sum(r.f1_test for r in active) / len(active)
        print("-" * 82)
        print(f"{'Mean F1 (active)':<20} {mean_f1:>8.4f}")
    print()

    # -- Step 6: Save JSON ---------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output = {
            "run_dir": args.run_dir,
            "step": args.step,
            "game": args.game,
            "source": args.source,
            "num_steps": args.num_steps,
            "seed": args.seed,
            "encoder_type": ckpt.encoder_type,
            "hidden_dim": ckpt.hidden_dim,
            "results": [
                {
                    "variable": r.variable,
                    "f1_test": r.f1_test,
                    "f1_train": r.f1_train,
                    "accuracy_test": r.accuracy_test,
                    "n_classes": r.n_classes,
                    "normalized_entropy": r.normalized_entropy,
                    "skipped": r.skipped,
                    "skip_reason": r.skip_reason,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
