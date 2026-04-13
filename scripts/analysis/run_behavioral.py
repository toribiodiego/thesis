#!/usr/bin/env python3
"""Behavioral analysis from replay buffer (experimental).

Computes per-checkpoint behavioral metrics from the replay buffer
without any forward passes or probe training. Lightweight and fast.

Metrics:
- action_entropy: Shannon entropy of action distribution (bits)
- action_max_pct: fraction of most-used action
- mean_return: mean episode return (from complete episodes)
- std_return: std of episode returns
- n_episodes: number of complete episodes in buffer
- reward_rate: fraction of steps with nonzero reward

Usage:
    python scripts/analysis/run_behavioral.py \\
        --run-dir experiments/dqn_atari/runs/spr_boxing_seed13 \\
        --steps all --output analysis/behavioral.csv
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def _resolve_steps(args_steps, run_dir):
    """Resolve step arguments to a sorted list of ints."""
    from src.analysis.checkpoint import discover_checkpoints

    if args_steps == ["all"]:
        steps = discover_checkpoints(run_dir)
        if not steps:
            raise ValueError(f"No checkpoints found in {run_dir}")
        return steps
    return sorted(int(s) for s in args_steps)


def _action_entropy(actions, num_actions):
    """Shannon entropy of action distribution in bits."""
    counts = np.bincount(actions, minlength=num_actions)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def _episode_stats(rewards, terminals):
    """Compute per-episode returns and lengths from replay buffer."""
    returns = []
    lengths = []
    ep_return = 0.0
    ep_length = 0
    for i in range(len(rewards)):
        ep_return += rewards[i]
        ep_length += 1
        if terminals[i]:
            returns.append(ep_return)
            lengths.append(ep_length)
            ep_return = 0.0
            ep_length = 0
    return (np.array(returns, dtype=np.float32),
            np.array(lengths, dtype=np.int32))


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral analysis from replay buffer (experimental)"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--steps", nargs="+", required=True,
                        help="Checkpoint steps (e.g., 10000 50000 100000) or 'all'")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save CSV files (e.g., run_dir/analysis/behavioral/)")
    args = parser.parse_args()

    import pandas as pd
    from src.analysis.replay_buffer import load_replay_buffer

    steps = _resolve_steps(args.steps, args.run_dir)
    print(f"Behavioral analysis: {args.run_dir}")
    print(f"  checkpoints: {steps}")

    action_rows = []
    return_rows = []
    length_rows = []

    for step in steps:
        t0 = time.time()
        replay = load_replay_buffer(args.run_dir, step)
        actions = replay.actions
        rewards = replay.rewards
        terminals = replay.terminals

        # Infer num_actions from max action value
        num_actions = int(actions.max()) + 1

        # Action distribution metrics
        entropy = _action_entropy(actions, num_actions)
        counts = np.bincount(actions, minlength=num_actions)
        max_action_pct = float(counts.max()) / len(actions)

        # Episode return and length metrics
        ep_returns, ep_lengths = _episode_stats(rewards, terminals)
        n_episodes = len(ep_returns)
        mean_return = float(ep_returns.mean()) if n_episodes > 0 else 0.0
        std_return = float(ep_returns.std()) if n_episodes > 1 else 0.0
        mean_length = float(ep_lengths.mean()) if n_episodes > 0 else 0.0
        std_length = float(ep_lengths.std()) if n_episodes > 1 else 0.0

        # Reward rate
        reward_rate = float((rewards != 0).sum()) / len(rewards)

        elapsed = time.time() - t0
        print(f"  step {step}: entropy={entropy:.3f}  "
              f"mean_ret={mean_return:.1f}  mean_len={mean_length:.0f}  "
              f"episodes={n_episodes}  ({elapsed:.1f}s)")

        action_rows.append({
            "step": step,
            "action_entropy": round(entropy, 6),
            "action_max_pct": round(max_action_pct, 6),
            "reward_rate": round(reward_rate, 6),
            "n_transitions": replay.add_count,
        })

        return_rows.append({
            "step": step,
            "mean_return": round(mean_return, 4),
            "std_return": round(std_return, 4),
            "n_episodes": n_episodes,
        })

        length_rows.append({
            "step": step,
            "mean_episode_length": round(mean_length, 2),
            "std_episode_length": round(std_length, 2),
            "n_episodes": n_episodes,
        })

        del replay

    # -- Save CSVs -----------------------------------------------------------
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        for name, rows in [
            ("action_distribution", action_rows),
            ("episode_returns", return_rows),
            ("episode_lengths", length_rows),
        ]:
            path = os.path.join(args.output_dir, f"{name}.csv")
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"  {path} ({len(rows)} rows)")

        print(f"\nResults saved to {args.output_dir}")
    else:
        print(f"\n{len(action_rows)} rows computed (use --output-dir to save)")


if __name__ == "__main__":
    main()
