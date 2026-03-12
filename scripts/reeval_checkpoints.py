#!/usr/bin/env python3
"""Re-run evaluations on saved checkpoints that are missing eval data.

Loads each checkpoint, creates the matching model and environment,
runs 30 evaluation episodes, and appends results to evaluations.csv.

Usage:
    python scripts/reeval_checkpoints.py                  # all runs with checkpoints
    python scripts/reeval_checkpoints.py run_name_1 ...   # specific runs only
"""

import argparse
import csv
import os
import re
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.atari_wrappers import make_atari_env
from src.models.dqn import DQN
from src.models.rainbow import RainbowDQN
from src.training.evaluation import evaluate

RUNS_DIR = "experiments/dqn_atari/runs"

EVAL_EPISODES = 30
EVAL_EPSILON = 0.05


def discover_runs(runs_dir):
    """Auto-discover run directories that have checkpoints and a config."""
    run_names = []
    if not os.path.isdir(runs_dir):
        return run_names
    for entry in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, entry)
        if not os.path.isdir(run_dir):
            continue
        config_path = os.path.join(run_dir, "config.yaml")
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        if os.path.isfile(config_path) and os.path.isdir(checkpoint_dir):
            run_names.append(entry)
    return run_names


def discover_checkpoint_steps(checkpoint_dir):
    """Scan checkpoint directory and return sorted list of step numbers."""
    steps = []
    if not os.path.isdir(checkpoint_dir):
        return steps
    for fname in os.listdir(checkpoint_dir):
        m = re.match(r"checkpoint_(\d+)\.pt$", fname)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def get_existing_eval_steps(run_dir):
    """Read evaluations.csv and return set of already-evaluated steps."""
    csv_path = os.path.join(run_dir, "eval", "evaluations.csv")
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    return {int(r["step"]) for r in rows}


def load_config(run_dir):
    """Load the run's config.yaml."""
    config_path = os.path.join(run_dir, "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def create_model(config, num_actions, device):
    """Create a model matching the run config (DQN or RainbowDQN)."""
    rainbow_cfg = config.get("rainbow", {})
    dropout = config.get("network", {}).get("dropout", 0.0)

    if rainbow_cfg.get("enabled", False):
        dist = rainbow_cfg.get("distributional", {})
        fc_hidden = config.get("network", {}).get("fc_hidden", 512)
        model = RainbowDQN(
            num_actions=num_actions,
            num_atoms=dist.get("num_atoms", 51),
            v_min=dist.get("v_min", -10.0),
            v_max=dist.get("v_max", 10.0),
            noisy=rainbow_cfg.get("noisy_nets", True),
            dueling=rainbow_cfg.get("dueling", True),
            dropout=dropout,
            fc_hidden=fc_hidden,
        )
    else:
        model = DQN(
            num_actions=num_actions,
            dropout=dropout,
        )

    return model.to(device)


def append_eval_row(run_dir, step, results, training_epsilon):
    """Append a row to evaluations.csv."""
    csv_path = os.path.join(run_dir, "eval", "evaluations.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "step", "mean_return", "median_return", "std_return",
                "min_return", "max_return", "episodes", "eval_epsilon",
                "training_epsilon",
            ])
        writer.writerow([
            step,
            results["mean_return"],
            results["median_return"],
            results["std_return"],
            results["min_return"],
            results["max_return"],
            results["num_episodes"],
            EVAL_EPSILON,
            training_epsilon,
        ])


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Re-evaluate checkpoints that are missing eval data.",
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run names to evaluate (default: auto-discover all runs with checkpoints)",
    )
    parser.add_argument(
        "--runs-dir",
        default=RUNS_DIR,
        help=f"Base directory containing run folders (default: {RUNS_DIR})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.runs:
        run_names = args.runs
    else:
        run_names = discover_runs(args.runs_dir)
        if not run_names:
            print(f"No runs with checkpoints found in {args.runs_dir}")
            return

    device = "cpu"
    print(f"Device: {device}")
    print(f"Evaluation episodes: {EVAL_EPISODES}, epsilon: {EVAL_EPSILON}")
    print(f"Runs to process: {len(run_names)}")
    print()

    for run_name in run_names:
        run_dir = os.path.join(args.runs_dir, run_name)
        if not os.path.exists(run_dir):
            print(f"SKIP {run_name}: directory not found")
            continue

        config = load_config(run_dir)
        env_id = config["environment"]["env_id"]
        existing_steps = get_existing_eval_steps(run_dir)

        # Find checkpoints that need evaluation
        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        all_steps = discover_checkpoint_steps(checkpoint_dir)
        missing_steps = [s for s in all_steps if s not in existing_steps]

        if not missing_steps:
            print(f"SKIP {run_name}: all checkpoints already evaluated")
            continue

        rainbow_enabled = config.get("rainbow", {}).get("enabled", False)
        spr_enabled = config.get("spr", {}).get("enabled", False)
        model_type = "RainbowDQN" if rainbow_enabled else "DQN"
        if spr_enabled:
            model_type += "+SPR"

        print(f"RUN  {run_name}")
        print(f"     env={env_id}, model={model_type}, missing steps: {missing_steps}")

        # Create environment
        env = make_atari_env(
            env_id=env_id,
            frame_size=84,
            num_stack=4,
            frame_skip=4,
            clip_rewards=False,
            episode_life=False,
            noop_max=30,
        )
        num_actions = env.action_space.n

        for step in sorted(missing_steps):
            cp_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            print(f"     step {step // 1000}K ... ", end="", flush=True)

            checkpoint = torch.load(cp_path, map_location=device, weights_only=False)

            model = create_model(config, num_actions, device)
            model.load_state_dict(checkpoint["online_model_state_dict"], strict=True)
            model.eval()

            training_epsilon = checkpoint.get("epsilon", 0.1)

            results = evaluate(
                env=env,
                model=model,
                num_episodes=EVAL_EPISODES,
                eval_epsilon=EVAL_EPSILON,
                num_actions=num_actions,
                device=device,
                step=step,
            )

            append_eval_row(run_dir, step, results, training_epsilon)
            print(f"mean={results['mean_return']:.1f} +/- {results['std_return']:.1f}")

        env.close()
        print()

    # Sort all eval CSVs by step
    print("Sorting eval CSVs by step...")
    for run_name in run_names:
        csv_path = os.path.join(args.runs_dir, run_name, "eval", "evaluations.csv")
        if not os.path.exists(csv_path):
            continue
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        rows.sort(key=lambda r: int(r["step"]))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    print("Done.")


if __name__ == "__main__":
    main()
