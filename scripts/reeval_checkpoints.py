#!/usr/bin/env python3
"""Re-run evaluations on saved checkpoints that are missing data.

Loads each checkpoint, creates the matching model and environment,
runs 30 evaluation episodes, and saves results to the eval/ directory.

Produces the same output format as inline evaluation:
  - eval/evaluations.csv (summary statistics per checkpoint)
  - eval/evaluations.jsonl (same data in JSONL)
  - eval/per_episode_returns.jsonl (raw per-episode data)
  - eval/detailed/eval_step_<step>.json (complete details)
  - videos/<game>_step_<step>_best_ep<N>_r<score>.mp4 (optional)

Usage:
    python scripts/reeval_checkpoints.py                       # all runs
    python scripts/reeval_checkpoints.py run_name_1 ...        # specific runs
    python scripts/reeval_checkpoints.py --device cuda         # use GPU
    python scripts/reeval_checkpoints.py --record-video        # save videos
    python scripts/reeval_checkpoints.py --force               # re-run all
"""

import argparse
import csv
import json
import os
import re
import sys

import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.atari_wrappers import make_atari_env
from src.models.dqn import DQN
from src.models.rainbow import RainbowDQN
from src.training.evaluation import evaluate

RUNS_DIR = "experiments/dqn_atari/runs"

EVAL_EPISODES = 30


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


def get_existing_steps(run_dir):
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


def get_run_eval_epsilon(config):
    """Determine the correct epsilon for evaluation.

    Rainbow with NoisyNets uses learned exploration -- evaluation
    should be fully greedy (epsilon=0.0) with noise disabled via
    model.eval(). DQN uses epsilon-greedy with 0.05 (Mnih et al.).
    """
    rainbow_cfg = config.get("rainbow", {})
    if rainbow_cfg.get("enabled", False) and rainbow_cfg.get("noisy_nets", True):
        return 0.0
    return 0.05


def get_repeat_action_probability(config):
    """Read sticky actions probability from config.

    Atari-100K uses 0.25 (Machado et al. 2018). Falls back to 0.0
    if not specified (standard ALE default).
    """
    return config.get("environment", {}).get("repeat_action_probability", 0.0)


def save_results(run_dir, step, results, run_eval_epsilon, training_epsilon):
    """Save results in all output formats matching EvaluationLogger.

    Writes:
      - eval/evaluations.csv
      - eval/evaluations.jsonl
      - eval/per_episode_returns.jsonl
      - eval/detailed/eval_step_<step>.json
    """
    eval_dir = os.path.join(run_dir, "eval")
    detailed_dir = os.path.join(eval_dir, "detailed")
    os.makedirs(detailed_dir, exist_ok=True)

    # 1. evaluations.csv
    csv_path = os.path.join(eval_dir, "evaluations.csv")
    csv_entry = {
        "step": step,
        "mean_return": results["mean_return"],
        "median_return": results["median_return"],
        "std_return": results["std_return"],
        "min_return": results["min_return"],
        "max_return": results["max_return"],
        "episodes": results["num_episodes"],
        "eval_epsilon": run_eval_epsilon,
        "training_epsilon": training_epsilon,
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_entry)

    # 2. evaluations.jsonl
    jsonl_path = os.path.join(eval_dir, "evaluations.jsonl")
    with open(jsonl_path, "a") as f:
        json.dump(csv_entry, f)
        f.write("\n")

    # 3. per_episode_returns.jsonl
    episodes_path = os.path.join(eval_dir, "per_episode_returns.jsonl")
    per_episode_entry = {
        "step": step,
        "episode_returns": [float(r) for r in results["episode_returns"]],
        "episode_lengths": [int(ln) for ln in results["episode_lengths"]],
    }
    with open(episodes_path, "a") as f:
        json.dump(per_episode_entry, f)
        f.write("\n")

    # 4. detailed/eval_step_<step>.json
    json_path = os.path.join(detailed_dir, f"eval_step_{step}.json")
    detailed_results = {
        "step": step,
        "statistics": {
            "mean_return": float(results["mean_return"]),
            "median_return": float(results["median_return"]),
            "std_return": float(results["std_return"]),
            "min_return": float(results["min_return"]),
            "max_return": float(results["max_return"]),
            "mean_length": float(results["mean_length"]),
        },
        "episode_returns": [float(r) for r in results["episode_returns"]],
        "episode_lengths": [int(ln) for ln in results["episode_lengths"]],
        "num_episodes": results["num_episodes"],
        "eval_epsilon": run_eval_epsilon,
        "training_epsilon": training_epsilon,
    }
    with open(json_path, "w") as f:
        json.dump(detailed_results, f, indent=2)


def write_reeval_progress(run_dir, run_name, step, steps_done, steps_total,
                          start_time, mean_return):
    """Write progress file for monitoring via Colab runner."""
    elapsed = time.time() - start_time
    remaining = steps_total - steps_done
    per_step = elapsed / steps_done if steps_done > 0 else 0
    eta = per_step * remaining

    progress = {
        "run": run_name,
        "step": step,
        "checkpoints_done": steps_done,
        "checkpoints_total": steps_total,
        "percent": round(100 * steps_done / steps_total, 1),
        "last_mean_return": mean_return,
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1),
        "status": "evaluating",
    }

    progress_path = os.path.join(run_dir, "eval", "reeval_progress.json")
    os.makedirs(os.path.dirname(progress_path), exist_ok=True)
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def sort_csv(run_dir):
    """Sort evaluations.csv by step number."""
    csv_path = os.path.join(run_dir, "eval", "evaluations.csv")
    if not os.path.exists(csv_path):
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return
    rows.sort(key=lambda r: int(r["step"]))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Re-evaluate checkpoints that are missing data.",
    )
    parser.add_argument(
        "runs",
        nargs="*",
        help="Run names (default: auto-discover all runs with checkpoints)",
    )
    parser.add_argument(
        "--runs-dir",
        default=RUNS_DIR,
        help=f"Base directory containing run folders (default: {RUNS_DIR})",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for model inference (default: cpu)",
    )
    parser.add_argument(
        "--record-video",
        action="store_true",
        help="Record best-episode video per checkpoint",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run all checkpoints even if data exists",
    )
    parser.add_argument(
        "--drive-dir",
        default=None,
        help="Google Drive thesis-runs directory. If set, copies "
             "eval output to Drive after each run completes. "
             "Auto-detected from /content/drive/MyDrive/thesis-runs "
             "if that path exists.",
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

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Auto-detect Google Drive mount
    drive_dir = args.drive_dir
    if drive_dir is None:
        default_drive = "/content/drive/MyDrive/thesis-runs"
        if os.path.isdir(default_drive):
            drive_dir = default_drive
            print(f"Drive auto-detected: {drive_dir}")

    print(f"Device: {device}")
    print(f"Episodes per checkpoint: {EVAL_EPISODES}")
    print(f"Record video: {args.record_video}")
    print(f"Force re-run: {args.force}")
    if drive_dir:
        print(f"Drive save: {drive_dir}")
    print(f"Runs to process: {len(run_names)}")
    print()

    for run_name in run_names:
        run_dir = os.path.join(args.runs_dir, run_name)
        if not os.path.exists(run_dir):
            print(f"SKIP {run_name}: directory not found")
            continue

        config = load_config(run_dir)
        env_id = config["environment"]["env_id"]

        # Determine which checkpoints need processing
        if args.force:
            existing_steps = set()
        else:
            existing_steps = get_existing_steps(run_dir)

        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        all_steps = discover_checkpoint_steps(checkpoint_dir)
        missing_steps = [s for s in all_steps if s not in existing_steps]

        if not missing_steps:
            print(f"SKIP {run_name}: all checkpoints already processed")
            continue

        # Determine model type and settings
        rainbow_enabled = config.get("rainbow", {}).get("enabled", False)
        spr_enabled = config.get("spr", {}).get("enabled", False)
        model_type = "RainbowDQN" if rainbow_enabled else "DQN"
        if spr_enabled:
            model_type += "+SPR"

        run_eval_epsilon = get_run_eval_epsilon(config)
        repeat_action_prob = get_repeat_action_probability(config)

        print(f"RUN  {run_name}")
        print(f"     env={env_id}, model={model_type}")
        print(f"     epsilon={run_eval_epsilon}, sticky={repeat_action_prob}")
        print(f"     steps: {missing_steps}")

        # Create environment with correct sticky actions
        # render_mode needed for video recording (env.render() returns frames)
        env_kwargs = {}
        if args.record_video:
            env_kwargs["render_mode"] = "rgb_array"
        env = make_atari_env(
            env_id=env_id,
            frame_size=84,
            num_stack=4,
            frame_skip=4,
            clip_rewards=False,
            episode_life=False,
            noop_max=30,
            repeat_action_probability=repeat_action_prob,
            **env_kwargs,
        )
        num_actions = env.action_space.n

        # Video directory
        video_dir = os.path.join(run_dir, "videos") if args.record_video else None
        if video_dir:
            os.makedirs(video_dir, exist_ok=True)

        sorted_steps = sorted(missing_steps)
        run_start_time = time.time()

        for step_idx, step in enumerate(sorted_steps):
            cp_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            print(f"     step {step // 1000}K ... ", end="", flush=True)

            checkpoint = torch.load(cp_path, map_location=device, weights_only=False)

            model = create_model(config, num_actions, device)
            model.load_state_dict(
                checkpoint["online_model_state_dict"], strict=True
            )
            # model.eval() disables NoisyNet noise (uses mean weights)
            model.eval()

            training_epsilon = checkpoint.get("epsilon", 0.1)

            results = evaluate(
                env=env,
                model=model,
                num_episodes=EVAL_EPISODES,
                eval_epsilon=run_eval_epsilon,
                num_actions=num_actions,
                device=device,
                step=step,
                record_video=args.record_video,
                video_dir=video_dir,
            )

            save_results(
                run_dir, step, results, run_eval_epsilon, training_epsilon
            )

            write_reeval_progress(
                run_dir, run_name, step,
                steps_done=step_idx + 1,
                steps_total=len(sorted_steps),
                start_time=run_start_time,
                mean_return=results["mean_return"],
            )

            video_msg = ""
            if results.get("video_info"):
                video_msg = f"  video=saved"
            print(
                f"mean={results['mean_return']:.1f} "
                f"+/- {results['std_return']:.1f}{video_msg}"
            )

        env.close()

        # Sort CSV by step after processing all checkpoints
        sort_csv(run_dir)

        # Copy eval output to Drive if configured
        if drive_dir:
            import shutil
            drive_run_dir = os.path.join(drive_dir, run_name)
            if os.path.isdir(drive_run_dir):
                drive_eval_dir = os.path.join(drive_run_dir, "eval")
                src_eval_dir = os.path.join(run_dir, "eval")
                if os.path.isdir(src_eval_dir):
                    if os.path.isdir(drive_eval_dir):
                        shutil.rmtree(drive_eval_dir)
                    shutil.copytree(src_eval_dir, drive_eval_dir)
                    print(f"     Saved eval to Drive: {drive_run_dir}/eval/")
                # Also copy videos if they were recorded
                src_videos = os.path.join(run_dir, "videos")
                if args.record_video and os.path.isdir(src_videos):
                    drive_videos = os.path.join(drive_run_dir, "videos")
                    if os.path.isdir(drive_videos):
                        shutil.rmtree(drive_videos)
                    shutil.copytree(src_videos, drive_videos)
                    print(f"     Saved videos to Drive")
            else:
                print(f"     WARN: {drive_run_dir} not found on Drive")

        print()

    print("Done.")


if __name__ == "__main__":
    main()
