#!/usr/bin/env python
"""
JAX/BBF Training Entry Point

Train BBF-family agents (BBF, BBFc, SR-SPR, SR-SPRc, SPR, SPRc, DER, DERc)
on Atari games using the ported BBF codebase.

Usage:
    python train.py --condition BBF --game boxing --seed 42
"""

import argparse
import csv
import datetime
import json
import os
import shutil
import subprocess
import sys
import time

# Unbuffered stdout for real-time progress in non-TTY environments (Colab, CI)
sys.stdout.reconfigure(line_buffering=True)

# Add src/ to path so BBF package imports work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gin
import jax
import numpy as np
import tensorflow as tf
from absl import logging
from dopamine.discrete_domains import atari_lib

from bigger_better_faster.bbf.agents.metric_agent import MetricBBFAgent


GIN_ROOT = os.path.join(os.path.dirname(__file__),
                        "src", "bigger_better_faster", "bbf", "configs")

# Maps each condition name to its base gin file.
CONDITION_BASE = {
    "BBF":     "BBF.gin",
    "BBFc":    "BBF.gin",
    "DER":     "SPR.gin",
    "DERc":    "SPR.gin",
    "SPR":     "SPR.gin",
    "SPRc":    "SPR.gin",
    "SR_SPR":  "SR_SPR.gin",
    "SR_SPRc": "SR_SPR.gin",
}

VALID_CONDITIONS = sorted(CONDITION_BASE.keys())


def resolve_gin_paths(condition, game):
    """Resolve condition and game names to gin file paths."""
    if condition not in CONDITION_BASE:
        raise SystemExit(
            f"Unknown condition '{condition}'. "
            f"Valid conditions: {', '.join(VALID_CONDITIONS)}")

    base_gin = os.path.join(GIN_ROOT, CONDITION_BASE[condition])
    condition_gin = os.path.join(GIN_ROOT, "conditions", f"{condition}.gin")
    game_gin = os.path.join(GIN_ROOT, "games", f"{game}.gin")

    for path, label in [(base_gin, "base"), (condition_gin, "condition"),
                        (game_gin, "game")]:
        if not os.path.isfile(path):
            raise SystemExit(f"{label} gin not found: {path}")

    return base_gin, condition_gin, game_gin


def parse_args():
    parser = argparse.ArgumentParser(description="Train BBF-family agent")
    parser.add_argument("--condition", required=True,
                        choices=VALID_CONDITIONS,
                        help="Experimental condition name")
    parser.add_argument("--game", required=True,
                        help="Atari game name (e.g., boxing)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gin_bindings", nargs="*", default=[],
                        help="Extra gin bindings (key=value)")
    parser.add_argument("--run_dir",
                        help="Output directory (default: auto-generated)")
    parser.add_argument("--total_steps", type=int, default=100_000,
                        help="Total environment steps (default: 100000)")
    return parser.parse_args()


def setup_gin(base_gin, condition_gin, game_gin, bindings):
    """Load gin configs in order: base -> condition -> game."""
    gin.clear_config()
    gin.parse_config_files_and_bindings(
        [base_gin, condition_gin, game_gin],
        bindings,
    )


def create_environment():
    """Create an Atari environment via dopamine's gin-configured factory."""
    env = atari_lib.create_atari_environment()
    return env


def create_agent(env, seed, run_dir):
    """Instantiate MetricBBFAgent with the environment's action count."""
    summary_dir = os.path.join(run_dir, "tb")
    os.makedirs(summary_dir, exist_ok=True)
    agent = MetricBBFAgent(
        num_actions=env.action_space.n,
        seed=seed,
        summary_writer=summary_dir,
    )
    agent.eval_mode = False
    return agent


def setup_run_dir(run_dir):
    """Create run directory and subdirectories."""
    os.makedirs(run_dir, exist_ok=True)
    for sub in ("checkpoints", "tb"):
        os.makedirs(os.path.join(run_dir, sub), exist_ok=True)


def save_config_snapshot(run_dir):
    """Save the resolved gin config to the run directory."""
    config_path = os.path.join(run_dir, "config.gin")
    with open(config_path, "w") as f:
        f.write(gin.config_str())


def _git_hash():
    """Return the short git commit hash, or 'unknown' if unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_meta(run_dir, condition, game, seed):
    """Write meta.json with run provenance at startup."""
    meta = {
        "condition": condition,
        "game": game,
        "seed": seed,
        "git_hash": _git_hash(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    path = os.path.join(run_dir, "meta.json")
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# Core fields (7) + extension columns (23) from MetricBBFAgent._last_metrics.
CSV_CORE = ["step", "fps", "loss", "grad_norm", "learning_rate",
            "epsilon", "replay_size"]
CSV_EXTENSIONS = [
    "TotalLoss", "DQNLoss", "TD Error", "SPRLoss",
    "QValueMean", "QValueMax",
    "GradNorm",
    "GradNorm/encoder", "GradNorm/transition_model",
    "GradNorm/projection", "GradNorm/predictor", "GradNorm/head",
    "PNorm",
    "Inter-batch time", "Training time", "Sampling time",
    "Set priority time", "Online Churn", "Target Churn",
    "Online-Target Agreement", "Online Off-Policy Rate",
    "Target Off-Policy Rate", "TargetDivergence",
]
CSV_HEADER = CSV_CORE + CSV_EXTENSIONS


def open_csv(run_dir):
    """Open the per-step CSV and write the header."""
    path = os.path.join(run_dir, "steps.csv")
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
    writer.writeheader()
    f.flush()
    return f, writer


EPISODE_HEADER = ["step", "episode", "episode_return", "episode_length"]


def open_episode_csv(run_dir):
    """Open the per-episode CSV and write the header."""
    path = os.path.join(run_dir, "episodes.csv")
    f = open(path, "w", newline="")
    writer = csv.DictWriter(f, fieldnames=EPISODE_HEADER)
    writer.writeheader()
    f.flush()
    return f, writer


def _get_live_lr(agent):
    """Extract the current learning rate from the optimizer state.

    When warmup is used, optax.inject_hyperparams stores the scheduled
    LR in state.hyperparams['learning_rate']. When no warmup is used,
    the optimizer state is a plain tuple and we fall back to the static
    constructor value.
    """
    for leaf in jax.tree_util.tree_leaves(agent.optimizer_state):
        if hasattr(leaf, 'hyperparams') and 'learning_rate' in leaf.hyperparams:
            return float(leaf.hyperparams['learning_rate'])
    return agent.learning_rate


def write_step_row(writer, csv_file, step, fps, agent):
    """Write one row to the per-step CSV using agent metrics."""
    metrics = agent._last_metrics
    if not metrics:
        return
    row = {
        "step": step,
        "fps": f"{fps:.1f}",
        "loss": metrics.get("TotalLoss", ""),
        "grad_norm": metrics.get("GradNorm", ""),
        "learning_rate": _get_live_lr(agent),
        "epsilon": 0.0 if agent._noisy else agent.epsilon_fn(
            agent.epsilon_decay_period, agent.training_steps,
            agent.min_replay_history, agent.epsilon_train),
        "replay_size": int(agent._replay.add_count),
    }
    for col in CSV_EXTENSIONS:
        if col not in row:
            row[col] = metrics.get(col, "")
    writer.writerow(row)
    csv_file.flush()


CHECKPOINT_INTERVAL = 10_000
PROGRESS_INTERVAL = 100


def validate_checkpoint(params_path, meta_path):
    """Verify checkpoint files exist and have nonzero size.

    Returns a dict with 'valid' (bool), 'files' (per-file details),
    and 'errors' (list of failure descriptions). Never raises -- logs
    errors so training can continue.
    """
    files = {}
    errors = []
    for path, label in [(params_path, "params"), (meta_path, "metadata")]:
        if not os.path.isfile(path):
            errors.append(f"{label} missing: {path}")
            files[label] = {"path": path, "exists": False, "size": 0}
        else:
            size = os.path.getsize(path)
            files[label] = {"path": path, "exists": True, "size": size}
            if size == 0:
                errors.append(f"{label} empty: {path}")

    return {"valid": len(errors) == 0, "files": files, "errors": errors}


def save_checkpoint(agent, step, run_dir):
    """Save online_params as msgpack and metadata as JSON."""
    from flax.serialization import msgpack_serialize, to_state_dict

    ckpt_dir = os.path.join(run_dir, "checkpoints")

    # Params -> msgpack (preserves pytree structure natively)
    state_dict = to_state_dict(agent.online_params)
    params_path = os.path.join(ckpt_dir, f"checkpoint_{step}.msgpack")
    with open(params_path, "wb") as f:
        f.write(msgpack_serialize(state_dict))

    # Metadata -> JSON sidecar
    meta = {
        "step": step,
        "training_steps": int(agent.training_steps),
        "cumulative_resets": int(agent.cumulative_resets),
        "cycle_grad_steps": int(agent.cycle_grad_steps),
        "gin_config": gin.config_str(),
    }
    meta_path = os.path.join(ckpt_dir, f"checkpoint_{step}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # Also trigger replay buffer save via the agent's bundle_and_checkpoint.
    # This saves replay_buffer_<iteration>.npz in the same directory.
    agent.bundle_and_checkpoint(ckpt_dir, step)

    # Write resets.json from the agent's captured reset events.
    if agent._reset_log:
        resets_path = os.path.join(run_dir, "resets.json")
        with open(resets_path, "w") as f:
            json.dump(agent._reset_log, f, indent=2)

    # Validate checkpoint files exist and have nonzero size
    validation = validate_checkpoint(params_path, meta_path)

    # Incremental Drive sync (Colab only)
    sync_to_drive(run_dir)

    status = "OK" if validation["valid"] else "FAILED"
    print(f"Checkpoint saved at step {step}: {params_path} [{status}]")
    if not validation["valid"]:
        print(f"  Validation errors: {validation['errors']}")

    return validation


DRIVE_BASE = "/content/drive/MyDrive/thesis-runs"


def sync_to_drive(run_dir):
    """Copy run directory to Google Drive if the mount exists."""
    if not os.path.isdir("/content/drive/MyDrive"):
        return
    run_name = os.path.basename(os.path.normpath(run_dir))
    dest = os.path.join(DRIVE_BASE, run_name)
    shutil.copytree(run_dir, dest, dirs_exist_ok=True)
    print(f"Synced to Drive: {dest}")


def write_progress(run_dir, step, total_steps, episode, fps,
                   start_time, status="training",
                   last_checkpoint_validation=None):
    """Write progress.json for remote monitoring via the Colab runner."""
    elapsed = time.time() - start_time
    percent = (step / total_steps) * 100 if total_steps > 0 else 0
    eta = (elapsed / step) * (total_steps - step) if step > 0 else 0
    progress = {
        "frame": step,
        "total_frames": total_steps,
        "percent": round(percent, 1),
        "episode": episode,
        "fps": round(fps, 1),
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1),
        "status": status,
    }
    if last_checkpoint_validation is not None:
        progress["last_checkpoint_validation"] = {
            "valid": last_checkpoint_validation["valid"],
            "errors": last_checkpoint_validation["errors"],
        }
    path = os.path.join(run_dir, "progress.json")
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    args = parse_args()
    logging.set_verbosity(logging.INFO)

    # Resolve gin paths from condition and game names
    base_gin, condition_gin, game_gin = resolve_gin_paths(
        args.condition, args.game)

    # Auto-generate run directory if not specified
    run_dir = args.run_dir
    if run_dir is None:
        condition_token = args.condition.lower().replace("_", "-")
        run_name = f"{condition_token}_{args.game}_seed{args.seed}"
        run_dir = os.path.join("experiments", "dqn_atari", "runs", run_name)

    # Setup
    set_seed(args.seed)
    setup_gin(base_gin, condition_gin, game_gin, args.gin_bindings)
    setup_run_dir(run_dir)
    save_config_snapshot(run_dir)
    write_meta(run_dir, args.condition, args.game, args.seed)

    # Create environment and agent
    env = create_environment()
    agent = create_agent(env, args.seed, run_dir)

    game_name = gin.query_parameter("DataEfficientAtariRunner.game_name")
    print(f"Training: game={game_name}, seed={args.seed}, "
          f"steps={args.total_steps}, run_dir={run_dir}")
    print(f"Agent: spr_weight={agent.spr_weight}, jumps={agent._jumps}, "
          f"replay_ratio={agent._replay_ratio}")

    # Open CSV files
    csv_file, csv_writer = open_csv(run_dir)
    ep_file, ep_writer = open_episode_csv(run_dir)

    # Initialize: reset environment and agent
    obs = env.reset()
    agent.reset_all(np.expand_dims(obs, 0))  # (1, 84, 84)

    # Training loop
    step = 0
    episode = 0
    episode_return = 0.0
    episode_length = 0
    start_time = time.time()
    last_ckpt_validation = None

    while step < args.total_steps:
        # Agent selects action (also trains if replay buffer is ready)
        actions = agent.step()
        action = int(actions[0])

        # Environment step
        obs, reward, done, info = env.step(action)

        # Clip reward to [-1, 1] per Atari-100K convention
        clipped_reward = np.clip(reward, -1.0, 1.0)

        episode_return += reward  # Track unclipped return
        episode_length += 1
        step += 1

        # Write per-step CSV row (only produces output when agent has metrics)
        elapsed = time.time() - start_time
        fps = step / elapsed if elapsed > 0 else 0
        write_step_row(csv_writer, csv_file, step, fps, agent)

        # Progress reporting
        if step % PROGRESS_INTERVAL == 0:
            write_progress(run_dir, step, args.total_steps,
                           episode, fps, start_time,
                           last_checkpoint_validation=last_ckpt_validation)

        # Log transition to agent (updates internal state and replay buffer)
        terminal = done
        episode_end = done
        agent.log_transition(
            np.expand_dims(obs, 0),  # (1, 84, 84)
            np.array([action]),
            np.array([clipped_reward]),
            np.array([terminal]),
            np.array([episode_end]),
        )

        # Checkpoint (after log_transition so params and buffer are consistent)
        if step % CHECKPOINT_INTERVAL == 0:
            last_ckpt_validation = save_checkpoint(agent, step, run_dir)

        # Episode boundary
        if done:
            episode += 1
            ep_writer.writerow({
                "step": step,
                "episode": episode,
                "episode_return": episode_return,
                "episode_length": episode_length,
            })
            ep_file.flush()
            print(f"Step {step}/{args.total_steps} | "
                  f"Episode {episode} | Return {episode_return:.0f} | "
                  f"Length {episode_length} | FPS {fps:.0f}")

            obs = env.reset()
            agent.reset_one(0)
            agent._record_observation(np.expand_dims(obs, 0))
            episode_return = 0.0
            episode_length = 0

    # Final checkpoint if not already saved at last interval
    if step % CHECKPOINT_INTERVAL != 0:
        last_ckpt_validation = save_checkpoint(agent, step, run_dir)

    csv_file.close()
    ep_file.close()
    elapsed = time.time() - start_time
    fps = step / elapsed if elapsed > 0 else 0
    write_progress(run_dir, step, args.total_steps,
                   episode, fps, start_time, status="complete",
                   last_checkpoint_validation=last_ckpt_validation)
    print(f"Training complete: {step} steps in {elapsed:.1f}s "
          f"({fps:.0f} FPS)")

    env.close()


if __name__ == "__main__":
    main()
