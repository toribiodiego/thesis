#!/usr/bin/env python3
"""
DQN training entry point.

Supports both full training runs and dry-run mode for testing the pipeline.
"""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf

from utils.repro import set_seed, save_run_metadata


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DQN on Atari games")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file (e.g., experiments/dqn_atari/configs/pong.yaml)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a short random rollout for testing (no training)"
    )
    parser.add_argument(
        "--dry-run-episodes",
        type=int,
        default=3,
        help="Number of episodes for dry run"
    )
    return parser.parse_args()


def load_config(config_path: str):
    """Load and merge config files."""
    config = OmegaConf.load(config_path)

    # If config has defaults, merge with base
    if "defaults" in config:
        base_path = Path(config_path).parent / "base.yaml"
        base_config = OmegaConf.load(base_path)
        config = OmegaConf.merge(base_config, config)
        # Remove defaults key from final config
        if "defaults" in config:
            del config["defaults"]

    return config


def create_env(config):
    """Create Atari environment with configured settings."""
    env = gym.make(
        config.env.id,
        frameskip=config.env.frameskip,
        repeat_action_probability=config.env.repeat_action_probability,
        full_action_space=False,  # Use minimal action set
    )
    return env


def dry_run(config, seed, num_episodes=3):
    """
    Execute a short random rollout for testing the environment setup.

    Saves:
    - A few preprocessed frame samples
    - List of available actions
    - Minimal evaluation report with episode statistics
    """
    print("=" * 80)
    print("DRY RUN MODE - Random Policy Rollout")
    print("=" * 80)
    print(f"Environment: {config.env.id}")
    print(f"Episodes: {num_episodes}")
    print(f"Seed: {seed}")
    print()

    # Create environment
    env = create_env(config)

    # Set seed
    set_seed(seed)

    # Get action space info
    action_space_size = env.action_space.n
    print(f"Action space size: {action_space_size}")

    # Create output directory
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Run random episodes
    episode_stats = []
    all_frames = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_length = 0
        done = False

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"  Initial observation shape: {obs.shape}")

        # Save first frame
        if episode == 0:
            all_frames.append(obs)

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_length += 1

            # Save a few frames from first episode
            if episode == 0 and len(all_frames) < 5 and episode_length % 10 == 0:
                all_frames.append(obs)

        episode_stats.append({
            "episode": episode + 1,
            "reward": float(episode_reward),
            "length": episode_length
        })

        print(f"  Reward: {episode_reward}, Length: {episode_length}")

    env.close()

    # Save frame samples
    print(f"\nSaving {len(all_frames)} frame samples to {frames_dir}/")
    for i, frame in enumerate(all_frames):
        np.save(frames_dir / f"frame_{i:03d}.npy", frame)

    # Create action list
    action_info = {
        "action_space_size": action_space_size,
        "action_meanings": [
            f"Action {i}" for i in range(action_space_size)
        ]
    }

    action_list_path = output_dir / "action_list.json"
    with open(action_list_path, "w") as f:
        json.dump(action_info, f, indent=2)
    print(f"Action list saved to: {action_list_path}")

    # Create minimal evaluation report
    mean_reward = np.mean([s["reward"] for s in episode_stats])
    std_reward = np.std([s["reward"] for s in episode_stats])
    mean_length = np.mean([s["length"] for s in episode_stats])

    eval_report = {
        "mode": "dry_run",
        "environment": config.env.id,
        "seed": seed,
        "num_episodes": num_episodes,
        "episode_stats": episode_stats,
        "summary": {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "mean_episode_length": float(mean_length),
            "total_frames": sum(s["length"] for s in episode_stats)
        }
    }

    eval_report_path = output_dir / "dry_run_report.json"
    with open(eval_report_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"Evaluation report saved to: {eval_report_path}")

    # Save run metadata
    ale_settings = {
        "frameskip": config.env.frameskip,
        "repeat_action_probability": config.env.repeat_action_probability,
        "full_action_space": False,
        "max_noop_start": config.env.get("max_noop_start", 30)
    }

    save_run_metadata(
        output_dir=output_dir,
        config=OmegaConf.to_container(config, resolve=True),
        seed=seed,
        ale_settings=ale_settings,
        extra_info={"mode": "dry_run", "episodes": num_episodes}
    )

    print("\n" + "=" * 80)
    print("DRY RUN COMPLETE")
    print("=" * 80)
    print(f"Summary: {mean_reward:.2f} ± {std_reward:.2f} reward over {num_episodes} episodes")
    print(f"Output directory: {output_dir}")

    return eval_report


def train(config, seed):
    """
    Main training loop (placeholder for now).

    This will be implemented in later subtasks.
    """
    print("=" * 80)
    print("TRAINING MODE")
    print("=" * 80)
    print("Full training not yet implemented.")
    print("Use --dry-run to test the environment setup.")
    print("=" * 80)


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Set seed
    seed = args.seed if args.seed is not None else config.experiment.seed

    if args.dry_run:
        dry_run(config, seed, num_episodes=args.dry_run_episodes)
    else:
        train(config, seed)


if __name__ == "__main__":
    main()
