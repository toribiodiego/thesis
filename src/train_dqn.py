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

from envs.atari_wrappers import make_atari_env
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


def create_env(config, save_samples=False, sample_dir=None, episode_life=None):
    """
    Create Atari environment with preprocessing and frame stacking.

    Args:
        config: Configuration object
        save_samples: Whether to save sample frames
        sample_dir: Directory to save samples
        episode_life: Override episode_life setting (None = use config)
    """
    # Use episode_life from config if not overridden
    if episode_life is None:
        episode_life = config.training.episode_life

    env = make_atari_env(
        env_id=config.env.id,
        frame_size=config.preprocess.frame_size,
        num_stack=config.preprocess.stack_size,
        frame_skip=config.env.frameskip,
        clip_rewards=config.training.reward_clip,
        episode_life=episode_life,
        noop_max=config.env.max_noop_start,
        save_samples=save_samples,
        sample_dir=sample_dir,
        frameskip=1,  # Disable built-in frameskip, use MaxAndSkipEnv instead
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

    # Create output directory
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    # Create environment with sample saving enabled (use evaluation mode)
    env = create_env(config, save_samples=True, sample_dir=frames_dir, episode_life=False)

    # Set seed
    set_seed(seed)

    # Get action space info
    action_space_size = env.action_space.n
    print(f"Action space size: {action_space_size}")
    print(f"Observation space: {env.observation_space.shape} (dtype: {env.observation_space.dtype})")
    print(f"Preprocessing: {config.preprocess.frame_size}x{config.preprocess.frame_size} grayscale")
    print(f"Frame stack: {config.preprocess.stack_size} frames")
    print(f"Frame skip: {config.env.frameskip} (action repeat with max-pooling)")
    print(f"No-op max: {config.env.max_noop_start} (random no-ops on reset)")
    print(f"Reward clipping: {'enabled' if config.training.reward_clip else 'disabled'}")
    print(f"Episode termination: full episode (life loss NOT terminal for dry run)")

    # Run random episodes and collect detailed statistics
    episode_stats = []
    all_rewards = []
    all_raw_rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode)
        episode_reward = 0
        episode_raw_reward = 0
        episode_length = 0
        episode_rewards = []
        done = False

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print(f"  Initial observation shape: {obs.shape} (dtype: {obs.dtype})")
        print(f"  Value range: [{obs.min()}, {obs.max()}]")

        step_count = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Track both clipped and raw rewards
            episode_reward += reward
            episode_rewards.append(float(reward))

            # Estimate raw reward (before clipping)
            # Note: This is approximate since we can't access the true raw reward
            # In actual training, we'd track this before the RewardClipper wrapper
            if reward > 0:
                raw_reward_estimate = 1.0  # Could be any positive value
            elif reward < 0:
                raw_reward_estimate = -1.0  # Could be any negative value
            else:
                raw_reward_estimate = 0.0
            episode_raw_reward += raw_reward_estimate

            episode_length += 1
            step_count += 1

        all_rewards.extend(episode_rewards)
        all_raw_rewards.append(episode_raw_reward)

        episode_stats.append({
            "episode": episode + 1,
            "reward": float(episode_reward),
            "length": episode_length,
            "rewards_collected": episode_rewards,
            "unique_rewards": list(set(episode_rewards))
        })

        print(f"  Reward: {episode_reward}, Length: {episode_length}")
        print(f"  Unique rewards seen: {sorted(set(episode_rewards))}")

    env.close()

    # Calculate statistics
    mean_reward = np.mean([s["reward"] for s in episode_stats])
    std_reward = np.std([s["reward"] for s in episode_stats])
    mean_length = np.mean([s["length"] for s in episode_stats])

    # Reward statistics
    reward_counts = {}
    for r in all_rewards:
        reward_counts[r] = reward_counts.get(r, 0) + 1

    # Create detailed rollout log
    rollout_log = {
        "metadata": {
            "mode": "dry_run",
            "environment": config.env.id,
            "seed": seed,
            "num_episodes": num_episodes,
            "timestamp": str(np.datetime64('now'))
        },
        "observation_space": {
            "shape": list(env.observation_space.shape),
            "dtype": str(env.observation_space.dtype),
            "description": f"Stacked {config.preprocess.stack_size} grayscale frames of {config.preprocess.frame_size}x{config.preprocess.frame_size}"
        },
        "action_space": {
            "size": action_space_size,
            "type": "Discrete"
        },
        "preprocessing": {
            "frame_size": config.preprocess.frame_size,
            "grayscale": True,
            "frame_stack": config.preprocess.stack_size,
            "frame_skip": config.env.frameskip,
            "max_pooling": "last 2 frames",
            "noop_max": config.env.max_noop_start,
            "reward_clipping": config.training.reward_clip,
            "episode_life": False  # Dry run uses full episodes
        },
        "reward_statistics": {
            "clipped_rewards": {
                "unique_values": sorted(list(reward_counts.keys())),
                "counts": reward_counts,
                "total_steps": len(all_rewards),
                "mean": float(np.mean(all_rewards)) if all_rewards else 0.0,
                "std": float(np.std(all_rewards)) if all_rewards else 0.0,
                "min": float(np.min(all_rewards)) if all_rewards else 0.0,
                "max": float(np.max(all_rewards)) if all_rewards else 0.0
            },
            "description": "Rewards are clipped to {-1, 0, +1}" if config.training.reward_clip else "Raw rewards (no clipping)"
        },
        "episode_statistics": episode_stats,
        "summary": {
            "mean_episode_reward": float(mean_reward),
            "std_episode_reward": float(std_reward),
            "mean_episode_length": float(mean_length),
            "total_frames": sum(s["length"] for s in episode_stats),
            "total_episodes": num_episodes
        }
    }

    # Save rollout log
    rollout_log_path = output_dir / "rollout_log.json"
    with open(rollout_log_path, "w") as f:
        json.dump(rollout_log, f, indent=2)
    print(f"\nRollout log saved to: {rollout_log_path}")
    print(f"  - Observation shape: {rollout_log['observation_space']['shape']}")
    print(f"  - Action repeat (frame skip): {config.env.frameskip}")
    print(f"  - Reward clipping: {rollout_log['reward_statistics']['description']}")
    print(f"  - Unique rewards: {rollout_log['reward_statistics']['clipped_rewards']['unique_values']}")
    print(f"  - Reward counts: {rollout_log['reward_statistics']['clipped_rewards']['counts']}")

    # Frame samples are automatically saved by FrameStack wrapper
    print(f"\nPreprocessed frame samples saved to {frames_dir}/")
    print(f"  - Shape: ({config.preprocess.stack_size}, {config.preprocess.frame_size}, {config.preprocess.frame_size})")
    print(f"  - Format: uint8 [0, 255]")
    print(f"  - Files: reset_*_frame_*.png")

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

    # Create minimal evaluation report (backward compatibility)
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
