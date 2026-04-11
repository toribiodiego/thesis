#!/usr/bin/env python
"""
JAX/BBF Training Entry Point

Train BBF-family agents (BBF, BBFc, SR-SPR, SR-SPRc, SPR, SPRc, DER, DERc)
on Atari games using the ported BBF codebase.

Usage:
    python train_bbf.py \
        --base_gin src/bigger_better_faster/bbf/configs/BBF.gin \
        --condition_gin src/bigger_better_faster/bbf/configs/conditions/BBF.gin \
        --game_gin src/bigger_better_faster/bbf/configs/games/boxing.gin \
        --seed 42 \
        --run_dir experiments/dqn_atari/runs/BBF_boxing_42
"""

import argparse
import json
import os
import sys
import time

# Unbuffered stdout for real-time progress in non-TTY environments (Colab, CI)
sys.stdout.reconfigure(line_buffering=True)

# Add src/ to path so BBF package imports work.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import gin
import numpy as np
import tensorflow as tf
from absl import logging
from dopamine.discrete_domains import atari_lib

from bigger_better_faster.bbf.agents.metric_agent import MetricBBFAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train BBF-family agent")
    parser.add_argument("--base_gin", required=True,
                        help="Path to base gin config (e.g., BBF.gin)")
    parser.add_argument("--condition_gin", required=True,
                        help="Path to condition overlay gin file")
    parser.add_argument("--game_gin", required=True,
                        help="Path to game gin file")
    parser.add_argument("--gin_bindings", nargs="*", default=[],
                        help="Extra gin bindings (key=value)")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run_dir", required=True,
                        help="Output directory for this run")
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


def set_seed(seed):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    args = parse_args()
    logging.set_verbosity(logging.INFO)

    # Setup
    set_seed(args.seed)
    setup_gin(args.base_gin, args.condition_gin, args.game_gin,
              args.gin_bindings)
    setup_run_dir(args.run_dir)
    save_config_snapshot(args.run_dir)

    # Create environment and agent
    env = create_environment()
    agent = create_agent(env, args.seed, args.run_dir)

    game_name = gin.query_parameter("DataEfficientAtariRunner.game_name")
    print(f"Training: game={game_name}, seed={args.seed}, "
          f"steps={args.total_steps}, run_dir={args.run_dir}")
    print(f"Agent: spr_weight={agent.spr_weight}, jumps={agent._jumps}, "
          f"replay_ratio={agent._replay_ratio}")

    # Initialize: reset environment and agent
    obs = env.reset()
    agent.reset_all(np.expand_dims(obs, 0))  # (1, 84, 84)

    # Training loop
    step = 0
    episode = 0
    episode_return = 0.0
    episode_length = 0
    start_time = time.time()

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

        # Episode boundary
        if done:
            episode += 1
            elapsed = time.time() - start_time
            fps = step / elapsed if elapsed > 0 else 0
            print(f"Step {step}/{args.total_steps} | "
                  f"Episode {episode} | Return {episode_return:.0f} | "
                  f"Length {episode_length} | FPS {fps:.0f}")

            obs = env.reset()
            agent.reset_one(0)
            agent._record_observation(np.expand_dims(obs, 0))
            episode_return = 0.0
            episode_length = 0

    elapsed = time.time() - start_time
    print(f"Training complete: {step} steps in {elapsed:.1f}s "
          f"({step / elapsed:.0f} FPS)")

    env.close()


if __name__ == "__main__":
    main()
