#!/usr/bin/env python
"""
DQN Training Entry Point

Train Deep Q-Network (DQN) on Atari games.

Usage:
    python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42
    python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \
        --set training.optimizer.lr=0.0005
    python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml \
        --resume experiments/dqn_atari/runs/pong_42/checkpoints/checkpoint_1000000.pt

For full documentation, see docs/design/config_cli.md
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from omegaconf import OmegaConf

from src.config.cli import main
from src.config.run_manager import setup_run_directory, print_run_info
from src.envs import make_atari_env
from src.models import DQN
from src.replay import ReplayBuffer
from src.training import (
    init_target_network,
    configure_optimizer,
    EpsilonScheduler,
    TargetNetworkUpdater,
    TrainingScheduler,
    FrameCounter,
    training_step,
    CheckpointManager,
    MetricsLogger,
    EvaluationScheduler,
    evaluate,
    get_rng_states,
    resume_from_checkpoint
)


def set_random_seeds(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_device(config):
    """
    Setup compute device with automatic fallback.

    Checks availability in order: CUDA > MPS > CPU
    If config specifies a device, attempts to use it with fallback to CPU.
    """
    requested_device = config.network.device

    # Auto-detect best available device
    if requested_device == 'auto' or requested_device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            return device
        elif requested_device == 'cuda':
            print("Warning: CUDA requested but not available. Falling back to CPU.")

    if requested_device == 'auto' or requested_device == 'mps':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Using MPS device (Apple Silicon GPU)")
            return device
        elif requested_device == 'mps':
            print("Warning: MPS requested but not available. Falling back to CPU.")

    # Default to CPU
    device = torch.device('cpu')
    print("Using CPU device")
    return device


def initialize_components(config, paths, device, resuming=False):
    """Initialize all training components."""
    # Create environment
    env = make_atari_env(
        env_id=config.environment.env_id,
        num_stack=config.environment.preprocessing.frame_stack,
        frame_skip=config.environment.action_repeat,
        noop_max=config.environment.episode.noop_max,
        episode_life=config.environment.episode.episodic_life,
        clip_rewards=config.environment.preprocessing.clip_rewards
    )

    # Set seed
    if config.seed.value is not None:
        env.reset(seed=config.seed.value)

    num_actions = env.action_space.n

    # Create evaluation environment (no episodic life, for true episode returns)
    eval_env = make_atari_env(
        env_id=config.environment.env_id,
        num_stack=config.environment.preprocessing.frame_stack,
        frame_skip=config.environment.action_repeat,
        noop_max=config.environment.episode.noop_max,
        episode_life=False,  # Full episodes for evaluation
        clip_rewards=config.environment.preprocessing.clip_rewards
    )

    # Set different seed for eval
    if config.seed.value is not None:
        eval_env.reset(seed=config.seed.value + 1000)

    # Create networks
    online_net = DQN(num_actions=num_actions).to(device)
    target_net = init_target_network(online_net, num_actions=num_actions).to(device)

    # Create optimizer
    optimizer = configure_optimizer(
        network=online_net,
        optimizer_type=config.training.optimizer.type,
        learning_rate=config.training.optimizer.lr,
        alpha=config.training.optimizer.rmsprop.alpha,
        eps=config.training.optimizer.rmsprop.eps,
        momentum=config.training.optimizer.rmsprop.momentum
    )

    # Create replay buffer
    replay_buffer = ReplayBuffer(
        capacity=config.replay.capacity,
        obs_shape=(config.environment.preprocessing.frame_stack, 84, 84),
        min_size=config.replay.min_size,
        device=str(device) if device is not None else None
    )

    # Create schedulers
    epsilon_scheduler = EpsilonScheduler(
        epsilon_start=config.exploration.schedule.start_epsilon,
        epsilon_end=config.exploration.schedule.end_epsilon,
        decay_frames=config.exploration.schedule.decay_frames
    )

    target_updater = TargetNetworkUpdater(
        update_interval=config.target_network.update_interval
    )

    training_scheduler = TrainingScheduler(
        train_every=config.training.train_every
    )

    frame_counter = FrameCounter(frameskip=config.environment.action_repeat)

    # Create loggers
    metrics_logger = MetricsLogger(
        log_dir=paths['run_dir'],
        enable_tensorboard=config.logging.get('tensorboard', {}).get('enabled', True),
        enable_wandb=config.logging.get('wandb', {}).get('enabled', False),
        enable_csv=config.logging.get('csv', {}).get('enabled', True),
        wandb_project=config.logging.get('wandb', {}).get('project', 'dqn-atari'),
        wandb_name=paths['run_dir'].name,
        wandb_config=OmegaConf.to_container(config, resolve=True),
        wandb_tags=config.logging.get('wandb', {}).get('tags', []),
        upload_artifacts=config.logging.get('wandb', {}).get('upload_artifacts', False)
    )

    checkpoint_manager = CheckpointManager(
        checkpoint_dir=paths['checkpoints'],
        save_interval=config.logging.checkpoint.save_every,
        keep_last_n=config.logging.checkpoint.keep_last_n,
        save_best=config.logging.checkpoint.save_best
    )

    eval_scheduler = EvaluationScheduler(
        eval_interval=config.evaluation.eval_every,
        num_episodes=config.evaluation.num_episodes,
        eval_epsilon=config.evaluation.epsilon
    )

    return {
        'env': env,
        'eval_env': eval_env,
        'num_actions': num_actions,
        'online_net': online_net,
        'target_net': target_net,
        'optimizer': optimizer,
        'replay_buffer': replay_buffer,
        'epsilon_scheduler': epsilon_scheduler,
        'target_updater': target_updater,
        'training_scheduler': training_scheduler,
        'frame_counter': frame_counter,
        'metrics_logger': metrics_logger,
        'checkpoint_manager': checkpoint_manager,
        'eval_scheduler': eval_scheduler
    }


def run_training(config, paths, device):
    """Main training loop."""
    # Set random seeds
    if config.seed.value is not None:
        set_random_seeds(config.seed.value)
        print(f"Random seed set to: {config.seed.value}")

    # Initialize components
    components = initialize_components(config, paths, device)

    env = components['env']
    eval_env = components['eval_env']
    num_actions = components['num_actions']
    online_net = components['online_net']
    target_net = components['target_net']
    optimizer = components['optimizer']
    replay_buffer = components['replay_buffer']
    epsilon_scheduler = components['epsilon_scheduler']
    target_updater = components['target_updater']
    training_scheduler = components['training_scheduler']
    frame_counter = components['frame_counter']
    metrics_logger = components['metrics_logger']
    checkpoint_manager = components['checkpoint_manager']
    eval_scheduler = components['eval_scheduler']

    # Training state
    episode_count = 0
    episode_return = 0.0
    episode_length = 0

    # Reset environment
    state, _ = env.reset()

    # Main training loop
    print("\n" + "="*80)
    print("Starting DQN Training")
    print("="*80)
    print(f"Total frames: {config.training.total_frames:,}")
    print(f"Replay capacity: {config.replay.capacity:,}")
    print(f"Warmup steps: {config.replay.warmup_steps:,}")
    print(f"Train every: {config.training.train_every} steps")
    print(f"Target update: every {config.target_network.update_interval:,} steps")
    print(f"Evaluation: every {config.evaluation.eval_every:,} frames")
    print("="*80 + "\n")

    start_time = time.time()
    last_log_time = start_time

    while frame_counter.frames < config.training.total_frames:
        # Execute training step
        step_result = training_step(
            env=env,
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            epsilon_scheduler=epsilon_scheduler,
            target_updater=target_updater,
            training_scheduler=training_scheduler,
            frame_counter=frame_counter,
            state=state,
            num_actions=num_actions,
            gamma=config.training.gamma,
            loss_type=config.training.loss.type,
            max_grad_norm=config.training.gradient_clip.max_norm,
            batch_size=config.replay.batch_size,
            device=device
        )

        # Update episode tracking
        episode_return += step_result['reward']
        episode_length += 1
        state = step_result['next_state']

        # Log step metrics
        if frame_counter.steps % config.logging.log_every_steps == 0:
            current_time = time.time()
            elapsed = current_time - last_log_time
            fps = (config.logging.log_every_steps * config.environment.action_repeat) / elapsed if elapsed > 0 else 0

            metrics_logger.log_step(
                step=frame_counter.frames,
                epsilon=step_result['epsilon'],
                replay_size=replay_buffer.size,
                fps=fps,
                loss=step_result['metrics'].loss if step_result['metrics'] else None,
                td_error=step_result['metrics'].td_error if step_result['metrics'] else None,
                grad_norm=step_result['metrics'].grad_norm if step_result['metrics'] else None,
                learning_rate=step_result['metrics'].learning_rate if step_result['metrics'] else None
            )

            last_log_time = current_time

            # Print progress
            if frame_counter.steps % (config.logging.log_every_steps * 10) == 0:
                progress = (frame_counter.frames / config.training.total_frames) * 100
                print(f"[{progress:5.1f}%] Frame {frame_counter.frames:>10,} | "
                      f"Episode {episode_count:>5} | "
                      f"ε={step_result['epsilon']:.3f} | "
                      f"FPS={fps:>6.0f} | "
                      f"Buffer={replay_buffer.size:>7,}")

        # Handle episode end
        if step_result['terminated'] or step_result['truncated']:
            episode_count += 1

            # Log episode metrics
            metrics_logger.log_episode(
                step=frame_counter.frames,
                episode=episode_count,
                episode_return=episode_return,
                episode_length=episode_length,
                epsilon=step_result['epsilon']
            )

            # Reset episode tracking
            episode_return = 0.0
            episode_length = 0
            state, _ = env.reset()

        # Periodic evaluation
        if eval_scheduler.should_evaluate(frame_counter.frames):
            print(f"\n{'='*80}")
            print(f"Evaluation at frame {frame_counter.frames:,}")
            print(f"{'='*80}")

            eval_results = evaluate(
                env=eval_env,
                model=online_net,
                num_episodes=config.evaluation.num_episodes,
                eval_epsilon=config.evaluation.epsilon,
                device=device,
                step=frame_counter.frames,
                record_video=config.evaluation.get('record_video', False),
                video_dir=str(paths['videos']) if 'videos' in paths else None,
                video_fps=30,
                export_gif=False
            )

            mean_return = np.mean(eval_results['episode_returns'])
            median_return = np.median(eval_results['episode_returns'])
            std_return = np.std(eval_results['episode_returns'])
            min_return = np.min(eval_results['episode_returns'])
            max_return = np.max(eval_results['episode_returns'])
            mean_length = np.mean(eval_results['episode_lengths'])

            print(f"Evaluation Results:")
            print(f"  Mean Return: {mean_return:.2f} +/- {std_return:.2f}")
            print(f"  Median Return: {median_return:.2f}")
            print(f"  Min/Max Return: {min_return:.2f} / {max_return:.2f}")
            print(f"  Mean Length: {mean_length:.1f}")
            print(f"  Episodes: {len(eval_results['episode_returns'])}")
            if 'video_info' in eval_results and eval_results['video_info']:
                print(f"  Video saved: {eval_results['video_info'].get('video_path', 'N/A')}")
            print(f"{'='*80}\n")

            # Save evaluation results to CSV/JSONL
            eval_csv_path = paths['eval'] / 'evaluations.csv'
            eval_json_path = paths['eval'] / 'evaluations.jsonl'

            # Write CSV header if file doesn't exist
            if not eval_csv_path.exists():
                with open(eval_csv_path, 'w') as f:
                    f.write('step,mean_return,median_return,std_return,min_return,max_return,mean_length,num_episodes,eval_epsilon\n')

            # Append evaluation results to CSV
            with open(eval_csv_path, 'a') as f:
                f.write(f"{frame_counter.frames},{mean_return:.4f},{median_return:.4f},{std_return:.4f},{min_return:.4f},{max_return:.4f},{mean_length:.2f},{config.evaluation.num_episodes},{config.evaluation.epsilon}\n")

            # Append to JSONL (one JSON object per line)
            eval_record = {
                'step': frame_counter.frames,
                'mean_return': float(mean_return),
                'median_return': float(median_return),
                'std_return': float(std_return),
                'min_return': float(min_return),
                'max_return': float(max_return),
                'mean_length': float(mean_length),
                'episode_returns': [float(r) for r in eval_results['episode_returns']],
                'episode_lengths': [int(l) for l in eval_results['episode_lengths']],
                'num_episodes': config.evaluation.num_episodes,
                'eval_epsilon': config.evaluation.epsilon,
                'video_info': eval_results.get('video_info', None)
            }
            with open(eval_json_path, 'a') as f:
                f.write(json.dumps(eval_record) + '\n')

            # Log evaluation metrics
            metrics_logger.log_evaluation(
                step=frame_counter.frames,
                mean_return=mean_return,
                median_return=median_return,
                std_return=std_return,
                min_return=min_return,
                max_return=max_return,
                mean_length=mean_length,
                num_episodes=config.evaluation.num_episodes
            )

            # Save best model
            if checkpoint_manager.save_best_enabled:
                is_new_best = checkpoint_manager.save_best(
                    step=frame_counter.frames,
                    episode=episode_count,
                    epsilon=step_result['epsilon'],
                    eval_return=mean_return,
                    online_model=online_net,
                    target_model=target_net,
                    optimizer=optimizer,
                    replay_buffer=replay_buffer,
                    rng_states=get_rng_states(env)
                )
                if is_new_best:
                    print(f"New best model saved (return: {mean_return:.2f})")

        # Periodic checkpoint
        if checkpoint_manager.should_save(frame_counter.frames):
            checkpoint_path = checkpoint_manager.save_checkpoint(
                step=frame_counter.frames,
                episode=episode_count,
                epsilon=step_result['epsilon'],
                online_model=online_net,
                target_model=target_net,
                optimizer=optimizer,
                replay_buffer=replay_buffer,
                rng_states=get_rng_states(env)
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    # Training complete
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Total frames: {frame_counter.frames:,}")
    print(f"Total episodes: {episode_count}")
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Average FPS: {frame_counter.frames/total_time:.1f}")
    print("="*80 + "\n")

    # Final evaluation
    print("Running final evaluation...")
    eval_results = evaluate(
        env=eval_env,
        model=online_net,
        num_episodes=config.evaluation.num_episodes,
        eval_epsilon=config.evaluation.epsilon,
        device=device
    )

    mean_return = np.mean(eval_results['episode_returns'])
    std_return = np.std(eval_results['episode_returns'])

    print(f"Final Evaluation Results:")
    print(f"  Mean Return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"  Episodes: {len(eval_results['episode_returns'])}")

    # Close environments
    env.close()
    eval_env.close()

    # Close logger
    metrics_logger.close()

    print(f"\nResults saved to: {paths['run_dir']}")


if __name__ == '__main__':
    # Load and validate configuration
    config_dict = main()

    # Convert to OmegaConf for easier access
    config = OmegaConf.create(config_dict)

    # Setup run directory and save config/metadata
    paths = setup_run_directory(config_dict)
    print_run_info(paths)

    # Setup device
    device = setup_device(config)

    # Run training
    run_training(config, paths, device)
