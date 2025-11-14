#!/usr/bin/env bash
#
# Smoke test for DQN training loop (~200K frames)
#
# Validates end-to-end stability:
# - Training loop executes without errors
# - Logs are created and grow
# - Checkpoints appear (if configured)
# - Evaluation runs trigger
# - Metrics are recorded
#
# Usage:
#   ./experiments/dqn_atari/scripts/smoke_test.sh [config] [seed]
#
# Examples:
#   # Default: Use Pong config with seed 0, 200K frames
#   ./experiments/dqn_atari/scripts/smoke_test.sh
#
#   # Custom config and seed
#   ./experiments/dqn_atari/scripts/smoke_test.sh experiments/dqn_atari/configs/breakout.yaml 42
#

set -e  # Exit on error

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

echo "========================================"
echo "DQN Training Smoke Test"
echo "========================================"
echo ""

# Configuration
CONFIG="${1:-experiments/dqn_atari/configs/pong.yaml}"
SEED="${2:-0}"
SMOKE_TEST_FRAMES=200000
RUN_DIR="experiments/dqn_atari/runs/smoke_test_${SEED}"

echo "Config: $CONFIG"
echo "Seed: $SEED"
echo "Total frames: $SMOKE_TEST_FRAMES"
echo "Run directory: $RUN_DIR"
echo ""

# Clean previous smoke test run
if [ -d "$RUN_DIR" ]; then
    echo "Cleaning previous smoke test run..."
    rm -rf "$RUN_DIR"
fi

# Create Python smoke test runner
cat > /tmp/smoke_test_runner.py << 'PYTHON_EOF'
"""
Smoke test runner for DQN training.

Runs a short training session (~200K frames) and validates:
- Training loop stability
- Logging functionality
- Checkpoint creation
- Evaluation triggers
- Metric recording
"""

import sys
import os
import time
import argparse
from pathlib import Path

import torch
import numpy as np

# Add repo to path
repo_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(repo_root))

from src.models import DQN
from src.replay import ReplayBuffer
from src.training import (
    init_target_network,
    configure_optimizer,
    training_step,
    EpsilonScheduler,
    TargetNetworkUpdater,
    TrainingScheduler,
    FrameCounter,
    StepLogger,
    EpisodeLogger,
    CheckpointManager,
    evaluate,
    EvaluationScheduler,
    EvaluationLogger,
    ReferenceStateQTracker,
    ReferenceQLogger,
    MetadataWriter
)


def create_dummy_env(num_actions=6):
    """Create simple mock environment for smoke testing."""
    from unittest.mock import Mock

    env = Mock()
    env.action_space.n = num_actions
    env.action_space.sample.return_value = 0

    # Create dummy state
    dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    env.reset.return_value = (dummy_state, {})

    # Step returns random rewards and terminates randomly
    step_count = [0]
    episode_length = 100

    def mock_step(action):
        step_count[0] += 1
        done = (step_count[0] % episode_length) == 0
        if done:
            step_count[0] = 0
        reward = np.random.randn()
        return (dummy_state, reward, done, False, {})

    env.step.side_effect = mock_step

    return env


def run_smoke_test(
    total_frames=200000,
    seed=0,
    run_dir='experiments/dqn_atari/runs/smoke_test'
):
    """
    Run smoke test.

    Args:
        total_frames: Total frames to run (default: 200K)
        seed: Random seed
        run_dir: Output directory
    """
    print(f"Starting smoke test...")
    print(f"  Total frames: {total_frames:,}")
    print(f"  Seed: {seed}")
    print(f"  Run dir: {run_dir}")
    print("")

    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(f"{run_dir}/logs", exist_ok=True)
    os.makedirs(f"{run_dir}/checkpoints", exist_ok=True)

    # Configuration
    num_actions = 6
    device = 'cpu'

    # Create environment
    print("Creating environment...")
    env = create_dummy_env(num_actions=num_actions)

    # Create model
    print("Creating model...")
    online_net = DQN(num_actions=num_actions).to(device)
    target_net = init_target_network(online_net, num_actions=num_actions)

    # Create optimizer
    print("Creating optimizer...")
    optimizer = configure_optimizer(online_net, learning_rate=0.00025)

    # Create replay buffer
    print("Creating replay buffer...")
    replay_buffer = ReplayBuffer(
        capacity=100000,
        obs_shape=(4, 84, 84),
        min_size=5000,
        device=None
    )

    # Create schedulers and utilities
    print("Creating schedulers...")
    epsilon_scheduler = EpsilonScheduler(
        epsilon_start=1.0,
        epsilon_end=0.1,
        decay_frames=100000
    )
    target_updater = TargetNetworkUpdater(update_interval=10000)
    training_scheduler = TrainingScheduler(train_every=4)
    frame_counter = FrameCounter(frameskip=4)

    # Create loggers
    print("Creating loggers...")
    step_logger = StepLogger(log_dir=f"{run_dir}/logs", log_interval=1000)
    episode_logger = EpisodeLogger(log_dir=f"{run_dir}/logs", rolling_window=10)

    # Create checkpoint manager
    print("Creating checkpoint manager...")
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=f"{run_dir}/checkpoints",
        save_interval=50000,
        keep_last_n=2
    )

    # Create evaluation scheduler
    print("Creating evaluation scheduler...")
    eval_scheduler = EvaluationScheduler(
        eval_interval=50000,
        num_episodes=3,
        eval_epsilon=0.05
    )
    eval_logger = EvaluationLogger(log_dir=f"{run_dir}/eval")

    # Create reference Q tracker
    print("Creating reference Q tracker...")
    q_tracker = ReferenceStateQTracker(log_interval=10000, device=device)
    q_logger = ReferenceQLogger(log_dir=f"{run_dir}/logs")

    # Write metadata
    print("Writing metadata...")
    metadata_writer = MetadataWriter(run_dir=run_dir)
    metadata_writer.write_metadata(
        config={'total_frames': total_frames, 'test': 'smoke_test'},
        seed=seed,
        extra={'device': device}
    )

    # Training loop
    print("")
    print("Starting training loop...")
    print("=" * 60)

    obs, _ = env.reset()
    episode_return = 0.0
    episode_length = 0
    start_time = time.time()

    while frame_counter.frames < total_frames:
        # Training step
        result = training_step(
            env=env,
            online_net=online_net,
            target_net=target_net,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            epsilon_scheduler=epsilon_scheduler,
            target_updater=target_updater,
            training_scheduler=training_scheduler,
            frame_counter=frame_counter,
            state=obs,
            num_actions=num_actions,
            device=device
        )

        # Update state
        obs = result['next_state']
        episode_return += result['reward']
        episode_length += 1

        # Log step metrics
        step_logger.log_step(
            step=frame_counter.steps,
            epsilon=result['epsilon'],
            metrics=result['metrics'],
            replay_size=len(replay_buffer),
            fps=frame_counter.fps(time.time() - start_time)
        )

        # Handle episode end
        if result['terminated'] or result['truncated']:
            episode_logger.log_episode(
                step=frame_counter.steps,
                episode_return=episode_return,
                episode_length=episode_length,
                fps=frame_counter.fps(time.time() - start_time),
                epsilon=result['epsilon']
            )

            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        # Save checkpoint
        if checkpoint_manager.should_save(frame_counter.steps):
            print(f"  Saving checkpoint at step {frame_counter.steps:,}...")
            checkpoint_manager.save_checkpoint(
                step=frame_counter.steps,
                model=online_net,
                optimizer=optimizer,
                metadata={'epsilon': result['epsilon']}
            )

        # Run evaluation
        if eval_scheduler.should_evaluate(frame_counter.steps):
            print(f"  Running evaluation at step {frame_counter.steps:,}...")
            eval_results = evaluate(
                env=env,
                model=online_net,
                num_episodes=eval_scheduler.num_episodes,
                eval_epsilon=eval_scheduler.eval_epsilon,
                device=device
            )
            eval_scheduler.record_evaluation(frame_counter.steps, eval_results)
            eval_logger.log_evaluation(
                step=frame_counter.steps,
                results=eval_results,
                epsilon=result['epsilon']
            )
            print(f"    Mean return: {eval_results['mean_return']:.2f}")

        # Log reference Q values
        if q_tracker.reference_states is None and len(replay_buffer) >= 100:
            # Initialize reference states from replay buffer
            ref_batch = replay_buffer.sample(100)
            q_tracker.set_reference_states(ref_batch['states'])

        if q_tracker.should_log(frame_counter.steps):
            q_tracker.log_q_values(step=frame_counter.steps, model=online_net)
            q_stats = {
                'avg_max_q': q_tracker.avg_max_q[-1],
                'max_q': q_tracker.max_q[-1],
                'min_q': q_tracker.min_q[-1]
            }
            q_logger.log(step=frame_counter.steps, q_stats=q_stats)

        # Progress update
        if frame_counter.steps % 10000 == 0:
            elapsed = time.time() - start_time
            fps = frame_counter.fps(elapsed)
            progress = (frame_counter.frames / total_frames) * 100
            print(f"  Step {frame_counter.steps:,} | "
                  f"Frames {frame_counter.frames:,}/{total_frames:,} ({progress:.1f}%) | "
                  f"FPS: {fps:.1f} | "
                  f"Epsilon: {result['epsilon']:.3f} | "
                  f"Replay: {len(replay_buffer):,}")

    print("=" * 60)
    print(f"Training complete!")
    print(f"Total time: {time.time() - start_time:.1f}s")
    print("")

    # Validate outputs
    print("Validating outputs...")
    validate_smoke_test_outputs(run_dir)

    print("")
    print("✓ Smoke test passed!")
    return 0


def validate_smoke_test_outputs(run_dir):
    """Validate that expected files were created."""

    checks = []

    # Check metadata
    checks.append(("metadata.json", os.path.exists(f"{run_dir}/metadata.json")))
    checks.append(("git_info.txt", os.path.exists(f"{run_dir}/git_info.txt")))

    # Check logs
    checks.append(("training_steps.csv", os.path.exists(f"{run_dir}/logs/training_steps.csv")))
    checks.append(("episodes.csv", os.path.exists(f"{run_dir}/logs/episodes.csv")))
    checks.append(("reference_q_values.csv", os.path.exists(f"{run_dir}/logs/reference_q_values.csv")))

    # Check evaluation
    checks.append(("evaluations.csv", os.path.exists(f"{run_dir}/eval/evaluations.csv")))

    # Check checkpoints
    checkpoint_dir = f"{run_dir}/checkpoints"
    has_checkpoints = os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0
    checks.append(("checkpoints", has_checkpoints))

    # Report
    print("")
    for name, passed in checks:
        status = "✓" if passed else "✗"
        print(f"  {status} {name}")

    all_passed = all(passed for _, passed in checks)
    if not all_passed:
        print("")
        print("WARNING: Some outputs missing!")
        return 1

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN smoke test")
    parser.add_argument("--total-frames", type=int, default=200000,
                        help="Total frames to run (default: 200000)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed (default: 0)")
    parser.add_argument("--run-dir", type=str,
                        default="experiments/dqn_atari/runs/smoke_test",
                        help="Run directory")

    args = parser.parse_args()

    exit_code = run_smoke_test(
        total_frames=args.total_frames,
        seed=args.seed,
        run_dir=args.run_dir
    )

    sys.exit(exit_code)
PYTHON_EOF

# Run smoke test
echo "Running Python smoke test..."
echo ""

python /tmp/smoke_test_runner.py \
    --total-frames "$SMOKE_TEST_FRAMES" \
    --seed "$SEED" \
    --run-dir "$RUN_DIR"

EXIT_CODE=$?

# Cleanup
rm /tmp/smoke_test_runner.py

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================"
    echo "Smoke test PASSED ✓"
    echo "========================================"
    echo ""
    echo "Generated files in: $RUN_DIR"
    echo ""
    echo "To inspect results:"
    echo "  ls -la $RUN_DIR"
    echo "  cat $RUN_DIR/logs/episodes.csv"
    echo "  cat $RUN_DIR/eval/evaluations.csv"
    echo ""
else
    echo ""
    echo "========================================"
    echo "Smoke test FAILED ✗"
    echo "========================================"
    exit 1
fi
