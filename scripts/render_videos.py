#!/usr/bin/env python3
"""
Render evaluation videos from existing DQN checkpoints.

Use this script to re-generate videos for completed training runs without
re-training. Useful for:
- Generating videos for runs that had video recording disabled
- Re-rendering videos at higher quality or different frame rates
- Creating videos for specific checkpoints

Usage:
    # Render video from best model
    python scripts/render_videos.py experiments/dqn_atari/runs/pong_42_20251116/ --best

    # Render videos from specific checkpoints
    python scripts/render_videos.py experiments/dqn_atari/runs/pong_42_20251116/ \
        --checkpoints checkpoint_250000.pt checkpoint_500000.pt

    # Render from all available checkpoints
    python scripts/render_videos.py experiments/dqn_atari/runs/pong_42_20251116/ --all
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import yaml
from omegaconf import OmegaConf

from src.envs import make_atari_env
from src.models import DQN
from src.training import evaluate


def load_config_from_run(run_dir: Path) -> OmegaConf:
    """Load config.yaml from run directory."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return OmegaConf.create(config_dict)


def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def setup_device() -> torch.device:
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def render_video_from_checkpoint(
    checkpoint_path: Path,
    config: OmegaConf,
    output_dir: Path,
    device: torch.device,
    num_episodes: int = 1,
    eval_epsilon: float = 0.05,
    video_fps: int = 30,
    export_gif: bool = False,
):
    """
    Load checkpoint and render evaluation video.

    Args:
        checkpoint_path: Path to .pt checkpoint file
        config: OmegaConf config object
        output_dir: Directory to save videos
        device: Compute device
        num_episodes: Number of episodes to record
        eval_epsilon: Epsilon for evaluation (0.05 default)
        video_fps: Video frame rate
        export_gif: Also export as GIF

    Returns:
        Dictionary with video info
    """
    # Extract step from checkpoint filename
    checkpoint_name = checkpoint_path.stem
    if checkpoint_name == "best_model":
        step = "best"
    else:
        # Parse step from filename like "checkpoint_250000"
        try:
            step = int(checkpoint_name.split("_")[-1])
        except ValueError:
            step = checkpoint_name

    print(f"\nRendering video from: {checkpoint_path.name}")
    print(f"  Step: {step}")

    # Create evaluation environment
    env = make_atari_env(
        env_id=config.environment.env_id,
        num_stack=config.environment.preprocessing.frame_stack,
        frame_skip=config.environment.action_repeat,
        noop_max=config.environment.episode.noop_max,
        episode_life=False,  # Full episodes for evaluation
        clip_rewards=config.environment.preprocessing.clip_rewards,
        render_mode="rgb_array",
    )

    # Get number of actions
    num_actions = env.action_space.n

    # Create model
    model = DQN(num_actions=num_actions).to(device)

    # Load checkpoint
    checkpoint_data = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint_data["online_model_state_dict"])
    model.eval()

    print("  Model loaded successfully")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluation with video recording
    eval_results = evaluate(
        env=env,
        model=model,
        num_episodes=num_episodes,
        eval_epsilon=eval_epsilon,
        device=device,
        step=step,
        record_video=True,
        video_dir=str(output_dir),
        video_fps=video_fps,
        export_gif=export_gif,
    )

    # Close environment
    env.close()

    # Print results
    print(f"  Mean Return: {eval_results['mean_return']:.2f} +/- {eval_results['std_return']:.2f}")
    if "video_info" in eval_results and eval_results["video_info"]:
        print(f"  Video saved: {eval_results['video_info'].get('video_path', 'N/A')}")
        if export_gif and "gif_path" in eval_results["video_info"]:
            print(f"  GIF saved: {eval_results['video_info']['gif_path']}")

    return eval_results


def main():
    parser = argparse.ArgumentParser(
        description="Render evaluation videos from DQN checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Render from best model
  python scripts/render_videos.py runs/pong_42/ --best

  # Render from specific checkpoints
  python scripts/render_videos.py runs/pong_42/ \\
    --checkpoints checkpoint_250000.pt checkpoint_500000.pt

  # Render from all checkpoints
  python scripts/render_videos.py runs/pong_42/ --all

  # Custom settings
  python scripts/render_videos.py runs/pong_42/ --best \\
    --episodes 3 --fps 60 --gif
        """,
    )

    parser.add_argument("run_dir", type=Path, help="Path to training run directory")

    # Checkpoint selection
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--best", action="store_true", help="Render video from best_model.pt"
    )
    checkpoint_group.add_argument(
        "--checkpoints",
        nargs="+",
        type=str,
        metavar="FILE",
        help="Specific checkpoint files to render (e.g., checkpoint_250000.pt)",
    )
    checkpoint_group.add_argument(
        "--all", action="store_true", help="Render from all available checkpoints"
    )

    # Video settings
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for videos (default: run_dir/videos)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Number of episodes to record per checkpoint (default: 1)",
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.05, help="Evaluation epsilon (default: 0.05)"
    )
    parser.add_argument(
        "--fps", type=int, default=30, help="Video frame rate (default: 30)"
    )
    parser.add_argument(
        "--gif", action="store_true", help="Also export as GIF (requires imageio)"
    )

    # Device settings
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu", "mps"],
        default="auto",
        help="Compute device (default: auto)",
    )

    args = parser.parse_args()

    # Validate run directory
    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}")
        sys.exit(1)

    # Load config
    print(f"Loading config from: {args.run_dir}")
    config = load_config_from_run(args.run_dir)

    # Setup device
    if args.device == "auto":
        device = setup_device()
    else:
        device = torch.device(args.device)
        print(f"Using device: {device}")

    # Determine output directory
    if args.output is None:
        output_dir = args.run_dir / "videos"
    else:
        output_dir = args.output

    # Find checkpoints to render
    checkpoint_dir = args.run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoints directory not found: {checkpoint_dir}")
        sys.exit(1)

    checkpoints_to_render = []

    if args.best:
        best_path = checkpoint_dir / "best_model.pt"
        if best_path.exists():
            checkpoints_to_render.append(best_path)
        else:
            print(f"Error: Best model not found: {best_path}")
            sys.exit(1)

    elif args.checkpoints:
        for cp_name in args.checkpoints:
            cp_path = checkpoint_dir / cp_name
            if cp_path.exists():
                checkpoints_to_render.append(cp_path)
            else:
                print(f"Warning: Checkpoint not found: {cp_path}")

        if not checkpoints_to_render:
            print("Error: No valid checkpoints found")
            sys.exit(1)

    elif args.all:
        # Find all checkpoint files
        checkpoints_to_render = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
        if not checkpoints_to_render:
            print("Error: No checkpoint files found")
            sys.exit(1)

    print(f"\nWill render {len(checkpoints_to_render)} video(s)")
    print(f"Output directory: {output_dir}")

    # Render videos
    all_results = []
    for checkpoint_path in checkpoints_to_render:
        results = render_video_from_checkpoint(
            checkpoint_path=checkpoint_path,
            config=config,
            output_dir=output_dir,
            device=device,
            num_episodes=args.episodes,
            eval_epsilon=args.epsilon,
            video_fps=args.fps,
            export_gif=args.gif,
        )
        all_results.append(results)

    print(f"\n{'='*60}")
    print("Video rendering complete!")
    print(f"{'='*60}")
    print(f"Total videos rendered: {len(all_results)}")
    print(f"Output directory: {output_dir}")

    # List generated files
    if output_dir.exists():
        videos = list(output_dir.glob("*.mp4")) + list(output_dir.glob("*.gif"))
        if videos:
            print("\nGenerated files:")
            for v in sorted(videos)[-10:]:  # Show last 10
                print(f"  - {v.name}")
            if len(videos) > 10:
                print(f"  ... and {len(videos) - 10} more")


if __name__ == "__main__":
    main()
