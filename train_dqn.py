#!/usr/bin/env python
"""
DQN Training Entry Point

Train Deep Q-Network (DQN) on Atari games.

Usage:
    python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42
    python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \\
        --set training.lr=0.0005
    python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml \\
        --resume experiments/dqn_atari/runs/pong_42/checkpoints/checkpoint_1000000.pt

For full documentation, see docs/design/config_cli.md
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.config.cli import main
from src.config.run_manager import setup_run_directory, print_run_info


if __name__ == '__main__':
    # Load and validate configuration
    config = main()

    # Setup run directory and save config/metadata
    paths = setup_run_directory(config)
    print_run_info(paths)

    # TODO: Implement training loop
    # For now, just confirm setup completed successfully
    print("✓ Run directory created and configuration saved")
    print(f"✓ Config snapshot: {paths['config_file']}")
    print(f"✓ Metadata: {paths['meta_file']}")
    print("\nTraining loop not yet implemented (will be added in future subtasks)")
    print("Run directory is ready for training.")
