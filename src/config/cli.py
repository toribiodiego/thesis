"""Command-line interface for DQN training.

Provides argument parsing and configuration loading for training experiments.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from .config_loader import (
    load_config,
    merge_cli_overrides,
    print_config,
    validate_config_exists
)


def create_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for DQN training CLI.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description='Train DQN on Atari games',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42
  
  # With inline overrides
  python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 \\
    --set training.lr=0.0005 training.gamma=0.95
  
  # Resume from checkpoint
  python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml \\
    --resume experiments/dqn_atari/runs/pong_42/checkpoints/checkpoint_1000000.pt
  
  # Dry run (print config without training)
  python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 42 --dry-run
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--cfg', '--config',
        type=str,
        required=True,
        metavar='PATH',
        dest='config',
        help='Path to configuration file (YAML)'
    )
    
    # Optional arguments
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        metavar='N',
        help='Random seed for reproducibility (overrides config)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to checkpoint file to resume training from'
    )
    
    parser.add_argument(
        '--set',
        type=str,
        nargs='+',
        default=[],
        metavar='KEY=VALUE',
        dest='overrides',
        help='Override config values using dot notation (e.g., training.lr=0.001)'
    )
    
    # Operational flags
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Load and print config without starting training'
    )
    
    parser.add_argument(
        '--print-config',
        action='store_true',
        help='Print resolved configuration and exit'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        choices=['cuda', 'cpu', 'mps'],
        help='Force specific device (cuda/cpu/mps, overrides config)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress non-essential output'
    )
    
    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Args:
        args: List of argument strings (None = use sys.argv)
        
    Returns:
        Parsed arguments namespace
    """
    parser = create_parser()
    return parser.parse_args(args)


def load_config_from_args(
    args: argparse.Namespace,
    print_resolved: bool = True
) -> Dict[str, Any]:
    """
    Load configuration from parsed CLI arguments.
    
    Handles:
    - Loading base config file
    - Applying --seed override
    - Applying --device override
    - Applying --set overrides
    - Printing resolved config if requested
    
    Args:
        args: Parsed command-line arguments
        print_resolved: Whether to print resolved config
        
    Returns:
        Fully resolved configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    # Validate config file exists
    validate_config_exists(args.config)
    
    # Load config with automatic base merging
    config = load_config(args.config)
    
    # Apply seed override if provided
    if args.seed is not None:
        if 'seed' not in config:
            config['seed'] = {}
        config['seed']['value'] = args.seed
    
    # Apply device override if provided
    if args.device is not None:
        if 'network' not in config:
            config['network'] = {}
        config['network']['device'] = args.device
    
    # Apply --set overrides
    if args.overrides:
        config = merge_cli_overrides(config, args.overrides)
    
    # Print resolved config if requested
    if print_resolved and not args.quiet:
        if args.dry_run or args.print_config:
            print_config(config, title='Resolved Configuration')
        elif args.verbose:
            print_config(config, title='Training Configuration')
    
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate loaded configuration has required fields.
    
    Args:
        config: Configuration dictionary
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = [
        ('environment', 'env_id'),
        ('training', 'total_frames'),
        ('training', 'optimizer', 'lr'),
        ('network', 'architecture'),
        ('replay', 'capacity'),
    ]
    
    errors = []
    
    for field_path in required_fields:
        current = config
        for key in field_path:
            if not isinstance(current, dict) or key not in current:
                errors.append(f"Missing required field: {'.'.join(field_path)}")
                break
            current = current[key]
    
    if errors:
        raise ValueError(
            "Configuration validation failed:\n" +
            "\n".join(f"  - {err}" for err in errors)
        )


def setup_from_args(
    args: Optional[argparse.Namespace] = None,
    argv: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Complete setup: parse args, load config, validate.
    
    This is the main entry point for CLI-based training setup.
    
    Args:
        args: Pre-parsed arguments (if None, will parse from argv)
        argv: Argument list (if None, uses sys.argv)
        
    Returns:
        Validated configuration dictionary
        
    Example:
        >>> # In train_dqn.py
        >>> from src.config.cli import setup_from_args
        >>> config = setup_from_args()
        >>> # Now use config for training
    """
    # Parse arguments if not provided
    if args is None:
        args = parse_args(argv)
    
    # Load config from arguments
    config = load_config_from_args(args, print_resolved=True)
    
    # Validate configuration
    validate_config(config)
    
    # Add CLI metadata to config
    if 'cli' not in config:
        config['cli'] = {}
    
    config['cli']['args'] = {
        'config_file': args.config,
        'seed': args.seed,
        'resume': args.resume,
        'overrides': args.overrides,
        'dry_run': args.dry_run,
        'device': args.device
    }
    
    return config


def print_startup_banner(config: Dict[str, Any]) -> None:
    """
    Print startup banner with key configuration details.
    
    Args:
        config: Configuration dictionary
    """
    env_id = config.get('environment', {}).get('env_id', 'Unknown')
    total_frames = config.get('training', {}).get('total_frames', 0)
    seed = config.get('seed', {}).get('value', 'Random')
    device = config.get('network', {}).get('device', 'auto')
    
    print("=" * 80)
    print("DQN Training".center(80))
    print("=" * 80)
    print(f"  Environment:   {env_id}")
    print(f"  Total Frames:  {total_frames:,}")
    print(f"  Seed:          {seed}")
    print(f"  Device:        {device}")
    print("=" * 80)
    print()


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Main CLI entry point.
    
    Args:
        argv: Command-line arguments (None = use sys.argv)
        
    Returns:
        Loaded configuration dictionary
        
    Raises:
        SystemExit: If --print-config or --dry-run specified
    """
    # Parse arguments
    args = parse_args(argv)
    
    try:
        # Setup configuration
        config = setup_from_args(args)
        
        # Handle special modes
        if args.print_config:
            print("\n✓ Configuration loaded successfully")
            sys.exit(0)
        
        if args.dry_run:
            print("\n✓ Dry run complete - configuration validated")
            sys.exit(0)
        
        # Print startup banner
        if not args.quiet:
            print_startup_banner(config)
        
        return config
        
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
