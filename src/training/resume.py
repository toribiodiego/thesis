"""Resume training from checkpoints with validation and state restoration."""

import os
import warnings
from typing import Dict, Any, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn

from .checkpoint_utils import set_rng_states
from .logging import CheckpointManager


def _get_nested_value(config: Dict[str, Any], path: str) -> Any:
    """
    Get value from nested dict using dot-separated path.

    Args:
        config: Configuration dictionary
        path: Dot-separated path (e.g., 'env.id')

    Returns:
        Value at path, or None if path doesn't exist
    """
    value = config
    for key in path.split('.'):
        if isinstance(value, dict):
            value = value.get(key, None)
        else:
            return None
    return value


def validate_config_compatibility(
    checkpoint_config: Dict[str, Any],
    current_config: Dict[str, Any],
    strict: bool = False
) -> Tuple[bool, list]:
    """
    Validate that checkpoint config is compatible with current config.

    Checks critical hyperparameters that would break training if changed:
    - Network architecture (action space)
    - Replay buffer capacity
    - Observation shape

    Args:
        checkpoint_config: Config from checkpoint metadata
        current_config: Current run configuration
        strict: If True, require exact match of all parameters

    Returns:
        Tuple of (is_compatible, list_of_warnings)
    """
    warnings_list = []
    is_compatible = True

    # Critical parameters that MUST match
    critical_params = {
        'env.id': 'Environment ID',
        'preprocess.frame_size': 'Frame size',
        'preprocess.stack_size': 'Frame stack size',
    }

    for param_path, param_name in critical_params.items():
        checkpoint_val = _get_nested_value(checkpoint_config, param_path)
        current_val = _get_nested_value(current_config, param_path)

        if checkpoint_val != current_val:
            is_compatible = False
            warnings_list.append(
                f"CRITICAL: {param_name} mismatch - "
                f"checkpoint: {checkpoint_val}, current: {current_val}"
            )

    # Important parameters that should match (warnings only)
    important_params = {
        'training.gamma': 'Discount factor',
        'training.learning_rate': 'Learning rate',
        'replay.capacity': 'Replay buffer capacity',
        'training.batch_size': 'Batch size',
        'training.target_update_interval': 'Target update interval',
    }

    for param_path, param_name in important_params.items():
        checkpoint_val = _get_nested_value(checkpoint_config, param_path)
        current_val = _get_nested_value(current_config, param_path)

        if checkpoint_val != current_val and checkpoint_val is not None and current_val is not None:
            warnings_list.append(
                f"WARNING: {param_name} differs - "
                f"checkpoint: {checkpoint_val}, current: {current_val}"
            )

    return is_compatible, warnings_list


def check_git_hash_mismatch(
    checkpoint_hash: str,
    current_hash: str
) -> Optional[str]:
    """
    Check if git commit hashes match.

    Args:
        checkpoint_hash: Git hash from checkpoint
        current_hash: Current git hash

    Returns:
        Warning message if mismatch, None if match
    """
    if checkpoint_hash == 'unknown' or current_hash == 'unknown':
        return "WARNING: Unable to verify git commit hash (not in git repo)"

    if checkpoint_hash != current_hash:
        return (
            f"WARNING: Git commit hash mismatch\n"
            f"  Checkpoint was saved at commit: {checkpoint_hash}\n"
            f"  Current commit: {current_hash}\n"
            f"  Code changes may affect reproducibility"
        )

    return None


def resume_from_checkpoint(
    checkpoint_path: str,
    online_model: nn.Module,
    target_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epsilon_scheduler: Any,
    replay_buffer: Any = None,
    env: Any = None,
    config: Dict[str, Any] = None,
    device: str = 'cpu',
    strict_config: bool = False
) -> Dict[str, Any]:
    """
    Resume training from checkpoint with full validation.

    This function:
    - Loads checkpoint with device-safe tensor mapping
    - Validates config compatibility
    - Warns on git commit hash mismatch
    - Restores all training state (models, optimizer, counters, epsilon, RNG)
    - Optionally restores replay buffer

    Args:
        checkpoint_path: Path to checkpoint file
        online_model: Online Q-network (will be modified in-place)
        target_model: Target Q-network (will be modified in-place)
        optimizer: Optimizer (will be modified in-place)
        epsilon_scheduler: Epsilon scheduler (will be modified in-place)
        replay_buffer: Optional replay buffer to restore
        env: Optional environment for RNG state restoration
        config: Current configuration for validation
        device: Device to map tensors to ('cpu', 'cuda', 'cuda:0', etc.)
        strict_config: If True, require exact config match (raise error on mismatch)

    Returns:
        Dict with resumed state:
            - step: Environment step to resume from
            - episode: Episode counter
            - epsilon: Current epsilon value
            - loaded_metadata: Full checkpoint metadata

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
        ValueError: If config is incompatible and strict_config=True
        RuntimeError: If checkpoint is corrupted

    Example:
        >>> resumed = resume_from_checkpoint(
        ...     checkpoint_path='checkpoints/checkpoint_1000000.pt',
        ...     online_model=online_model,
        ...     target_model=target_model,
        ...     optimizer=optimizer,
        ...     epsilon_scheduler=epsilon_scheduler,
        ...     replay_buffer=replay_buffer,
        ...     env=env,
        ...     config=config,
        ...     device='cuda'
        ... )
        >>> start_step = resumed['step']
        >>> print(f"Resuming from step {start_step}")
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"\n{'='*80}")
    print(f"RESUMING FROM CHECKPOINT")
    print(f"{'='*80}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load checkpoint with device mapping
    manager = CheckpointManager(checkpoint_dir=str(Path(checkpoint_path).parent))

    try:
        loaded_state = manager.load_checkpoint(
            checkpoint_path=checkpoint_path,
            online_model=online_model,
            target_model=target_model,
            optimizer=optimizer,
            replay_buffer=replay_buffer,
            device=device,
            strict=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # Extract state
    step = loaded_state['step']
    episode = loaded_state['episode']
    epsilon_value = loaded_state['epsilon']
    rng_states = loaded_state['rng_states']
    commit_hash = loaded_state['commit_hash']
    timestamp = loaded_state['timestamp']
    checkpoint_metadata = loaded_state.get('metadata', {})

    print(f"\nCheckpoint Info:")
    print(f"  Step: {step:,}")
    print(f"  Episode: {episode:,}")
    print(f"  Epsilon: {epsilon_value:.4f}")
    print(f"  Saved at: {timestamp}")
    print(f"  Commit: {commit_hash}")

    # Validate config compatibility if provided
    if config is not None and checkpoint_metadata.get('config') is not None:
        print(f"\nValidating config compatibility...")

        is_compatible, warnings_list = validate_config_compatibility(
            checkpoint_config=checkpoint_metadata['config'],
            current_config=config,
            strict=strict_config
        )

        if warnings_list:
            print(f"\n{'!'*80}")
            for warning in warnings_list:
                print(f"  {warning}")
            print(f"{'!'*80}")

        if not is_compatible:
            error_msg = "Config incompatibility detected (see warnings above)"
            if strict_config:
                raise ValueError(error_msg)
            else:
                print(f"\n{error_msg}")
                print("Continuing anyway (use --strict-resume to enforce compatibility)")

    # Check git hash mismatch
    from ..training.metadata import get_git_commit_hash
    current_hash = get_git_commit_hash()
    git_warning = check_git_hash_mismatch(commit_hash, current_hash)

    if git_warning:
        print(f"\n{git_warning}")

    # Restore epsilon scheduler state
    print(f"\nRestoring epsilon scheduler...")
    print(f"  Setting epsilon to: {epsilon_value:.4f}")
    print(f"  Setting frame counter to: {step}")

    # Update epsilon scheduler to match checkpoint state
    # The scheduler needs to be at the correct frame count
    epsilon_scheduler.frame_counter = step
    epsilon_scheduler.current_epsilon = epsilon_value

    # Verify epsilon is correct after restoration
    restored_epsilon = epsilon_scheduler.get_epsilon(step)
    if abs(restored_epsilon - epsilon_value) > 1e-6:
        warnings.warn(
            f"Epsilon mismatch after restoration: "
            f"checkpoint={epsilon_value:.6f}, restored={restored_epsilon:.6f}"
        )

    # Restore RNG states for reproducibility
    if rng_states:
        print(f"\nRestoring RNG states for reproducibility...")
        try:
            set_rng_states(rng_states, env)
            print(f"  ✓ Python random state restored")
            print(f"  ✓ NumPy random state restored")
            print(f"  ✓ PyTorch random state restored")
            if 'torch_cuda' in rng_states and torch.cuda.is_available():
                print(f"  ✓ CUDA random state restored")
            if 'env' in rng_states:
                print(f"  ✓ Environment random state restored")
        except Exception as e:
            warnings.warn(f"Failed to restore RNG states: {e}")
            print(f"  ⚠ RNG state restoration failed - training may not be fully deterministic")

    # Replay buffer info
    if replay_buffer is not None:
        print(f"\nReplay buffer state:")
        print(f"  Size: {replay_buffer.size:,} / {replay_buffer.capacity:,}")
        print(f"  Write index: {replay_buffer.index}")
        if replay_buffer.size < replay_buffer.min_size:
            print(f"  ⚠ Buffer below min_size ({replay_buffer.min_size:,}) - will warm up before training")

    # Optimizer info
    print(f"\nOptimizer state restored:")
    print(f"  Type: {type(optimizer).__name__}")
    param_groups = optimizer.param_groups
    if param_groups:
        print(f"  Learning rate: {param_groups[0]['lr']}")

    # Model info
    print(f"\nModel weights restored:")
    print(f"  Online model parameters: {sum(p.numel() for p in online_model.parameters()):,}")
    print(f"  Target model parameters: {sum(p.numel() for p in target_model.parameters()):,}")
    print(f"  Device: {device}")

    print(f"\n{'='*80}")
    print(f"RESUME COMPLETE - Starting from step {step + 1:,}")
    print(f"{'='*80}\n")

    return {
        'step': step,
        'episode': episode,
        'epsilon': epsilon_value,
        'loaded_metadata': loaded_state,
        'next_step': step + 1,  # Resume from next step
        'warnings': warnings_list if config is not None else [],
    }


def add_resume_args(parser):
    """
    Add resume-related arguments to argument parser.

    Args:
        parser: argparse.ArgumentParser instance

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_resume_args(parser)
        >>> args = parser.parse_args()
    """
    resume_group = parser.add_argument_group('Resume Options')

    resume_group.add_argument(
        '--resume',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to checkpoint file to resume from (e.g., checkpoints/checkpoint_1000000.pt)'
    )

    resume_group.add_argument(
        '--strict-resume',
        action='store_true',
        help='Enforce strict config compatibility when resuming (error on mismatch)'
    )

    resume_group.add_argument(
        '--no-restore-rng',
        action='store_true',
        help='Skip RNG state restoration (training will not be deterministic)'
    )

    return parser
