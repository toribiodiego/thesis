"""Checkpoint utilities for capturing and restoring RNG states."""

import random
import numpy as np
import torch
from typing import Dict, Any, Optional


def get_rng_states(env: Any = None) -> Dict[str, Any]:
    """
    Capture all RNG states for reproducibility.

    Captures states from:
    - Python random module
    - NumPy random
    - PyTorch CPU random
    - PyTorch CUDA random (if available)
    - Environment (if provided and has get_rng_state method)

    Args:
        env: Optional environment object with get_rng_state() method

    Returns:
        dict: Dictionary containing all RNG states
    """
    rng_states = {
        'python_random': random.getstate(),
        'numpy_random': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }

    # Capture CUDA RNG state if available
    if torch.cuda.is_available():
        rng_states['torch_cuda'] = torch.cuda.get_rng_state_all()

    # Capture environment RNG state if available
    if env is not None and hasattr(env, 'get_rng_state'):
        rng_states['env'] = env.get_rng_state()

    return rng_states


def set_rng_states(rng_states: Dict[str, Any], env: Any = None):
    """
    Restore RNG states from checkpoint.

    Args:
        rng_states: Dictionary of RNG states (from get_rng_states)
        env: Optional environment object to restore state into
    """
    if 'python_random' in rng_states:
        random.setstate(rng_states['python_random'])

    if 'numpy_random' in rng_states:
        np.random.set_state(rng_states['numpy_random'])

    if 'torch_cpu' in rng_states:
        torch.set_rng_state(rng_states['torch_cpu'])

    if 'torch_cuda' in rng_states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_states['torch_cuda'])

    if env is not None and 'env' in rng_states and hasattr(env, 'set_rng_state'):
        env.set_rng_state(rng_states['env'])


def verify_checkpoint_integrity(checkpoint_path: str) -> bool:
    """
    Verify checkpoint file integrity by attempting to load it.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        bool: True if checkpoint is valid, False otherwise
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Check required fields
        required_fields = [
            'schema_version',
            'step',
            'episode',
            'epsilon',
            'online_model_state_dict',
            'target_model_state_dict',
            'optimizer_state_dict'
        ]

        for field in required_fields:
            if field not in checkpoint:
                print(f"Missing required field: {field}")
                return False

        return True

    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return False
