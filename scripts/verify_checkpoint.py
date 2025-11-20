#!/usr/bin/env python3
"""
Verify checkpoint can be loaded and contains all required components.
"""
import argparse
import torch
from pathlib import Path


def verify_checkpoint(checkpoint_path):
    """Load and verify checkpoint structure."""
    print("=" * 80)
    print("CHECKPOINT VERIFICATION")
    print("=" * 80)

    ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        print(f"\n✗ ERROR: Checkpoint not found: {ckpt_path}")
        return False

    print(f"\nFile: {ckpt_path.name}")
    print(f"Size: {ckpt_path.stat().st_size / (1024**2):.2f} MB")

    try:
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        print("\n✓ Checkpoint loaded successfully")

    except Exception as e:
        print(f"\n✗ ERROR loading checkpoint: {e}")
        return False

    # Required keys for resuming training
    required_keys = [
        'env_step',
        'episode',
        'epsilon',
        'online_net_state',
        'target_net_state',
        'optimizer_state',
        'rng_states'
    ]

    print("\nRequired components:")
    all_present = True
    for key in required_keys:
        present = key in checkpoint
        status = "✓" if present else "✗"
        print(f"  {status} {key}")
        if not present:
            all_present = False

    if not all_present:
        print("\n✗ INVALID: Missing required keys")
        return False

    # Print metadata
    print("\nCheckpoint metadata:")
    print(f"  Training step: {checkpoint['env_step']:,}")
    print(f"  Episode: {checkpoint['episode']:,}")
    print(f"  Epsilon: {checkpoint['epsilon']:.6f}")

    # Verify network states have weights
    online_params = sum(1 for v in checkpoint['online_net_state'].values() if isinstance(v, torch.Tensor))
    target_params = sum(1 for v in checkpoint['target_net_state'].values() if isinstance(v, torch.Tensor))

    print(f"\nNetwork weights:")
    print(f"  Online network: {online_params} parameter tensors")
    print(f"  Target network: {target_params} parameter tensors")

    if online_params == 0 or target_params == 0:
        print("\n✗ INVALID: Network states are empty")
        return False

    # Verify RNG states
    if 'torch' not in checkpoint['rng_states'] or 'numpy' not in checkpoint['rng_states']:
        print("\n⚠ WARNING: Incomplete RNG states (determinism may be affected)")
    else:
        print("\n✓ RNG states present (deterministic resume supported)")

    print("\n" + "=" * 80)
    print("VERDICT: ✓ VALID CHECKPOINT - Ready for resume")
    print("=" * 80)

    return True


def main():
    parser = argparse.ArgumentParser(description='Verify checkpoint integrity')
    parser.add_argument('checkpoint', help='Path to checkpoint file')

    args = parser.parse_args()

    success = verify_checkpoint(args.checkpoint)
    exit(0 if success else 1)


if __name__ == '__main__':
    main()
