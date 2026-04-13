#!/usr/bin/env python3
"""Structural health analysis script (M11).

Computes per-layer dead neuron fraction and effective rank
(participation ratio of singular values) for encoder layers.
Uses Flax's capture_intermediates to extract activations at
stage/layer boundaries, then applies ReLU to match post-
activation state.

Dead neuron: a channel whose mean post-ReLU activation across
all spatial positions and all observations is below a threshold
(default 0.025). Measures plasticity loss.

Effective rank: participation ratio of the singular value
spectrum of the (N, C) channel-mean activation matrix.
Measures representational diversity.

Handles both encoder types:
- Nature CNN: 3 Conv layers (capture Conv outputs, apply ReLU)
- IMPALA: 3 ResidualStages (capture stage outputs, apply ReLU)

Usage:
    # From replay buffer (fastest, no environment)
    python scripts/run_structural_health.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step 10000 --source replay

    # From random policy
    python scripts/run_structural_health.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step 10000 --game CrazyClimber --source random --num-steps 500
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import sys
import time

import flax.linen as nn
import jax
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEAD_NEURON_THRESHOLD = 0.025


def _get_capture_filter(encoder_type):
    """Return a capture_intermediates filter for the encoder type."""
    if encoder_type == "impala":
        try:
            from bigger_better_faster.bbf.spr_networks import ResidualStage
        except ImportError:
            from src.bigger_better_faster.bbf.spr_networks import ResidualStage
        return lambda m, _: isinstance(m, ResidualStage)
    else:
        # Nature CNN: capture Conv outputs
        return lambda m, _: isinstance(m, nn.Conv)


def _extract_layer_activations(ckpt, observations, batch_size=64, seed=0):
    """Run observations through the encoder, capturing per-layer activations.

    For IMPALA: captures ResidualStage outputs (3 stages).
    For Nature CNN: captures Conv outputs (3 layers).
    Applies ReLU to captured outputs to match post-activation state.

    Returns:
        List of (layer_name, activations) where activations is
        (N, H, W, C) float32 post-ReLU.
    """
    net = ckpt.network_def
    params = {"params": ckpt.online_params}
    support = ckpt.support
    capture_filter = _get_capture_filter(ckpt.encoder_type)

    @jax.jit
    def _forward_single(obs, key):
        _, state = net.apply(
            params, obs, support=support, eval_mode=True, key=key,
            rngs={"dropout": key},
            capture_intermediates=capture_filter,
            mutable=["intermediates"],
        )
        return state["intermediates"]

    _forward_batch = jax.jit(jax.vmap(_forward_single))

    rng = jax.random.PRNGKey(seed)
    n = len(observations)

    # Discover layer names from first batch
    first_size = min(batch_size, n)
    first_chunk = observations[:first_size].astype(np.float32) / 255.0
    rng, rng_batch = jax.random.split(rng)
    keys = jax.random.split(rng_batch, first_size)
    first_ints = _forward_batch(first_chunk, keys)

    layer_names = []
    layer_acts = {}

    def _walk(d, prefix=""):
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, dict):
                _walk(v, prefix + k + "/")
            else:
                name = prefix + k
                if name.startswith("encoder/"):
                    layer_names.append(name)
                    layer_acts[name] = [np.asarray(v[0])]

    _walk(dict(first_ints))

    # Process remaining batches
    for start in range(batch_size, n, batch_size):
        end = min(start + batch_size, n)
        chunk = observations[start:end].astype(np.float32) / 255.0
        rng, rng_batch = jax.random.split(rng)
        keys = jax.random.split(rng_batch, end - start)
        ints = _forward_batch(chunk, keys)

        def _collect(d, prefix=""):
            for k in sorted(d.keys()):
                v = d[k]
                if isinstance(v, dict):
                    _collect(v, prefix + k + "/")
                else:
                    name = prefix + k
                    if name in layer_acts:
                        layer_acts[name].append(np.asarray(v[0]))

        _collect(dict(ints))

    # Concatenate and apply ReLU (captured outputs are pre-ReLU)
    results = []
    for name in layer_names:
        acts = np.concatenate(layer_acts[name], axis=0)  # (N, H, W, C)
        acts = np.maximum(acts, 0)  # post-ReLU
        results.append((name, acts))

    return results


def dead_neuron_fraction(activations, threshold=DEAD_NEURON_THRESHOLD):
    """Compute fraction of dead channels using ReDo-style normalized scores.

    Following Sokar et al. 2023 (ReDo):
    1. Per-unit mean absolute activation across batch and spatial dims
    2. Normalize scores by the layer mean (scale-invariant threshold)
    3. A unit is dead if its normalized score <= threshold

    Args:
        activations: (N, H, W, C) float32 post-ReLU activations.
        threshold: Dead neuron threshold on normalized scores (default 0.025).

    Returns:
        (dead_fraction, num_dead, num_channels)
    """
    # Step 1: per-channel mean absolute activation
    scores = np.abs(activations).mean(axis=(0, 1, 2))  # (C,)
    # Step 2: normalize by layer mean
    layer_mean = scores.mean()
    if layer_mean > 0:
        scores = scores / layer_mean
    # Step 3: threshold
    num_dead = int((scores <= threshold).sum())
    num_channels = len(scores)
    return num_dead / num_channels, num_dead, num_channels


def effective_rank(activations):
    """Compute effective rank via participation ratio of singular values.

    Global-average-pools spatial dims to get (N, C) matrix, centers,
    computes SVD, and returns participation ratio:
        eff_rank = (sum(s))^2 / sum(s^2)

    Args:
        activations: (N, H, W, C) float32 post-ReLU activations.

    Returns:
        (eff_rank, num_channels)
    """
    pooled = activations.mean(axis=(1, 2))  # (N, C)
    pooled = pooled - pooled.mean(axis=0)
    s = np.linalg.svd(pooled, compute_uv=False)
    s_sq_sum = (s ** 2).sum()
    if s_sq_sum == 0:
        return 0.0, pooled.shape[1]
    return float((s.sum() ** 2) / s_sq_sum), pooled.shape[1]


def main():
    parser = argparse.ArgumentParser(
        description="Structural health: dead neurons and effective rank (M11)"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--game", type=str, default=None,
                        help="Game name (required for greedy/random)")
    parser.add_argument("--source", choices=["greedy", "random", "replay"],
                        default="replay")
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=DEAD_NEURON_THRESHOLD)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.source in ("greedy", "random") and args.game is None:
        parser.error("--game is required for greedy and random sources")

    # -- Load checkpoint -----------------------------------------------------
    print(f"Loading checkpoint: {args.run_dir} step {args.step}")
    from src.analysis.checkpoint import load_checkpoint
    ckpt = load_checkpoint(args.run_dir, args.step)
    print(f"  encoder: {ckpt.encoder_type}, hidden_dim: {ckpt.hidden_dim}")

    # -- Get observations ----------------------------------------------------
    print(f"Loading observations (source={args.source})...")
    t0 = time.time()

    if args.source == "greedy":
        from src.analysis.observations import collect_greedy
        observations = collect_greedy(
            ckpt, game=args.game, num_steps=args.num_steps,
            seed=args.seed, noop_max=30,
        ).observations
    elif args.source == "random":
        from src.analysis.observations import collect_random
        observations = collect_random(
            game=args.game, num_actions=ckpt.num_actions,
            num_steps=args.num_steps, seed=args.seed, noop_max=30,
        ).observations
    else:
        from src.analysis.replay_buffer import load_replay_buffer
        replay = load_replay_buffer(args.run_dir, args.step)
        frames, terms = replay.observations, replay.terminals
        obs_list = []
        for i in range(3, len(frames)):
            if not any(terms[i - 3 : i]):
                obs_list.append(np.stack(frames[i - 3 : i + 1], axis=-1))
        observations = np.array(obs_list, dtype=np.uint8)

    print(f"  {len(observations)} observations ({time.time() - t0:.1f}s)")

    # -- Extract per-layer activations (CPU-intensive) -----------------------
    print(f"Extracting per-layer activations (batch_size={args.batch_size})...")
    print("  (this may take a few minutes on CPU)")
    t0 = time.time()
    layer_data = _extract_layer_activations(
        ckpt, observations, batch_size=args.batch_size, seed=args.seed,
    )
    print(f"  {len(layer_data)} layers ({time.time() - t0:.1f}s)")

    # -- Compute metrics -----------------------------------------------------
    print()
    print(f"{'Layer':<35} {'Shape':>14} {'Dead':>5} {'Dead%':>7} "
          f"{'EffRank':>8} {'Ch':>5}")
    print("-" * 80)

    results = []
    for name, acts in layer_data:
        dead_frac, n_dead, n_ch = dead_neuron_fraction(acts, args.threshold)
        eff_r, _ = effective_rank(acts)
        short = name.replace("encoder/", "").replace("/__call__", "")
        shape = f"{acts.shape[1]}x{acts.shape[2]}x{acts.shape[3]}"
        print(f"{short:<35} {shape:>14} {n_dead:>5} {100*dead_frac:>6.1f}% "
              f"{eff_r:>8.1f} {n_ch:>5}")
        results.append({
            "layer": short, "channels": n_ch,
            "spatial": [int(acts.shape[1]), int(acts.shape[2])],
            "dead_neurons": n_dead, "dead_fraction": round(dead_frac, 4),
            "effective_rank": round(eff_r, 2),
        })

    print()

    # -- Save JSON -----------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output = {
            "run_dir": args.run_dir, "step": args.step,
            "source": args.source, "num_observations": len(observations),
            "encoder_type": ckpt.encoder_type, "threshold": args.threshold,
            "layers": results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
