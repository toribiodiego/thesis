#!/usr/bin/env python3
"""Transition model evaluation script (M16).

Evaluates autoregressive transition model prediction accuracy at
each horizon step k=1..K via cosine similarity in projected space.
Only runs on +SPR checkpoints (refuses -SPR with error).

For each valid K-step sequence from the replay buffer:
1. Encode obs_0 (4-frame stack at position 0)
2. Run transition model autoregressively for k steps
3. Compare k-th prediction against encode_project(obs_k)
4. Report per-step and mean cosine similarity

Note: CPU-intensive (~3-5 min for IMPALA on full replay buffer).

Usage:
    python scripts/analysis/run_transition_eval.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step 10000

    python scripts/analysis/run_transition_eval.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --step 10000 --K 3 --output output/probing/transition_eval.json
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import json
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def _build_stacked_sequences(replay, K):
    """Build 4-frame-stacked obs and actions for K-step evaluation.

    Needs K+4 consecutive non-terminal frames. Returns:
        obs_stacked: (M, K+1, 84, 84, 4) uint8 HWC stacked frames
        actions: (M, K) int32 actions for the K transition steps
    """
    frames = replay.observations
    terms = replay.terminals
    n = len(frames)
    window = K + 3  # need 3 context + K+1 eval frames = K+4 total frames

    obs_list, act_list = [], []
    for i in range(n - window):
        if any(terms[i : i + window]):
            continue
        # Build K+1 stacked observations
        stacks = []
        for k in range(K + 1):
            stack = np.stack(frames[i + k : i + k + 4], axis=-1)  # (84, 84, 4)
            stacks.append(stack)
        obs_list.append(np.stack(stacks))  # (K+1, 84, 84, 4)
        # Actions for K transition steps: from position i+3 to i+3+K-1
        act_list.append(replay.actions[i + 3 : i + 3 + K])

    if not obs_list:
        return (
            np.empty((0, K + 1, 84, 84, 4), dtype=np.uint8),
            np.empty((0, K), dtype=np.int32),
        )
    return np.array(obs_list, dtype=np.uint8), np.array(act_list, dtype=np.int32)


def _evaluate_all_steps(checkpoint, obs_stacked, actions, K, batch_size, seed):
    """Evaluate all K steps in a single JIT-compiled function.

    Uses fixed-size K action sequences to avoid recompilation per step.
    Returns (M, K) cosine similarities.
    """
    net = checkpoint.network_def
    online_params = {"params": checkpoint.online_params}
    target_params_dict = checkpoint.target_params
    target_params = {"params": target_params_dict} if target_params_dict else online_params

    def _predict_all_steps(self, obs, actions_seq, key):
        """Predict K steps forward, return all K prediction vectors."""
        spatial = self.encode(obs, eval_mode=True)
        _, pred_latents = self.transition_model(spatial, actions_seq)
        # pred_latents: (K, H, W, C)
        representations = self.flatten_spatial_latent(pred_latents, has_batch=True)
        # representations: (K, flat_dim)
        # Project + predict each step (spr_predict is vmapped over axis 0)
        predictions = self.spr_predict(representations, key, True)
        return predictions  # (K, hidden_dim)

    def _target_single(self, obs, key):
        return self.encode_project(obs, key, True)

    @jax.jit
    def _eval_batch(obs0_batch, acts_batch, obs_targets_batch, keys):
        """obs0: (B, 84, 84, 4), acts: (B, K), obs_targets: (B, K, 84, 84, 4)."""
        def _single(ob0, acts, ob_tgts, key):
            # Predictions: (K, hidden_dim)
            preds = net.apply(
                online_params, ob0, acts, key,
                method=_predict_all_steps, rngs={"dropout": key},
            )
            # Targets: vmap encode_project over K target observations
            def _encode_target(ob_tgt):
                return net.apply(
                    target_params, ob_tgt, key, True,
                    method=net.encode_project, rngs={"dropout": key},
                )
            targets = jax.vmap(_encode_target)(ob_tgts)  # (K, hidden_dim)

            # Cosine similarity per step
            pred_norm = preds / (jnp.linalg.norm(preds, axis=-1, keepdims=True) + 1e-8)
            tgt_norm = targets / (jnp.linalg.norm(targets, axis=-1, keepdims=True) + 1e-8)
            return jnp.sum(pred_norm * tgt_norm, axis=-1)  # (K,)

        return jax.vmap(_single)(obs0_batch, acts_batch, obs_targets_batch, keys)

    rng = jax.random.PRNGKey(seed)
    M = len(obs_stacked)
    all_sims = []

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        o0 = obs_stacked[start:end, 0].astype(np.float32) / 255.0  # (B, 84, 84, 4)
        ak = actions[start:end]  # (B, K)
        # Target observations at steps 1..K
        ot = obs_stacked[start:end, 1:].astype(np.float32) / 255.0  # (B, K, 84, 84, 4)

        rng, rng_batch = jax.random.split(rng)
        keys = jax.random.split(rng_batch, end - start)

        sims = _eval_batch(o0, ak, ot, keys)  # (B, K)
        all_sims.append(np.asarray(sims))

    return np.concatenate(all_sims, axis=0)  # (M, K)


def main():
    parser = argparse.ArgumentParser(
        description="Transition model evaluation: per-step cosine similarity (M16)"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--K", type=int, default=5,
                        help="Number of forward prediction steps (default: 5)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # -- Load checkpoint -----------------------------------------------------
    print(f"Loading checkpoint: {args.run_dir} step {args.step}")
    from src.analysis.checkpoint import load_checkpoint
    ckpt = load_checkpoint(args.run_dir, args.step)
    print(f"  encoder: {ckpt.encoder_type}, hidden_dim: {ckpt.hidden_dim}")

    # -- Check for transition model (refuse -SPR checkpoints) ----------------
    if "transition_model" not in ckpt.online_params:
        print("ERROR: No transition_model in checkpoint params. "
              "This checkpoint is from a -SPR condition and has no "
              "transition model to evaluate.")
        sys.exit(1)

    # -- Load replay buffer and build stacked sequences ----------------------
    print("Loading replay buffer...")
    from src.analysis.replay_buffer import load_replay_buffer
    replay = load_replay_buffer(args.run_dir, args.step)
    print(f"  {replay.add_count} entries, "
          f"{replay.terminals.sum()} episode boundaries")

    print(f"Building {args.K}-step stacked sequences...")
    t0 = time.time()
    obs_stacked, actions = _build_stacked_sequences(replay, args.K)
    print(f"  {len(obs_stacked)} valid sequences ({time.time() - t0:.1f}s)")

    if len(obs_stacked) == 0:
        print("ERROR: No valid sequences found.")
        sys.exit(1)

    # -- Evaluate all K steps at once (single JIT compilation) ----------------
    print(f"Evaluating transition model at steps k=1..{args.K}")
    print("  (this may take a few minutes on CPU -- single JIT compile)")
    t0 = time.time()

    all_sims = _evaluate_all_steps(
        ckpt, obs_stacked, actions, args.K,
        batch_size=args.batch_size, seed=args.seed,
    )  # (M, K)
    elapsed = time.time() - t0
    print(f"  done ({elapsed:.1f}s)")

    per_step_results = []
    for k in range(args.K):
        sims_k = all_sims[:, k]
        per_step_results.append({
            "step": k + 1,
            "mean_cosine_sim": round(float(sims_k.mean()), 6),
            "std_cosine_sim": round(float(sims_k.std()), 6),
            "n_sequences": len(sims_k),
        })
        print(f"  k={k+1}: cosine_sim={sims_k.mean():.4f} +/- {sims_k.std():.4f}")

    overall_mean = np.mean([r["mean_cosine_sim"] for r in per_step_results])
    print(f"\n  Mean across steps: {overall_mean:.4f}")
    print()

    # -- Save JSON -----------------------------------------------------------
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        output = {
            "run_dir": args.run_dir,
            "step": args.step,
            "K": args.K,
            "encoder_type": ckpt.encoder_type,
            "hidden_dim": ckpt.hidden_dim,
            "num_sequences": len(obs_stacked),
            "overall_mean_cosine_sim": round(overall_mean, 6),
            "per_step": per_step_results,
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
