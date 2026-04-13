#!/usr/bin/env python3
"""Batch analysis runner -- process one run across all checkpoints.

Iterates over all checkpoints in a run directory, applying all
applicable analysis methods with shared representation extraction.
Per-checkpoint results are saved as CSVs to run_dir/analysis/.

Skips:
- Existing output files (resumable)
- Transition model assessment for -SPR conditions (no transition_model)
- AtariARI probing for unannotated games

CPU/GPU intensive: representation extraction runs the full encoder
forward pass for each checkpoint. Budget ~3 min per checkpoint for
IMPALA, ~30s for Nature CNN on CPU.

Usage:
    python scripts/analysis/run_all.py \\
        --run-dir experiments/dqn_atari/runs/bbf_crazy_climber_seed13 \\
        --game CrazyClimber --source replay

    # Override observation source and count
    python scripts/analysis/run_all.py \\
        --run-dir experiments/dqn_atari/runs/spr_boxing_seed13 \\
        --game Boxing --source greedy --num-steps 5000
"""

import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import argparse
import csv
import json
import re
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

# AtariARI-annotated games (lowercase, no separators)
_ATARIARI_GAMES = None


def _get_atariari_games():
    global _ATARIARI_GAMES
    if _ATARIARI_GAMES is None:
        try:
            from atariari.benchmark.wrapper import atari_dict
            _ATARIARI_GAMES = set(atari_dict.keys())
        except ImportError:
            _ATARIARI_GAMES = set()
    return _ATARIARI_GAMES


def _game_is_annotated(game):
    """Check if a game has AtariARI annotations."""
    normalized = game.lower().replace("_", "").replace(" ", "")
    return normalized in _get_atariari_games()


def _discover_checkpoint_steps(run_dir):
    """Find all checkpoint steps from msgpack files."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    steps = []
    for f in os.listdir(ckpt_dir):
        m = re.match(r"checkpoint_(\d+)\.msgpack$", f)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def _read_effective_gamma(run_dir, step):
    csv_path = os.path.join(run_dir, "steps.csv")
    if not os.path.isfile(csv_path):
        return 0.99
    df = pd.read_csv(csv_path)
    if "effective_gamma" not in df.columns:
        return 0.99
    row = df.iloc[(df["step"] - step).abs().argsort()[:1]]
    return float(row["effective_gamma"].values[0])


def _stack_replay_frames(replay):
    """Stack replay single frames into (M, 84, 84, 4) HWC + index mapping."""
    frames, terms = replay.observations, replay.terminals
    obs_list, idx_list = [], []
    for i in range(3, len(frames)):
        if not any(terms[i - 3 : i]):
            obs_list.append(np.stack(frames[i - 3 : i + 1], axis=-1))
            idx_list.append(i)
    return np.array(obs_list, dtype=np.uint8), np.array(idx_list)


def _write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _run_reward_probing(reps, rewards, out_dir):
    """M10: reward probing."""
    out_path = os.path.join(out_dir, "reward_probing.csv")
    if os.path.exists(out_path):
        return "skip (exists)"

    from src.analysis.probing import train_probe
    labels = (rewards > 0).astype(np.int32)
    result = train_probe(reps, labels, variable_name="reward_binary",
                         entropy_threshold=0.0)
    _write_csv(out_path, [{
        "variable": result.variable,
        "f1_test": result.f1_test,
        "f1_train": result.f1_train,
        "accuracy_test": result.accuracy_test,
        "n_classes": result.n_classes,
        "skipped": result.skipped,
    }], ["variable", "f1_test", "f1_train", "accuracy_test", "n_classes", "skipped"])
    return f"f1={result.f1_test:.4f}"


def _run_atariari_probing(ckpt, game, obs_source, num_steps, seed, out_dir):
    """M9: AtariARI probing (only for annotated games)."""
    out_path = os.path.join(out_dir, "atariari_probing.csv")
    if os.path.exists(out_path):
        return "skip (exists)"

    if obs_source == "replay":
        return "skip (replay source, no labels)"

    from src.analysis.observations import collect_greedy, collect_random
    if obs_source == "greedy":
        data = collect_greedy(ckpt, game=game, num_steps=num_steps,
                              seed=seed, noop_max=30, collect_labels=True)
    else:
        data = collect_random(game=game, num_actions=ckpt.num_actions,
                              num_steps=num_steps, seed=seed, noop_max=30,
                              collect_labels=True)

    from src.analysis.representations import extract_representations
    label_reps = extract_representations(ckpt, data.observations, seed=seed)

    from src.analysis.probing import train_probes_multi
    results = train_probes_multi(label_reps, data.labels)

    rows = [{
        "variable": r.variable,
        "f1_test": r.f1_test,
        "f1_train": r.f1_train,
        "accuracy_test": r.accuracy_test,
        "n_classes": r.n_classes,
        "normalized_entropy": r.normalized_entropy,
        "skipped": r.skipped,
        "skip_reason": r.skip_reason or "",
    } for r in results]
    _write_csv(out_path, rows,
               ["variable", "f1_test", "f1_train", "accuracy_test",
                "n_classes", "normalized_entropy", "skipped", "skip_reason"])

    active = [r for r in results if not r.skipped]
    mean_f1 = sum(r.f1_test for r in active) / len(active) if active else 0
    return f"mean_f1={mean_f1:.4f} ({len(active)} vars)"


def _run_inverse_dynamics(ckpt, replay, out_dir, seed):
    """M14: inverse dynamics probing."""
    out_path = os.path.join(out_dir, "inverse_dynamics.csv")
    if os.path.exists(out_path):
        return "skip (exists)"

    frames, terms = replay.observations, replay.terminals
    obs_t_list, obs_next_list, act_list = [], [], []
    for i in range(3, len(frames) - 1):
        if any(terms[i - 3 : i + 1]):
            continue
        obs_t_list.append(np.stack(frames[i - 3 : i + 1], axis=-1))
        obs_next_list.append(np.stack(frames[i - 2 : i + 2], axis=-1))
        act_list.append(replay.actions[i])

    if not obs_t_list:
        return "skip (no valid transitions)"

    obs_t = np.array(obs_t_list, dtype=np.uint8)
    obs_next = np.array(obs_next_list, dtype=np.uint8)
    actions = np.array(act_list, dtype=np.int32)

    from src.analysis.representations import extract_representations
    reps_t = extract_representations(ckpt, obs_t, seed=seed)
    reps_next = extract_representations(ckpt, obs_next, seed=seed)
    features = np.concatenate([reps_t, reps_next], axis=1)

    from src.analysis.probing import train_probe
    result = train_probe(features, actions, variable_name="action",
                         entropy_threshold=0.0)
    _write_csv(out_path, [{
        "variable": result.variable,
        "f1_test": result.f1_test,
        "f1_train": result.f1_train,
        "accuracy_test": result.accuracy_test,
        "n_classes": result.n_classes,
        "chance": round(1.0 / max(result.n_classes, 1), 4),
        "skipped": result.skipped,
    }], ["variable", "f1_test", "f1_train", "accuracy_test",
         "n_classes", "chance", "skipped"])
    return f"acc={result.accuracy_test:.4f}"


def _run_structural_health(ckpt, observations, out_dir, seed):
    """M11: dead neurons and effective rank."""
    out_path = os.path.join(out_dir, "structural_health.csv")
    if os.path.exists(out_path):
        return "skip (exists)"

    import flax.linen as nn
    import jax
    net = ckpt.network_def
    params = {"params": ckpt.online_params}
    support = ckpt.support

    if ckpt.encoder_type == "impala":
        try:
            from bigger_better_faster.bbf.spr_networks import ResidualStage
        except ImportError:
            from src.bigger_better_faster.bbf.spr_networks import ResidualStage
        capture_filter = lambda m, _: isinstance(m, ResidualStage)
    else:
        capture_filter = lambda m, _: isinstance(m, nn.Conv)

    @jax.jit
    def _fwd(obs, key):
        _, state = net.apply(
            params, obs, support=support, eval_mode=True, key=key,
            rngs={"dropout": key},
            capture_intermediates=capture_filter,
            mutable=["intermediates"],
        )
        return state["intermediates"]

    _fwd_batch = jax.jit(jax.vmap(_fwd))

    rng = jax.random.PRNGKey(seed)
    n = len(observations)
    batch_size = 32
    layer_acts = {}

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = observations[start:end].astype(np.float32) / 255.0
        rng, rng_batch = jax.random.split(rng)
        keys = jax.random.split(rng_batch, end - start)
        ints = _fwd_batch(chunk, keys)

        def _collect(d, prefix=""):
            for k in sorted(d.keys()):
                v = d[k]
                if isinstance(v, dict):
                    _collect(v, prefix + k + "/")
                else:
                    name = prefix + k
                    if name.startswith("encoder/"):
                        layer_acts.setdefault(name, []).append(
                            np.asarray(np.maximum(v[0], 0)))

        _collect(dict(ints))

    rows = []
    for name in sorted(layer_acts.keys()):
        acts = np.concatenate(layer_acts[name], axis=0)
        scores = np.abs(acts).mean(axis=(0, 1, 2))
        layer_mean = scores.mean()
        if layer_mean > 0:
            scores = scores / layer_mean
        n_dead = int((scores <= 0.025).sum())
        n_ch = len(scores)

        pooled = acts.mean(axis=(1, 2))
        pooled = pooled - pooled.mean(axis=0)
        s = np.linalg.svd(pooled, compute_uv=False)
        s_sq = (s ** 2).sum()
        eff_rank = float((s.sum() ** 2) / s_sq) if s_sq > 0 else 0.0

        short = name.replace("encoder/", "").replace("/__call__", "")
        rows.append({
            "layer": short, "channels": n_ch,
            "dead_neurons": n_dead,
            "dead_fraction": round(n_dead / n_ch, 4),
            "effective_rank": round(eff_rank, 2),
        })

    _write_csv(out_path, rows,
               ["layer", "channels", "dead_neurons", "dead_fraction", "effective_rank"])
    return f"{len(rows)} layers"


def _run_filter_frequency(ckpt, out_dir):
    """M12: filter frequency analysis (weights only, fast)."""
    out_path = os.path.join(out_dir, "filter_frequency.csv")
    if os.path.exists(out_path):
        return "skip (exists)"

    encoder_params = ckpt.online_params["encoder"]

    def _collect_kernels(d, prefix=""):
        kernels = []
        for k in sorted(d.keys()):
            v = d[k]
            if isinstance(v, dict):
                kernels.extend(_collect_kernels(v, prefix + k + "/"))
            elif k == "kernel" and np.array(v).ndim == 4:
                kernels.append((prefix.rstrip("/"), np.array(v)))
        return kernels

    kernels = _collect_kernels(encoder_params)
    rows = []
    for name, kernel in kernels:
        h, w, c_in, c_out = kernel.shape
        filters = kernel.transpose(2, 3, 0, 1).reshape(-1, h, w)
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(filters), axes=(-2, -1)))
        power = (fft_mag ** 2).mean(axis=0)
        cy, cx = h // 2, w // 2
        dc_power = float(power[cy, cx])
        total = float(power.sum())
        dc_frac = dc_power / total if total > 0 else 0
        rows.append({
            "layer": name, "kernel_h": h, "kernel_w": w,
            "c_in": c_in, "c_out": c_out,
            "dc_power": round(dc_power, 6),
            "dc_fraction": round(dc_frac, 4),
        })

    _write_csv(out_path, rows,
               ["layer", "kernel_h", "kernel_w", "c_in", "c_out",
                "dc_power", "dc_fraction"])
    return f"{len(rows)} layers"


def _run_q_accuracy(ckpt, replay, observations, valid_indices, step, run_dir, out_dir, seed):
    """M15: Q-value accuracy."""
    out_path = os.path.join(out_dir, "q_accuracy.csv")
    if os.path.exists(out_path):
        return "skip (exists)"

    gamma = _read_effective_gamma(run_dir, step)

    from src.analysis.returns import compute_returns
    returns = compute_returns(replay, gamma)

    from src.analysis.representations import extract_q_values
    q_all = extract_q_values(ckpt, observations, seed=seed)
    actions_at_valid = replay.actions[valid_indices]
    q_taken = q_all[np.arange(len(q_all)), actions_at_valid]

    returns_at_valid = returns[valid_indices]
    mask = np.isfinite(returns_at_valid)
    q_matched = q_taken[mask]
    g_matched = returns_at_valid[mask]

    if len(q_matched) < 2:
        return "skip (too few matched pairs)"

    from scipy import stats
    spearman_r, spearman_p = stats.spearmanr(q_matched, g_matched)
    signed_error = q_matched - g_matched

    _write_csv(out_path, [{
        "gamma": gamma,
        "spearman_r": round(float(spearman_r), 6),
        "spearman_p": float(spearman_p),
        "mean_signed_error": round(float(signed_error.mean()), 6),
        "rmse": round(float(np.sqrt((signed_error ** 2).mean())), 6),
        "q_mean": round(float(q_matched.mean()), 6),
        "g_mean": round(float(g_matched.mean()), 6),
        "n_pairs": len(q_matched),
    }], ["gamma", "spearman_r", "spearman_p", "mean_signed_error",
         "rmse", "q_mean", "g_mean", "n_pairs"])
    return f"r={float(spearman_r):.4f}"


def _run_transition_assessment(ckpt, replay, out_dir, seed):
    """M16: transition model prediction assessment (SPR conditions only)."""
    out_path = os.path.join(out_dir, "transition_eval.csv")
    if os.path.exists(out_path):
        return "skip (exists)"

    from src.analysis.representations import evaluate_transition_model

    frames, terms = replay.observations, replay.terminals
    obs_t_list, obs_next_list, act_list = [], [], []
    for i in range(3, len(frames) - 1):
        if any(terms[i - 3 : i + 1]):
            continue
        obs_t_list.append(np.stack(frames[i - 3 : i + 1], axis=-1))
        obs_next_list.append(np.stack(frames[i - 2 : i + 2], axis=-1))
        act_list.append(replay.actions[i])

    if not obs_t_list:
        return "skip (no valid transitions)"

    obs_t = np.array(obs_t_list, dtype=np.uint8)
    obs_next = np.array(obs_next_list, dtype=np.uint8)
    actions = np.array(act_list, dtype=np.int32)

    sims = evaluate_transition_model(ckpt, obs_t, actions, obs_next, seed=seed)
    mean_sim = float(sims.mean())

    _write_csv(out_path, [{
        "step_k": 1,
        "mean_cosine_sim": round(mean_sim, 6),
        "std_cosine_sim": round(float(sims.std()), 6),
        "n_transitions": len(sims),
    }], ["step_k", "mean_cosine_sim", "std_cosine_sim", "n_transitions"])
    return f"sim={mean_sim:.4f}"


def main():
    parser = argparse.ArgumentParser(
        description="Batch analysis: all methods on all checkpoints for one run"
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--game", required=True,
                        help="Game name (CamelCase) for observation collection")
    parser.add_argument("--source", choices=["greedy", "random", "replay"],
                        default="replay")
    parser.add_argument("--num-steps", type=int, default=10000,
                        help="Steps for greedy/random collection (default: 10000)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    # -- Read run metadata ---------------------------------------------------
    meta_path = os.path.join(args.run_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        condition = meta.get("condition", "unknown")
        game_meta = meta.get("game", "unknown")
    else:
        condition = "unknown"
        game_meta = "unknown"

    print(f"Run: {args.run_dir}")
    print(f"  condition: {condition}, game: {game_meta}")

    # -- Discover checkpoints ------------------------------------------------
    steps = _discover_checkpoint_steps(args.run_dir)
    print(f"  checkpoints: {steps}")

    if not steps:
        print("No checkpoints found.")
        return

    has_atariari = _game_is_annotated(args.game)
    print(f"  AtariARI annotated: {has_atariari}")
    print()

    # -- Process each checkpoint ---------------------------------------------
    for step in steps:
        print(f"=== Checkpoint {step} ===")
        out_dir = os.path.join(args.run_dir, "analysis", f"checkpoint_{step}")
        os.makedirs(out_dir, exist_ok=True)

        t0 = time.time()

        # Load checkpoint
        from src.analysis.checkpoint import load_checkpoint
        ckpt = load_checkpoint(args.run_dir, step)
        has_transition_model = "transition_model" in ckpt.online_params

        # Load replay buffer
        from src.analysis.replay_buffer import load_replay_buffer
        replay = load_replay_buffer(args.run_dir, step)

        # Stack replay frames for shared use
        observations, valid_indices = _stack_replay_frames(replay)
        print(f"  loaded: {len(observations)} stacked obs ({time.time() - t0:.1f}s)")

        # Extract shared representations (once per checkpoint)
        from src.analysis.representations import extract_representations
        t1 = time.time()
        reps = extract_representations(
            ckpt, observations, batch_size=args.batch_size, seed=args.seed,
        )
        print(f"  representations: {reps.shape} ({time.time() - t1:.1f}s)")

        rewards_at_valid = replay.rewards[valid_indices]

        # -- Run each method -------------------------------------------------
        # M10: Reward probing
        t1 = time.time()
        status = _run_reward_probing(reps, rewards_at_valid, out_dir)
        print(f"  M10 reward probing: {status} ({time.time() - t1:.1f}s)")

        # M9: AtariARI probing (annotated games only)
        if has_atariari and args.source != "replay":
            t1 = time.time()
            status = _run_atariari_probing(ckpt, args.game, args.source,
                                           args.num_steps, args.seed, out_dir)
            print(f"  M9 AtariARI probing: {status} ({time.time() - t1:.1f}s)")
        else:
            reason = "unannotated game" if not has_atariari else "replay source"
            print(f"  M9 AtariARI probing: skip ({reason})")

        # M14: Inverse dynamics
        t1 = time.time()
        status = _run_inverse_dynamics(ckpt, replay, out_dir, args.seed)
        print(f"  M14 inverse dynamics: {status} ({time.time() - t1:.1f}s)")

        # M11: Structural health
        t1 = time.time()
        status = _run_structural_health(ckpt, observations, out_dir, args.seed)
        print(f"  M11 structural health: {status} ({time.time() - t1:.1f}s)")

        # M12: Filter frequency (weights only, fast)
        t1 = time.time()
        status = _run_filter_frequency(ckpt, out_dir)
        print(f"  M12 filter frequency: {status} ({time.time() - t1:.1f}s)")

        # M15: Q-value accuracy
        t1 = time.time()
        status = _run_q_accuracy(ckpt, replay, observations, valid_indices,
                                 step, args.run_dir, out_dir, args.seed)
        print(f"  M15 Q-value accuracy: {status} ({time.time() - t1:.1f}s)")

        # M16: Transition model assessment (SPR conditions only)
        if has_transition_model:
            t1 = time.time()
            status = _run_transition_assessment(ckpt, replay, out_dir, args.seed)
            print(f"  M16 transition model: {status} ({time.time() - t1:.1f}s)")
        else:
            print(f"  M16 transition model: skip (-SPR condition)")

        print(f"  total: {time.time() - t0:.1f}s")
        print()

    print("Done.")


if __name__ == "__main__":
    main()
