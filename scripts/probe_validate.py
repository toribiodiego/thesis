#!/usr/bin/env python3
"""Validate probing methodology before scaling.

Three checks:
1. Random CNN baseline (untrained encoder) -- is there signal
   above convolutional inductive bias?
2. Bin count sensitivity (8, 16, 32, 256) -- is 256 too aggressive?
3. Label sanity check -- do RAM addresses track expected values?

Usage:
    python scripts/probe_validate.py
"""

import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.atari_wrappers import make_atari_env
from src.models.dqn import DQN
from scripts.probe import (
    ATARI_ARI_LABELS,
    apply_labels,
    create_model,
    encode_observations,
    run_probes,
)

DATA_DIR = "output/probe_data/boxing"
DQN_RUN = "experiments/dqn_atari/runs/atari100k_boxing_42_20260310_170320"
SPR_RUN = "experiments/dqn_atari/runs/atari100k_boxing_spr_42_20260324_182503"


def check_1_random_baseline(observations, labels, device):
    """Check 1: Random CNN baseline.

    Create an untrained DQN, encode observations, probe.
    This measures the floor from convolutional inductive bias.
    """
    print("\n" + "=" * 60)
    print("CHECK 1: Random CNN baseline (untrained encoder)")
    print("=" * 60)

    env = make_atari_env(
        env_id="BoxingNoFrameskip-v4", frame_size=84, num_stack=4,
        frame_skip=4, clip_rewards=False, episode_life=False, noop_max=30,
    )
    num_actions = env.action_space.n
    env.close()

    model = DQN(num_actions=num_actions, dropout=0.0).to(device)
    model.eval()

    print("  Encoding observations with random CNN...")
    features = encode_observations(model, observations, device)
    print(f"  Feature dim: {features.shape[1]}")

    for num_bins in [8, 16, 32]:
        results = run_probes(features, labels, num_bins=num_bins)
        mean_f1 = np.mean([r["f1_test"] for r in results]) if results else 0
        print(f"\n  Bins={num_bins}: mean F1={mean_f1:.4f}")
        for r in results:
            print(f"    {r['variable']:<16} F1={r['f1_test']:.4f}  "
                  f"acc={r['accuracy']:.4f}")


def check_2_bin_sensitivity(observations, labels, device):
    """Check 2: Bin count sensitivity for trained encoders."""
    print("\n" + "=" * 60)
    print("CHECK 2: Bin count sensitivity (DQN and DQN+SPR)")
    print("=" * 60)

    for run_dir, label in [(DQN_RUN, "DQN"), (SPR_RUN, "DQN+SPR")]:
        config_path = os.path.join(run_dir, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        env = make_atari_env(
            env_id="BoxingNoFrameskip-v4", frame_size=84, num_stack=4,
            frame_skip=4, clip_rewards=False, episode_life=False,
            noop_max=30,
        )
        num_actions = env.action_space.n
        env.close()

        model = create_model(config, num_actions, device)
        is_rainbow = config.get("rainbow", {}).get("enabled", False)

        cp_path = os.path.join(run_dir, "checkpoints", "checkpoint_400000.pt")
        if not os.path.exists(cp_path):
            cp_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
        if not os.path.exists(cp_path):
            print(f"\n  {label}: SKIP (no checkpoint)")
            continue

        checkpoint = torch.load(cp_path, map_location=device,
                                weights_only=False)
        model.load_state_dict(
            checkpoint["online_model_state_dict"], strict=True
        )
        if hasattr(model, "set_eval_mode"):
            model.set_eval_mode()
        else:
            model.eval()

        features = encode_observations(
            model, observations, device, is_rainbow
        )

        print(f"\n  {label}:")
        for num_bins in [8, 16, 32, 256]:
            results = run_probes(features, labels, num_bins=num_bins)
            mean_f1 = (np.mean([r["f1_test"] for r in results])
                       if results else 0)
            var_scores = "  ".join(
                f"{r['variable'][:8]}={r['f1_test']:.3f}"
                for r in results
            )
            print(f"    Bins={num_bins:>3}: mean F1={mean_f1:.4f}  "
                  f"{var_scores}")


def check_4_conv_vs_fc(observations, labels, device):
    """Check 4: Conv_output (pre-FC) vs features (post-FC).

    Tests whether spatial structure survives training at the
    conv layer even if lost at the FC layer.
    """
    print("\n" + "=" * 60)
    print("CHECK 4: Conv_output (3136-dim) vs features (512-dim)")
    print("=" * 60)

    num_bins = 8  # Use 8 bins where we saw the best signal

    # Random CNN at both layers
    env = make_atari_env(
        env_id="BoxingNoFrameskip-v4", frame_size=84, num_stack=4,
        frame_skip=4, clip_rewards=False, episode_life=False, noop_max=30,
    )
    num_actions = env.action_space.n
    env.close()

    random_model = DQN(num_actions=num_actions, dropout=0.0).to(device)
    random_model.eval()

    for layer_name in ["features", "conv_output"]:
        feats = encode_observations(
            random_model, observations, device, layer=layer_name
        )
        results = run_probes(feats, labels, num_bins=num_bins)
        mean_f1 = np.mean([r["f1_test"] for r in results]) if results else 0
        print(f"\n  Random CNN {layer_name} ({feats.shape[1]}-dim): "
              f"mean F1={mean_f1:.4f}")
        for r in results:
            print(f"    {r['variable']:<16} F1={r['f1_test']:.4f}")

    # Trained models at both layers
    for run_dir, label in [(DQN_RUN, "DQN"), (SPR_RUN, "DQN+SPR")]:
        config_path = os.path.join(run_dir, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        env = make_atari_env(
            env_id="BoxingNoFrameskip-v4", frame_size=84, num_stack=4,
            frame_skip=4, clip_rewards=False, episode_life=False,
            noop_max=30,
        )
        num_actions = env.action_space.n
        env.close()

        model = create_model(config, num_actions, device)
        is_rainbow = config.get("rainbow", {}).get("enabled", False)

        cp_path = os.path.join(run_dir, "checkpoints", "checkpoint_400000.pt")
        if not os.path.exists(cp_path):
            cp_path = os.path.join(run_dir, "checkpoints", "best_model.pt")
        if not os.path.exists(cp_path):
            print(f"\n  {label}: SKIP (no checkpoint)")
            continue

        checkpoint = torch.load(cp_path, map_location=device,
                                weights_only=False)
        model.load_state_dict(
            checkpoint["online_model_state_dict"], strict=True
        )
        if hasattr(model, "set_eval_mode"):
            model.set_eval_mode()
        else:
            model.eval()

        for layer_name in ["features", "conv_output"]:
            feats = encode_observations(
                model, observations, device, is_rainbow, layer=layer_name
            )
            results = run_probes(feats, labels, num_bins=num_bins)
            mean_f1 = (np.mean([r["f1_test"] for r in results])
                       if results else 0)
            print(f"\n  {label} {layer_name} ({feats.shape[1]}-dim): "
                  f"mean F1={mean_f1:.4f}")
            for r in results:
                print(f"    {r['variable']:<16} F1={r['f1_test']:.4f}")


def check_3_label_sanity(ram_snapshots):
    """Check 3: Do RAM addresses track expected values?"""
    print("\n" + "=" * 60)
    print("CHECK 3: Label sanity check")
    print("=" * 60)

    label_map = ATARI_ARI_LABELS["BoxingNoFrameskip-v4"]
    labels = apply_labels(ram_snapshots, label_map)

    print("\n  Variable distributions (random policy, "
          f"{ram_snapshots.shape[0]} steps):")
    print(f"  {'Variable':<16} {'Min':<6} {'Max':<6} {'Mean':<8} "
          f"{'Std':<8} {'Unique':<8} {'Non-zero%':<10}")
    print(f"  {'-' * 62}")

    for name, vals in labels.items():
        nz_pct = (vals != 0).mean() * 100
        print(f"  {name:<16} {vals.min():<6.0f} {vals.max():<6.0f} "
              f"{vals.mean():<8.1f} {vals.std():<8.1f} "
              f"{len(np.unique(vals)):<8} {nz_pct:<10.1f}")

    # Temporal smoothness: positions should change gradually
    print("\n  Temporal smoothness (mean |delta| between steps):")
    for name, vals in labels.items():
        deltas = np.abs(np.diff(vals))
        print(f"    {name:<16} mean_delta={deltas.mean():.2f}  "
              f"max_delta={deltas.max():.0f}")


def main():
    device = torch.device("cpu")

    if not os.path.exists(DATA_DIR):
        print(f"ERROR: No collected data at {DATA_DIR}")
        print("Run: python scripts/probe.py collect --game boxing")
        sys.exit(1)

    observations = np.load(os.path.join(DATA_DIR, "observations.npy"))
    ram_snapshots = np.load(os.path.join(DATA_DIR, "ram.npy"))
    print(f"Loaded {observations.shape[0]} observations from {DATA_DIR}")

    label_map = ATARI_ARI_LABELS["BoxingNoFrameskip-v4"]
    labels = apply_labels(ram_snapshots, label_map)

    check_3_label_sanity(ram_snapshots)
    check_4_conv_vs_fc(observations, labels, device)

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
