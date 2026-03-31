#!/usr/bin/env python3
"""Linear probing script for encoder representations.

Two-phase approach following AtariARI (Anand et al. 2019):
1. Collect observations and RAM using a random policy (shared
   across all encoders)
2. For each encoder, pass the saved observations through to
   get representations, then train linear probes

Usage:
    # Collect shared observations for a game
    python scripts/probe.py collect --game boxing --num-episodes 20

    # Probe a specific run against the collected observations
    python scripts/probe.py probe --game boxing --run-dirs RUN_DIR

    # Probe multiple runs and compare
    python scripts/probe.py probe --game boxing --run-dirs RUN_1 RUN_2
"""

import argparse
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.atari_wrappers import make_atari_env
from src.models.dqn import DQN
from src.models.rainbow import RainbowDQN

# ---------------------------------------------------------------------------
# Game name -> env_id mapping
# ---------------------------------------------------------------------------
GAME_ENV_IDS = {
    "boxing": "BoxingNoFrameskip-v4",
    "frostbite": "FrostbiteNoFrameskip-v4",
    "pong": "PongNoFrameskip-v4",
    "breakout": "BreakoutNoFrameskip-v4",
    "crazy_climber": "CrazyClimberNoFrameskip-v4",
    "kangaroo": "KangarooNoFrameskip-v4",
    "road_runner": "RoadRunnerNoFrameskip-v4",
    "up_n_down": "UpNDownNoFrameskip-v4",
}

# ---------------------------------------------------------------------------
# AtariARI label maps: address -> variable name
# Source: github.com/mila-iqia/atari-representation-learning
# ---------------------------------------------------------------------------
ATARI_ARI_LABELS = {
    "BoxingNoFrameskip-v4": {
        32: "player_x",
        34: "player_y",
        33: "enemy_x",
        35: "enemy_y",
        18: "player_score",
        19: "enemy_score",
        17: "clock",
    },
}

# Number of bins for discretizing continuous variables (AtariARI default)
NUM_BINS = 256


def create_model(config, num_actions, device):
    """Create a model matching the run config."""
    rainbow_cfg = config.get("rainbow", {})
    dropout = config.get("network", {}).get("dropout", 0.0)

    if rainbow_cfg.get("enabled", False):
        dist = rainbow_cfg.get("distributional", {})
        fc_hidden = config.get("network", {}).get("fc_hidden", 512)
        model = RainbowDQN(
            num_actions=num_actions,
            num_atoms=dist.get("num_atoms", 51),
            v_min=dist.get("v_min", -10.0),
            v_max=dist.get("v_max", 10.0),
            noisy=rainbow_cfg.get("noisy_nets", True),
            dueling=rainbow_cfg.get("dueling", True),
            dropout=dropout,
            fc_hidden=fc_hidden,
        )
    else:
        model = DQN(num_actions=num_actions, dropout=dropout)

    return model.to(device)


# ---------------------------------------------------------------------------
# Phase 1: Collect observations and RAM with random policy
# ---------------------------------------------------------------------------

def collect_observations(env_id, num_episodes, output_dir,
                         repeat_action_probability=0.25):
    """Run random-policy episodes, saving observations and RAM.

    Saves:
        output_dir/observations.npy  -- (N, 4, 84, 84) uint8
        output_dir/ram.npy           -- (N, 128) uint8
        output_dir/meta.yaml         -- collection metadata
    """
    env = make_atari_env(
        env_id=env_id,
        frame_size=84,
        num_stack=4,
        frame_skip=4,
        clip_rewards=False,
        episode_life=False,
        noop_max=30,
        repeat_action_probability=repeat_action_probability,
    )

    all_obs = []
    all_ram = []
    total_steps = 0

    for ep in range(num_episodes):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            all_obs.append(obs)
            ram = env.unwrapped.ale.getRAM().copy()
            all_ram.append(ram)

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        total_steps += steps
        print(f"  Episode {ep + 1}/{num_episodes}: {steps} steps")

    env.close()

    observations = np.stack(all_obs)
    ram_snapshots = np.stack(all_ram)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "observations.npy"), observations)
    np.save(os.path.join(output_dir, "ram.npy"), ram_snapshots)

    meta = {
        "env_id": env_id,
        "num_episodes": num_episodes,
        "total_steps": total_steps,
        "obs_shape": list(observations.shape),
        "repeat_action_probability": repeat_action_probability,
        "policy": "random",
    }
    with open(os.path.join(output_dir, "meta.yaml"), "w") as f:
        yaml.dump(meta, f)

    print(f"\nSaved {total_steps} observations to {output_dir}")
    print(f"  observations.npy: {observations.shape}")
    print(f"  ram.npy: {ram_snapshots.shape}")

    return observations, ram_snapshots


# ---------------------------------------------------------------------------
# Phase 2: Encode observations and run probes
# ---------------------------------------------------------------------------

def encode_observations(model, observations, device, is_rainbow=False,
                        batch_size=256, layer="features"):
    """Pass saved observations through an encoder.

    Args:
        model: DQN or RainbowDQN
        observations: (N, 4, 84, 84) uint8 numpy array
        device: torch device
        is_rainbow: if True and layer="features", extract from value_fc
        layer: "features" (512-dim post-FC) or "conv_output" (3136-dim
               flattened spatial features)

    Returns:
        features: (N, feat_dim) numpy array
    """
    all_features = []
    n = observations.shape[0]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = observations[start:end]

        obs_tensor = (
            torch.from_numpy(batch.astype(np.float32) / 255.0)
            .to(device)
        )

        with torch.no_grad():
            out = model(obs_tensor)

        if layer == "conv_output":
            conv_out = out["conv_output"]
            h_t = conv_out.reshape(conv_out.size(0), -1)
        elif is_rainbow:
            conv_out = out["conv_output"]
            flat = conv_out.reshape(conv_out.size(0), -1)
            h_t = model.drop(torch.relu(model.value_fc(flat)))
        else:
            h_t = out["features"]

        all_features.append(h_t.cpu().numpy())

    return np.concatenate(all_features, axis=0)


def apply_labels(ram_snapshots, label_map):
    """Extract named variables from RAM snapshots."""
    labels = {}
    for addr, name in label_map.items():
        labels[name] = ram_snapshots[:, addr].astype(np.float32)
    return labels


def discretize(values, num_bins=NUM_BINS):
    """Bin continuous values into discrete classes.

    Following AtariARI: uniform bins across the observed range.
    Returns integer class labels.
    """
    v_min, v_max = values.min(), values.max()
    if v_min == v_max:
        return np.zeros_like(values, dtype=np.int64)

    bins = np.linspace(v_min, v_max, num_bins + 1)
    binned = np.digitize(values, bins) - 1
    binned = np.clip(binned, 0, num_bins - 1)
    return binned.astype(np.int64)


def run_probes(features, labels, test_fraction=0.2, num_bins=NUM_BINS):
    """Train linear probes per variable following AtariARI protocol.

    Discretizes all variables and uses LogisticRegression.
    Reports F1 (macro) per variable.

    Returns:
        results: list of dicts
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    n = features.shape[0]
    indices = np.arange(n)

    train_idx, test_idx = train_test_split(
        indices, test_size=test_fraction, random_state=42
    )

    X_train = features[train_idx]
    X_test = features[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = []

    for var_name, y_raw in labels.items():
        y = discretize(y_raw, num_bins=num_bins)
        y_train = y[train_idx]
        y_test = y[test_idx]

        n_classes = len(np.unique(y_train))

        if n_classes <= 1:
            print(f"  {var_name}: skipped (constant value)")
            continue

        # Filter low-entropy variables (AtariARI threshold: 0.6)
        counts = np.bincount(y_train)
        probs = counts[counts > 0] / counts.sum()
        entropy = -(probs * np.log2(probs)).sum()
        max_entropy = np.log2(n_classes)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0

        if norm_entropy < 0.6:
            print(f"  {var_name}: skipped (low entropy: {norm_entropy:.2f})")
            continue

        clf = LogisticRegression(
            max_iter=1000, random_state=42, solver="lbfgs",
        )
        clf.fit(X_train, y_train)

        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)

        f1_train = f1_score(y_train, y_train_pred, average="macro",
                            zero_division=0)
        f1_test = f1_score(y_test, y_test_pred, average="macro",
                           zero_division=0)
        accuracy_test = clf.score(X_test, y_test)

        results.append({
            "variable": var_name,
            "f1_train": f1_train,
            "f1_test": f1_test,
            "accuracy": accuracy_test,
            "n_classes": n_classes,
            "entropy": norm_entropy,
        })

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def cmd_collect(args):
    """Collect observations with random policy."""
    if args.game not in GAME_ENV_IDS:
        print(f"ERROR: Unknown game '{args.game}'")
        print(f"Available: {list(GAME_ENV_IDS.keys())}")
        sys.exit(1)

    env_id = GAME_ENV_IDS[args.game]

    if env_id not in ATARI_ARI_LABELS:
        print(f"WARNING: No AtariARI labels for {env_id}. "
              f"Collection will proceed but probing will fail.")

    output_dir = os.path.join("output", "probe_data", args.game)
    print(f"Game: {env_id}")
    print(f"Collecting {args.num_episodes} episodes with random policy...\n")

    collect_observations(env_id, args.num_episodes, output_dir)


def cmd_probe(args):
    """Probe one or more runs against collected observations."""
    game = args.game
    data_dir = os.path.join("output", "probe_data", game)

    if not os.path.exists(data_dir):
        print(f"ERROR: No collected data for '{game}'")
        print(f"Run: python scripts/probe.py collect --game {game}")
        sys.exit(1)

    observations = np.load(os.path.join(data_dir, "observations.npy"))
    ram_snapshots = np.load(os.path.join(data_dir, "ram.npy"))

    with open(os.path.join(data_dir, "meta.yaml")) as f:
        meta = yaml.safe_load(f)

    env_id = meta["env_id"]
    print(f"Game: {env_id}")
    print(f"Loaded {observations.shape[0]} observations from {data_dir}")

    if env_id not in ATARI_ARI_LABELS:
        print(f"ERROR: No AtariARI labels for {env_id}")
        sys.exit(1)

    label_map = ATARI_ARI_LABELS[env_id]
    labels = apply_labels(ram_snapshots, label_map)

    print(f"\nLabel ranges:")
    for name, vals in labels.items():
        print(f"  {name}: min={vals.min():.0f}, max={vals.max():.0f}, "
              f"unique={len(np.unique(vals))}")

    device = torch.device(args.device)

    all_results = {}

    for run_dir in args.run_dirs:
        run_name = os.path.basename(run_dir)
        print(f"\n{'=' * 60}")
        print(f"Run: {run_name}")

        config_path = os.path.join(run_dir, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)

        run_env_id = config["environment"]["env_id"]
        if run_env_id != env_id:
            print(f"  SKIP: env_id mismatch ({run_env_id} != {env_id})")
            continue

        # Get num_actions from a temp environment
        env = make_atari_env(
            env_id=env_id, frame_size=84, num_stack=4, frame_skip=4,
            clip_rewards=False, episode_life=False, noop_max=30,
        )
        num_actions = env.action_space.n
        env.close()

        model = create_model(config, num_actions, device)
        is_rainbow = config.get("rainbow", {}).get("enabled", False)

        # Find checkpoint
        if args.checkpoint_step is not None:
            cp_name = f"checkpoint_{args.checkpoint_step}.pt"
        else:
            cp_name = "best_model.pt"
            cp_path = os.path.join(run_dir, "checkpoints", cp_name)
            if not os.path.exists(cp_path):
                cp_dir = os.path.join(run_dir, "checkpoints")
                if os.path.isdir(cp_dir):
                    cps = sorted(
                        [f for f in os.listdir(cp_dir)
                         if f.startswith("checkpoint_")],
                        key=lambda f: int(
                            f.replace("checkpoint_", "").replace(".pt", "")
                        )
                    )
                    if cps:
                        cp_name = cps[-1]

        cp_path = os.path.join(run_dir, "checkpoints", cp_name)
        if not os.path.exists(cp_path):
            print(f"  SKIP: no checkpoint found at {cp_path}")
            continue

        print(f"  Checkpoint: {cp_name}")
        print(f"  Model: {'RainbowDQN' if is_rainbow else 'DQN'}")

        checkpoint = torch.load(cp_path, map_location=device,
                                weights_only=False)
        model.load_state_dict(
            checkpoint["online_model_state_dict"], strict=True
        )
        if hasattr(model, "set_eval_mode"):
            model.set_eval_mode()
        else:
            model.eval()

        print(f"  Encoding {observations.shape[0]} observations...")
        features = encode_observations(
            model, observations, device, is_rainbow
        )
        print(f"  Feature dim: {features.shape[1]}")

        results = run_probes(features, labels, num_bins=args.num_bins)
        all_results[run_name] = results

        print(f"\n  {'Variable':<16} {'F1 test':<10} {'Acc':<10} "
              f"{'Classes':<10} {'Entropy':<10}")
        print(f"  {'-' * 56}")
        for r in results:
            print(f"  {r['variable']:<16} {r['f1_test']:<10.4f} "
                  f"{r['accuracy']:<10.4f} {r['n_classes']:<10} "
                  f"{r['entropy']:<10.2f}")

    # Comparison table
    if len(all_results) > 1:
        print(f"\n{'=' * 60}")
        print("COMPARISON (F1 test)")
        print(f"{'=' * 60}")

        all_vars = []
        for results in all_results.values():
            for r in results:
                if r["variable"] not in all_vars:
                    all_vars.append(r["variable"])

        run_names = list(all_results.keys())
        short_names = [n.replace("atari100k_", "").rsplit("_", 2)[0]
                       for n in run_names]
        header = f"{'Variable':<16}" + "".join(
            f"{s:<20}" for s in short_names
        )
        print(header)
        print("-" * len(header))

        for var in all_vars:
            row = f"{var:<16}"
            for rn in run_names:
                match = [r for r in all_results[rn]
                         if r["variable"] == var]
                if match:
                    row += f"{match[0]['f1_test']:<20.4f}"
                else:
                    row += f"{'--':<20}"
            print(row)


def main():
    parser = argparse.ArgumentParser(
        description="Linear probing for encoder representations."
    )
    subparsers = parser.add_subparsers(dest="command")

    collect_p = subparsers.add_parser(
        "collect", help="Collect observations with random policy"
    )
    collect_p.add_argument("--game", required=True)
    collect_p.add_argument("--num-episodes", type=int, default=20)

    probe_p = subparsers.add_parser(
        "probe", help="Probe runs against collected observations"
    )
    probe_p.add_argument("--game", required=True)
    probe_p.add_argument("--run-dirs", nargs="+", required=True)
    probe_p.add_argument("--checkpoint-step", type=int, default=None)
    probe_p.add_argument("--device", default="cpu")
    probe_p.add_argument("--num-bins", type=int, default=256,
                         help="Bins for discretization (default: 256)")

    args = parser.parse_args()

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "probe":
        cmd_probe(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
