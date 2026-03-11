#!/usr/bin/env python3
"""Programmatic RAM address discovery for Atari games.

Records all 128 ALE RAM bytes across scripted action sequences (idle,
move in each direction, random play) and computes per-byte statistics
to identify candidate addresses for game state variables like player
position, score, and lives.

Usage:
    python scripts/ram_discover.py --game Pong
    python scripts/ram_discover.py --game CrazyClimber --num-random 2000
    python scripts/ram_discover.py --game Boxing --output output/ram/boxing.json
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.atari_wrappers import make_atari_env

# Short game name -> env_id mapping for our target games.
# Keys are lowercased with no separators for flexible matching.
GAME_ENV_IDS = {
    "pong": "PongNoFrameskip-v4",
    "breakout": "BreakoutNoFrameskip-v4",
    "boxing": "BoxingNoFrameskip-v4",
    "frostbite": "FrostbiteNoFrameskip-v4",
    "crazyclimber": "CrazyClimberNoFrameskip-v4",
    "kangaroo": "KangarooNoFrameskip-v4",
    "roadrunner": "RoadRunnerNoFrameskip-v4",
    "upndown": "UpNDownNoFrameskip-v4",
}


def resolve_env_id(game):
    """Convert a game name to a Gymnasium env_id.

    Accepts short names (Pong, CrazyClimber, road_runner) or full
    env_ids (PongNoFrameskip-v4).
    """
    key = game.lower().replace("_", "").replace("-", "").replace(" ", "")
    if key in GAME_ENV_IDS:
        return GAME_ENV_IDS[key]
    return game


def find_action(meanings, target):
    """Find the action index for a target direction name.

    Prefers exact matches ('RIGHT') over compound matches
    ('RIGHTFIRE'). Returns None if no match found.
    """
    for i, name in enumerate(meanings):
        if name == target:
            return i
    for i, name in enumerate(meanings):
        if name.startswith(target):
            return i
    return None


def get_ram(env):
    """Read current ALE RAM (128 bytes) as a numpy array."""
    return env.unwrapped.ale.getRAM().copy()


def get_lives(env):
    """Read current lives count from ALE."""
    return env.unwrapped.ale.lives()


def run_phase(env, action, num_steps):
    """Run a fixed action for num_steps, recording RAM, reward, and lives.

    Records the RAM state BEFORE each step, then steps with the given
    action. This means ram[i] is the state that produced reward[i].

    Returns:
        ram: (num_steps, 128) uint8 array
        rewards: (num_steps,) float array
        lives: (num_steps,) int array
    """
    ram_list = []
    rewards = []
    lives = []

    for _ in range(num_steps):
        ram_list.append(get_ram(env))
        lives.append(get_lives(env))
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            env.reset()

    return (
        np.array(ram_list, dtype=np.uint8),
        np.array(rewards, dtype=np.float32),
        np.array(lives, dtype=np.int32),
    )


def run_random_phase(env, num_steps):
    """Run random actions for num_steps, recording RAM and events."""
    ram_list = []
    rewards = []
    lives = []
    actions = []

    for _ in range(num_steps):
        ram_list.append(get_ram(env))
        lives.append(get_lives(env))
        action = env.action_space.sample()
        actions.append(int(action))
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(float(reward))
        if terminated or truncated:
            env.reset()

    return (
        np.array(ram_list, dtype=np.uint8),
        np.array(rewards, dtype=np.float32),
        np.array(lives, dtype=np.int32),
        np.array(actions, dtype=np.int32),
    )


def compute_entropy(values):
    """Compute Shannon entropy (bits) of a discrete value sequence."""
    _, counts = np.unique(values, return_counts=True)
    probs = counts / len(values)
    return float(-np.sum(probs * np.log2(probs + 1e-10)))


def compute_byte_statistics(phases, random_data):
    """Compute per-byte statistics across all recorded phases.

    Args:
        phases: dict mapping phase name -> (ram, rewards, lives)
        random_data: (ram, rewards, lives, actions) from random phase

    Returns:
        List of 128 dicts with per-byte statistics.
    """
    random_ram, random_rewards, random_lives, _ = random_data

    # Combine all RAM for global statistics
    all_ram_parts = [random_ram]
    for ram, _, _ in phases.values():
        all_ram_parts.append(ram)
    all_ram = np.concatenate(all_ram_parts, axis=0)

    stats = []
    for byte_idx in range(128):
        s = {"index": byte_idx}

        # --- Global statistics ---
        values = all_ram[:, byte_idx]
        s["min"] = int(values.min())
        s["max"] = int(values.max())
        s["mean"] = round(float(values.mean()), 2)
        s["std"] = round(float(values.std()), 2)
        s["unique_values"] = int(len(np.unique(values)))
        s["entropy"] = round(compute_entropy(values), 3)

        # Change frequency: fraction of consecutive frames where byte changed
        changes = np.diff(values.astype(np.int16)) != 0
        s["change_freq"] = round(float(changes.mean()), 4) if len(changes) > 0 else 0.0

        # --- Per-phase change frequency ---
        phase_freqs = {}
        for phase_name, (ram, _, _) in phases.items():
            pv = ram[:, byte_idx]
            pc = np.diff(pv.astype(np.int16)) != 0
            phase_freqs[phase_name] = round(float(pc.mean()), 4) if len(pc) > 0 else 0.0

        rv = random_ram[:, byte_idx]
        rc = np.diff(rv.astype(np.int16)) != 0
        phase_freqs["random"] = round(float(rc.mean()), 4) if len(rc) > 0 else 0.0
        s["phase_change_freq"] = phase_freqs

        # --- Directional mean diffs ---
        dir_diffs = {}
        for direction in ["right", "left", "up", "down"]:
            if direction in phases:
                pv = phases[direction][0][:, byte_idx]
                diffs = np.diff(pv.astype(np.int16))
                dir_diffs[direction] = round(float(diffs.mean()), 4) if len(diffs) > 0 else 0.0
        s["directional_mean_diff"] = dir_diffs

        # --- Score correlation ---
        # Compare byte change rate during reward frames vs non-reward frames.
        # changes[i] aligns with rewards[i]: both describe the step from frame i.
        if len(rc) > 0:
            reward_mask = random_rewards[:-1] != 0
            no_reward_mask = ~reward_mask

            if reward_mask.sum() > 0 and no_reward_mask.sum() > 0:
                rate_with = float(rc[reward_mask].mean())
                rate_without = float(rc[no_reward_mask].mean())
                s["score_correlation"] = round(rate_with - rate_without, 4)
            else:
                s["score_correlation"] = 0.0
        else:
            s["score_correlation"] = 0.0

        # --- Death correlation ---
        # Does this byte change when lives decrease?
        life_diffs = np.diff(random_lives)
        life_lost_mask = life_diffs < 0
        if life_lost_mask.sum() > 0:
            n = min(len(life_lost_mask), len(rc))
            s["death_correlation"] = round(float(rc[:n][life_lost_mask[:n]].mean()), 4)
        else:
            s["death_correlation"] = 0.0

        stats.append(s)

    return stats


def identify_candidates(stats):
    """Classify bytes into candidate variable types.

    Uses heuristics on per-byte statistics to suggest which addresses
    might correspond to position, score, lives, or timer variables.
    """
    candidates = {
        "position_x": [],
        "position_y": [],
        "score": [],
        "lives": [],
        "timer_or_counter": [],
        "constant": [],
    }

    for s in stats:
        idx = s["index"]

        if s["unique_values"] <= 1:
            candidates["constant"].append(idx)
            continue

        diffs = s.get("directional_mean_diff", {})

        # Position X: byte increases during right, decreases during left
        # (or vice versa -- screen coordinates vary by game)
        right_d = diffs.get("right", 0)
        left_d = diffs.get("left", 0)
        if abs(right_d) > 0.1 and abs(left_d) > 0.1:
            if (right_d > 0) != (left_d > 0):
                candidates["position_x"].append({
                    "index": idx,
                    "right_diff": right_d,
                    "left_diff": left_d,
                    "entropy": s["entropy"],
                })

        # Position Y: byte increases during up, decreases during down
        up_d = diffs.get("up", 0)
        down_d = diffs.get("down", 0)
        if abs(up_d) > 0.1 and abs(down_d) > 0.1:
            if (up_d > 0) != (down_d > 0):
                candidates["position_y"].append({
                    "index": idx,
                    "up_diff": up_d,
                    "down_diff": down_d,
                    "entropy": s["entropy"],
                })

        # Score: byte changes disproportionately when reward is non-zero
        if s.get("score_correlation", 0) > 0.2:
            candidates["score"].append({
                "index": idx,
                "score_correlation": s["score_correlation"],
                "entropy": s["entropy"],
            })

        # Lives: byte changes when lives decrease
        if s.get("death_correlation", 0) > 0.3:
            candidates["lives"].append({
                "index": idx,
                "death_correlation": s["death_correlation"],
                "unique_values": s["unique_values"],
            })

        # Timer/counter: moderate entropy, steady changes, not score-linked
        idle_freq = s.get("phase_change_freq", {}).get("idle", 0)
        if (2.0 < s["entropy"] < 7.0
                and s["change_freq"] > 0.3
                and idle_freq > 0.2
                and abs(s.get("score_correlation", 0)) < 0.2):
            candidates["timer_or_counter"].append({
                "index": idx,
                "entropy": s["entropy"],
                "change_freq": s["change_freq"],
            })

    # Sort by strongest signal
    candidates["position_x"].sort(
        key=lambda x: abs(x["right_diff"]) + abs(x["left_diff"]), reverse=True)
    candidates["position_y"].sort(
        key=lambda x: abs(x["up_diff"]) + abs(x["down_diff"]), reverse=True)
    candidates["score"].sort(
        key=lambda x: x["score_correlation"], reverse=True)
    candidates["lives"].sort(
        key=lambda x: x["death_correlation"], reverse=True)

    return candidates


def print_summary(game, stats, candidates, action_meanings):
    """Print a human-readable summary of discovery results."""
    print(f"\n{'=' * 60}")
    print(f"RAM Discovery Results: {game}")
    print(f"{'=' * 60}")

    print(f"\nAction space: {len(action_meanings)} actions")
    print(f"  {', '.join(action_meanings)}")

    active = [s for s in stats if s["unique_values"] > 1]
    print(f"\nActive bytes: {len(active)} / 128"
          f" ({128 - len(active)} constant)")

    # Top entropy bytes
    by_entropy = sorted(active, key=lambda s: s["entropy"], reverse=True)
    print(f"\nTop 10 highest-entropy bytes:")
    print(f"  {'Idx':>4}  {'Entropy':>8}  {'ChgFreq':>8}"
          f"  {'Min':>4}  {'Max':>4}  {'Unique':>6}")
    for s in by_entropy[:10]:
        print(f"  {s['index']:>4}  {s['entropy']:>8.3f}"
              f"  {s['change_freq']:>8.4f}  {s['min']:>4}"
              f"  {s['max']:>4}  {s['unique_values']:>6}")

    # Candidate tables
    sections = [
        ("position_x", "Position X candidates"),
        ("position_y", "Position Y candidates"),
        ("score", "Score candidates"),
        ("lives", "Lives candidates"),
        ("timer_or_counter", "Timer / counter candidates"),
    ]
    for key, label in sections:
        items = candidates[key]
        if items:
            print(f"\n{label}:")
            for item in items[:5]:
                parts = [f"  ram[{item['index']:>3}]"]
                for k, v in item.items():
                    if k != "index":
                        if isinstance(v, float):
                            parts.append(f"{k}={v:.4f}")
                        else:
                            parts.append(f"{k}={v}")
                print("  ".join(parts))
        else:
            print(f"\n{label}: (none found)")

    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Discover RAM address mappings for Atari games",
    )
    parser.add_argument(
        "--game", required=True,
        help="Game name (e.g., Pong, CrazyClimber) or full env_id",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON path (default: output/ram/<game>.json)",
    )
    parser.add_argument(
        "--num-idle", type=int, default=30,
        help="Steps for idle (NOOP) phase (default: 30)",
    )
    parser.add_argument(
        "--num-directional", type=int, default=40,
        help="Steps per directional phase (default: 40)",
    )
    parser.add_argument(
        "--num-random", type=int, default=1000,
        help="Steps for random play phase (default: 1000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    env_id = resolve_env_id(args.game)
    print(f"Game: {args.game} -> {env_id}")

    # Create environment with minimal randomness for reproducibility.
    # clip_rewards=False to preserve raw reward magnitudes for score
    # correlation analysis. noop_max=0 for deterministic resets.
    env = make_atari_env(
        env_id=env_id,
        frame_size=84,
        num_stack=4,
        frame_skip=4,
        clip_rewards=False,
        episode_life=False,
        noop_max=0,
    )
    env.reset(seed=args.seed)

    action_meanings = env.unwrapped.get_action_meanings()
    print(f"Actions ({len(action_meanings)}): {action_meanings}")

    # Map direction names to action indices
    noop = find_action(action_meanings, "NOOP") or 0
    fire = find_action(action_meanings, "FIRE")

    directions = {}
    for d in ["RIGHT", "LEFT", "UP", "DOWN"]:
        a = find_action(action_meanings, d)
        if a is not None:
            directions[d.lower()] = a
    print(f"Directions available: {directions}")

    # Fire once to start the game (some games require this)
    if fire is not None:
        env.step(fire)

    # --- Scripted phases ---
    phases = {}

    print(f"\nPhase: idle ({args.num_idle} steps)...")
    phases["idle"] = run_phase(env, noop, args.num_idle)

    for direction, action in directions.items():
        print(f"Phase: {direction} ({args.num_directional} steps)...")
        phases[direction] = run_phase(env, action, args.num_directional)

    # Reset for random phase to get fresh score/death events
    print(f"Phase: random ({args.num_random} steps)...")
    env.reset(seed=args.seed + 1)
    if fire is not None:
        env.step(fire)
    random_data = run_random_phase(env, args.num_random)

    env.close()

    # --- Analysis ---
    print("\nComputing per-byte statistics...")
    stats = compute_byte_statistics(phases, random_data)
    candidates = identify_candidates(stats)

    print_summary(args.game, stats, candidates, action_meanings)

    # --- Save JSON output ---
    if args.output is None:
        game_key = args.game.lower().replace(" ", "_")
        output_path = os.path.join("output", "ram", f"{game_key}.json")
    else:
        output_path = args.output

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output = {
        "game": args.game,
        "env_id": env_id,
        "action_meanings": action_meanings,
        "directions_found": {k: int(v) for k, v in directions.items()},
        "phase_config": {
            "idle_steps": args.num_idle,
            "directional_steps": args.num_directional,
            "random_steps": args.num_random,
        },
        "seed": args.seed,
        "byte_statistics": stats,
        "candidates": {
            k: v if k == "constant" else v
            for k, v in candidates.items()
        },
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Full statistics saved to {output_path}")


if __name__ == "__main__":
    main()
