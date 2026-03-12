#!/usr/bin/env python3
"""Programmatic RAM address discovery and verification for Atari games.

Discovery mode: records all 128 ALE RAM bytes across scripted action
sequences and computes per-byte statistics to identify candidate
addresses for game state variables.

Verification mode (--verify): checks known RAM address mappings
(from OCAtari or AtariARI) against actual gameplay behavior. Outputs
a per-address confidence rating: VERIFIED, SUSPECT, or FAILED.

Usage:
    python scripts/ram_discover.py --game Pong
    python scripts/ram_discover.py --game Kangaroo --verify
    python scripts/ram_discover.py --game CrazyClimber --verify --num-random 2000
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

# Known RAM address mappings from OCAtari and AtariARI.
# Each game maps to a source label and a list of address entries.
# Types: position_x, position_y (player axes with directional response),
# score, lives, level, timer, state (discrete), object (NPC/item byte).
KNOWN_MAPPINGS = {
    "kangaroo": {
        "source": "OCAtari",
        "addresses": [
            {"addr": 17, "name": "player_x", "type": "position_x"},
            {"addr": 16, "name": "player_y", "type": "position_y"},
            {"addr": 18, "name": "player_state", "type": "state"},
            {"addr": 39, "name": "score_hi", "type": "score"},
            {"addr": 40, "name": "score_lo", "type": "score"},
            {"addr": 45, "name": "lives", "type": "lives"},
            {"addr": 36, "name": "level", "type": "level"},
            {"addr": 59, "name": "time_remaining", "type": "timer"},
            {"addr": 54, "name": "crash_state", "type": "state"},
            {"addr": 83, "name": "child_x", "type": "object"},
            {"addr": 15, "name": "monkey_0_x", "type": "object"},
            {"addr": 14, "name": "monkey_1_x", "type": "object"},
            {"addr": 13, "name": "monkey_2_x", "type": "object"},
            {"addr": 12, "name": "monkey_3_x", "type": "object"},
            {"addr": 11, "name": "monkey_0_y", "type": "object"},
            {"addr": 10, "name": "monkey_1_y", "type": "object"},
            {"addr": 9, "name": "monkey_2_y", "type": "object"},
            {"addr": 8, "name": "monkey_3_y", "type": "object"},
            {"addr": 3, "name": "monkey_0_state", "type": "state"},
            {"addr": 2, "name": "monkey_1_state", "type": "state"},
            {"addr": 1, "name": "monkey_2_state", "type": "state"},
            {"addr": 0, "name": "monkey_3_state", "type": "state"},
            {"addr": 25, "name": "coconut_0_y", "type": "object"},
            {"addr": 26, "name": "coconut_1_y", "type": "object"},
            {"addr": 27, "name": "coconut_2_y", "type": "object"},
            {"addr": 28, "name": "coconut_0_x", "type": "object"},
            {"addr": 29, "name": "coconut_1_x", "type": "object"},
            {"addr": 30, "name": "coconut_2_x", "type": "object"},
            {"addr": 33, "name": "falling_coconut_y", "type": "object"},
            {"addr": 34, "name": "falling_coconut_x", "type": "object"},
            {"addr": 41, "name": "bell_status", "type": "state"},
            {"addr": 42, "name": "fruit_0_state", "type": "state"},
            {"addr": 43, "name": "fruit_1_state", "type": "state"},
            {"addr": 44, "name": "fruit_2_state", "type": "state"},
        ],
    },
    "upndown": {
        "source": "OCAtari",
        "addresses": [
            {"addr": 40, "name": "player_x", "type": "position_x"},
            {"addr": 56, "name": "player_y", "type": "position_y"},
            {"addr": 0, "name": "score_0", "type": "score"},
            {"addr": 1, "name": "score_1", "type": "score"},
            {"addr": 2, "name": "score_2", "type": "score"},
            {"addr": 6, "name": "lives", "type": "lives"},
            {"addr": 4, "name": "hud_flags", "type": "state"},
            {"addr": 36, "name": "object_type_0", "type": "state"},
            {"addr": 37, "name": "object_type_1", "type": "state"},
            {"addr": 38, "name": "object_type_2", "type": "state"},
            {"addr": 39, "name": "object_type_3", "type": "state"},
            {"addr": 41, "name": "object_1_x", "type": "object"},
            {"addr": 42, "name": "object_2_x", "type": "object"},
            {"addr": 43, "name": "object_3_x", "type": "object"},
            {"addr": 57, "name": "object_1_y", "type": "object"},
            {"addr": 58, "name": "object_2_y", "type": "object"},
            {"addr": 59, "name": "object_3_y", "type": "object"},
        ],
    },
    "crazyclimber": {
        "source": "OCAtari",
        "addresses": [
            {"addr": 24, "name": "player_x", "type": "position_x"},
            {"addr": 42, "name": "lives", "type": "lives"},
            {"addr": 58, "name": "floor_offset", "type": "object"},
            {"addr": 9, "name": "helicopter_y", "type": "object"},
            {"addr": 34, "name": "helicopter_bird_x", "type": "object"},
            {"addr": 14, "name": "enemy_x", "type": "object"},
            {"addr": 15, "name": "enemy_y", "type": "object"},
            {"addr": 16, "name": "enemy_height", "type": "object"},
            {"addr": 84, "name": "enemy_type", "type": "state"},
            {"addr": 85, "name": "ball_x", "type": "object"},
        ],
    },
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


def verify_address(addr_entry, byte_stats):
    """Verify a single claimed RAM address against observed byte statistics.

    Applies type-specific checks to determine whether the byte at the
    claimed address behaves consistently with its claimed variable type.

    Args:
        addr_entry: dict with 'addr' (int), 'name' (str), 'type' (str)
        byte_stats: list of 128 per-byte statistic dicts

    Returns:
        dict with addr, name, type, rating (VERIFIED/SUSPECT/FAILED),
        checks list, and observed statistics summary.
    """
    addr = addr_entry["addr"]
    name = addr_entry["name"]
    claimed_type = addr_entry["type"]
    s = byte_stats[addr]

    # Each check: (name, passed, detail, severity)
    # Severity: "critical" -> FAIL means FAILED overall
    #           "strong"   -> FAIL means SUSPECT at best
    #           "moderate" -> FAIL noted but doesn't downgrade alone
    #           "info"     -> informational, ignored for rating
    checks = []

    is_constant = s["unique_values"] <= 1
    random_freq = s.get("phase_change_freq", {}).get("random", 0)
    diffs = s.get("directional_mean_diff", {})

    if claimed_type == "position_x":
        _checks_position(s, checks, diffs, "x",
                         ("right", "left"), random_freq, is_constant)

    elif claimed_type == "position_y":
        _checks_position(s, checks, diffs, "y",
                         ("up", "down"), random_freq, is_constant)

    elif claimed_type == "score":
        _checks_score(s, checks, is_constant)

    elif claimed_type == "lives":
        _checks_lives(s, checks, is_constant)

    elif claimed_type == "level":
        _checks_level(s, checks)

    elif claimed_type == "timer":
        _checks_timer(s, checks, is_constant)

    else:  # state, object, misc
        _checks_generic(s, checks, is_constant)

    # --- Determine overall rating ---
    critical_fails = any(
        not c[1] for c in checks if c[3] == "critical")
    strong_fails = any(
        not c[1] for c in checks if c[3] == "strong")
    non_info = [c for c in checks if c[3] != "info"]
    all_pass = all(c[1] for c in non_info) if non_info else True

    if critical_fails:
        rating = "FAILED"
    elif all_pass:
        rating = "VERIFIED"
    elif strong_fails:
        rating = "SUSPECT"
    else:
        rating = "VERIFIED"

    return {
        "addr": addr,
        "name": name,
        "type": claimed_type,
        "rating": rating,
        "checks": [
            {"check": c[0], "result": "PASS" if c[1] else "FAIL",
             "detail": c[2]}
            for c in checks if c[3] != "info"
        ],
        "observed": {
            "min": s["min"],
            "max": s["max"],
            "entropy": round(s["entropy"], 3),
            "change_freq": round(s["change_freq"], 4),
            "unique_values": s["unique_values"],
        },
    }


def _checks_position(s, checks, diffs, axis, dir_pair, random_freq,
                      is_constant):
    """Position checks for player X or Y axes."""
    pos_name, neg_name = dir_pair

    if is_constant:
        checks.append(("not_constant", False,
                        f"constant at {s['min']}", "critical"))
    else:
        checks.append(("not_constant", True,
                        f"{s['unique_values']} unique", "info"))

    # Entropy: position bytes should span many values
    if s["entropy"] > 1.0:
        checks.append(("entropy", True,
                        f"{s['entropy']:.2f} bits", "moderate"))
    else:
        checks.append(("entropy", False,
                        f"{s['entropy']:.2f} bits (expected > 1.0)",
                        "moderate"))

    # Directional correlation: opposite diffs for paired directions
    pos_d = diffs.get(pos_name, 0)
    neg_d = diffs.get(neg_name, 0)
    if abs(pos_d) > 0.05 and abs(neg_d) > 0.05:
        if (pos_d > 0) != (neg_d > 0):
            checks.append(("directional", True,
                            f"{pos_name}={pos_d:+.3f} "
                            f"{neg_name}={neg_d:+.3f}", "strong"))
        else:
            checks.append(("directional", False,
                            f"same sign: {pos_name}={pos_d:+.3f} "
                            f"{neg_name}={neg_d:+.3f}", "moderate"))
    else:
        checks.append(("directional", False,
                        f"no signal: {pos_name}={pos_d:+.3f} "
                        f"{neg_name}={neg_d:+.3f}", "moderate"))

    # Active during random play
    if random_freq > 0.02:
        checks.append(("active_in_play", True,
                        f"freq={random_freq:.4f}", "moderate"))
    else:
        checks.append(("active_in_play", False,
                        f"freq={random_freq:.4f} (expected > 0.02)",
                        "moderate"))


def _checks_score(s, checks, is_constant):
    """Score byte checks: correlation with reward events."""
    if is_constant:
        checks.append(("not_constant", False,
                        f"constant at {s['min']}", "critical"))
    else:
        checks.append(("not_constant", True,
                        f"{s['unique_values']} unique", "info"))

    score_corr = s.get("score_correlation", 0)
    if score_corr > 0.1:
        checks.append(("score_event_corr", True,
                        f"corr={score_corr:.4f}", "strong"))
    elif score_corr > 0.02:
        checks.append(("score_event_corr", True,
                        f"weak corr={score_corr:.4f}", "moderate"))
    else:
        checks.append(("score_event_corr", False,
                        f"corr={score_corr:.4f} (expected > 0.1)",
                        "strong"))


def _checks_lives(s, checks, is_constant):
    """Lives byte checks: death correlation and low cardinality."""
    death_corr = s.get("death_correlation", 0)
    if death_corr > 0.3:
        checks.append(("death_event_corr", True,
                        f"corr={death_corr:.4f}", "strong"))
    elif is_constant:
        # No deaths may have occurred -- inconclusive, not failure
        checks.append(("death_event_corr", False,
                        f"constant (no deaths observed?)", "moderate"))
    elif death_corr > 0.0:
        checks.append(("death_event_corr", True,
                        f"weak corr={death_corr:.4f}", "moderate"))
    else:
        checks.append(("death_event_corr", False,
                        f"corr={death_corr:.4f}", "strong"))

    if s["unique_values"] <= 8:
        checks.append(("low_cardinality", True,
                        f"{s['unique_values']} unique", "moderate"))
    else:
        checks.append(("low_cardinality", False,
                        f"{s['unique_values']} unique (expected <= 8)",
                        "strong"))


def _checks_level(s, checks):
    """Level/stage byte checks: low cardinality, infrequent changes."""
    if s["unique_values"] <= 20:
        checks.append(("low_cardinality", True,
                        f"{s['unique_values']} unique", "moderate"))
    else:
        checks.append(("low_cardinality", False,
                        f"{s['unique_values']} unique (expected <= 20)",
                        "moderate"))

    if s["change_freq"] < 0.1:
        checks.append(("low_change_freq", True,
                        f"freq={s['change_freq']:.4f}", "moderate"))
    else:
        checks.append(("low_change_freq", False,
                        f"freq={s['change_freq']:.4f} (expected < 0.1)",
                        "moderate"))


def _checks_timer(s, checks, is_constant):
    """Timer/clock checks: frequent changes, active during idle."""
    if is_constant:
        checks.append(("not_constant", False,
                        f"constant at {s['min']}", "critical"))
    else:
        checks.append(("not_constant", True,
                        f"{s['unique_values']} unique", "info"))

    if s["change_freq"] > 0.1:
        checks.append(("high_change_freq", True,
                        f"freq={s['change_freq']:.4f}", "strong"))
    else:
        checks.append(("high_change_freq", False,
                        f"freq={s['change_freq']:.4f} (expected > 0.1)",
                        "strong"))

    idle_freq = s.get("phase_change_freq", {}).get("idle", 0)
    if idle_freq > 0.05:
        checks.append(("changes_during_idle", True,
                        f"idle_freq={idle_freq:.4f}", "moderate"))
    else:
        checks.append(("changes_during_idle", False,
                        f"idle_freq={idle_freq:.4f} (expected > 0.05)",
                        "moderate"))


def _checks_generic(s, checks, is_constant):
    """Generic checks for state, object, and misc bytes."""
    random_freq = s.get("phase_change_freq", {}).get("random", 0)
    if is_constant:
        checks.append(("active", False,
                        f"constant at {s['min']}", "moderate"))
    else:
        checks.append(("active", True,
                        f"{s['unique_values']} unique, "
                        f"freq={random_freq:.4f}", "moderate"))


def run_verification(game, byte_stats):
    """Run verification for all known addresses of a game.

    Args:
        game: game name (will be normalized for lookup)
        byte_stats: list of 128 per-byte statistic dicts

    Returns:
        (results, mapping) tuple where results is a list of per-address
        verification dicts, or (None, None) if no mapping exists.
    """
    key = game.lower().replace("_", "").replace("-", "").replace(" ", "")
    if key not in KNOWN_MAPPINGS:
        return None, None

    mapping = KNOWN_MAPPINGS[key]
    results = []
    for addr_entry in mapping["addresses"]:
        result = verify_address(addr_entry, byte_stats)
        results.append(result)

    return results, mapping


def print_verification_report(game, results, mapping):
    """Print a formatted verification report."""
    source = mapping["source"]

    print(f"\n{'=' * 70}")
    print(f"Verification Report: {game} (source: {source})")
    print(f"{'=' * 70}")
    print(f"\n{'Addr':>4}  {'Name':<22} {'Type':<12} {'Rating':<10} "
          f"{'Key Detail'}")
    print(f"{'----':>4}  {'----':<22} {'----':<12} {'------':<10} "
          f"{'----------'}")

    for r in results:
        # Pick the most informative check detail for the summary column
        detail = ""
        for c in r["checks"]:
            if c["result"] == "FAIL":
                detail = c["detail"]
                break
        if not detail and r["checks"]:
            detail = r["checks"][0]["detail"]

        rating_str = r["rating"]
        print(f"{r['addr']:>4}  {r['name']:<22} {r['type']:<12} "
              f"{rating_str:<10} {detail}")

    # Summary counts
    counts = {"VERIFIED": 0, "SUSPECT": 0, "FAILED": 0}
    for r in results:
        counts[r["rating"]] += 1

    print(f"\nSummary: {counts['VERIFIED']} VERIFIED, "
          f"{counts['SUSPECT']} SUSPECT, {counts['FAILED']} FAILED "
          f"(out of {len(results)} addresses)")
    print(f"{'=' * 70}\n")


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
    parser.add_argument(
        "--verify", action="store_true",
        help="Run verification against known address mappings",
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

    # --- Verification mode ---
    verification_results = None
    verification_mapping = None
    if args.verify:
        verification_results, verification_mapping = run_verification(
            args.game, stats)
        if verification_results is None:
            game_key = args.game.lower().replace(
                "_", "").replace("-", "").replace(" ", "")
            available = ", ".join(sorted(KNOWN_MAPPINGS.keys()))
            print(f"\nNo known mappings for '{args.game}' "
                  f"(key: '{game_key}').")
            print(f"Available games: {available}")
            print("Run without --verify for discovery mode.\n")
        else:
            print_verification_report(
                args.game, verification_results, verification_mapping)
    else:
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

    if verification_results is not None:
        output["verification"] = {
            "source": verification_mapping["source"],
            "results": verification_results,
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Full statistics saved to {output_path}")


if __name__ == "__main__":
    main()
