#!/usr/bin/env python3
"""Generate SPR convergence plots (D3 figures).

Shows cosine similarity over training for SPR conditions, zoomed
to reveal differences between conditions. One figure per game.

Usage:
    python scripts/plot_spr_convergence.py

Output:
    output/plots/<game>_spr_convergence.png
"""

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

RUNS_DIR = "experiments/dqn_atari/runs"
OUTPUT_DIR = "output/plots"

# (game, condition) -> run directory
# Only include runs that have SPR (cosine_similarity column)
RUN_NAMES = {
    ("Boxing", "DQN+SPR"): "atari100k_boxing_spr_42_20260324_182503",
    ("Boxing", "DQN+Both"): "atari100k_boxing_both_42_20260324_185917",
    ("Crazy Climber", "DQN+SPR"): "atari100k_crazy_climber_spr_42_20260323_160044",
    ("Crazy Climber", "DQN+Both"): "atari100k_crazy_climber_both_42_20260323_160044",
    ("Crazy Climber", "Rainbow+SPR"): "atari100k_crazy_climber_rainbow_spr_42_20260323_160045",
    ("Frostbite", "DQN+SPR"): "atari100k_frostbite_spr_42_20260324_182504",
    ("Frostbite", "DQN+Both"): "atari100k_frostbite_both_42_20260324_185917",
    ("Kangaroo", "DQN+SPR"): "atari100k_kangaroo_spr_42_20260324_182504",
    ("Kangaroo", "DQN+Both"): "atari100k_kangaroo_both_42_20260324_185918",
    ("Road Runner", "DQN+SPR"): "atari100k_road_runner_spr_42_20260324_182505",
    ("Road Runner", "DQN+Both"): "atari100k_road_runner_both_42_20260324_185918",
    ("Up N Down", "DQN+SPR"): "atari100k_up_n_down_spr_42_20260324_182505",
    ("Up N Down", "DQN+Both"): "atari100k_up_n_down_both_42_20260324_185918",
}

GAMES = [
    "Crazy Climber", "Road Runner", "Boxing",
    "Kangaroo", "Frostbite", "Up N Down",
]

CONDITIONS = ["DQN+SPR", "DQN+Both", "Rainbow+SPR"]
COLORS = {
    "DQN+SPR": "#55A868",
    "DQN+Both": "#C44E52",
    "Rainbow+SPR": "#8172B3",
}


def smooth(values, window=5):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    pad = len(values) - len(smoothed)
    return np.concatenate([values[:pad], smoothed])


def load_cosine_similarity(run_dir):
    """Load cosine similarity timeseries from training CSV."""
    csv_path = os.path.join(run_dir, "csv", "training_steps.csv")
    if not os.path.exists(csv_path):
        return None, None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    steps = []
    cos_sims = []
    for r in rows:
        cs = r.get("cosine_similarity", "")
        if cs:
            val = float(cs)
            if val > 0:  # skip initial negative values (pre-convergence)
                steps.append(int(r["step"]) / 1000)
                cos_sims.append(val)
    if not steps:
        return None, None
    return np.array(steps), np.array(cos_sims)


def plot_game(game):
    """Generate SPR convergence plot for one game."""
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
        "legend.frameon": True, "legend.fontsize": 11,
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    has_data = False
    all_mins = []

    for cond in CONDITIONS:
        run_name = RUN_NAMES.get((game, cond))
        if run_name is None:
            continue
        run_dir = os.path.join(RUNS_DIR, run_name)
        steps, cos_sims = load_cosine_similarity(run_dir)
        if steps is None:
            print(f"  WARNING: missing data for {game} / {cond}")
            continue
        smoothed = smooth(cos_sims)
        ax.plot(steps, smoothed, label=cond, color=COLORS[cond],
                linewidth=2.0)
        has_data = True
        all_mins.append(np.min(smoothed))

    if not has_data:
        print(f"  SKIP {game}: no SPR data")
        plt.close()
        return False

    ax.set_title(f"SPR Convergence: {game} (seed 42)",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Frames (K)")
    ax.set_ylabel("Cosine Similarity")

    # Fixed y-axis range across all games for comparability
    ax.set_ylim(0.975, 1.005)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    game_slug = game.lower().replace(" ", "_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{game_slug}_spr_convergence.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{game_slug}_spr_convergence.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved {game_slug}_spr_convergence.{{png,pdf}}")
    return True


def main():
    for game in GAMES:
        plot_game(game)


if __name__ == "__main__":
    main()
