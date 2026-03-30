#!/usr/bin/env python3
"""Generate training stability plots (D4 figures).

Shows smoothed TD error over training for all conditions. Higher
TD error means less stable value estimates.

Usage:
    python scripts/plot_training_stability.py

Output:
    output/plots/<game>_training_stability.png
"""

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from run_registry import COLORS, DQN_CONDITIONS, GAMES, RUNS, RUNS_DIR

matplotlib.use("Agg")

OUTPUT_DIR = "output/plots"

CONDITIONS = DQN_CONDITIONS


def smooth(values, window=7):
    """Moving average smoothing."""
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    pad = len(values) - len(smoothed)
    return np.concatenate([values[:pad], smoothed])


def load_td_error(run_dir):
    """Load TD error timeseries from training CSV."""
    csv_path = os.path.join(run_dir, "csv", "training_steps.csv")
    if not os.path.exists(csv_path):
        return None, None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    steps = []
    td_errors = []
    for r in rows:
        td = r.get("td_error", "")
        if td:
            steps.append(int(r["step"]) / 1000)
            td_errors.append(float(td))
    if not steps:
        return None, None
    return np.array(steps), np.array(td_errors)


def plot_game(game):
    """Generate training stability plot for one game."""
    plt.rcParams.update({
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.grid": True, "grid.alpha": 0.3,
        "legend.frameon": True, "legend.fontsize": 11,
    })

    fig, ax = plt.subplots(figsize=(8, 5))

    has_data = False
    all_stable_maxes = []
    for cond in CONDITIONS:
        run_name = RUNS.get((game, cond))
        if run_name is None:
            continue
        run_dir = os.path.join(RUNS_DIR, run_name)
        steps, td_errors = load_td_error(run_dir)
        if steps is None:
            continue
        smoothed = smooth(td_errors)
        ax.plot(steps, smoothed, label=cond, color=COLORS[cond],
                linewidth=2.0)
        has_data = True
        # Track stable-region max (after 20% of training)
        cutoff = len(smoothed) // 5
        all_stable_maxes.append(np.max(smoothed[cutoff:]))

    if not has_data:
        print(f"  SKIP {game}: no TD error data")
        plt.close()
        return False

    ax.set_title(f"Training Stability: {game} (seed 42)",
                 fontsize=15, fontweight="bold")
    ax.set_xlabel("Frames (K)")
    ax.set_ylabel("Mean |TD Error|")

    # Cap y-axis to stable region, removing initial transient spike
    if all_stable_maxes:
        y_max = max(all_stable_maxes) * 1.2
        ax.set_ylim(0, y_max)

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    game_slug = game.lower().replace(" ", "_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{game_slug}_training_stability.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{game_slug}_training_stability.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved {game_slug}_training_stability.{{png,pdf}}")
    return True


def main():
    for game in GAMES:
        plot_game(game)


if __name__ == "__main__":
    main()
