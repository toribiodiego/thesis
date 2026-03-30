#!/usr/bin/env python3
"""Generate DQN vs Rainbow capability scatter plot (B3 figure).

One point per game: x = DQN final score, y = Rainbow final score.
Points above the diagonal mean Rainbow outperforms DQN.

Usage:
    python scripts/plot_capability_scatter.py

Output:
    output/plots/dqn_vs_rainbow_scatter.png
"""

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from run_registry import COLORS, GAMES, RUNS, RUNS_DIR

matplotlib.use("Agg")

OUTPUT_DIR = "output/plots"


def get_final_score(game, condition):
    """Get the mean_return at the last checkpoint."""
    run_name = RUNS.get((game, condition))
    if run_name is None:
        return None
    csv_path = os.path.join(RUNS_DIR, run_name, "eval", "evaluations.csv")
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return float(rows[-1]["mean_return"])


def main():
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })

    fig, ax = plt.subplots(figsize=(7, 7))

    dqn_scores = []
    rainbow_scores = []
    labels = []

    for game in GAMES:
        dqn = get_final_score(game, "DQN")
        rainbow = get_final_score(game, "Rainbow")
        if dqn is None or rainbow is None:
            print(f"  SKIP {game}: missing DQN or Rainbow eval")
            continue
        dqn_scores.append(dqn)
        rainbow_scores.append(rainbow)
        labels.append(game)

    if not dqn_scores:
        print("No data to plot")
        return

    dqn_scores = np.array(dqn_scores)
    rainbow_scores = np.array(rainbow_scores)

    # Parity line
    all_scores = np.concatenate([dqn_scores, rainbow_scores])
    margin = (max(all_scores) - min(all_scores)) * 0.1
    lo = min(all_scores) - margin
    hi = max(all_scores) + margin
    ax.plot([lo, hi], [lo, hi], color="gray", linestyle="--",
            linewidth=1, alpha=0.6, zorder=1)

    # Scatter points
    ax.scatter(dqn_scores, rainbow_scores, s=120, zorder=5,
               color=COLORS["Rainbow"], edgecolors="white", linewidth=1.5)

    # Label each point
    for i, game in enumerate(labels):
        x, y = dqn_scores[i], rainbow_scores[i]
        # Offset labels to avoid overlap with points
        xoff, yoff = 12, -5
        if game == "Boxing":
            xoff, yoff = 12, 8
        elif game == "Frostbite":
            xoff, yoff = 12, -12
        ax.annotate(game, (x, y), textcoords="offset points",
                    xytext=(xoff, yoff), fontsize=11)

    ax.set_xlabel("DQN Final Score")
    ax.set_ylabel("Rainbow Final Score")
    ax.set_title("DQN vs Rainbow: Per-Game Comparison (seed 42)",
                 fontsize=14, fontweight="bold")

    # Add region labels
    ax.text(0.95, 0.05, "DQN better", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10, color="gray",
            style="italic")
    ax.text(0.05, 0.95, "Rainbow better", transform=ax.transAxes,
            ha="left", va="top", fontsize=10, color="gray",
            style="italic")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "dqn_vs_rainbow_scatter.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "dqn_vs_rainbow_scatter.pdf"),
                bbox_inches="tight")
    plt.close()
    print("Saved dqn_vs_rainbow_scatter.{png,pdf}")


if __name__ == "__main__":
    main()
