#!/usr/bin/env python3
"""Generate SPR lift comparison: DQN vs Rainbow (B2 figure).

Paired bars per game showing how much SPR helps each base agent.
Left bar: DQN+SPR - DQN, Right bar: Rainbow+SPR - Rainbow.

Usage:
    python scripts/plot_rainbow_lift.py

Output:
    output/plots/<game>_spr_lift_comparison.png  (per-game)
    output/plots/all_games_spr_lift_comparison.png  (summary)
"""

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from run_registry import GAMES, RUNS, RUNS_DIR

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


def plot_all_games():
    """Summary bar chart: SPR lift on DQN vs Rainbow across all games."""
    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    dqn_color = "#4C72B0"
    rainbow_color = "#8172B3"

    game_labels = []
    dqn_lifts = []
    rainbow_lifts = []
    has_rainbow_spr = False

    for game in GAMES:
        dqn = get_final_score(game, "DQN")
        dqn_spr = get_final_score(game, "+ SPR")
        rainbow = get_final_score(game, "Rainbow")
        rainbow_spr = get_final_score(game, "Rainbow+SPR")

        if dqn is None or dqn_spr is None:
            continue

        game_labels.append(game)
        dqn_lifts.append(dqn_spr - dqn)

        if rainbow is not None and rainbow_spr is not None:
            rainbow_lifts.append(rainbow_spr - rainbow)
            has_rainbow_spr = True
        else:
            rainbow_lifts.append(None)

    if not game_labels:
        print("No data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(game_labels))
    width = 0.35

    # DQN SPR lift bars
    ax.bar(x - width / 2, dqn_lifts, width, label="SPR lift on DQN",
           color=dqn_color, alpha=0.8, edgecolor=dqn_color)

    # Rainbow SPR lift bars (if available)
    if has_rainbow_spr:
        rb_vals = [v if v is not None else 0 for v in rainbow_lifts]
        rb_mask = [v is not None for v in rainbow_lifts]
        bars = ax.bar(x + width / 2, rb_vals, width,
                      label="SPR lift on Rainbow",
                      color=rainbow_color, alpha=0.8,
                      edgecolor=rainbow_color)
        # Gray out bars with no data
        for i, has_data in enumerate(rb_mask):
            if not has_data:
                bars[i].set_color("#CCCCCC")
                bars[i].set_edgecolor("#999999")

    # Labels on bars
    for i, val in enumerate(dqn_lifts):
        sign = "+" if val > 0 else ""
        label = f"{sign}{val:.0f}" if abs(val) >= 10 else f"{sign}{val:.1f}"
        ax.text(x[i] - width / 2, val + (20 if val >= 0 else -40),
                label, ha="center", va="bottom" if val >= 0 else "top",
                fontsize=10, fontweight="bold", color=dqn_color)

    for i, val in enumerate(rainbow_lifts):
        if val is None:
            continue
        sign = "+" if val > 0 else ""
        label = f"{sign}{val:.0f}" if abs(val) >= 10 else f"{sign}{val:.1f}"
        ax.text(x[i] + width / 2, val + (20 if val >= 0 else -40),
                label, ha="center", va="bottom" if val >= 0 else "top",
                fontsize=10, fontweight="bold", color=rainbow_color)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(game_labels)
    ax.set_ylabel("Score Change from Adding SPR")
    ax.set_title("SPR Lift: DQN vs Rainbow (seed 42)",
                 fontsize=16, fontweight="bold")
    ax.legend(loc="best", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "all_games_spr_lift_comparison.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "all_games_spr_lift_comparison.pdf"),
                bbox_inches="tight")
    plt.close()
    print("Saved all_games_spr_lift_comparison.{png,pdf}")


def main():
    plot_all_games()


if __name__ == "__main__":
    main()
