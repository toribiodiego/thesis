#!/usr/bin/env python3
"""Generate interaction plots (A3 figures).

Shows whether augmentation and SPR have independent or interacting
effects. Two lines per game: "Without SPR" and "With SPR", plotted
across x = {No Augmentation, With Augmentation}. Parallel lines
mean independent effects; crossing/converging lines mean interaction.

Usage:
    python scripts/plot_interaction.py

Output:
    output/plots/<game>_interaction.png  (per-game figures)
"""

import csv
import os

import matplotlib
import matplotlib.pyplot as plt

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


def plot_interaction(game):
    """Generate interaction plot for one game."""
    baseline = get_final_score(game, "DQN")
    aug = get_final_score(game, "+ Aug")
    spr = get_final_score(game, "+ SPR")
    both = get_final_score(game, "+ Both")

    if baseline is None or aug is None:
        print(f"  SKIP {game}: missing Baseline or +Aug eval data")
        return False

    has_spr = spr is not None and both is not None

    if not has_spr:
        print(f"  SKIP {game}: missing SPR or Both eval data")
        return False

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
        "legend.frameon": True,
        "legend.fontsize": 11,
    })

    no_spr_color = "#7CA1D4"   # blue
    with_spr_color = "#6ABF6A" # green

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(f"Interaction Plot: {game} (seed 42)",
                 fontsize=15, fontweight="bold")

    x = [0, 1]
    x_labels = ["No Augmentation", "With Augmentation"]

    # Without SPR line: Baseline -> +Aug
    no_spr_scores = [baseline, aug]
    ax.plot(x, no_spr_scores, color=no_spr_color, linewidth=2.2,
            marker="o", markersize=10, markerfacecolor="white",
            markeredgecolor=no_spr_color, markeredgewidth=2,
            label="Without SPR", zorder=5)

    # With SPR line: +SPR -> +Both
    with_spr_scores = [spr, both]
    ax.plot(x, with_spr_scores, color=with_spr_color, linewidth=2.2,
            marker="s", markersize=10, markerfacecolor="white",
            markeredgecolor=with_spr_color, markeredgewidth=2,
            label="With SPR", zorder=5)

    # Add score labels with smart positioning to avoid overlaps
    # Each entry: (x, y, color, x_offset, y_offset)
    # Determine offsets based on relative positions
    points = [
        (0, baseline, no_spr_color, "no_spr_left"),
        (1, aug, no_spr_color, "no_spr_right"),
        (0, spr, with_spr_color, "with_spr_left"),
        (1, both, with_spr_color, "with_spr_right"),
    ]

    # Sort left-side points by y to determine which is on top
    left_top = max(baseline, spr)
    right_gap = abs(aug - both)

    for px, py, color, pos in points:
        if abs(py) >= 100:
            label = f"{py:.0f}"
        else:
            label = f"{py:.1f}"

        # Default: label above and to the left
        xoff, yoff = -40, 8

        if pos == "no_spr_left":
            if baseline > spr:
                xoff, yoff = -40, 8   # top-left: above
            else:
                xoff, yoff = -40, -18  # bottom-left: below
        elif pos == "with_spr_left":
            if spr > baseline:
                xoff, yoff = -40, 8
            else:
                xoff, yoff = -40, -18
        elif pos == "no_spr_right":
            if aug > both:
                xoff, yoff = 18, 8
            else:
                xoff, yoff = 18, -18
        elif pos == "with_spr_right":
            if both > aug:
                xoff, yoff = 18, 8
            else:
                xoff, yoff = 18, -18

        ax.annotate(label, (px, py), textcoords="offset points",
                    xytext=(xoff, yoff), fontsize=12, fontweight="bold",
                    color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlim(-0.3, 1.45)
    ax.set_ylabel("Mean Eval Return")

    # Pad y-axis so labels at extremes don't get clipped
    all_scores = [baseline, aug, spr, both]
    y_min = min(all_scores)
    y_max = max(all_scores)
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.12 * y_range, y_max + 0.12 * y_range)
    ax.legend(loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    game_slug = game.lower().replace(" ", "_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{game_slug}_interaction.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{game_slug}_interaction.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved {game_slug}_interaction.{{png,pdf}}")
    return True


def main():
    for game in GAMES:
        plot_interaction(game)


if __name__ == "__main__":
    main()
