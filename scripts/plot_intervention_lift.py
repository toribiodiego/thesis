#!/usr/bin/env python3
"""Generate intervention lift bar charts (A2 figures).

For each game, shows how adding SPR or augmentation changes the
final eval score. Two panels per game:
  Left:  Adding SPR (without aug, with aug)
  Right: Adding Augmentation (without SPR, with SPR)

Usage:
    python scripts/plot_intervention_lift.py

Output:
    output/plots/<game>_spr_aug_lift.png  (per-game figures)
"""

import csv
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

RUNS_DIR = "experiments/dqn_atari/runs"
OUTPUT_DIR = "output/plots"

# Map (game, condition) -> run directory name
RUN_NAMES = {
    ("Boxing", "Baseline"): "atari100k_boxing_42_20260310_170320",
    ("Boxing", "+ Aug"): "atari100k_boxing_aug_42_20260310_202944",
    ("Boxing", "+ SPR"): "atari100k_boxing_spr_42_20260312_022320",
    ("Boxing", "+ Both"): "atari100k_boxing_both_42_20260312_030847",
    ("Crazy Climber", "Baseline"): "atari100k_crazy_climber_42_20260310_164841",
    ("Crazy Climber", "+ Aug"): "atari100k_crazy_climber_aug_42_20260310_201115",
    ("Crazy Climber", "+ SPR"): "atari100k_crazy_climber_spr_42_20260323_160044",
    ("Crazy Climber", "+ Both"): "atari100k_crazy_climber_both_42_20260323_160044",
    ("Frostbite", "Baseline"): "atari100k_frostbite_42_20260310_173243",
    ("Frostbite", "+ Aug"): "atari100k_frostbite_aug_42_20260310_212155",
    ("Frostbite", "+ SPR"): "atari100k_frostbite_spr_42_20260324_182504",
    ("Frostbite", "+ Both"): "atari100k_frostbite_both_42_20260324_185917",
    ("Kangaroo", "Baseline"): "atari100k_kangaroo_42_20260310_173011",
    ("Kangaroo", "+ Aug"): "atari100k_kangaroo_aug_42_20260310_212156",
    ("Kangaroo", "+ SPR"): "atari100k_kangaroo_spr_42_20260324_182504",
    ("Kangaroo", "+ Both"): "atari100k_kangaroo_both_42_20260324_185918",
    ("Road Runner", "Baseline"): "atari100k_road_runner_42_20260310_165021",
    ("Road Runner", "+ Aug"): "atari100k_road_runner_aug_42_20260310_201202",
    ("Road Runner", "+ SPR"): "atari100k_road_runner_spr_42_20260324_182505",
    ("Road Runner", "+ Both"): "atari100k_road_runner_both_42_20260324_185918",
    ("Up N Down", "Baseline"): "atari100k_up_n_down_42_20260310_174441",
    ("Up N Down", "+ Aug"): "atari100k_up_n_down_aug_42_20260310_213744",
    ("Up N Down", "+ SPR"): "atari100k_up_n_down_spr_42_20260324_182505",
    ("Up N Down", "+ Both"): "atari100k_up_n_down_both_42_20260324_185918",
}

GAMES = [
    "Crazy Climber", "Road Runner", "Boxing",
    "Kangaroo", "Frostbite", "Up N Down",
]


def get_final_score(game, condition):
    """Get the mean_return at the last checkpoint."""
    run_name = RUN_NAMES.get((game, condition))
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


def plot_game_lift(game):
    """Generate the two-panel intervention lift figure for one game."""
    baseline = get_final_score(game, "Baseline")
    aug = get_final_score(game, "+ Aug")
    spr = get_final_score(game, "+ SPR")
    both = get_final_score(game, "+ Both")

    if baseline is None or aug is None:
        print(f"  SKIP {game}: missing Baseline or +Aug eval data")
        return False

    has_spr = spr is not None and both is not None

    # Compute deltas and percentages
    aug_lift_no_spr = aug - baseline           # adding aug without SPR
    spr_lift_no_aug = spr - baseline if has_spr else None  # adding SPR without aug
    aug_lift_with_spr = both - spr if has_spr else None    # adding aug with SPR
    spr_lift_with_aug = both - aug if has_spr else None     # adding SPR with aug

    def pct(delta, base):
        if base == 0:
            return None
        return (delta / abs(base)) * 100

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(f"Intervention Lift: {game} (seed 42)", fontsize=17,
                 fontweight="bold", y=0.98)

    pos_color = "#4C72B0"       # blue = helped
    pos_color_light = "#A8C4E0"
    neg_color = "#DD8452"       # orange = hurt
    neg_color_light = "#F5D4B8"

    def bar_colors(val):
        if val >= 0:
            return pos_color_light, pos_color
        return neg_color_light, neg_color

    def add_bar_label(ax, x, val):
        sign = "+" if val > 0 else ""
        if abs(val) >= 100:
            label = f"{sign}{val:.0f}"
        else:
            label = f"{sign}{val:.1f}"
        _, edge = bar_colors(val)
        ax.text(x, val / 2, label, ha="center", va="center",
                fontsize=13, fontweight="bold", color=edge)

    def plot_bar(ax, x, val):
        fill, edge = bar_colors(val)
        ax.bar(x, val, color=fill, edgecolor=edge,
               linewidth=1.5, width=0.6)
        add_bar_label(ax, x, val)

    # Left panel: Adding Augmentation
    ax1.set_title("Adding Augmentation", fontsize=15)
    plot_bar(ax1, 0, aug_lift_no_spr)
    if has_spr:
        plot_bar(ax1, 1, aug_lift_with_spr)
        ax1.set_xticks([0, 1])
        ax1.set_xticklabels(["Without\nSPR", "With\nSPR"])
    else:
        ax1.set_xticks([0])
        ax1.set_xticklabels(["Without\nSPR"])

    ax1.axhline(y=0, color="black", linewidth=0.8)
    ax1.set_ylabel("Score Change")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Right panel: Adding SPR
    ax2.set_title("Adding SPR", fontsize=15)
    if has_spr:
        plot_bar(ax2, 0, spr_lift_no_aug)
        plot_bar(ax2, 1, spr_lift_with_aug)
        ax2.set_xticks([0, 1])
        ax2.set_xticklabels(["Without\nAugmentation", "With\nAugmentation"])
    else:
        ax2.text(0.5, 0.5, "SPR eval\npending", ha="center", va="center",
                 fontsize=14, color="gray", transform=ax2.transAxes)
        ax2.set_xticks([])

    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_ylabel("Score Change")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    game_slug = game.lower().replace(" ", "_")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, f"{game_slug}_spr_aug_lift.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{game_slug}_spr_aug_lift.pdf"),
                bbox_inches="tight")
    plt.close()
    print(f"Saved {game_slug}_spr_aug_lift.{{png,pdf}}")
    return True


def plot_all_games_aug_effect():
    """Bar chart of augmentation lift across all 6 games."""
    pos_color = "#4C72B0"
    pos_color_light = "#A8C4E0"
    neg_color = "#DD8452"
    neg_color_light = "#F5D4B8"

    plt.rcParams.update({
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })

    deltas = []
    labels = []
    for game in GAMES:
        baseline = get_final_score(game, "Baseline")
        aug = get_final_score(game, "+ Aug")
        if baseline is None or aug is None:
            continue
        deltas.append(aug - baseline)
        labels.append(game)

    if not deltas:
        print("  SKIP all_games_aug_effect: no data")
        return

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(labels))
    fills = [pos_color_light if d >= 0 else neg_color_light for d in deltas]
    edges = [pos_color if d >= 0 else neg_color for d in deltas]
    ax.bar(x, deltas, color=fills, edgecolor=edges,
           linewidth=1.5, width=0.6)

    for i, val in enumerate(deltas):
        sign = "+" if val > 0 else ""
        if abs(val) >= 100:
            label = f"{sign}{val:.0f}"
        else:
            label = f"{sign}{val:.1f}"
        text_color = pos_color if val >= 0 else neg_color
        ax.text(i, val / 2, label, ha="center", va="center",
                fontsize=12, fontweight="bold", color=text_color)

    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score Change from Adding Augmentation")
    ax.set_title("Augmentation Effect Across Games (seed 42)",
                 fontsize=16, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, "all_games_aug_effect.png"),
                dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "all_games_aug_effect.pdf"),
                bbox_inches="tight")
    plt.close()
    print("Saved all_games_aug_effect.{png,pdf}")


def main():
    for game in GAMES:
        plot_game_lift(game)
    plot_all_games_aug_effect()


if __name__ == "__main__":
    main()
