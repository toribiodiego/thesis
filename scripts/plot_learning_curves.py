#!/usr/bin/env python3
"""Generate learning curve plots for the DQN isolation study.

Reads eval CSVs from experiment runs and produces a 6-panel figure
with smoothed lines and shaded std bands in the standard RL paper style.

Usage:
    python scripts/plot_learning_curves.py

Output:
    output/plots/dqn_learning_curves.png
    output/plots/dqn_learning_curves.pdf
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
    ("Crazy Climber", "Baseline"): "atari100k_crazy_climber_42_20260310_164841",
    ("Road Runner", "Baseline"): "atari100k_road_runner_42_20260310_165021",
    ("Boxing", "Baseline"): "atari100k_boxing_42_20260310_170320",
    ("Kangaroo", "Baseline"): "atari100k_kangaroo_42_20260310_173011",
    ("Frostbite", "Baseline"): "atari100k_frostbite_42_20260310_173243",
    ("Up N Down", "Baseline"): "atari100k_up_n_down_42_20260310_174441",
    ("Crazy Climber", "+ Aug"): "atari100k_crazy_climber_aug_42_20260310_201115",
    ("Road Runner", "+ Aug"): "atari100k_road_runner_aug_42_20260310_201202",
    ("Boxing", "+ Aug"): "atari100k_boxing_aug_42_20260310_202944",
    ("Kangaroo", "+ Aug"): "atari100k_kangaroo_aug_42_20260310_212156",
    ("Frostbite", "+ Aug"): "atari100k_frostbite_aug_42_20260310_212155",
    ("Up N Down", "+ Aug"): "atari100k_up_n_down_aug_42_20260310_213744",
    ("Crazy Climber", "+ SPR"): "atari100k_crazy_climber_spr_42_20260323_160044",
    ("Road Runner", "+ SPR"): "atari100k_road_runner_spr_42_20260324_182505",
    ("Boxing", "+ SPR"): "atari100k_boxing_spr_42_20260324_182503",
    ("Kangaroo", "+ SPR"): "atari100k_kangaroo_spr_42_20260324_182504",
    ("Frostbite", "+ SPR"): "atari100k_frostbite_spr_42_20260324_182504",
    ("Up N Down", "+ SPR"): "atari100k_up_n_down_spr_42_20260324_182505",
    ("Crazy Climber", "+ Both"): "atari100k_crazy_climber_both_42_20260323_160044",
    ("Road Runner", "+ Both"): "atari100k_road_runner_both_42_20260324_185918",
    ("Boxing", "+ Both"): "atari100k_boxing_both_42_20260324_185917",
    ("Kangaroo", "+ Both"): "atari100k_kangaroo_both_42_20260324_185918",
    ("Frostbite", "+ Both"): "atari100k_frostbite_both_42_20260324_185917",
    ("Up N Down", "+ Both"): "atari100k_up_n_down_both_42_20260324_185918",
}

# INVALID: These runs used vanilla DQN with Rainbow hyperparameters,
# not actual Rainbow (train_dqn.py ignored config.rainbow.enabled).
# Moved to invalid/ subfolder. Will be replaced after Task 42.
RAINBOW_RUN_NAMES = {
    ("Crazy Climber", "Rainbow"): "invalid/atari100k_crazy_climber_rainbow_42_20260311_035227",
    ("Road Runner", "Rainbow"): "invalid/atari100k_road_runner_rainbow_42_20260311_033324",
    ("Boxing", "Rainbow"): "invalid/atari100k_boxing_rainbow_42_20260311_034322",
    ("Kangaroo", "Rainbow"): "invalid/atari100k_kangaroo_rainbow_42_20260311_041254",
    ("Frostbite", "Rainbow"): "invalid/atari100k_frostbite_rainbow_42_20260311_041254",
    ("Up N Down", "Rainbow"): "invalid/atari100k_up_n_down_rainbow_42_20260311_042140",
}

GAMES = [
    "Crazy Climber", "Road Runner", "Boxing",
    "Kangaroo", "Frostbite", "Up N Down",
]
CONDITIONS = ["Baseline", "+ Aug", "+ SPR", "+ Both"]
COLORS = {
    "Baseline": "#4C72B0",
    "+ Aug": "#DD8452",
    "+ SPR": "#55A868",
    "+ Both": "#C44E52",
    "Rainbow": "#8172B3",
}


def smooth(values, window=3):
    """Simple moving average smoothing."""
    if len(values) < window:
        return values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    pad = len(values) - len(smoothed)
    return np.concatenate([values[:pad], smoothed])


def load_eval_csv(run_dir):
    """Load evaluation data from a run's eval CSV."""
    csv_path = os.path.join(run_dir, "eval", "evaluations.csv")
    if not os.path.exists(csv_path):
        return None
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    steps = np.array([int(r["step"]) / 4000 for r in rows])  # env steps in K
    means = np.array([float(r["mean_return"]) for r in rows])
    stds = np.array([float(r["std_return"]) for r in rows])
    return steps, means, stds


def main():
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 2.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "legend.fontsize": 12,
    })

    # Split into three 1x2 figures (pairs) for readability
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pairs = [GAMES[0:2], GAMES[2:4], GAMES[4:6]]

    for pair_idx, pair_games in enumerate(pairs):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for i, game in enumerate(pair_games):
            ax = axes[i]
            for cond in CONDITIONS:
                run_name = RUN_NAMES.get((game, cond))
                if run_name is None:
                    continue
                run_dir = os.path.join(RUNS_DIR, run_name)
                data = load_eval_csv(run_dir)
                if data is None:
                    print(f"  WARNING: missing eval data for {game} / {cond}")
                    continue
                steps, means, stds = data
                smoothed_means = smooth(means)
                smoothed_stds = smooth(stds)
                ax.plot(steps, smoothed_means, label=cond,
                        color=COLORS[cond], linewidth=2.2)
                ax.fill_between(steps,
                                smoothed_means - smoothed_stds,
                                smoothed_means + smoothed_stds,
                                color=COLORS[cond], alpha=0.12)

            ax.set_title(game, fontsize=15, fontweight="bold")
            ax.set_xlabel("Env Steps (K)", fontsize=13)
            ax.set_ylabel("Mean Return", fontsize=13)
            ax.tick_params(labelsize=12)
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x, _: f"{int(x):,}" if x == int(x) else f"{x:.0f}"
                )
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=13,
                   bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.08, 1, 1.0])

        suffix = chr(ord("a") + pair_idx)
        plt.savefig(os.path.join(OUTPUT_DIR, f"dqn_learning_curves_{suffix}.png"),
                    dpi=300, bbox_inches="tight")
        plt.savefig(os.path.join(OUTPUT_DIR, f"dqn_learning_curves_{suffix}.pdf"),
                    bbox_inches="tight")
        plt.close()
        print(f"Saved dqn_learning_curves_{suffix}.{{png,pdf}}")


def plot_rainbow_comparison():
    """Generate paired figures comparing DQN Baseline vs Rainbow."""
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "lines.linewidth": 2.0,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "legend.frameon": False,
        "legend.fontsize": 12,
    })

    conditions = ["Baseline", "Rainbow"]
    all_runs = {**RUN_NAMES, **RAINBOW_RUN_NAMES}

    # Pair 1: Games where Rainbow helps (Crazy Climber, Up N Down)
    pairs = [
        (["Crazy Climber", "Up N Down"], "rainbow_helps",
         "Games where Rainbow improves over DQN."),
        (["Road Runner", "Frostbite"], "rainbow_hurts",
         "Games where Rainbow regresses from DQN."),
        (["Kangaroo", "Boxing"], "rainbow_flat",
         "Games where neither agent learns."),
    ]

    for games, filename, _desc in pairs:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for i, game in enumerate(games):
            ax = axes[i]
            for cond in conditions:
                run_name = all_runs.get((game, cond))
                if run_name is None:
                    continue
                run_dir = os.path.join(RUNS_DIR, run_name)
                data = load_eval_csv(run_dir)
                if data is None:
                    print(f"  WARNING: missing eval data for {game} / {cond}")
                    continue
                steps, means, stds = data
                smoothed_means = smooth(means)
                smoothed_stds = smooth(stds)
                ax.plot(steps, smoothed_means, label=cond,
                        color=COLORS[cond], linewidth=2.0,
                        marker="o" if cond == "Rainbow" else None,
                        markersize=5)
                ax.fill_between(steps,
                                smoothed_means - smoothed_stds,
                                smoothed_means + smoothed_stds,
                                color=COLORS[cond], alpha=0.12)

            ax.set_title(game, fontsize=13, fontweight="bold")
            ax.set_xlabel("Env Steps (K)")
            if i == 0:
                ax.set_ylabel("Mean Return")
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(
                    lambda x, _: f"{int(x):,}" if x == int(x) else f"{x:.0f}"
                )
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=13,
                   bbox_to_anchor=(0.5, -0.03))
        plt.tight_layout(rect=[0, 0.08, 1, 1.0])

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.png"),
                    dpi=150, bbox_inches="tight")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{filename}.pdf"),
                    bbox_inches="tight")
        plt.close()
        print(f"Saved {filename}.{{png,pdf}}")


if __name__ == "__main__":
    main()
    plot_rainbow_comparison()
