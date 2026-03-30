"""Single source of truth for experiment run directories.

All plotting scripts import from here instead of hardcoding run
names. Update this file when runs are replaced or added.
"""

import os

RUNS_DIR = "experiments/dqn_atari/runs"

GAMES = [
    "Crazy Climber", "Road Runner", "Boxing",
    "Kangaroo", "Frostbite", "Up N Down",
]

# Condition groups
DQN_CONDITIONS = ["DQN", "+ Aug", "+ SPR", "+ Both"]
RAINBOW_CONDITIONS = ["Rainbow", "Rainbow+SPR"]
ALL_CONDITIONS = DQN_CONDITIONS + RAINBOW_CONDITIONS

# Standard colors (consistent across all figures)
COLORS = {
    "DQN": "#4C72B0",
    "+ Aug": "#DD8452",
    "+ SPR": "#55A868",
    "+ Both": "#C44E52",
    "Rainbow": "#8172B3",
    "Rainbow+SPR": "#937860",
}

# All seed-42 runs: (game, condition) -> directory name
# fmt: off
RUNS = {
    # Boxing
    ("Boxing", "DQN"):          "atari100k_boxing_42_20260310_170320",
    ("Boxing", "+ Aug"):        "atari100k_boxing_aug_42_20260310_202944",
    ("Boxing", "+ SPR"):        "atari100k_boxing_spr_42_20260324_182503",
    ("Boxing", "+ Both"):       "atari100k_boxing_both_42_20260324_185917",
    ("Boxing", "Rainbow"):      "atari100k_boxing_rainbow_42_20260327_192211",

    # Crazy Climber
    ("Crazy Climber", "DQN"):          "atari100k_crazy_climber_42_20260310_164841",
    ("Crazy Climber", "+ Aug"):        "atari100k_crazy_climber_aug_42_20260310_201115",
    ("Crazy Climber", "+ SPR"):        "atari100k_crazy_climber_spr_42_20260323_160044",
    ("Crazy Climber", "+ Both"):       "atari100k_crazy_climber_both_42_20260323_160044",
    ("Crazy Climber", "Rainbow"):      "atari100k_crazy_climber_rainbow_42_20260323_160045",
    ("Crazy Climber", "Rainbow+SPR"):  "atari100k_crazy_climber_rainbow_spr_42_20260323_160045",

    # Frostbite
    ("Frostbite", "DQN"):          "atari100k_frostbite_42_20260310_173243",
    ("Frostbite", "+ Aug"):        "atari100k_frostbite_aug_42_20260310_212155",
    ("Frostbite", "+ SPR"):        "atari100k_frostbite_spr_42_20260324_182504",
    ("Frostbite", "+ Both"):       "atari100k_frostbite_both_42_20260324_185917",
    ("Frostbite", "Rainbow"):      "atari100k_frostbite_rainbow_42_20260327_192213",

    # Kangaroo
    ("Kangaroo", "DQN"):          "atari100k_kangaroo_42_20260310_173011",
    ("Kangaroo", "+ Aug"):        "atari100k_kangaroo_aug_42_20260310_212156",
    ("Kangaroo", "+ SPR"):        "atari100k_kangaroo_spr_42_20260324_182504",
    ("Kangaroo", "+ Both"):       "atari100k_kangaroo_both_42_20260324_185918",
    ("Kangaroo", "Rainbow"):      "atari100k_kangaroo_rainbow_42_20260327_192214",

    # Road Runner
    ("Road Runner", "DQN"):          "atari100k_road_runner_42_20260310_165021",
    ("Road Runner", "+ Aug"):        "atari100k_road_runner_aug_42_20260310_201202",
    ("Road Runner", "+ SPR"):        "atari100k_road_runner_spr_42_20260324_182505",
    ("Road Runner", "+ Both"):       "atari100k_road_runner_both_42_20260324_185918",
    ("Road Runner", "Rainbow"):      "atari100k_road_runner_rainbow_42_20260327_192215",

    # Up N Down
    ("Up N Down", "DQN"):          "atari100k_up_n_down_42_20260310_174441",
    ("Up N Down", "+ Aug"):        "atari100k_up_n_down_aug_42_20260310_213744",
    ("Up N Down", "+ SPR"):        "atari100k_up_n_down_spr_42_20260324_182505",
    ("Up N Down", "+ Both"):       "atari100k_up_n_down_both_42_20260324_185918",
    ("Up N Down", "Rainbow"):      "atari100k_up_n_down_rainbow_42_20260327_192217",
}
# fmt: on


def get_run_dir(game, condition):
    """Return full path to a run directory, or None if not registered."""
    name = RUNS.get((game, condition))
    if name is None:
        return None
    return os.path.join(RUNS_DIR, name)
