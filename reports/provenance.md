# Figure and Table Provenance

How each figure and table in `working-results.tex` was produced.
Update this file whenever you regenerate a result.

Run data lives on Google Drive at `My Drive/thesis-runs/<run-name>/`.
Download locally with `bash scripts/pull-run.sh <run-name>`, or pull
groups with `--group` (e.g. `--group rainbow`), or all runs with
`--all`. Run `pull-run.sh --list` to see what is available.
Run directory names are documented in
`experiments/dqn_atari/runs/INDEX.md`.


## Figures

### Figure 1: aug_helps (Kangaroo, Up N Down)

- **File**: `output/plots/aug_helps.pdf`
- **Script**: `python scripts/plot_learning_curves.py`
- **Data**: eval CSVs from 8 runs (2 games x 4 conditions), seed 42
- **Produced**: 2026-03-10
- **Notes**: Script generates all figure PDFs in one pass. The
  `PANEL_GROUPS` dict in the script controls which games go into
  which output file.

### Figure 2: aug_hurts_spr_holds (Crazy Climber, Road Runner)

- **File**: `output/plots/aug_hurts_spr_holds.pdf`
- **Script**: `python scripts/plot_learning_curves.py`
- **Data**: eval CSVs from 8 runs (2 games x 4 conditions), seed 42
- **Produced**: 2026-03-10

### Figure 3: minimal_learning (Boxing, Frostbite)

- **File**: `output/plots/minimal_learning.pdf`
- **Script**: `python scripts/plot_learning_curves.py`
- **Data**: eval CSVs from 8 runs (2 games x 4 conditions), seed 42
- **Produced**: 2026-03-10


## Tables

### Table 1: Game selection (6 games and challenges)

- **Source**: Hand-written from game descriptions
- **No regeneration needed**

### Table 2: Experimental conditions (4 DQN conditions)

- **Source**: Hand-written from experiment design
- **No regeneration needed**

### Table 3: Training configuration

- **Source**: Hand-written from `experiments/dqn_atari/configs/base.yaml`
- **No regeneration needed**

### Table 4: Rainbow conditions

- **Source**: Hand-written from experiment design
- **No regeneration needed**

### Table 5: Summary results (tab:summary)

- **Source**: Final-checkpoint eval CSVs (step 400000) from 24 DQN
  runs, seed 42. Each cell is mean +/- std over 30 eval episodes.
- **Data location**: Google Drive `thesis-runs/<run-name>/eval/evaluations.csv`
- **How numbers were extracted**: Read the last row (step=400000)
  of each run's `evaluations.csv`. `mean_return` and `std_return`
  columns give the values in the table.
- **Run directories**: See `experiments/dqn_atari/runs/INDEX.md`,
  DQN Isolation Study section (24 runs).
- **Rainbow columns**: Pending -- original runs were invalid (see
  Appendix in working-results.tex).
- **Produced**: 2026-03-10

### Table 6: Atari-100K game list by reward density (app:games)

- **Source**: Hand-written from Schwarzer et al. (2021), Table 4
- **No regeneration needed**

### Table 7: Invalid Rainbow results (tab:invalid-rainbow)

- **Source**: Final-checkpoint eval CSVs from 6 invalid Rainbow runs
  in `thesis-runs/invalid/` on Google Drive.
- **Run directories**: See `experiments/dqn_atari/runs/INDEX.md`,
  invalid/ section.
- **Produced**: 2026-03-11
- **Notes**: These runs used vanilla DQN with Rainbow hyperparameters
  due to a bug in train_dqn.py. Kept for reference only.


## Regeneration

To regenerate all figures from local run data:

```bash
# Download all runs from Google Drive (requires rclone)
bash scripts/pull-run.sh --all

# Or download by group
bash scripts/pull-run.sh --group base      # vanilla DQN runs
bash scripts/pull-run.sh --group rainbow   # Rainbow runs
bash scripts/pull-run.sh --group spr       # DQN+SPR runs

# Or download a single run
bash scripts/pull-run.sh atari100k_crazy_climber_42_20260310_164841

# Skip checkpoints for faster/smaller downloads
bash scripts/pull-run.sh --all --no-checkpoints

# Generate plots
python scripts/plot_learning_curves.py
```

To regenerate table numbers, read the final row of each run's
`eval/evaluations.csv` after downloading.
