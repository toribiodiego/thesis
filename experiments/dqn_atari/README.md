# DQN Atari Experiment

This directory will host the scripts, configs, and notes for reproducing the original DQN Atari results.

Planned contents:

- `configs/` – Shared base config plus per-game overrides.
- `scripts/train.py` – Entry point for training runs.
- `scripts/eval.py` – Evaluation harness that generates episode rollouts and score tables.
- `notes.md` – Chronological experiment log (hyperparameter adjustments, observations, follow-ups).

All core implementation lives under `src/`, keeping experiment directories thin and focused on wiring plus documentation.
