# DQN Atari Experiment

Experiment-specific assets that sit on top of the shared `src/` modules.

## Layout
- `configs/` – `base.yaml` plus per-game overrides; see `configs/README.md`.
- `scripts/` – Shell entry points like `run_dqn.sh` (training), with planned `eval` and `dry_run` helpers.
- `notes.md` – Chronological experiment log for hyperparameter tweaks and observations.

Core implementations (agents, replay, env wrappers) stay under `src/`, keeping this directory focused on wiring, configs, and documentation.
