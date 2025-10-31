# Experiments

Each experiment lives in its own subdirectory with scripts, configs, and notes that operate on the shared `src/` codebase.

Recommended layout:

- `configs/` – Hydra/OmegaConf configuration files (base + overrides).
- `scripts/` – Training/evaluation entrypoints specific to the experiment.
- `notes.md` – Lightweight log of hypotheses, parameter changes, and observations.
- `artifacts/` – Optional local outputs (checkpoints, metrics) ignored by git.

The first experiment, `dqn_atari/`, will replicate “Playing Atari with Deep Reinforcement Learning” to establish the baseline tooling. Future experiments (e.g., CURL, DrQ, EfficientZero) should mirror this structure so they can reuse the shared environment and utilities.
