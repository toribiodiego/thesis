# WP1 – Deep-Dive & Planning Checklist

This checklist tracks the concrete actions required to complete Work Package 1 for the DQN reproduction.

## Reading & Knowledge Capture
- [x] Gather primary reference: “Playing Atari with Deep Reinforcement Learning” (2013) and Nature appendix (2015).
- [x] Extract baseline hyperparameters and implementation nuances (`docs/papers/dqn_2013_notes.md`).
- [ ] Collect secondary replication resources (e.g., OpenAI Baselines, Dopamine) for cross-validation.

## Documentation Artifacts
- [x] Roadmap updated to reflect WP structure (`docs/dqn_roadmap.md`).
- [x] Paper notes established under `docs/papers/`.
- [ ] Create architecture outline draft in `docs/design/architecture_outline.md`.
- [ ] Define logging/reporting expectations in `docs/design/metrics_strategy.md`.

## Repository Foundations
- [x] Directory structure for shared modules (`src/`), experiments (`experiments/`), environments (`envs/`), and docs.
- [ ] Add placeholder `requirements.txt` capturing initial package set.
- [ ] Prepare `setup_env.sh` script for virtualenv creation.
- [ ] Configure `.gitignore` entries for virtualenvs, logs, checkpoints (pending once base requirements defined).

## Planning Decisions
- [ ] Choose configuration framework (Hydra vs. pure dataclasses) and document rationale.
- [ ] Decide on logging stack (Weights & Biases vs. local dashboards) respecting network constraints.
- [ ] Specify evaluation cadence (frames-per-eval) and metrics to track.
- [ ] Identify hardware targets (GPU/CPU) and estimate run durations.

Update this checklist as tasks complete to keep WP1 focused and transparent.
