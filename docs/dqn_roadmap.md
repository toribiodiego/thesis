# DQN Roadmap

Plan for reproducing “Playing Atari with Deep Reinforcement Learning” while building a reusable experimentation scaffold for future sample-efficient RL work.

## 1. Objectives

- **Reproduce canonical DQN results** on a small Atari suite (initially Pong, Breakout, and Space Invaders) using modern tooling.
- **Establish a reusable experimentation stack** (shared configs, environment setup, logging) that future MuZero, EfficientZero, CURL, DrQ, and SPR experiments can inherit.
- **Develop evaluation workflows** (metrics, tables, plots) to compare future methods under consistent protocols.

## 2. Outputs & Artifacts

- `envs/` – Python virtual environment scripts/requirements; CUDA vs. CPU variants.
- `src/` – Reusable modules (buffer, networks, training loops, evaluation harness).
- `experiments/dqn_atari/` – DQN-specific configs, scripts, and experiment notes.
- `reports/` – Generated figures, tables, and experiment cards.
- `docs/` – Living design docs, experiment logs, and paper notes.

## 3. Work Packages

### WP1 – Understand the Baseline
- Re-read DQN + Nature appendix, note implementation quirks and evaluation rules.
- Lock in baseline hyperparameters and preprocessing pipeline.
- Sketch repo architecture, configuration approach, and logging needs.

**Outputs:** Paper notes in `docs/papers/`, planning artifacts in `docs/design/`.

### WP2 – Build the Tooling
- Create pinned `requirements.txt` and virtualenv setup script.
- Script Atari ROM acquisition/validation and smoke-test installs.
- Stand up configuration utilities and logging hooks shared by all experiments.

**Outputs:** `envs/requirements.txt`, `envs/setup_env.sh`, `src/config/`, `src/utils/logging.py`.

### WP3 – Implement the Learner
- Write preprocessing wrappers (grayscale, resize, frame stacking, action repeat).
- Implement replay buffer, target network, ε-greedy policy, and Nature CNN.
- Validate training loop on controllable tasks (e.g., CartPole) before Atari.

**Outputs:** `src/agents/dqn.py`, `src/replay/`, unit tests for core utilities.

### WP4 – Evaluate on Atari
- Integrate ALE-specific logic (life-loss terminals, no-op starts, evaluation runners).
- Automate checkpointing, evaluation sweeps, and metrics aggregation.
- Produce learning curves, score tables, and rollout summaries.

**Outputs:** `experiments/dqn_atari/train.py`, `experiments/dqn_atari/eval.py`, generated plots in `reports/dqn_baseline/`.

### WP5 – Document & Compare
- Capture exact run commands, hardware footprint, and runtime budgets.
- Compare reproduced scores against published DQN baselines.
- Record lessons learned and priorities for the next algorithm.

**Outputs:** Final report in `reports/dqn_baseline/report.md`, reusable plotting scripts.

## 4. Milestones & Checkpoints

| Milestone | Criteria | Status Signals |
|-----------|----------|----------------|
| M1 | Environment + tooling smoke test green | `venv` reproducible install; ALE ROM loader verified |
| M2 | CartPole DQN convergence | Training script reaches >195 reward across 100 episodes |
| M3 | Pong score ≥ Original paper benchmark | Evaluation matches reported score within tolerance |
| M4 | Three-game Atari suite reproduced with plots + tables | Reports generated in `reports/dqn_baseline/` |

## 5. Risk Management

- **Compute constraints:** Track GPU availability; maintain CPU-friendly configs for quick tests.
- **Environment drift:** Pin emulator + gym versions; archive ROM acquisition instructions securely.
- **Training instability:** Include gradient check utilities, reward clipping toggles, and seeded runs.
- **Data logging overload:** Adopt rolling log retention and periodic artifact pruning.

## 6. Future Extensions

- Swap DQN backbone with CURL / SPR for representation learning comparisons.
- Layer image augmentation hooks (DrQ) on top of the shared preprocessing pipeline.
- Benchmark EfficientZero-style planning modules using the same evaluation harness.
- Explore robustness tests (distribution shift, goal misgeneralization) by reusing metrics utilities.

## 7. Next Actions

1. Create `envs/requirements.txt` with PyTorch + ALE stack; validate installation instructions.
2. Draft module skeletons in `src/` and `experiments/dqn_atari/`.
3. Implement CartPole smoke test and minimal training loop scaffold.
