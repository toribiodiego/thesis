# DQN Foundation Setup

Central reference for everything completed in Subtask 1 (game selection, pinned dependencies, evaluation settings, seeding, and dry-run tooling). Update this file whenever the foundation changes so collaborators know exactly how to bootstrap the project.

## Selected Games
- Pong (`ALE/Pong-v5`, minimal action set)
- Breakout (`ALE/Breakout-v5`, minimal action set)
- Beam Rider (`ALE/BeamRider-v5`, minimal action set)

Each game has a config stub in `experiments/dqn_atari/configs/` with overrides for frame budgets, evaluation cadence, and logging paths.

## Dependencies & Environment
- Python: 3.10.13 (documented in `envs/README.md`)
- PyTorch: 2.4.1 (CUDA 12.1 build where available)
- Gymnasium: 0.29.1
- ale-py: 0.8.1
- AutoROM: 0.6.1 (accept license flag)

Pinned versions live in `envs/requirements.txt`. Install via:
```bash
bash envs/setup_env.sh
source .venv/bin/activate
```

## ROM Acquisition
Use the helper script (to be added under `scripts/setup_roms.sh`) or run directly:
```bash
python -m AutoROM --accept-license
```
Document ROM storage path and verification steps in the script once created.

## Evaluation Protocol
Defined in `experiments/dqn_atari/configs/base.yaml`:
- ε-eval = 0.05 (configurable)
- Training termination: life-loss treated as terminal (`episode_life=true`)
- Evaluation termination: full episode
- `eval.episodes = 10`
- Reward clipping: {−1, 0, +1}
- Frame budgets: 10–20M for full runs, ≤1M for smoke tests

## Seeding & Metadata
Implement `src/utils/repro.py` with `set_seed(seed, deterministic=False)` covering Python, NumPy, Torch (CPU/GPU), and env seeds. Each run saves a `meta.json` containing:
- Git commit hash
- Merged config snapshot
- Seed value
- ALE settings (action set, frameskip, repeat_action_probability)

## Dry-Run Instructions
Once `scripts/run_dqn.sh` supports `--dry-run`:
```bash
bash experiments/dqn_atari/scripts/run_dqn.sh experiments/dqn_atari/configs/pong.yaml --dry-run
```
The dry run must:
- Execute a short random-policy rollout
- Save preprocessed frame stacks to `experiments/dqn_atari/artifacts/frames/<game>/`
- Log available actions and observation stats
- Write a minimal evaluation report (returns, episode length)

## Troubleshooting Notes
- If AutoROM fails, confirm `pip install AutoROM.accept-rom-license` succeeded and rerun with `--install-dir` pointing to a writable location.
- When Gymnasium/ALE versions drift, rerun `scripts/capture_env.sh` to snapshot system info and update this doc.
- For headless GPU servers, ensure SDL dependencies are installed before running Atari environments; document the package list here when known.
