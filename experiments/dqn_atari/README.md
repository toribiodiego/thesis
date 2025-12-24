# DQN Atari Experiment

Experiment-specific assets that sit on top of the shared `src/` modules.

## Selected Games

Initial games for DQN reproduction (Mnih et al., 2013):

| Game        | Environment ID        | Action Set | Config File      | Purpose                           |
|-------------|-----------------------|------------|------------------|-----------------------------------|
| Pong        | `ALE/Pong-v5`         | minimal    | `pong.yaml`      | Simple game, fast convergence     |
| Breakout    | `ALE/Breakout-v5`     | minimal    | `breakout.yaml`  | Moderate complexity, brick-breaking strategy |
| Beam Rider  | `ALE/BeamRider-v5`    | minimal    | `beam_rider.yaml`| More complex, multi-object tracking |

**Action Set:** Using `minimal` action space (game-specific legal actions only, excluding NOOPs and redundant actions).

## ROM Setup

Atari ROMs are required to run ALE environments. Install them using:

```bash
../../setup/setup_roms.sh
```

This script calls `python -m AutoROM --accept-license` to download legally-redistributable Atari 2600 ROMs.

## ALE Runtime Settings

Deterministic configuration for reproducibility:

| Setting                      | Value   | Purpose                                      |
|------------------------------|---------|----------------------------------------------|
| `repeat_action_probability`  | `0.0`   | Disable stochastic frame skipping            |
| `frameskip`                  | `4`     | Action repeated 4 times, rewards accumulated |
| `full_action_space`          | `false` | Use minimal action set per game              |

These settings are applied when creating environments and ensure deterministic behavior across runs.

## Running DQN

### Quick Start

```bash
# 1. Install dependencies (first time only)
pip install -r requirements.txt

# 2. Download ROMs (first time only)
./setup/setup_roms.sh

# 3. Validate setup with dry run
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# 4. Start training
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --seed 123
```

### Training Scripts

All scripts are in `scripts/` and should be run from repository root:

- **`run_dqn.sh`** – Main training entry point with dry-run support
  ```bash
  # Dry run (validates preprocessing, saves debug frames)
  ./experiments/dqn_atari/scripts/run_dqn.sh \
    experiments/dqn_atari/configs/pong.yaml --dry-run

  # Full training
  ./experiments/dqn_atari/scripts/run_dqn.sh \
    experiments/dqn_atari/configs/pong.yaml --seed 42
  ```

- **`setup_roms.sh`** – One-time ROM installation via AutoROM
  ```bash
  ./setup/setup_roms.sh
  ```

- **`capture_env.sh`** – Record system info and package versions
  ```bash
  ./experiments/dqn_atari/scripts/capture_env.sh
  ```

See `scripts/README.md` for detailed documentation, all CLI flags, and common workflows.

## Layout
- `configs/` – `base.yaml` plus per-game overrides; see `configs/README.md`.
- `scripts/` – Training and setup utilities (`run_dqn.sh`, `setup_roms.sh`, `capture_env.sh`); see `scripts/README.md`.
- `notes.md` – Chronological experiment log for hyperparameter tweaks and observations.

Core implementations (agents, replay, env wrappers) stay under `src/`, keeping this directory focused on wiring, configs, and documentation.
