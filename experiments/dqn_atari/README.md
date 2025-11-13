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
./scripts/setup_roms.sh
```

This script calls `python -m AutoROM --accept-license` to download legally-redistributable Atari 2600 ROMs.

## Layout
- `configs/` – `base.yaml` plus per-game overrides; see `configs/README.md`.
- `scripts/` – Shell entry points like `run_dqn.sh` (training), with planned `eval` and `dry_run` helpers.
- `notes.md` – Chronological experiment log for hyperparameter tweaks and observations.

Core implementations (agents, replay, env wrappers) stay under `src/`, keeping this directory focused on wiring, configs, and documentation.
