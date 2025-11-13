# Environment Specifications

This folder stores reproducible environment definitions shared across experiments.

## Files
- `requirements.txt` – Pinned base stack (PyTorch 2.4.1, Gymnasium/ALE, AutoROM helper, plotting/logging utilities).
- `setup_env.sh` – Creates `.venv/`, installs dependencies, and runs `python -m AutoROM --accept-license`.

## Usage
```bash
bash envs/setup_env.sh         # creates .venv and installs everything
source .venv/bin/activate      # activate when working in the repo
```

Pin any experiment-specific extras via additional `requirements-*.txt` files or pip-compile inputs as the project grows.
