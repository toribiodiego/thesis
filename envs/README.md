# Environment Specifications

This folder stores reproducible environment definitions (e.g., `requirements.txt`, pip-tools `*.in`, optional Dockerfiles) shared across experiments.

Planned files:

- `requirements.txt` – Primary development stack (Python, PyTorch, Gymnasium/ALE, logging stack) for use with `python -m venv`.
- `requirements-cpu.txt` / `requirements-gpu.txt` – Optional variants tuned for hardware constraints.
- `setup_env.sh` – Helper script to create/activate the virtual environment and install Atari ROM tooling.

Document any external steps (ROM acquisition, CUDA driver requirements) in comments inside the environment files to keep setup friction low.
