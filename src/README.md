# Shared Source Modules

Reusable components that power multiple experiments live here. Suggested submodules:

- `agents/` – Policy/Value learners (e.g., `dqn.py`, future `curl.py`, `drq.py`).
- `envs/` – Wrappers, preprocessing pipelines, and environment registration.
- `replay/` – Buffers and data iterators shared across algorithms.
- `config/` – Configuration schemas and helpers (Hydra/OmegaConf, dataclasses).
- `utils/` – Logging, seeding, checkpointing, metrics utilities.

Keeping common logic centralized avoids duplication across `experiments/*` directories and speeds up comparisons among algorithms.
