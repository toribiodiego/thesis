# Environment Setup (MOVED)

All environment setup assets have been moved to `setup/`.

## New Location

- **Setup script**: `setup/setup_env.sh`
- **Requirements files**: `setup/requirements*.txt`
- **Documentation**: See project README and `docs/guides/workflows.md`

## Usage

```bash
# Create virtual environment and install dependencies
bash setup/setup_env.sh

# For GPU support
bash setup/setup_env.sh --gpu

# Activate environment
source .venv/bin/activate
```

---

**Note**: This directory (`envs/`) is deprecated and will be removed in a future cleanup. Please update any scripts or workflows to use `setup/` instead.
