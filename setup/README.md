# Environment Setup

This directory contains all scripts and dependency files for setting up the development environment.

## Quick Start

```bash
# 1. Create virtual environment and install dependencies
bash setup/setup_env.sh

# 2. Activate the environment
source .venv/bin/activate

# 3. Install Atari ROMs (one-time)
bash setup/setup_roms.sh

# 4. (Optional) Capture system info
bash setup/capture_env.sh
```

## Scripts

### setup_env.sh

Creates a Python virtual environment and installs all dependencies.

**Usage:**
```bash
# CPU-only installation (default)
bash setup/setup_env.sh

# GPU installation (CUDA 12.1)
bash setup/setup_env.sh --gpu
```

**What it does:**
- Auto-detects Python 3.11 (or falls back to python3)
- Creates `.venv/` virtual environment
- Installs pip and wheel
- Installs dependencies from requirements files
- Installs Atari ROMs via AutoROM

**Requirements files used:**
- Default: `setup/requirements.txt` (CPU-only PyTorch)
- GPU mode: `setup/requirements-gpu.txt` (PyTorch with CUDA 12.1)

**Output:**
- `.venv/` directory at repository root
- Confirmation of PyTorch, CUDA, and Gymnasium versions

### setup_roms.sh

Downloads legally-redistributable Atari 2600 ROMs for ALE environments.

**Usage:**
```bash
bash setup/setup_roms.sh
```

**What it does:**
- Runs `python -m AutoROM --accept-license`
- Downloads ROMs to AutoROM package directory
- Imports ROMs into ale-py

**Verification:**
```bash
python -c "import ale_py; print(ale_py.roms.list())"
```

### capture_env.sh

Captures system and environment information for reproducibility.

**Usage:**
```bash
# Use default output location
bash setup/capture_env.sh

# Specify custom output file
bash setup/capture_env.sh /path/to/output.txt
```

**Default output:** `experiments/dqn_atari/system_info.txt`

**What it captures:**
- System information (OS, kernel, architecture, date, hostname)
- Python version and executable path
- Key package versions (PyTorch, NumPy, Gymnasium, ALE-py, CUDA)
- Git repository state (commit, branch, status)
- Full pip freeze output

**When to use:**
- Before starting production training runs
- When reporting bugs or reproducibility issues
- When documenting experimental conditions

## Dependency Files

### requirements.txt

Base dependencies for CPU-only environments.

**Key packages:**
- PyTorch 2.4.1 (CPU-only)
- Gymnasium 0.29.1
- ALE-py 0.8.1
- NumPy 1.26.4
- AutoROM (Atari ROM downloader)

### requirements-cpu.txt

Alternative CPU-only specification (currently matches requirements.txt).

### requirements-gpu.txt

GPU dependencies with CUDA 12.1 support.

**Key differences from CPU version:**
- PyTorch 2.4.1+cu121 (CUDA 12.1)
- torchvision 0.19.1+cu121
- torchaudio 2.4.1+cu121

**Platform support:**
- Tested on: Google Colab A100 GPU
- CUDA version: 12.1

## Troubleshooting

### Python version not found

**Error:** `No Python 3 installation found`

**Solution:** Install Python 3.11 or Python 3:
```bash
# macOS
brew install python@3.11

# Ubuntu/Debian
sudo apt-get install python3.11

# Check version
python3.11 --version
```

### ROM installation fails

**Error:** AutoROM fails or ROMs not found

**Solution:** Manually install ROMs:
```bash
# Activate venv first
source .venv/bin/activate

# Install AutoROM
pip install autorom[accept-rom-license]

# Download ROMs
AutoROM --accept-license

# Import into ale-py
ale-import-roms ~/.local/lib/python3.11/site-packages/AutoROM/roms
```

### CUDA not available (GPU mode)

**Error:** `torch.cuda.is_available()` returns False

**Solution:**
1. Verify NVIDIA driver is installed: `nvidia-smi`
2. Check CUDA version matches PyTorch: `nvcc --version`
3. Reinstall PyTorch with correct CUDA version
4. On Colab: Ensure GPU runtime is selected (Runtime > Change runtime type > GPU)

### Virtual environment activation fails

**Error:** `.venv/bin/activate` not found

**Solution:**
```bash
# Recreate virtual environment
rm -rf .venv
bash setup/setup_env.sh
```

## Development Workflow

### First-time setup

```bash
# 1. Clone repository
git clone <repository-url>
cd thesis

# 2. Set up environment (choose CPU or GPU)
bash setup/setup_env.sh          # CPU
bash setup/setup_env.sh --gpu    # GPU

# 3. Activate environment
source .venv/bin/activate

# 4. Verify installation
python -c "import torch, gymnasium, ale_py; print('Success!')"

# 5. Install ROMs
bash setup/setup_roms.sh

# 6. (Optional) Capture system info
bash setup/capture_env.sh
```

### Updating dependencies

When `requirements.txt` changes:

```bash
# Activate environment
source .venv/bin/activate

# Update packages
pip install -r setup/requirements.txt --upgrade

# Verify versions
pip list
```

### Fresh environment rebuild

```bash
# Remove old environment
rm -rf .venv

# Recreate from scratch
bash setup/setup_env.sh

# Activate and verify
source .venv/bin/activate
python -c "import torch; print(torch.__version__)"
```

## Environment Variables

The setup scripts do not set environment variables by default. For W&B integration or other services, create a `.env` file at the repository root:

```bash
# .env (gitignored)
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=dqn-atari
WANDB_ENTITY=your_entity
```

Load environment variables:
```bash
set -a && source .env && set +a
```

## Platform-Specific Notes

### macOS (Apple Silicon)

- Use CPU-only PyTorch (MPS support not yet validated)
- ROM installation may require Rosetta 2 for x86 packages

### Linux (Ubuntu/Debian)

- May require `python3-venv` package: `sudo apt-get install python3-venv`
- GPU mode requires NVIDIA driver + CUDA 12.1

### Google Colab

- Use GPU mode: `bash setup/setup_env.sh --gpu`
- Environment persists only during session (must reinstall on reconnect)
- Pre-installed packages may conflict; use fresh venv

### Windows (WSL)

- Use WSL 2 for best compatibility
- Follow Linux setup instructions
- GPU support requires WSL 2 with CUDA toolkit

## See Also

- [Main README](../README.md) - Project overview and usage
- [Quick Start Guide](../docs/guides/quick-start.md) - Getting started with training
- [Workflows Guide](../docs/guides/workflows.md) - Common development tasks
- [Troubleshooting Guide](../docs/guides/troubleshooting.md) - Detailed problem solutions
- [Environment Notes](../docs/reference/dqn-setup.md) - Technical environment details
