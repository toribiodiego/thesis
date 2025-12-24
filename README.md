# Thesis RL Experiments

Masters thesis on sample- and data-efficient reinforcement learning. First milestone: reproduce DQN (Mnih et al., 2013) with reusable tooling for future algorithms (MuZero, EfficientZero, CURL, DrQ, SPR).

## Quick Start

### Environment Setup

**1. Create and activate virtual environment:**

```bash
# Create .venv and install all dependencies
bash setup/setup_env.sh

# Activate the virtual environment
source .venv/bin/activate
```

The `setup_env.sh` script:
- Creates a Python virtual environment at `.venv/`
- Installs all pinned dependencies from `setup/requirements.txt`
- Sets up Atari ROM tooling (AutoROM)

**Pinned Dependencies** (see `setup/requirements.txt` for authoritative versions):
- **PyTorch 2.4.1** (CUDA 12.1)
- **Gymnasium 0.29.1** (with Atari ROM license acceptance)
- **ALE-py 0.8.1** (Atari emulator)
- **NumPy 1.26.4**, **SciPy 1.13.1**
- **OpenCV 4.10.0** (image preprocessing)
- **matplotlib 3.9.1** (plotting)
- **OmegaConf 2.3.0** (config management)
- Additional utilities: tqdm, rich, typing-extensions
- Testing: pytest (install separately with `pip install pytest`)

**Important:** Always activate the virtual environment before running any commands:
```bash
source .venv/bin/activate
```

**2. Verify installation:**

Once the virtual environment is active, verify all dependencies are correctly installed:

```bash
# Check core dependencies are importable
python -c "import torch, gymnasium, ale_py"

# Verify pytest is available (install if needed: pip install pytest)
pytest --version

# Verify PyTorch version
python -c "import torch; print(f'PyTorch {torch.__version__}')"

# Check Gymnasium and ALE-py versions
python -c "import gymnasium; import ale_py; print(f'Gymnasium {gymnasium.__version__}, ALE-py {ale_py.__version__}')"
```

Expected output:
- No import errors from the first command
- pytest version 8.x or higher
- PyTorch 2.4.1
- Gymnasium 0.29.1, ALE-py 0.8.1

**3. Additional setup (optional):**

```bash
# Capture system info for reproducibility
./setup/capture_env.sh
```

### Run a Training Job

The complete DQN training pipeline is ready to use (Subtask 6 complete):

```bash
# 1. Validate training loop with smoke test (~5-10 min)
./experiments/dqn_atari/scripts/smoke_test.sh

# 2. Validate preprocessing with dry run (3 episodes, real env)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# 3. Run unit tests (optional, verify components)
pytest tests/test_dqn_trainer.py -k "training_step" -v

# 4. Start full training (10M frames for Pong)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --seed 123

# 5. Monitor training progress
tail -f experiments/dqn_atari/runs/pong_123/csv/episodes.csv
```

**Config overrides** (adjust runs without editing YAML):

```bash
# Override learning rate
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 7 \
  --set training.optimizer.lr=0.001

# Multiple overrides (repeat --set flag)
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 7 \
  --set training.total_frames=2000000 \
  --set training.gamma=0.95

# Disable target network (2013 NIPS DQN)
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 7 \
  --set target_network.update_interval=null
```

**Note:** `--set` is repeatable - use one `--set` flag per override.

See [experiments/dqn_atari/configs/README.md](experiments/dqn_atari/configs/README.md) for complete CLI reference.

**What's included:**
- [x] Complete training loop with epsilon-greedy exploration
- [x] Structured logging (steps, episodes, evaluation, Q-values)
- [x] Periodic evaluation with low-ε policy
- [x] Checkpoint management (periodic + best model)
- [x] Metadata persistence (git hash, config, seed)
- [x] Smoke test for fast validation
- [x] 163+ unit tests for all components

### Logging & Plots

Training metrics are logged to **three backends simultaneously**: TensorBoard, Weights & Biases (W&B), and CSV files.

**Enable W&B logging:**
```bash
# Set API key (one-time setup)
export WANDB_API_KEY="your_key_here"

# Configure in training config
# experiments/dqn_atari/configs/pong.yaml
logging:
  wandb:
    enabled: true
    project: "dqn-atari"
    entity: "my-team"      # optional
    upload_artifacts: true
    tags: ["baseline", "pong"]  # optional run tags

# Or use CLI flags (tags merge with config tags, not overwrite)
python train_dqn.py --cfg configs/pong.yaml \
  --set logging.wandb.enabled=true \
  --tags experiment-v2 --tags ablation

# Use offline mode (sync later)
export WANDB_MODE=offline
```

**View TensorBoard logs:**
```bash
# Launch TensorBoard
tensorboard --logdir experiments/dqn_atari/runs/

# Logs are written to:
# experiments/dqn_atari/runs/<game>_<seed>_<timestamp>/tensorboard/
```

**CSV logs location:**
```bash
# Per-step metrics (loss, epsilon, FPS)
experiments/dqn_atari/runs/<game>_<seed>_<timestamp>/csv/training_steps.csv

# Per-episode metrics (return, length)
experiments/dqn_atari/runs/<game>_<seed>_<timestamp>/csv/episodes.csv

# Quick check
tail -f experiments/dqn_atari/runs/pong_42_20251115/csv/episodes.csv
```

**Generate plots:**
```bash
# From local CSV files
python scripts/plot_results.py \
  --episodes experiments/dqn_atari/runs/pong_42_20251115/csv/episodes.csv \
  --steps experiments/dqn_atari/runs/pong_42_20251115/csv/training_steps.csv \
  --output plots/pong \
  --game-name pong

# From W&B artifacts
python scripts/plot_results.py \
  --wandb-project dqn-atari \
  --wandb-run abc123 \
  --wandb-artifact training_logs_step_1000000:latest \
  --output plots/pong

# Multi-seed aggregation (with 95% CI)
python scripts/plot_results.py \
  --multi-seed experiments/dqn_atari/runs/pong_42_20251115/csv/episodes.csv \
               experiments/dqn_atari/runs/pong_43_20251115/csv/episodes.csv \
               experiments/dqn_atari/runs/pong_44_20251115/csv/episodes.csv \
  --output plots/pong_multi_seed \
  --game-name pong
```

**Export results table:**
```bash
# Generate Markdown/CSV summary tables
python scripts/export_results_table.py \
  --runs-dir experiments/dqn_atari/runs/ \
  --output output/summary

# Outputs:
# - output/summary/results_summary.csv
# - output/summary/results_summary.md
```

**See also:**
- [docs/reference/logging-pipeline.md](docs/reference/logging-pipeline.md) - Complete logging & plotting documentation
- [scripts/plot_results.py](scripts/plot_results.py) - Full CLI reference with `--help`
- [scripts/export_results_table.py](scripts/export_results_table.py) - Results table generator

## Documentation Map

**Complete Documentation**: **[docs/README.md](docs/README.md)** is the main entry point for all project documentation.

**Quick Navigation**:
- **Getting Started**: [Quick Start](#quick-start) (this file) → [docs/guides/quick-start.md](docs/guides/quick-start.md) (detailed setup)
- **Common Tasks**: [docs/guides/workflows.md](docs/guides/workflows.md) - Training, debugging, testing workflows
- **Troubleshooting**: [docs/guides/troubleshooting.md](docs/guides/troubleshooting.md) - Problem diagnosis and fixes
- **Architecture**: [docs/guides/architecture.md](docs/guides/architecture.md) - System design overview
- **Component Specs**: [docs/reference/](docs/reference/) - Detailed technical specifications
- **Progress Tracking**: [TODO](TODO) (local file) - Current tasks and roadmap

**Documentation Structure**:
```
docs/
├── README.md              # Documentation index and navigation guide
├── guides/                # High-level task-oriented guides
├── reference/             # Technical component specifications
├── plans/                 # Experiment and analysis plans
├── reports/               # Experiment results and analysis
├── ops/                   # Maintenance procedures and checklists
└── thesis/                # Thesis-ready artifacts and integration
```

For the full documentation index with descriptions of each file, see **[docs/README.md](docs/README.md)**.

---

## Detailed Documentation References

**New to the project?** Start with the [Quick Start](#quick-start) above, then explore:

### Navigation Hub
- **[docs/README.md](docs/README.md)** - Complete documentation index with navigation guide
- **[docs/guides/workflows.md](docs/guides/workflows.md)** - Task-oriented guides for common operations
- **[docs/guides/troubleshooting.md](docs/guides/troubleshooting.md)** - Quick reference for problem diagnosis and solutions

### Planning & Progress
- **[TODO](TODO)** - Current roadmap and task tracker (source of truth for project status)
- **[docs/changelog.md](docs/changelog.md)** - Timeline of major completions and updates

**Note:** The `TODO` file is untracked (local workspace) and contains the authoritative task list. For high-level progress and completed work, see `docs/changelog.md`.

### Design Documentation
Core implementation specifications (read in this order):

**Setup & Environment:**
- **[docs/reference/dqn-setup.md](docs/reference/dqn-setup.md)** - Environment setup, dependencies, ROM installation, deterministic mode
- **[docs/reference/atari-env-wrapper.md](docs/reference/atari-env-wrapper.md)** - Wrapper chain specification and preprocessing pipeline
- **[docs/reference/config-cli.md](docs/reference/config-cli.md)** - Configuration system and CLI reference (hierarchical configs, overrides, validation, run artifacts)

**Core Components:**
- **[docs/reference/dqn-model.md](docs/reference/dqn-model.md)** - Q-network architecture and forward pass details
- **[docs/reference/replay-buffer.md](docs/reference/replay-buffer.md)** - Experience replay storage and sampling
- **[docs/reference/dqn-training.md](docs/reference/dqn-training.md)** - Q-learning update flow, loss functions, debugging

**Training & Evaluation:**
- **[docs/reference/episode-handling.md](docs/reference/episode-handling.md)** - Episode management, termination policies, training vs. evaluation
- **[docs/reference/training-loop-runtime.md](docs/reference/training-loop-runtime.md)** - Training loop orchestration, logging, evaluation, troubleshooting
- **[docs/reference/checkpointing.md](docs/reference/checkpointing.md)** - Checkpoint/resume system, metadata schema, deterministic seeding

### Quick References
- **[experiments/dqn_atari/README.md](experiments/dqn_atari/README.md)** - Experiment setup and game selection
- **[experiments/dqn_atari/scripts/README.md](experiments/dqn_atari/scripts/README.md)** - CLI tools and script usage
- **[tests/README.md](tests/README.md)** - Test suite documentation and running instructions

### Running DQN

See [experiments/dqn_atari/README.md](experiments/dqn_atari/README.md) for experiment-specific details and [experiments/dqn_atari/scripts/README.md](experiments/dqn_atari/scripts/README.md) for complete CLI documentation.

**Key scripts:**
- `experiments/dqn_atari/scripts/run_dqn.sh` – Training and dry-run validation
- `experiments/dqn_atari/scripts/smoke_test.sh` – Fast end-to-end validation (~200K frames)
- `setup/setup_roms.sh` – One-time ROM installation
- `setup/capture_env.sh` – System and package information capture

### Testing

See [tests/README.md](tests/README.md) for complete test documentation.

**Running tests** (assumes virtual environment is active):

```bash
# 1. Activate virtual environment (if not already active)
source .venv/bin/activate

# 2. Install pytest if not present
pip install pytest

# 3. Run tests
# All tests
pytest tests/ -v

# Training loop tests (Subtask 6)
pytest tests/test_dqn_trainer.py -v

# Targeted component tests
pytest tests/test_dqn_trainer.py -k "scheduler" -v

# Smoke test example (fast validation)
pytest tests/test_dqn_trainer.py -k "smoke_test" -v
```

**Complete workflow for contributors:**
1. Activate venv: `source .venv/bin/activate`
2. Install dependencies: `bash setup/setup_env.sh` (first time only)
3. Verify install: `python -c "import torch, gymnasium, ale_py"`
4. Run tests: `pytest tests/`

## Structure

```
├── train_dqn.py                 # Main training entry point
├── TODO                         # Current roadmap and task tracker (untracked)
├── docs/                        # Project documentation
│   ├── guides/                 # Task-oriented guides
│   ├── reference/              # Technical specifications
│   ├── plans/                  # Experiment plans
│   ├── reports/                # Analysis and results
│   └── ops/                    # Maintenance procedures
├── setup/                       # Environment setup and dependencies
│   ├── setup_env.sh            # Virtual environment creation
│   ├── setup_roms.sh           # Atari ROM installation
│   ├── capture_env.sh          # System info capture
│   └── requirements*.txt       # Pinned dependencies
├── src/                         # Reusable RL modules
│   ├── models/                 # Neural network architectures
│   ├── replay/                 # Experience replay buffers
│   ├── envs/                   # Atari wrappers and preprocessing
│   ├── training/               # DQN trainer and update logic
│   └── config/                 # Configuration system
├── scripts/                     # Global analysis utilities
│   ├── plot_results.py         # Plotting from CSV logs
│   ├── export_results_table.py # Results table generation
│   └── analyze_results.py      # Statistical analysis
├── experiments/dqn_atari/       # DQN experiment
│   ├── configs/                # YAML configs for each game
│   ├── scripts/                # DQN-specific wrappers
│   └── runs/                   # Training outputs (gitignored)
├── tests/                       # Unit and integration tests
├── output/                      # Analysis outputs (gitignored)
│   ├── plots/                  # Generated visualizations
│   └── summary/                # Results tables
└── wandb/                       # W&B local cache (gitignored)
```

## Workflow

1. Check `TODO` file for current tasks and priorities
2. Review relevant design docs for specifications
3. Implement following checklist items
4. Run tests and dry-run validation
5. Use commit prefixes from `docs/guides/git-commit-guide.md`
6. Mark completed items in `TODO` file
