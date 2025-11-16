# Thesis RL Experiments

Masters thesis on sample- and data-efficient reinforcement learning. First milestone: reproduce DQN (Mnih et al., 2013) with reusable tooling for future algorithms (MuZero, EfficientZero, CURL, DrQ, SPR).

## Quick Start

### Environment Setup

**1. Create and activate virtual environment:**

```bash
# Create .venv and install all dependencies
bash envs/setup_env.sh

# Activate the virtual environment
source .venv/bin/activate
```

The `setup_env.sh` script:
- Creates a Python virtual environment at `.venv/`
- Installs all pinned dependencies from `envs/requirements.txt`
- Sets up Atari ROM tooling (AutoROM)

**Pinned Dependencies** (see `envs/requirements.txt` for authoritative versions):
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
./experiments/dqn_atari/scripts/capture_env.sh
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

# Multiple overrides
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 7 \
  --set training.total_frames=2000000 training.gamma=0.95

# Disable target network (2013 NIPS DQN)
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml --seed 7 \
  --set target_network.update_interval=null
```

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
tensorboard --logdir results/logs/pong/

# Logs are written to:
# results/logs/<game>/<run_id>/tensorboard/
```

**CSV logs location:**
```bash
# Per-step metrics (loss, epsilon, FPS)
results/logs/<game>/<run_id>/csv/training_steps.csv

# Per-episode metrics (return, length)
results/logs/<game>/<run_id>/csv/episodes.csv

# Quick check
tail -f results/logs/pong/pong_seed42/csv/episodes.csv
```

**Generate plots:**
```bash
# From local CSV files
python scripts/plot_results.py \
  --episodes results/logs/pong/run_123/csv/episodes.csv \
  --steps results/logs/pong/run_123/csv/training_steps.csv \
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
  --multi-seed results/logs/pong/seed1/csv/episodes.csv \
               results/logs/pong/seed2/csv/episodes.csv \
               results/logs/pong/seed3/csv/episodes.csv \
  --output plots/pong_multi_seed \
  --game-name pong
```

**Export results table:**
```bash
# Generate Markdown/CSV summary tables
python scripts/export_results_table.py \
  --runs-dir results/logs/pong/ \
  --output results/summary

# Outputs:
# - results/summary/results_summary.csv
# - results/summary/results_summary.md
```

**See also:**
- [docs/design/logging_pipeline.md](docs/design/logging_pipeline.md) - Complete logging & plotting documentation
- [scripts/plot_results.py](scripts/plot_results.py) - Full CLI reference with `--help`
- [scripts/export_results_table.py](scripts/export_results_table.py) - Results table generator

## Documentation

**New to the project?** Start with the [Quick Start](#quick-start) above, then explore:

### Navigation Hub
- **[docs/index.md](docs/index.md)** - Complete documentation index with navigation guide
- **[docs/workflows.md](docs/workflows.md)** - Task-oriented guides for common operations
- **[docs/troubleshooting.md](docs/troubleshooting.md)** - Quick reference for problem diagnosis and solutions

### Planning & Progress
- **[docs/roadmap.md](docs/roadmap.md)** - Complete project plan with 21 subtasks and progress tracking
- **[docs/changelog.md](docs/changelog.md)** - Timeline of major completions and updates

### Design Documentation
Core implementation specifications (read in this order):

**Setup & Environment:**
- **[docs/design/dqn_setup.md](docs/design/dqn_setup.md)** - Environment setup, dependencies, ROM installation, deterministic mode
- **[docs/design/atari_env_wrapper.md](docs/design/atari_env_wrapper.md)** - Wrapper chain specification and preprocessing pipeline
- **[docs/design/config_cli.md](docs/design/config_cli.md)** - Configuration system and CLI reference (hierarchical configs, overrides, validation, run artifacts)

**Core Components:**
- **[docs/design/dqn_model.md](docs/design/dqn_model.md)** - Q-network architecture and forward pass details
- **[docs/design/replay_buffer.md](docs/design/replay_buffer.md)** - Experience replay storage and sampling
- **[docs/design/dqn_training.md](docs/design/dqn_training.md)** - Q-learning update flow, loss functions, debugging

**Training & Evaluation:**
- **[docs/design/episode_handling.md](docs/design/episode_handling.md)** - Episode management, termination policies, training vs. evaluation
- **[docs/design/training_loop_runtime.md](docs/design/training_loop_runtime.md)** - Training loop orchestration, logging, evaluation, troubleshooting
- **[docs/design/checkpointing.md](docs/design/checkpointing.md)** - Checkpoint/resume system, metadata schema, deterministic seeding

### Quick References
- **[experiments/dqn_atari/README.md](experiments/dqn_atari/README.md)** - Experiment setup and game selection
- **[experiments/dqn_atari/scripts/README.md](experiments/dqn_atari/scripts/README.md)** - CLI tools and script usage
- **[tests/README.md](tests/README.md)** - Test suite documentation and running instructions

### Running DQN

See [experiments/dqn_atari/README.md](experiments/dqn_atari/README.md) for experiment-specific details and [experiments/dqn_atari/scripts/README.md](experiments/dqn_atari/scripts/README.md) for complete CLI documentation.

**Key scripts:**
- `run_dqn.sh` – Training and dry-run validation
- `smoke_test.sh` – Fast end-to-end validation (~200K frames)
- `setup_roms.sh` – One-time ROM installation
- `capture_env.sh` – System and package information capture

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
2. Install dependencies: `bash envs/setup_env.sh` (first time only)
3. Verify install: `python -c "import torch, gymnasium, ale_py"`
4. Run tests: `pytest tests/`

## Structure

```
├── docs/
│   ├── roadmap.md              # Project plan with 21 subtasks
│   └── design/                 # Architecture and implementation specs
├── envs/                        # Dependencies and setup scripts
├── src/                         # Reusable RL modules
│   ├── models/                 # Neural network architectures
│   ├── replay/                 # Experience replay buffers
│   ├── envs/                   # Atari wrappers and preprocessing
│   └── training/               # DQN trainer and update logic
├── experiments/dqn_atari/       # DQN configs and training scripts
│   ├── configs/                # YAML configs for each game
│   └── scripts/                # Training and setup utilities
└── tests/                       # Unit tests for all modules
```

## Workflow

1. Check `docs/roadmap.md` for current subtask
2. Review relevant design docs for specifications
3. Implement following checklist items
4. Run tests and dry-run validation
5. Use commit prefixes from `docs/git_commit_guide.md`
6. Mark completed items in roadmap
