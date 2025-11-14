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

**2. Additional setup (optional):**

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
tail -f experiments/dqn_atari/runs/pong_123/logs/episodes.csv
```

**What's included:**
- [x] Complete training loop with epsilon-greedy exploration
- [x] Structured logging (steps, episodes, evaluation, Q-values)
- [x] Periodic evaluation with low-ε policy
- [x] Checkpoint management (periodic + best model)
- [x] Metadata persistence (git hash, config, seed)
- [x] Smoke test for fast validation
- [x] 163+ unit tests for all components

## Documentation

### Roadmap
- **[docs/roadmap.md](docs/roadmap.md)** – Complete project plan with 21 subtasks and progress tracking

### Design Docs
Core implementation specifications and guides:

- **[docs/design/dqn_setup.md](docs/design/dqn_setup.md)** – Environment setup, dependencies, and ROM installation
- **[docs/design/atari_env_wrapper.md](docs/design/atari_env_wrapper.md)** – Wrapper chain specification and preprocessing pipeline
- **[docs/design/dqn_model.md](docs/design/dqn_model.md)** – Q-network architecture and forward pass details
- **[docs/design/replay_buffer.md](docs/design/replay_buffer.md)** – Experience replay storage and sampling
- **[docs/design/dqn_training.md](docs/design/dqn_training.md)** – Q-learning update flow, loss functions, and debugging guide
- **[docs/design/episode_handling.md](docs/design/episode_handling.md)** – Episode management, termination policies, training vs. evaluation
- **[docs/design/training_loop_runtime.md](docs/design/training_loop_runtime.md)** – Training loop orchestration, logging, evaluation, troubleshooting

### Running DQN

See [experiments/dqn_atari/README.md](experiments/dqn_atari/README.md) for experiment-specific details and [experiments/dqn_atari/scripts/README.md](experiments/dqn_atari/scripts/README.md) for complete CLI documentation.

**Key scripts:**
- `run_dqn.sh` – Training and dry-run validation
- `smoke_test.sh` – Fast end-to-end validation (~200K frames)
- `setup_roms.sh` – One-time ROM installation
- `capture_env.sh` – System and package information capture

### Testing

See [tests/README.md](tests/README.md) for complete test documentation.

**Run tests:**
```bash
# All tests
pytest tests/ -v

# Training loop tests (Subtask 6)
pytest tests/test_dqn_trainer.py -v

# Targeted component tests
pytest tests/test_dqn_trainer.py -k "scheduler" -v
```

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
