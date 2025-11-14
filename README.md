# Thesis RL Experiments

Masters thesis on sample- and data-efficient reinforcement learning. First milestone: reproduce DQN (Mnih et al., 2013) with reusable tooling for future algorithms (MuZero, EfficientZero, CURL, DrQ, SPR).

## Quick Start

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Download Atari ROMs
./experiments/dqn_atari/scripts/setup_roms.sh

# 3. Validate preprocessing with dry run
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# 4. Start training
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --seed 123
```

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

### Running DQN

See [experiments/dqn_atari/README.md](experiments/dqn_atari/README.md) for experiment-specific details and [experiments/dqn_atari/scripts/README.md](experiments/dqn_atari/scripts/README.md) for complete CLI documentation.

**Key scripts:**
- `run_dqn.sh` – Training and dry-run validation
- `setup_roms.sh` – One-time ROM installation
- `capture_env.sh` – System and package information capture

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
