# Thesis: Data-Efficient DQN with Self-Supervised Representation Learning

Masters thesis investigating how self-supervised representation learning
(SPR) and data augmentation interact to improve data efficiency for DQN
on the Atari-100K benchmark. Includes a 2x2 factorial study isolating
SPR on vanilla DQN, augmentation interaction analysis, and
interpretability probes into learned representations.

> **Quick links:** [Quick Start](#quick-start) ·
> [Documentation](docs/README.md) ·
> [Architecture](docs/guides/architecture.md) ·
> [Config Reference](docs/reference/config-cli.md)

<br><br>

## Quick Start

### Environment Setup

**1. Create and activate virtual environment:**

```bash
bash setup/setup_env.sh
source .venv/bin/activate
```

The `setup_env.sh` script creates a Python virtual environment at
`.venv/`, installs all pinned dependencies from
`setup/requirements.txt`, and sets up Atari ROM tooling (`AutoROM`).

**2. Verify installation:**

```bash
python -c "import torch, gymnasium, ale_py"
pytest --version
```

**3. Capture system info (optional):**

```bash
./setup/capture_env.sh
```

<br><br>

### Run a Training Job

```bash
# Smoke test (~5-10 min)
./experiments/dqn_atari/scripts/smoke_test.sh

# Dry run (3 episodes, real environment)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --dry-run

# Full training (100K interaction steps)
./experiments/dqn_atari/scripts/run_dqn.sh \
  experiments/dqn_atari/configs/pong.yaml --seed 123
```

**Config overrides** (adjust runs without editing YAML):

```bash
python train_dqn.py --cfg experiments/dqn_atari/configs/pong.yaml \
  --seed 7 \
  --set training.optimizer.lr=0.001 \
  --set training.total_frames=2000000
```

Use one `--set` flag per override. See
[`experiments/dqn_atari/configs/README.md`](experiments/dqn_atari/configs/README.md)
for the complete CLI reference.

<br><br>

### Logging and Plots

Training metrics log to TensorBoard, Weights & Biases, and CSV files
simultaneously.

```bash
# TensorBoard
tensorboard --logdir experiments/dqn_atari/runs/

# CSV logs
tail -f experiments/dqn_atari/runs/<run_dir>/csv/episodes.csv

# Generate plots from CSV
python scripts/plot_results.py \
  --episodes experiments/dqn_atari/runs/<run_dir>/csv/episodes.csv \
  --steps experiments/dqn_atari/runs/<run_dir>/csv/training_steps.csv \
  --output plots/pong --game-name pong

# Multi-seed aggregation (with 95% CI)
python scripts/plot_results.py \
  --multi-seed runs/pong_42/csv/episodes.csv \
               runs/pong_43/csv/episodes.csv \
               runs/pong_44/csv/episodes.csv \
  --output plots/pong_multi_seed --game-name pong

# Results summary table
python scripts/export_results_table.py \
  --runs-dir experiments/dqn_atari/runs/ --output output/summary
```

Enable W&B by setting `WANDB_API_KEY` and
`logging.wandb.enabled=true` in the config. See
[`docs/reference/logging-pipeline.md`](docs/reference/logging-pipeline.md)
for full logging documentation.

<br><br>

### Testing

```bash
source .venv/bin/activate

# All tests
pytest tests/ -x

# Specific component
pytest tests/test_dqn_trainer.py -k "training_step" -v
```

See [`tests/README.md`](tests/README.md) for the full test suite
documentation.

<br><br>

## Documentation

[`docs/README.md`](docs/README.md) is the documentation index. Key
entry points:

| Category | Document | Description |
|----------|----------|-------------|
| Getting started | [`docs/guides/quick-start.md`](docs/guides/quick-start.md) | Detailed environment and dependency setup |
| Workflows | [`docs/guides/workflows.md`](docs/guides/workflows.md) | Training, debugging, testing procedures |
| Troubleshooting | [`docs/guides/troubleshooting.md`](docs/guides/troubleshooting.md) | Problem diagnosis and fixes |
| Architecture | [`docs/guides/architecture.md`](docs/guides/architecture.md) | System design overview |
| Config & CLI | [`docs/reference/config-cli.md`](docs/reference/config-cli.md) | Configuration system and CLI reference |
| Training loop | [`docs/reference/training-loop-runtime.md`](docs/reference/training-loop-runtime.md) | Loop orchestration, logging, evaluation |
| Experiment plans | [`docs/plans/`](docs/plans/) | Ablation, multi-game, and report plans |
| Results | [`docs/reports/`](docs/reports/) | Experiment results and analysis |
| Standards | [`docs/standards/`](docs/standards/) | Documentation and engineering conventions |

<br><br>

## Structure

```text
train_dqn.py                     # Main training entry point
src/                             # Core implementation
  models/                        # Neural network architectures
  replay/                        # Experience replay buffers
  envs/                          # Atari wrappers and preprocessing
  training/                      # DQN trainer and update logic
  config/                        # Configuration system
experiments/dqn_atari/           # DQN experiment
  configs/                       # YAML configs per game
  scripts/                       # Training and validation scripts
docs/                            # Project documentation
  guides/                        # Task-oriented guides
  reference/                     # Technical component specs
  plans/                         # Experiment plans
  reports/                       # Results and analysis
  standards/                     # Documentation and engineering standards
  thesis/                        # Thesis manuscript assets
scripts/                         # Analysis utilities (plotting, export)
tests/                           # Unit and integration tests
setup/                           # Environment setup and dependencies
```

<br><br>

## Dependencies

See `setup/requirements.txt` for pinned versions. Core dependencies:

- **PyTorch 2.4.1** (CUDA 12.1)
- **Gymnasium 0.29.1** with **ALE-py 0.8.1**
- **NumPy 1.26.4**, **SciPy 1.13.1**
- **OpenCV 4.10.0**, **matplotlib 3.9.1**
- **OmegaConf 2.3.0**
- **pytest** (testing)
