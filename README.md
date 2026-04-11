# MEng Thesis

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch 2.4](https://img.shields.io/badge/pytorch-2.4-ee4c2c)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Masters thesis investigating when self-supervised representation
learning (SPR) improves data-efficient reinforcement learning and what
determines whether the learned representations are useful. A DQN
isolation study (augmentation x SPR, four conditions) reveals that SPR
barely helps vanilla DQN, while published results show large gains on
Rainbow -- suggesting the base agent's capability determines whether
self-supervised representations are exploitable. Interpretability
probes (linear probing, latent visualization, transition model
analysis) compare representations across agent capability levels to
explain why.

**Status:**
- DQN isolation study complete (24 runs, single seed)
- Rainbow training integrated
- Next: Rainbow runs, DQN + Rainbow multi-seed runs, interpretability

> **Quick links:** [Quick Start](#quick-start) ·
> [Documentation](docs/README.md) ·
> [Architecture](docs/guides/architecture.md) ·
> [Config Reference](docs/reference/config-cli.md)

<br><br>

## Quick Start

### Setup

```bash
bash setup/setup_env.sh
source .venv/bin/activate
python -c "import torch, gymnasium, ale_py"
```

<br><br>

### Training

```bash
# Vanilla DQN (100K steps)
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/atari100k_boxing.yaml --seed 42

# DQN + augmentation + SPR
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/atari100k_boxing_both.yaml --seed 42

# Rainbow DQN
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/atari100k_boxing_rainbow.yaml --seed 42

# Config overrides (without editing YAML)
python train_dqn.py \
  --cfg experiments/dqn_atari/configs/atari100k_boxing.yaml --seed 7 \
  --set training.optimizer.lr=0.001
```

See [`experiments/dqn_atari/configs/README.md`](experiments/dqn_atari/configs/README.md)
for the full config and CLI reference.

<br><br>

### Plotting

```bash
# Learning curves across conditions (used in working results)
python scripts/plot_learning_curves.py

# Per-run plots from CSV
python scripts/plot_results.py \
  --episodes experiments/dqn_atari/runs/<run_dir>/csv/episodes.csv \
  --output output/plots/run_name --game-name boxing

# Re-evaluate checkpoints (auto-discovers runs)
python scripts/eval_checkpoints.py
python scripts/eval_checkpoints.py run_name_1 run_name_2
```

<br><br>

### Testing

```bash
pytest tests/ -x
pytest tests/test_rainbow_model.py -v
```

<br><br>

## Reproducibility

- **Seeded RNG** -- Centralized seeding across Python, NumPy, and
  PyTorch; full RNG state saved in checkpoints and restored on resume
- **Pinned dependencies** -- All packages pinned in
  `setup/requirements.txt`
- **Metadata tracking** -- Every run records git commit, merged config,
  and system info
- **Provenance** -- [`reports/provenance.md`](reports/provenance.md)
  documents how each figure and table was produced

See [Engineering Standards](docs/standards/engineering.md) for details.

<br><br>

## Documentation

[`docs/README.md`](docs/README.md) is the documentation index. Key
entry points:

| Category | Document | Description |
|----------|----------|-------------|
| Getting started | [`docs/guides/quick-start.md`](docs/guides/quick-start.md) | Environment and dependency setup |
| Workflows | [`docs/guides/workflows.md`](docs/guides/workflows.md) | Training, debugging, testing |
| Architecture | [`docs/guides/architecture.md`](docs/guides/architecture.md) | System design overview |
| Config & CLI | [`docs/reference/config-cli.md`](docs/reference/config-cli.md) | Configuration and CLI reference |
| SPR | [`docs/reference/spr-architecture.md`](docs/reference/spr-architecture.md) | SPR components and integration |
| Rainbow | [`docs/reference/rainbow-architecture.md`](docs/reference/rainbow-architecture.md) | Rainbow components and integration |
| Standards | [`docs/standards/`](docs/standards/) | Documentation and engineering conventions |

<br><br>

## Structure

```text
train_dqn.py                     # Training entry point
src/
  models/                        # DQN, RainbowDQN, SPR, NoisyLinear, EMA
  replay/                        # Uniform and prioritized replay buffers
  envs/                          # Atari wrappers and preprocessing
  training/                      # Training loop, losses, evaluation, logging
  config/                        # Configuration loading and validation
experiments/dqn_atari/
  configs/                       # YAML configs (baseline, aug, spr, rainbow)
  scripts/                       # Training and validation scripts
  runs/                          # Run data (Google Drive, gitignored)
scripts/                         # Plotting, analysis, checkpoint re-evaluation
tests/                           # Unit and integration tests
reports/                         # Working results and provenance records
writing/                         # Thesis manuscript (LaTeX)
presentation/                    # Defense slides (LaTeX/Beamer)
docs/                            # Project documentation
setup/                           # Environment setup and dependencies
```

<br><br>

## Dependencies

See `setup/requirements.txt` for pinned versions. Core:

- **PyTorch 2.4.1** (CUDA 12.1)
- **Gymnasium 0.29.1** with **ALE-py 0.8.1**
- **NumPy 1.26.4**, **matplotlib 3.9.1**
- **OmegaConf 2.3.0**
