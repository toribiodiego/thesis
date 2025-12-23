# Documentation

Complete reference for the DQN Atari reproduction thesis project.

## Quick Start

- **New to the project?** Start with [Architecture Overview](guides/architecture.md) → [Quick Start Guide](guides/quick-start.md)
- **Need to do something?** Check [Workflows](guides/workflows.md) for task-oriented guides
- **Stuck on an issue?** See [Troubleshooting](guides/troubleshooting.md)
- **Track progress?** See [TODO](../TODO) in repo root (untracked file with current tasks)

---

## Docs Structure

This documentation is organized into the following sections:

### guides/
High-level task-oriented guides:

- **[architecture.md](guides/architecture.md)** - System design and component interactions (start here)
- **[quick-start.md](guides/quick-start.md)** - Setup and first training run
- **[workflows.md](guides/workflows.md)** - Common task guides (train, debug, test)
- **[troubleshooting.md](guides/troubleshooting.md)** - Problem diagnosis and fixes
- **[testing.md](guides/testing.md)** - Test suite documentation
- **[git-commit-guide.md](guides/git-commit-guide.md)** - Commit message conventions
- **[colab-guide.md](guides/colab-guide.md)** - Google Colab setup for GPU training

### reference/
Detailed technical specifications for each component:

- Environment wrappers and preprocessing
- DQN model architecture
- Replay buffer implementation
- Training loop and Q-learning updates
- Checkpointing and resume
- Config system and CLI
- Evaluation harness
- Logging and plotting pipeline
- Environment and stability notes

See [reference/README.md](reference/README.md) for the full index (to be created).

### plans/
Experiment and analysis plans:

- Game suite selection
- Ablation studies
- Report outline
- Reproduction recipe

### reports/
Experiment results and analysis:

- GPU validation results
- Multi-seed comparisons
- DQN baseline results

### ops/
Operational procedures and maintenance:

- Code quality and testing guide
- Pre-commit checklists

### resear../research/papers/
Notes and summaries from research papers:

- DQN 2013 paper notes
- Related work references

### Other
- **[changelog.md](changelog.md)** - Project timeline (kept in root)

---

## Naming Conventions

### Primary Entrypoint
- **`docs/README.md`** - Main documentation entry point (start here)
- **`README.md`** (repo root) - Project overview and quick links

### File Naming
All documentation files use **snake_case** with descriptive names:

**Top-level guides:**
- `quick-start.md` - Getting started guide
- `git-commit-guide.md` - Commit message conventions
- `colab-guide.md` - Platform-specific setup

**Design/reference docs:**
- `dqn-model.md` - Component specification
- `config-cli.md` - System configuration
- `plan-ablations.md` - Experiment plans

**Analysis/reports:**
- `design/report-gpu-validation.md` - Performance analysis
- `design/report-results-comparison.md` - Multi-run analysis

### Folder Structure
- `docs/` - Top-level guides and overviews
- `docs/design/` - Technical specifications
- `do../research/papers/` - Research paper notes
- `docs/reports/` - Experiment results
- `docs/maintenance/` - Operational guides

---

## Docs Scope

### What Belongs in docs/

Documentation in `docs/` should be:
- **Stable reference material** - Guides, specs, and procedures
- **Reusable across experiments** - Not run-specific
- **Version controlled** - Important for reproducibility

**Include in docs/:**
- Architecture and design decisions
- Implementation guides and workflows
- Testing and troubleshooting procedures
- Configuration and setup instructions
- Paper summaries and research notes

### What Belongs Elsewhere

**experiments/** - Experiment-specific content:
- Config files for specific runs
- Run scripts and job submissions
- Experiment metadata and logs

**results/** - Generated outputs:
- Plots and visualizations
- Analysis summaries and tables
- Trained model checkpoints

**notes/** - Personal workspace:
- Scratch notes and ideas
- Planning checklists
- Temporary investigation notes
- Draft content before formalization

**TODO (repo root)** - Project roadmap and task tracker:
- Untracked file (not in git)
- Source of truth for current tasks and priorities
- Personal workspace for tracking progress
- See [TODO](../TODO) for active task list

**Key principle:** If it documents *how the system works* → `docs/`. If it describes *what a specific run produced* → `results/` or `experiments/`.

---

## Navigation Tips

1. **Getting started:** [architecture.md](guides/architecture.md) → [quick-start.md](guides/quick-start.md)
2. **Implementing features:** Check relevant doc in [reference/](reference/)
3. **Running experiments:** [workflows.md](guides/workflows.md) or [quick-start.md](guides/quick-start.md)
4. **Debugging issues:** [troubleshooting.md](guides/troubleshooting.md)
5. **Testing:** [testing.md](guides/testing.md) for test suite documentation

---

## Docs Reorganization Plan

**Status:** PLANNED - This section documents the upcoming docs restructure (Task 02)

The following table maps current file locations to their planned destinations. This serves as a reference during the reorganization to ensure all files are tracked.

### File Move Map

| Current Path | New Path | Category | Notes |
|--------------|----------|----------|-------|
| **Top-Level Guides** | | | |
| `docs/architecture.md` | `docs/guides/architecture.md` | Guide | System design overview |
| `docs/quick-start.md` | `docs/guides/quick-start.md` | Guide | Rename to kebab-case |
| `docs/workflows.md` | `docs/guides/workflows.md` | Guide | Task-oriented guides |
| `docs/troubleshooting.md` | `docs/guides/troubleshooting.md` | Guide | Problem diagnosis |
| `docs/testing.md` | `docs/guides/testing.md` | Guide | Test suite docs |
| `docs/colab-guide.md` | `docs/guides/colab-setup.md` | Guide | Rename to kebab-case |
| `docs/git-commit-guide.md` | `docs/guides/git-commit-guide.md` | Guide | Rename to kebab-case |
| `docs/changelog.md` | `docs/changelog.md` | Root | Keep in root |
| **Design/Reference Docs** | | | |
| `docs/design/dqn-setup.md` | `docs/reference/dqn-setup.md` | Reference | Rename to kebab-case |
| `docs/design/atari-env-wrapper.md` | `docs/reference/atari-env-wrapper.md` | Reference | Rename to kebab-case |
| `docs/design/dqn-model.md` | `docs/reference/dqn-model.md` | Reference | Rename to kebab-case |
| `docs/design/replay-buffer.md` | `docs/reference/replay-buffer.md` | Reference | Rename to kebab-case |
| `docs/design/dqn-training.md` | `docs/reference/dqn-training.md` | Reference | Rename to kebab-case |
| `docs/design/episode-handling.md` | `docs/reference/episode-handling.md` | Reference | Rename to kebab-case |
| `docs/design/training-loop-runtime.md` | `docs/reference/training-loop-runtime.md` | Reference | Rename to kebab-case |
| `docs/design/checkpointing.md` | `docs/reference/checkpointing.md` | Reference | Rename to kebab-case |
| `docs/design/config-cli.md` | `docs/reference/config-cli.md` | Reference | Rename to kebab-case |
| `docs/design/eval-harness.md` | `docs/reference/eval-harness.md` | Reference | Rename to kebab-case |
| `docs/design/logging-pipeline.md` | `docs/reference/logging-pipeline.md` | Reference | Rename to kebab-case |
| `docs/design/environment-notes.md` | `docs/reference/environment-notes.md` | Reference | Rename to kebab-case |
| `docs/design/stability-notes.md` | `docs/reference/stability-notes.md` | Reference | Rename to kebab-case |
| **Plans** | | | |
| `docs/design/plan-game-suite.md` | `docs/plans/plan-game-suite.md` | Plan | Rename to kebab-case |
| `docs/design/plan-ablations.md` | `docs/plans/plan-ablations.md` | Plan | Rename to kebab-case |
| `docs/design/plan-report-outline.md` | `docs/plans/plan-report-outline.md` | Plan | Rename to kebab-case |
| `docs/design/plan-reproduce-recipe.md` | `docs/plans/plan-reproduce-recipe.md` | Plan | Rename to kebab-case |
| **Reports** | | | |
| `docs/design/report-gpu-validation.md` | `docs/reports/report-gpu-validation.md` | Report | Rename to kebab-case |
| `docs/design/report-results-comparison.md` | `docs/reports/report-results-comparison.md` | Report | Rename to kebab-case |
| `docs/reports/report-dqn-results.md` | `docs/reports/report-dqn-results.md` | Report | Rename to kebab-case |
| **Operational Docs** | | | |
| `docs/design/code-quality.md` | `docs/ops/code-quality.md` | Ops | Rename to kebab-case |
| `docs/maintenance/checklists.md` | `docs/ops/checklists.md` | Ops | Consolidate maintenance/ into ops/ |
| **Research/Papers** | | | |
| `docs/resear../research/papers/README.md` | `docs/research/resear../research/papers/README.md` | Research | Move into research/ |
| `docs/resear../research/papers/dqn-2013-notes.md` | `docs/research/resear../research/papers/dqn-2013-notes.md` | Research | Rename to kebab-case |
| **Index/README** | | | |
| `docs/index.md` | `docs/README.md` | Root | Merge into existing README.md |
| `docs/design/README.md` | `docs/reference/README.md` | Reference | Consolidate design index |

### New Folder Structure

After reorganization, the docs tree will be:

```
docs/
├── README.md              # Main entry point (merge index.md content)
├── changelog.md           # Project timeline (keep in root)
├── guides/                # High-level task-oriented guides
│   ├── architecture.md
│   ├── quick-start.md
│   ├── workflows.md
│   ├── troubleshooting.md
│   ├── testing.md
│   ├── colab-setup.md
│   └── git-commit-guide.md
├── reference/             # Technical component specifications
│   ├── README.md
│   ├── dqn-setup.md
│   ├── atari-env-wrapper.md
│   ├── dqn-model.md
│   ├── replay-buffer.md
│   ├── dqn-training.md
│   ├── episode-handling.md
│   ├── training-loop-runtime.md
│   ├── checkpointing.md
│   ├── config-cli.md
│   ├── eval-harness.md
│   ├── logging-pipeline.md
│   ├── environment-notes.md
│   └── stability-notes.md
├── plans/                 # Experiment and analysis plans
│   ├── plan-game-suite.md
│   ├── plan-ablations.md
│   ├── plan-report-outline.md
│   └── plan-reproduce-recipe.md
├── reports/               # Experiment results and analysis
│   ├── report-gpu-validation.md
│   ├── report-results-comparison.md
│   └── report-dqn-results.md
├── ops/                   # Operational procedures and maintenance
│   ├── code-quality.md
│   └── checklists.md
└── research/              # Research paper notes and references
    └── papers/
        ├── README.md
        └── dqn-2013-notes.md
```

### Legacy Folders to Remove

After files are moved:
- `docs/design/` (empty after moving all contents)
- `docs/maintenance/` (merged into ops/)
- `do../research/papers/` (moved to resear../research/papers/)

---

## Contributing to Docs

When adding or updating documentation:

1. Place high-level guides in `docs/guides/`
2. Place technical specs in `docs/reference/`
3. Place experiment reports in `docs/reports/`
4. Place experiment plans in `docs/plans/`
5. Place operational guides in `docs/ops/`
6. Update the relevant README/index when adding new docs
7. Follow the git commit guide for doc changes: `docs: <description>`
