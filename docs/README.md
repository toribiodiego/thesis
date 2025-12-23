# Documentation

Complete reference for the DQN Atari reproduction thesis project.

## Quick Start

- **New to the project?** Start with [Architecture Overview](architecture.md) → [Quick Start Guide](quick_start.md)
- **Need to do something?** Check [Workflows](workflows.md) for task-oriented guides
- **Stuck on an issue?** See [Troubleshooting](troubleshooting.md)
- **Track progress?** See [TODO](../TODO) in repo root (untracked file with current tasks)
- **Full documentation index:** See [index.md](index.md)

---

## Docs Structure

This documentation is organized into the following sections:

### Top-Level Guides
High-level guides and references in the root `docs/` folder:

- **[architecture.md](architecture.md)** - System design and component interactions (start here)
- **[quick_start.md](quick_start.md)** - Setup and first training run
- **[workflows.md](workflows.md)** - Common task guides (train, debug, test)
- **[troubleshooting.md](troubleshooting.md)** - Problem diagnosis and fixes
- **[testing.md](testing.md)** - Test suite documentation
- **[git_commit_guide.md](git_commit_guide.md)** - Commit message conventions
- **[changelog.md](changelog.md)** - Project change history
- **[colab_guide.md](colab_guide.md)** - Google Colab setup for GPU training

### design/
Detailed technical specifications for each component:

- Environment wrappers and preprocessing
- DQN model architecture
- Replay buffer implementation
- Training loop and Q-learning updates
- Checkpointing and resume
- Config system and CLI
- Evaluation harness
- Logging and plotting pipeline
- GPU validation and performance
- Game suite plan and ablations

See [design/README.md](design/README.md) for the full index.

### papers/
Notes and summaries from research papers:

- DQN 2013 paper notes
- Related work references

### reports/
Experiment results and analysis:

- Training run summaries
- Multi-seed comparisons
- Performance benchmarks

### maintenance/
Operational checklists and maintenance guides:

- Pre-commit checklists
- Release procedures
- Dependency updates

---

## Naming Conventions

### Primary Entrypoint
- **`docs/README.md`** - Main documentation entry point (start here)
- **`README.md`** (repo root) - Project overview and quick links

### File Naming
All documentation files use **snake_case** with descriptive names:

**Top-level guides:**
- `quick_start.md` - Getting started guide
- `git_commit_guide.md` - Commit message conventions
- `colab_guide.md` - Platform-specific setup

**Design/reference docs:**
- `dqn_model.md` - Component specification
- `config_cli.md` - System configuration
- `ablations_plan.md` - Experiment plans

**Analysis/reports:**
- `design/gpu_validation.md` - Performance analysis
- `design/results_comparison.md` - Multi-run analysis

### Folder Structure
- `docs/` - Top-level guides and overviews
- `docs/design/` - Technical specifications
- `docs/papers/` - Research paper notes
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

1. **Getting started:** [architecture.md](architecture.md) → [quick_start.md](quick_start.md)
2. **Implementing features:** Check relevant doc in [design/](design/)
3. **Running experiments:** [workflows.md](workflows.md) or [quick_start.md](quick_start.md)
4. **Debugging issues:** [troubleshooting.md](troubleshooting.md)
5. **Full details:** [index.md](index.md) for complete documentation index

---

## Docs Reorganization Plan

**Status:** PLANNED - This section documents the upcoming docs restructure (Task 02)

The following table maps current file locations to their planned destinations. This serves as a reference during the reorganization to ensure all files are tracked.

### File Move Map

| Current Path | New Path | Category | Notes |
|--------------|----------|----------|-------|
| **Top-Level Guides** | | | |
| `docs/architecture.md` | `docs/guides/architecture.md` | Guide | System design overview |
| `docs/quick_start.md` | `docs/guides/quick-start.md` | Guide | Rename to kebab-case |
| `docs/workflows.md` | `docs/guides/workflows.md` | Guide | Task-oriented guides |
| `docs/troubleshooting.md` | `docs/guides/troubleshooting.md` | Guide | Problem diagnosis |
| `docs/testing.md` | `docs/guides/testing.md` | Guide | Test suite docs |
| `docs/colab_guide.md` | `docs/guides/colab-setup.md` | Guide | Rename to kebab-case |
| `docs/git_commit_guide.md` | `docs/guides/git-commit-guide.md` | Guide | Rename to kebab-case |
| `docs/changelog.md` | `docs/changelog.md` | Root | Keep in root |
| **Design/Reference Docs** | | | |
| `docs/design/dqn_setup.md` | `docs/reference/dqn-setup.md` | Reference | Rename to kebab-case |
| `docs/design/atari_env_wrapper.md` | `docs/reference/atari-env-wrapper.md` | Reference | Rename to kebab-case |
| `docs/design/dqn_model.md` | `docs/reference/dqn-model.md` | Reference | Rename to kebab-case |
| `docs/design/replay_buffer.md` | `docs/reference/replay-buffer.md` | Reference | Rename to kebab-case |
| `docs/design/dqn_training.md` | `docs/reference/dqn-training.md` | Reference | Rename to kebab-case |
| `docs/design/episode_handling.md` | `docs/reference/episode-handling.md` | Reference | Rename to kebab-case |
| `docs/design/training_loop_runtime.md` | `docs/reference/training-loop-runtime.md` | Reference | Rename to kebab-case |
| `docs/design/checkpointing.md` | `docs/reference/checkpointing.md` | Reference | Rename to kebab-case |
| `docs/design/config_cli.md` | `docs/reference/config-cli.md` | Reference | Rename to kebab-case |
| `docs/design/eval_harness.md` | `docs/reference/eval-harness.md` | Reference | Rename to kebab-case |
| `docs/design/logging_pipeline.md` | `docs/reference/logging-pipeline.md` | Reference | Rename to kebab-case |
| `docs/design/environment_notes.md` | `docs/reference/environment-notes.md` | Reference | Rename to kebab-case |
| `docs/design/stability_notes.md` | `docs/reference/stability-notes.md` | Reference | Rename to kebab-case |
| **Plans** | | | |
| `docs/design/game_suite_plan.md` | `docs/plans/game-suite-plan.md` | Plan | Rename to kebab-case |
| `docs/design/ablations_plan.md` | `docs/plans/ablations-plan.md` | Plan | Rename to kebab-case |
| `docs/design/report_outline.md` | `docs/plans/report-outline.md` | Plan | Rename to kebab-case |
| `docs/design/reproduce_recipe.md` | `docs/plans/reproduce-recipe.md` | Plan | Rename to kebab-case |
| **Reports** | | | |
| `docs/design/gpu_validation.md` | `docs/reports/gpu-validation.md` | Report | Rename to kebab-case |
| `docs/design/results_comparison.md` | `docs/reports/results-comparison.md` | Report | Rename to kebab-case |
| `docs/reports/dqn_results.md` | `docs/reports/dqn-results.md` | Report | Rename to kebab-case |
| **Operational Docs** | | | |
| `docs/design/code_quality.md` | `docs/ops/code-quality.md` | Ops | Rename to kebab-case |
| `docs/maintenance/checklists.md` | `docs/ops/checklists.md` | Ops | Consolidate maintenance/ into ops/ |
| **Research/Papers** | | | |
| `docs/papers/README.md` | `docs/research/papers/README.md` | Research | Move into research/ |
| `docs/papers/dqn_2013_notes.md` | `docs/research/papers/dqn-2013-notes.md` | Research | Rename to kebab-case |
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
│   ├── game-suite-plan.md
│   ├── ablations-plan.md
│   ├── report-outline.md
│   └── reproduce-recipe.md
├── reports/               # Experiment results and analysis
│   ├── gpu-validation.md
│   ├── results-comparison.md
│   └── dqn-results.md
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
- `docs/papers/` (moved to research/papers/)

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
