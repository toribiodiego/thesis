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

## Contributing to Docs

When adding or updating documentation:

1. Place high-level guides in `docs/` root
2. Place technical specs in `docs/design/`
3. Place experiment reports in `docs/reports/`
4. Update the relevant README/index when adding new docs
5. Follow the git commit guide for doc changes: `docs: <description>`
