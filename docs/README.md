# Documentation

Complete reference for the DQN Atari reproduction thesis project.

## Quick Start

- **New to the project?** Start with [Architecture Overview](architecture.md) → [Quick Start Guide](quick_start.md)
- **Need to do something?** Check [Workflows](workflows.md) for task-oriented guides
- **Stuck on an issue?** See [Troubleshooting](troubleshooting.md)
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
