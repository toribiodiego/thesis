# Documentation

Complete reference for the DQN Atari reproduction thesis project.

> **Last Updated**: February 2026
> **When to Read**: Start here when joining the project, setting up your environment, or looking for specific component documentation. Revisit after major milestones or when workflow changes.

## Quick Start

- **New to the project?** Start with [Architecture Overview](guides/architecture.md) → [Quick Start Guide](guides/quick-start.md)
- **Need to do something?** Check [Workflows](guides/workflows.md) for task-oriented guides
- **Stuck on an issue?** See [Troubleshooting](guides/troubleshooting.md)

<br><br>

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

- **[dqn-setup.md](reference/dqn-setup.md)** - Environment setup, game selection, and reproducibility
- **[atari-env-wrapper.md](reference/atari-env-wrapper.md)** - Preprocessing, frame stacking, reward clipping
- **[dqn-model.md](reference/dqn-model.md)** - CNN architecture and weight initialization
- **[replay-buffer.md](reference/replay-buffer.md)** - Circular buffer with uniform sampling
- **[dqn-training.md](reference/dqn-training.md)** - TD targets, loss computation, and optimization
- **[training-loop-runtime.md](reference/training-loop-runtime.md)** - Main training loop orchestration
- **[episode-handling.md](reference/episode-handling.md)** - Life-loss vs full-episode termination
- **[checkpointing.md](reference/checkpointing.md)** - Save/resume and deterministic seeding
- **[config-cli.md](reference/config-cli.md)** - Configuration system and command-line interface
- **[eval-harness.md](reference/eval-harness.md)** - Periodic evaluation, metrics, and video capture
- **[logging-pipeline.md](reference/logging-pipeline.md)** - Multi-backend logging, plotting, and results analysis
- **[environment-notes.md](reference/environment-notes.md)** - Toolchain differences and compatibility
- **[stability-notes.md](reference/stability-notes.md)** - Hyperparameter choices and stability observations

### reports/
Experiment results and analysis:

- **[foundation-validation-summary.md](reports/foundation-validation-summary.md)** - Implementation validation recap
- **[report-gpu-validation.md](reports/report-gpu-validation.md)** - GPU vs CPU performance

### ops/
Operational procedures and maintenance:

- **[code-quality.md](ops/code-quality.md)** - Code quality and testing guide

### research/papers/
Notes and summaries from research papers:

- **[dqn-2013-notes.md](research/papers/dqn-2013-notes.md)** - DQN 2013 paper implementation notes
- **[README.md](research/papers/README.md)** - Research paper index

<br><br>

## Naming Conventions

### Primary Entrypoint
- **`docs/README.md`** - Main documentation entry point (start here)
- **`README.md`** (repo root) - Project overview and quick links

### File Naming
All documentation files use **kebab-case** with descriptive names:

**Top-level guides:**
- `quick-start.md` - Getting started guide
- `git-commit-guide.md` - Commit message conventions
- `colab-guide.md` - Platform-specific setup

**Design/reference docs:**
- `dqn-model.md` - Component specification
- `config-cli.md` - System configuration
- `report-gpu-validation.md` - Performance analysis

### Folder Structure
- `docs/` - Documentation root
- `docs/guides/` - High-level task-oriented guides
- `docs/reference/` - Technical component specifications
- `docs/plans/` - Experiment and analysis plans
- `docs/reports/` - Experiment results and analysis
- `docs/ops/` - Operational procedures and maintenance
- `docs/research/papers/` - Research paper notes

<br><br>

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
- [notes.md](../experiments/dqn_atari/notes.md) - Experiment log for tracking observations and follow-ups during training runs

**output/** - Generated analysis outputs:
- Plots and visualizations
- Analysis summaries and tables
- Statistical comparisons

**Key principle:** If it documents *how the system works* --> `docs/`. If it describes *what a specific run produced* --> `output/` or `experiments/`.

<br><br>

## Navigation Tips

1. **Getting started:** [architecture.md](guides/architecture.md) → [quick-start.md](guides/quick-start.md)
2. **Implementing features:** Check relevant doc in [reference/](reference/)
3. **Running experiments:** [workflows.md](guides/workflows.md) or [quick-start.md](guides/quick-start.md)
4. **Debugging issues:** [troubleshooting.md](guides/troubleshooting.md)
5. **Testing:** [testing.md](guides/testing.md) for test suite documentation

<br><br>

## Contributing to Docs

When adding or updating documentation:

1. Place high-level guides in `docs/guides/`
2. Place technical specs in `docs/reference/`
3. Place experiment reports in `docs/reports/`
4. Place experiment plans in `docs/plans/`
5. Place operational guides in `docs/ops/`
6. Update the relevant README/index when adding new docs
7. Follow the git commit guide for doc changes: `docs: <description>`
8. Use the minimal outline template below for new docs

### Minimal Outline Template

All new documentation should start with this minimal structure to keep scope tight and focused:

```markdown
# Document Title

> **Status**: DRAFT | ACTIVE | REFERENCE
> **Purpose**: One-sentence description of what this document provides.

## Purpose

What problem does this document solve? What questions does it answer?
(2-3 sentences)

## Inputs/Outputs

**Inputs** (what readers need before using this doc):
- Prerequisites, prior knowledge, or context required
- Related docs to read first

**Outputs** (what readers gain after reading):
- Concrete outcomes, decisions, or capabilities
- What they can now do that they couldn't before

## [Main Content Sections]

... (document-specific content) ...

## Next Steps

Where should readers go after this document?
- Links to related docs
- Follow-up actions or tasks
- Further reading

<br><br>

**Last Updated**: YYYY-MM-DD
```

**Key principles**:
- **Purpose** answers "Why does this doc exist?"
- **Inputs/Outputs** sets clear expectations for readers
- **Next Steps** provides navigation and prevents dead-ends
- Keep each section concise (prefer bullet points over prose)
- Add **Status** indicator: DRAFT (incomplete), ACTIVE (actively used), REFERENCE (stable/archival)

<br><br>

## Docs Conventions

### Naming

**Files**:
- Use **kebab-case** for all documentation files (e.g., `quick-start.md`, `dqn-model.md`)
- Prefix reports with `report-` (e.g., `report-gpu-validation.md`)
- Prefix plans with `plan-` (e.g., `plan-experiment.md`)
- Use descriptive, searchable names that indicate content

**Folders**:
- Follow the established structure: `guides/`, `reference/`, `plans/`, `reports/`, `ops/`, `research/`
- Keep folder names concise and self-explanatory

### Headings and Structure

**H1 Titles**:
- Match the filename (e.g., `quick-start.md` → `# Quick Start`)
- Use Title Case for H1 headings
- Include status callout for plans and research docs

**Section Hierarchy**:
- Use proper heading levels (H2 for main sections, H3 for subsections)
- Keep nesting to 3 levels maximum (H1 → H2 → H3)
- Use parallel structure within a section (all bullets or all paragraphs)

**Content Style**:
- Prefer bullet points over long paragraphs
- Use code blocks for commands, configs, and examples
- Include concrete examples over abstract descriptions
- Add cross-references to related docs

### Update Cadence

**When to Update**:
- **Immediately**: When implementation changes behavior documented in reference docs
- **Before commit**: When workflow changes affect guides or quickstarts
- **After milestones**: Update reports and summaries when experiments complete
- **Quarterly review**: Check all docs for stale references and broken links

**Update Checklist**:
- [ ] Update "Last Updated" date at bottom of modified docs
- [ ] Check cross-references still point to correct sections
- [ ] Update version numbers or commit hashes if referenced
- [ ] Verify code examples still work with current implementation
- [ ] Update navigation in `docs/README.md` if adding new docs

**Maintenance**:
- Mark outdated docs with `> **Status**: DEPRECATED` callout
- Archive obsolete docs to `docs/archive/` rather than deleting
- Keep documentation current with major project changes
- Run periodic link checks (manual or scripted) to find broken references
