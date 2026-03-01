# Contributing

Guide to the development workflow, testing, and documentation
conventions for the DQN Atari-100K thesis project.

<br><br>

## Project Overview

This project implements and evaluates data-efficient deep reinforcement
learning on the Atari-100K benchmark. The codebase covers DQN training,
evaluation, data augmentation, self-supervised representation
learning (SPR), and interpretability analysis.

```text
thesis/
  src/              Core implementation (models, replay, training, utils)
  tests/            Unit and integration tests
  docs/             Design docs, guides, plans, reports, thesis
  experiments/      Configs, run scripts, artifacts
  output/           Plots, logs, reports (gitignored)
  scripts/          Utility scripts (plotting, analysis, setup)
  setup/            Environment setup, requirements
```

**Key files:**

- `train_dqn.py` -- main training entry point
- `experiments/dqn_atari/configs/` -- YAML configuration files
- `docs/guides/architecture.md` -- system design overview
- `docs/reports/reporting-requirements.md` -- thesis reporting backlog

<br><br>

## Development Environment

### Local setup

```bash
cd /path/to/thesis
pip install -r setup/requirements.txt
```

For CPU-only or GPU-specific installs, use `setup/requirements-cpu.txt`
or `setup/requirements-gpu.txt`.

### ROM setup

```bash
bash setup/setup_roms.sh
```

### Google Colab

See [Colab Guide](docs/guides/colab-guide.md) for notebook setup,
checkpointing, and auto-shutdown patterns.

<br><br>

## Testing

Run all tests, stopping on first failure:

```bash
pytest tests/ -x
```

Run a specific test file:

```bash
pytest tests/test_dqn_model.py -x
```

Rules:

- Run relevant tests after every implementation change
- Fix all test failures before committing
- Add tests when coverage is insufficient

<br><br>

## Documentation Standards

All files under `docs/` follow consistent formatting
conventions. Key rules:

- Active voice, imperative mood for instructions
- Language-tagged code blocks (`bash`, `python`, `yaml` -- never bare)
- Backticks for all technical references (class names, file paths,
  commands, config keys)
- `<br><br>` spacing between major sections (no horizontal rules `---`)
- Lowercase-hyphen (kebab-case) filenames
- No vague qualifiers ("very", "really", "quite", "fairly", "easily")

See [Documentation Standards](docs/standards/documentation.md) for the
full reference with examples, visual aid conventions, and
cross-referencing patterns.

See [Engineering Standards](docs/standards/engineering.md) for
reproducibility, compute efficiency, experiment hygiene, and
evidence-based decision making.

<br><br>

## Commit Messages

Use exactly one prefix (`feat:`, `fix:`, `refactor:`, `perf:`, `test:`,
`docs:`, `style:`, `build:`, `chore:`, `wip:`), imperative mood, and
keep subject lines at 72 characters or fewer. Focus on one logical
change per commit and explain WHY in the body.

### Prefix table

| Prefix      | When to use                                 |
|-------------|---------------------------------------------|
| `feat:`     | New user-facing functionality                |
| `fix:`      | Corrects broken behavior                     |
| `refactor:` | Restructures code without changing behavior  |
| `perf:`     | Improves performance measurably              |
| `test:`     | Adds or modifies tests only                  |
| `docs:`     | Documentation changes only                   |
| `style:`    | Formatting/linting only                      |
| `build:`    | Build system, CI, or packaging               |
| `chore:`    | Repo maintenance (deps, configs, cleanup)    |
| `wip:`      | Incomplete work (squash before merge)        |

### Message structure

```text
<prefix> <imperative verb> <what changed>

<WHY this change was needed -- not HOW>
<wrap lines at 72 characters>
```

<br><br>

## Questions or Issues?

Review existing issues on GitHub or open a new issue with detailed
reproduction steps. For research direction questions, check
[`docs/guides/architecture.md`](docs/guides/architecture.md).
