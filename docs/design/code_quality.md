# Code Quality and Testing Guide

This document provides the authoritative guide for testing, linting, formatting, type checking, and maintaining code quality in the DQN reproduction project.

## Overview

Code quality is maintained through:
1. **Comprehensive test suite** (335+ tests)
2. **Code formatting** (Black, isort)
3. **Linting** (Ruff/flake8)
4. **Type checking** (mypy, planned)
5. **Coverage tracking** (pytest-cov)
6. **CI automation** (GitHub Actions, planned)

---

## Testing

### Test Suite Structure

```
tests/
├── test_dqn_model.py         # CNN architecture tests
├── test_dqn_trainer.py       # Training loop tests
├── test_atari_wrappers.py    # Environment wrapper tests
├── test_replay_buffer.py     # Experience replay tests
├── test_config_manager.py    # Configuration loading tests
├── test_metrics_logger.py    # Logging infrastructure tests
├── test_video_recorder.py    # Video recording tests
├── test_evaluation.py        # Evaluation harness tests
└── conftest.py               # Shared fixtures
```

### Running Tests

**Full test suite:**
```bash
pytest tests/ -x
```

**With verbose output:**
```bash
pytest tests/ -v
```

**Specific test file:**
```bash
pytest tests/test_dqn_trainer.py -v
```

**Specific test function:**
```bash
pytest tests/test_dqn_trainer.py::test_trainer_initialization -v
```

**Run tests matching pattern:**
```bash
pytest tests/ -k "replay" -v
```

**Stop on first failure:**
```bash
pytest tests/ -x --tb=short
```

### Test Markers

Tests are organized with pytest markers:

```bash
# Run only slow tests
pytest tests/ -m slow

# Skip slow tests
pytest tests/ -m "not slow"

# Run GPU tests (if available)
pytest tests/ -m gpu
```

Available markers:
- `slow`: Tests that take >1 second
- `gpu`: Tests requiring CUDA
- `integration`: Integration tests

### Test Fixtures

Common fixtures in `conftest.py`:

```python
@pytest.fixture
def sample_config():
    """Minimal valid configuration."""
    return {
        "environment": {"game": "Pong", ...},
        "training": {"batch_size": 32, ...},
        ...
    }

@pytest.fixture
def temp_run_dir(tmp_path):
    """Temporary directory for run outputs."""
    return tmp_path / "test_run"

@pytest.fixture
def seeded_env():
    """Deterministically seeded environment."""
    env = make_atari_env("PongNoFrameskip-v4", seed=42)
    return env
```

### Writing New Tests

Follow these patterns:

```python
import pytest
import torch

from src.models.dqn_model import DQNModel

class TestDQNModel:
    """Test DQN CNN architecture."""

    def test_forward_pass_shape(self):
        """Verify output dimensions match action space."""
        model = DQNModel(observation_shape=(4, 84, 84), num_actions=6)
        batch = torch.randn(32, 4, 84, 84)
        output = model(batch)
        assert output.shape == (32, 6)

    def test_parameter_count(self):
        """Verify expected parameter count."""
        model = DQNModel((4, 84, 84), 6)
        params = sum(p.numel() for p in model.parameters())
        # ~1.68M parameters
        assert 1_600_000 < params < 1_700_000

    @pytest.mark.slow
    def test_gradient_flow(self):
        """Ensure gradients flow through all layers."""
        model = DQNModel((4, 84, 84), 6)
        # ... test implementation
```

### Test Coverage

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

Generate HTML report:
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

**Coverage targets:**
- Core modules (models, training, replay): >= 80%
- Utils and wrappers: >= 75%
- Overall: >= 75%

Current coverage status:
```
src/models/           92%
src/training/         85%
src/replay/           88%
src/utils/            78%
Overall:              84%
```

---

## Code Formatting

### Black (Code Formatter)

Configuration in `pyproject.toml`:
```toml
[tool.black]
line-length = 88
target-version = ['py311']
```

**Format all code:**
```bash
black src/ tests/ scripts/
```

**Check formatting without changes:**
```bash
black --check src/ tests/
```

**Show diff without applying:**
```bash
black --diff src/ tests/
```

### isort (Import Sorting)

Configuration in `pyproject.toml`:
```toml
[tool.isort]
profile = "black"
line_length = 88
```

**Sort imports:**
```bash
isort src/ tests/
```

**Check without changes:**
```bash
isort --check-only src/ tests/
```

### Combined Formatting

```bash
# Format everything
black src/ tests/ scripts/ && isort src/ tests/

# Check everything
black --check src/ tests/ && isort --check-only src/ tests/
```

---

## Linting

### Ruff (Fast Linter)

Configuration in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 88
target-version = "py311"

select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
]

ignore = [
    "E501",  # line too long (handled by black)
]
```

**Run linter:**
```bash
ruff check src/ tests/
```

**Auto-fix issues:**
```bash
ruff check --fix src/ tests/
```

### Common Lint Issues

1. **Unused imports**: Remove or mark with `# noqa: F401`
2. **Undefined names**: Check variable scope
3. **Line too long**: Let Black handle it
4. **Complexity**: Refactor complex functions

---

## Type Checking

### mypy (Static Type Checker)

Configuration in `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.11"
strict = false
warn_return_any = true
warn_unused_configs = true
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "src.*"
disallow_untyped_defs = true
```

**Run type checker:**
```bash
mypy src/
```

**Check specific module:**
```bash
mypy src/models/
```

### Type Annotations

Core modules should have complete type annotations:

```python
# Good: Fully typed function
def compute_td_loss(
    q_values: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    next_q_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Compute TD loss for DQN."""
    ...

# Types for complex objects
from typing import Dict, List, Optional, Tuple
import numpy.typing as npt

ObservationType = npt.NDArray[np.uint8]
ActionType = int
RewardType = float
```

### Annotation Progress

Track annotation coverage:
```bash
mypy src/ --html-report mypy_report/
```

Priority for type annotations:
1. `src/models/` - Core neural network
2. `src/training/` - Training loop
3. `src/replay/` - Replay buffer
4. `src/utils/` - Helper functions
5. `scripts/` - Utility scripts

---

## Pre-commit Hooks

### Setup

```bash
pip install pre-commit
pre-commit install
```

Configuration in `.pre-commit-config.yaml`:
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
        language_version: python3.11

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.0
    hooks:
      - id: ruff

  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
```

### Usage

Hooks run automatically on `git commit`. Manual run:
```bash
pre-commit run --all-files
```

---

## Continuous Integration (CI)

### GitHub Actions Workflow

File: `.github/workflows/ci.yml`
```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt
          pip install pytest pytest-cov black isort ruff

      - name: Format check
        run: |
          black --check src/ tests/
          isort --check-only src/ tests/

      - name: Lint
        run: ruff check src/ tests/

      - name: Test
        run: pytest tests/ -x --cov=src --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
```

### Local CI Simulation

Run all CI checks locally:
```bash
# scripts/run_ci.sh
#!/bin/bash
set -e

echo "=== Format Check ==="
black --check src/ tests/
isort --check-only src/ tests/

echo "=== Lint ==="
ruff check src/ tests/

echo "=== Type Check ==="
mypy src/ --ignore-missing-imports

echo "=== Tests ==="
pytest tests/ -x --cov=src --cov-report=term-missing

echo "=== All checks passed ==="
```

---

## Quality Metrics

### Code Health Dashboard

Track these metrics:
1. **Test coverage**: >= 75% (target 80%)
2. **Lint warnings**: 0 (clean codebase)
3. **Type errors**: 0 in annotated modules
4. **CI status**: All checks passing

### Technical Debt

Common debt items to address:
- [ ] Add missing type annotations
- [ ] Increase test coverage for edge cases
- [ ] Remove dead code
- [ ] Simplify complex functions (cyclomatic complexity)
- [ ] Update deprecated dependencies

---

## Troubleshooting

### Common Issues

**Black/isort conflicts:**
```bash
# Use isort with black profile
isort --profile black src/
```

**Pytest import errors:**
```bash
# Ensure src is in PYTHONPATH
PYTHONPATH=. pytest tests/
```

**mypy can't find module:**
```bash
# Add to pyproject.toml
[[tool.mypy.overrides]]
module = "problem_module"
ignore_missing_imports = true
```

**Coverage too low:**
```bash
# Identify uncovered code
pytest tests/ --cov=src --cov-report=term-missing
# Focus on lines marked with "!"
```

### Getting Help

- Test documentation: `tests/README.md`
- Python style guide: PEP 8
- Black documentation: https://black.readthedocs.io/
- Pytest documentation: https://docs.pytest.org/
- mypy documentation: https://mypy.readthedocs.io/

---

## Best Practices

### Writing Quality Code

1. **Follow DRY principle**: Don't repeat yourself
2. **Use descriptive names**: `compute_td_target` not `calc`
3. **Keep functions focused**: Single responsibility
4. **Document edge cases**: In docstrings and tests
5. **Handle errors gracefully**: Use try/except appropriately

### Code Review Checklist

Before submitting PR:
- [ ] All tests pass (`pytest tests/ -x`)
- [ ] Code formatted (`black --check src/`)
- [ ] No lint errors (`ruff check src/`)
- [ ] New features have tests
- [ ] Documentation updated if needed
- [ ] No hardcoded paths or secrets
- [ ] Commit messages follow convention

### Maintaining Quality

1. **Run tests frequently**: After every significant change
2. **Keep dependencies updated**: Check for security patches
3. **Monitor coverage**: Don't let it decrease
4. **Review before merge**: Self-review at minimum
5. **Document decisions**: In code comments or docs/

---

## References

- Project test suite: `tests/`
- Configuration: `pyproject.toml`
- CI workflows: `.github/workflows/` (planned)
- Python style: https://pep8.org/
- Testing guide: https://docs.pytest.org/
