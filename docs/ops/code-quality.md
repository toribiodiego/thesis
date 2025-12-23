# Code Quality

This document provides a pragmatic guide for testing and maintaining code quality in the DQN reproduction thesis project.

> **See Also**: [Testing Guide](../guides/testing.md) - Comprehensive test suite documentation with detailed examples and debugging tips

## Overview

For this thesis project, focus on:
1. **Testing** - Run tests before important commits (335+ tests available)
2. **Optional formatting** - Use when convenient (Black, isort)
3. **Optional linting** - Use when debugging issues (Ruff)

Note: This is a thesis project, not production software. Prioritize working code over perfect code.

---

## Testing

### Quick Start

Run all tests before important commits:
```bash
pytest tests/ -x
```

This runs the full test suite and stops on first failure.

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

### Common Test Commands

**Full test suite (stop on first failure):**
```bash
pytest tests/ -x
```

**Verbose output:**
```bash
pytest tests/ -v
```

**Specific test file:**
```bash
pytest tests/test_dqn_trainer.py -v
```

**Skip slow tests:**
```bash
pytest tests/ -m "not slow"
```

**Run specific pattern:**
```bash
pytest tests/ -k "replay" -v
```

### Test Markers

Available markers for filtering tests:
- `slow`: Tests that take >1 second
- `gpu`: Tests requiring CUDA
- `integration`: Integration tests

### When to Run Tests

**Essential:**
- Before committing major changes
- After modifying core training/model code
- Before running expensive multi-hour experiments

**Optional:**
- After small config changes
- For documentation-only changes

---

## Code Formatting (Optional)

These tools help maintain consistency but are not required for thesis work.

### Black (Auto-formatter)

Format code when you feel like it:
```bash
black src/ tests/ scripts/
```

Check without modifying:
```bash
black --check src/ tests/
```

Configuration in `pyproject.toml`:
```toml
[tool.black]
line-length = 88
target-version = ['py311']
```

### isort (Import Sorting)

Sort imports:
```bash
isort src/ tests/
```

Check only:
```bash
isort --check-only src/ tests/
```

Configuration in `pyproject.toml`:
```toml
[tool.isort]
profile = "black"
line_length = 88
```

### Combined Formatting

Run both together:
```bash
black src/ tests/ scripts/ && isort src/ tests/
```

---

## Linting (Optional)

### Ruff

Use Ruff to catch potential bugs:
```bash
ruff check src/ tests/
```

Auto-fix simple issues:
```bash
ruff check --fix src/ tests/
```

Configuration in `pyproject.toml`:
```toml
[tool.ruff]
line-length = 88
target-version = "py311"

select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "B",   # flake8-bugbear
]

ignore = [
    "E501",  # line too long (handled by black)
]
```

---

## Test Coverage (Optional)

Check test coverage when curious:
```bash
pytest tests/ --cov=src --cov-report=term-missing
```

Generate HTML report:
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

Current coverage (as of latest run):
```
src/models/           92%
src/training/         85%
src/replay/           88%
src/utils/            78%
Overall:              84%
```

Don't obsess over coverage percentages for thesis work.

---

## Type Checking (Optional)

### mypy

If you want static type checking:
```bash
mypy src/ --ignore-missing-imports
```

Configuration in `pyproject.toml`:
```toml
[tool.mypy]
python_version = "3.11"
strict = false
warn_return_any = true
ignore_missing_imports = true
```

Type annotations are helpful but not required for all code.

---

## Recommended Workflow

### Before Important Commits

Run this minimal checklist:
```bash
# 1. Run tests
pytest tests/ -x

# 2. (Optional) Format code if you modified multiple files
black src/ tests/

# 3. Done - commit your changes
```

### Before Long Experiments

Extra validation before expensive GPU runs:
```bash
# Run full test suite
pytest tests/ -v

# Run a quick smoke test if available
python train_dqn.py --dry-run
```

### When Debugging Issues

Use linting to catch potential bugs:
```bash
# Check for undefined variables, unused imports, etc.
ruff check src/ tests/
```

---

## Troubleshooting

### Common Issues

**Pytest import errors:**
```bash
# Ensure src is in PYTHONPATH
PYTHONPATH=. pytest tests/
```

**Tests fail after environment changes:**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

**Coverage report not showing:**
```bash
# Install pytest-cov
pip install pytest-cov
```

---

## Philosophy for Thesis Work

### Do:
- Run tests before important commits
- Fix failing tests before moving on
- Use tests to validate experimental changes
- Format code when convenient

### Don't:
- Obsess over 100% coverage
- Block work for minor lint warnings
- Enforce strict formatting rules
- Over-engineer quality processes

### Remember:
This is research code for a thesis, not production software. The goal is correct, reproducible results - not perfect code style.

---

## References

- [Testing Guide](../guides/testing.md) - Comprehensive test suite documentation
- Test suite: `tests/`
- Configuration: `pyproject.toml`
- Python style: https://pep8.org/
- Pytest docs: https://docs.pytest.org/
