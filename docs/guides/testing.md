# Testing Guide

This document describes the test suite organization, how to run tests, and what each test module covers.

## Overview

The test suite uses **pytest** and is organized into focused test modules:

```
tests/
├── test_dqn_model.py          # DQN CNN architecture tests
├── test_dqn_trainer.py         # Core training utilities tests
├── test_evaluation.py          # Evaluation harness tests (NEW)
├── test_video_recorder.py      # Video recording tests (NEW)
├── test_replay_buffer.py       # Replay buffer tests
├── test_checkpoint.py          # Checkpointing tests
├── test_resume.py              # Resume functionality tests
├── test_config_loader.py       # Configuration system tests
├── test_cli.py                 # CLI argument parsing tests
└── ...
```

**Total:** ~194 tests across 13 test modules

---

## Quick Start

**Run all tests:**
```bash
pytest tests/
```

**Run specific test module:**
```bash
pytest tests/test_evaluation.py -v
```

**Run tests matching pattern:**
```bash
pytest tests/ -k "evaluate" -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=src --cov-report=html
```

---

## Evaluation Tests

### Test Modules

#### `test_evaluation.py` (19 tests)

Tests for the evaluation harness: `evaluate()` function, `EvaluationScheduler`, and `EvaluationLogger`.

**What it covers:**
- Core evaluate() function with various configurations
- Episode statistics aggregation (mean, median, std, min, max)
- Metadata inclusion (seed, step, eval_epsilon)
- Lives tracking (optional Atari-specific feature)
- Train/eval mode switching (model.train() / model.eval())
- Frame-based evaluation scheduling (every N steps)
- Wall-clock evaluation scheduling (every N seconds)
- Schedule metadata and history tracking
- Performance trend detection (improving/declining/stable)
- CSV output format and schema
- JSON output format (detailed evaluations)
- JSONL output format (streaming-friendly)
- Per-episode sidecar file
- Multi-format output verification

**Run all evaluation tests:**
```bash
pytest tests/test_evaluation.py -v
```

**Run specific evaluation test categories:**
```bash
# Core evaluation function tests
pytest tests/test_evaluation.py -k "test_evaluate" -v

# Scheduler tests
pytest tests/test_evaluation.py -k "test_evaluation_scheduler" -v

# Logger tests
pytest tests/test_evaluation.py -k "test_evaluation_logger" -v
```

**Example output:**
```
tests/test_evaluation.py::test_evaluate_basic PASSED                     [  5%]
tests/test_evaluation.py::test_evaluate_greedy PASSED                    [ 10%]
tests/test_evaluation.py::test_evaluate_with_metadata PASSED             [ 15%]
tests/test_evaluation.py::test_evaluate_lives_tracking PASSED            [ 21%]
tests/test_evaluation.py::test_evaluation_scheduler_interval PASSED      [ 26%]
tests/test_evaluation.py::test_evaluation_scheduler_wall_clock PASSED    [ 31%]
tests/test_evaluation.py::test_evaluation_logger_csv PASSED              [ 36%]
tests/test_evaluation.py::test_evaluation_logger_jsonl PASSED            [ 42%]
...
============================= 19 passed in 3.24s ==============================
```

#### `test_video_recorder.py` (10 tests)

Tests for video recording during evaluation.

**What it covers:**
- Basic frame capture and MP4 encoding
- Grayscale and RGB frame handling
- Float32 frame normalization
- Empty recorder handling (no frames captured)
- Directory creation for video outputs
- Different frame rates (15, 30, 60 FPS)
- Different resolutions (84×84, 210×160, 128×128)
- Integration with evaluate() function
- Video recording enable/disable
- First-episode-only recording verification

**Run all video recorder tests:**
```bash
pytest tests/test_video_recorder.py -v
```

**Run specific video test categories:**
```bash
# VideoRecorder component tests
pytest tests/test_video_recorder.py -k "test_video_recorder" -v

# Integration tests with evaluate()
pytest tests/test_video_recorder.py -k "test_evaluate" -v
```

**Example output:**
```
tests/test_video_recorder.py::test_video_recorder_basic PASSED           [ 10%]
tests/test_video_recorder.py::test_video_recorder_grayscale PASSED       [ 20%]
tests/test_video_recorder.py::test_video_recorder_float_frames PASSED    [ 30%]
tests/test_video_recorder.py::test_video_recorder_different_fps PASSED   [ 40%]
tests/test_video_recorder.py::test_evaluate_with_video_recording PASSED  [ 90%]
============================= 10 passed in 1.82s ==============================
```

**Note:** Video tests require OpenCV:
```bash
pip install opencv-python
```

---

## Common Test Commands

### Run evaluation tests only

```bash
# All evaluation-related tests
pytest tests/test_evaluation.py tests/test_video_recorder.py -v
```

### Run with verbose output

```bash
# Show test names and results
pytest tests/test_evaluation.py -v

# Show print statements
pytest tests/test_evaluation.py -v -s
```

### Run specific test function

```bash
# Run single test
pytest tests/test_evaluation.py::test_evaluate_basic -v

# Run multiple specific tests
pytest tests/test_evaluation.py::test_evaluate_basic tests/test_evaluation.py::test_evaluate_greedy -v
```

### Run tests matching pattern

```bash
# All tests with "scheduler" in name
pytest tests/ -k "scheduler" -v

# All tests with "video" in name
pytest tests/ -k "video" -v

# Multiple patterns (OR)
pytest tests/ -k "evaluate or video" -v

# Exclude pattern (NOT)
pytest tests/ -k "evaluate and not video" -v
```

### Run with coverage

```bash
# Coverage for evaluation module
pytest tests/test_evaluation.py --cov=src/training/evaluation --cov-report=term-missing

# Coverage for all training modules
pytest tests/ --cov=src/training --cov-report=html
# Open htmlcov/index.html in browser
```

### Run failed tests only

```bash
# Run tests that failed last time
pytest --lf -v

# Run failed tests first, then others
pytest --ff -v
```

### Stop on first failure

```bash
pytest tests/ -x
```

### Show slowest tests

```bash
pytest tests/ --durations=10
```

---

## Test Organization by Component

### DQN Model Tests (`test_dqn_model.py`)

- CNN architecture forward/backward pass
- Output shapes for different action spaces
- Weight initialization (Kaiming normal)
- Checkpoint save/load

### Training Utilities Tests (`test_dqn_trainer.py`)

- Target network hard updates
- TD target computation
- Loss functions (MSE, Huber)
- Optimizer configuration (RMSProp, Adam)
- Gradient clipping
- Training schedulers
- Stability checks

### Replay Buffer Tests (`test_replay_buffer.py`)

- Circular buffer storage
- Uniform sampling
- Episode boundary handling
- Memory optimization (uint8 storage)

### Checkpoint Tests (`test_checkpoint.py`)

- Model state saving/loading
- Optimizer state persistence
- RNG state capture
- Metadata recording

### Config Tests (`test_config_loader.py`)

- YAML config loading
- Config merging (base + overrides)
- Schema validation
- CLI overrides

---

## Continuous Integration

Tests run automatically on every commit via GitHub Actions (if configured).

**Local pre-commit check:**
```bash
# Run full test suite before committing
pytest tests/ -v

# Run quick smoke test
pytest tests/test_evaluation.py tests/test_video_recorder.py -v
```

---

## Debugging Test Failures

### View full traceback

```bash
pytest tests/test_evaluation.py -v --tb=long
```

### Drop into debugger on failure

```bash
pytest tests/test_evaluation.py -v --pdb
```

### Print pytest internals

```bash
pytest tests/test_evaluation.py -v --debug
```

### Show local variables in traceback

```bash
pytest tests/test_evaluation.py -v --showlocals
```

---

## Adding New Tests

### File naming convention

- Test files: `test_*.py`
- Test functions: `def test_*():`
- Test classes: `class Test*:`

### Example test structure

```python
def test_evaluation_feature():
    """Test description explaining what is being verified."""
    from src.training import evaluate
    from unittest.mock import Mock

    # Setup
    env = Mock()
    model = create_test_model()

    # Execute
    results = evaluate(env, model, num_episodes=10)

    # Verify
    assert 'mean_return' in results
    assert results['num_episodes'] == 10
```

### Best practices

1. **One test per feature**: Test one specific behavior per test function
2. **Descriptive names**: `test_evaluate_includes_seed_metadata` not `test_evaluate_2`
3. **Clear assertions**: Use specific assertions (`assert x == 10` not `assert x`)
4. **Clean setup/teardown**: Use fixtures or context managers for cleanup
5. **Mock external dependencies**: Use `unittest.mock` for environments, file I/O
6. **Document expectations**: Include docstring explaining what test verifies

---

## Test Coverage Goals

### Current coverage

```bash
pytest tests/ --cov=src --cov-report=term

# Expected output:
# src/training/evaluation.py    95%
# src/training/loss.py           92%
# src/models/dqn.py              88%
```

### Coverage targets

- **Critical modules (≥90%)**: evaluation, loss, optimization, replay
- **Core modules (≥80%)**: models, training loop, schedulers
- **Utilities (≥70%)**: logging, metadata, checkpointing

**Generate HTML coverage report:**
```bash
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

---

## Troubleshooting

### ImportError: No module named 'src'

**Solution:** Run pytest from repository root
```bash
cd /path/to/thesis
pytest tests/
```

### OpenCV not found (video tests fail)

**Solution:** Install OpenCV
```bash
pip install opencv-python
```

### CUDA out of memory (model tests)

**Solution:** Run tests on CPU
```bash
pytest tests/ -v  # Tests default to CPU
```

### Tests hang indefinitely

**Solution:** Add timeout
```bash
pytest tests/ -v --timeout=60
```

### Fixtures not found

**Solution:** Check conftest.py or import fixtures
```python
# In test file
from conftest import mock_env_fixture
```

---

## Related Documentation

- [Evaluation Harness Design](../reference/eval_harness.md) - Complete evaluation system documentation
- [Testing Best Practices](https://docs.pytest.org/en/stable/goodpractices.html) - Pytest conventions
- [Coverage.py](https://coverage.readthedocs.io/) - Coverage measurement tool

---

## Quick Reference

**Run all evaluation tests:**
```bash
pytest tests/test_evaluation.py tests/test_video_recorder.py -v
```

**Run with coverage:**
```bash
pytest tests/test_evaluation.py --cov=src/training/evaluation --cov-report=term-missing
```

**Run specific test:**
```bash
pytest tests/test_evaluation.py::test_evaluate_basic -v
```

**Run tests matching pattern:**
```bash
pytest tests/ -k "evaluate" -v
```

**Debug test failure:**
```bash
pytest tests/test_evaluation.py::test_failing -v --pdb
```
