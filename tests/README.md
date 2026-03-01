# Test Suite Documentation

Comprehensive test coverage for DQN implementation. All tests can be run from the repository root using pytest.

## Test Files

| Test File | Lines | Tests | Coverage |
|-----------|-------|-------|----------|
| `test_dqn_model.py` | ~500 | 20+ | DQN model architecture, forward/backward pass, checkpointing |
| `test_replay_buffer.py` | ~400 | 15+ | Circular buffer, sampling, episode boundaries, memory efficiency |
| `test_dqn_trainer.py` | 4,127 | 163+ | **Complete training infrastructure** (Subtask 5-6) |
| `test_atari_wrappers.py` | ~300 | 20+ | Preprocessing pipeline, frame stacking, reward clipping |
| `test_checkpoint.py` | ~500 | 16 | Checkpoint save/load, atomic writes, best model tracking (Subtask 7) |
| `test_resume.py` | ~560 | 15 | Resume from checkpoint, config validation, RNG restoration (Subtask 7) |
| `test_seeding.py` | ~350 | 17 | Deterministic seeding, RNG state management (Subtask 7) |
| `test_determinism.py` | ~350 | 20 | Determinism configuration, cuDNN flags (Subtask 7) |
| `test_save_resume_determinism.py` | ~550 | 1 | End-to-end save/resume smoke test (Subtask 7) |
| `test_metrics_logger.py` | ~900 | 28 | Multi-backend logging (TensorBoard/W&B/CSV), flush, artifacts (Subtask 10) |
| `test_plot_results.py` | ~490 | 20 | Plotting script, CSV loading, smoothing, metadata (Subtask 10) |

**Total:** 335+ unit tests across all modules

<br><br>

## Quick Start

### Run All Tests

```bash
# Run entire test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Parallel execution (faster)
pytest tests/ -n auto
```

### Run Specific Test Files

```bash
# Model tests only
pytest tests/test_dqn_model.py -v

# Replay buffer tests
pytest tests/test_replay_buffer.py -v

# Training infrastructure tests (Subtask 5-6)
pytest tests/test_dqn_trainer.py -v

# Wrapper tests
pytest tests/test_atari_wrappers.py -v
```

<br><br>

## Training Loop Tests (Subtask 6)

The `test_dqn_trainer.py` file contains 163+ tests covering all training components introduced in Subtasks 5-6.

### Targeted Test Suites

**By Component:**

```bash
# Schedulers (epsilon decay, training frequency, target sync)
pytest tests/test_dqn_trainer.py -k "scheduler" -v

# Training step orchestration
pytest tests/test_dqn_trainer.py -k "training_step" -v

# Logging (step logger, episode logger, checkpoints)
pytest tests/test_dqn_trainer.py -k "logger" -v

# Evaluation system
pytest tests/test_dqn_trainer.py -k "evaluation" -v

# Reference Q-value tracking
pytest tests/test_dqn_trainer.py -k "reference" -v

# Metadata and reproducibility
pytest tests/test_dqn_trainer.py -k "metadata or git" -v
```

**By Feature:**

```bash
# Loss computation and TD targets
pytest tests/test_dqn_trainer.py -k "loss or td" -v

# Target network management
pytest tests/test_dqn_trainer.py -k "target" -v

# Stability checks (NaN/Inf detection)
pytest tests/test_dqn_trainer.py -k "stability or nan or inf" -v

# Optimizer and gradient clipping
pytest tests/test_dqn_trainer.py -k "optimizer or gradient" -v
```

### Test Coverage Breakdown

| Component | Tests | File Location |
|-----------|-------|---------------|
| **Target Network** | 6 | Lines 100-250 |
| **Loss Functions** | 12 | Lines 251-600 |
| **Optimization** | 8 | Lines 601-850 |
| **Schedulers** | 18 | Lines 851-1500 |
| **Stability Checks** | 21 | Lines 1501-2100 |
| **Metrics** | 15 | Lines 2101-2500 |
| **Training Loop** | 22 | Lines 2501-3200 |
| **Logging** | 24 | Lines 3201-3700 |
| **Evaluation** | 18 | Lines 3701-4000 |
| **Q Tracking** | 9 | Lines 4001-4100 |
| **Metadata** | 8 | Lines 4101-4127 |
| **Integration** | 2 | Throughout |

<br><br>

## Common Test Commands

### Development Workflow

```bash
# Run tests for component you're working on
pytest tests/test_dqn_trainer.py -k "component_name" -v

# Run with automatic re-run on file changes
pytest tests/test_dqn_trainer.py -k "component_name" -f

# Stop on first failure (fast feedback)
pytest tests/test_dqn_trainer.py -x

# Run last failed tests only
pytest tests/ --lf
```

### CI/CD

```bash
# Fast subset for CI (~30s)
pytest tests/test_dqn_trainer.py -k "not slow" -v

# Full suite with coverage (for PRs)
pytest tests/ --cov=src --cov-report=xml

# Parallel execution
pytest tests/ -n auto --dist loadscope
```

### Debugging

```bash
# Verbose output with print statements
pytest tests/test_dqn_trainer.py::test_function_name -vvs

# Drop into debugger on failure
pytest tests/test_dqn_trainer.py --pdb

# Show local variables on failure
pytest tests/test_dqn_trainer.py -l

# Show test durations
pytest tests/ --durations=10
```

<br><br>

## Test Organization

### Unit Tests

**Purpose:** Verify individual components in isolation
**Speed:** Fast (seconds to minutes)
**Dependencies:** Mocked or minimal
**Example:** `test_epsilon_scheduler`, `test_compute_td_loss`

### Integration Tests

**Purpose:** Verify components work together correctly
**Speed:** Medium (minutes)
**Dependencies:** Multiple real components
**Example:** `test_training_step_integration`, `test_checkpoint_save_and_load`

### Smoke Tests

**Purpose:** Validate end-to-end training pipeline
**Speed:** Slow (~5-10 minutes)
**Dependencies:** All components, mock environment
**Location:** `experiments/dqn_atari/scripts/smoke_test.sh`
**Run:** `./experiments/dqn_atari/scripts/smoke_test.sh`

<br><br>

## Feature-to-Test Mapping

### Subtask 3: DQN Model

**Tests:** `test_dqn_model.py`
**Verify:**
- Model architecture (conv layers, FC layers)
- Forward pass shapes
- Gradient flow
- Checkpoint save/load

### Subtask 4: Replay Buffer

**Tests:** `test_replay_buffer.py`
**Verify:**
- Circular buffer behavior
- Episode boundary tracking
- Sampling correctness
- Memory efficiency (uint8 storage)

### Subtask 5: Q-Learning Loss & Optimizer

**Tests:** `test_dqn_trainer.py` (lines 100-2100)
**Verify:**
- TD target computation
- Loss functions (MSE, Huber)
- Optimizer configuration
- Gradient clipping
- Target network synchronization
- Stability checks

### Subtask 6: Training Loop & Evaluation

**Tests:** `test_dqn_trainer.py` (lines 2101-4127)
**Verify:**
- Epsilon scheduling
- Frame counter accuracy
- Training step orchestration
- Logging (steps, episodes, checkpoints)
- Evaluation system
- Reference Q tracking
- Metadata persistence

### Subtask 7: Checkpoint/Resume & Deterministic Seeding

**Tests:** `test_checkpoint.py`, `test_resume.py`, `test_seeding.py`, `test_determinism.py`, `test_save_resume_determinism.py`

<br><br>

## Logging & Plotting Tests (Subtask 10)

### Metrics Logger Tests

**File:** `test_metrics_logger.py` (28 tests, ~900 lines)

**Run all logging tests:**
```bash
# Full test suite
pytest tests/test_metrics_logger.py -v

# Specific backend tests
pytest tests/test_metrics_logger.py -k "tensorboard" -v
pytest tests/test_metrics_logger.py -k "wandb" -v
pytest tests/test_metrics_logger.py -k "csv" -v
```

**Test coverage:**

**TensorBoard Backend:**
```bash
# Run TensorBoard-specific tests
pytest tests/test_metrics_logger.py::test_tensorboard_backend_initialization -v
pytest tests/test_metrics_logger.py::test_tensorboard_backend_log_scalar -v
pytest tests/test_metrics_logger.py::test_tensorboard_backend_log_scalars -v
pytest tests/test_metrics_logger.py::test_tensorboard_backend_handles_none_values -v
```

**W&B Backend:**
```bash
# Run W&B-specific tests (with graceful degradation)
pytest tests/test_metrics_logger.py::test_wandb_backend_handles_import_error -v
pytest tests/test_metrics_logger.py::test_wandb_backend_disabled_operations -v

# Note: W&B tests handle missing wandb package gracefully
# No need to install wandb for tests to pass
```

**CSV Backend:**
```bash
# Run CSV-specific tests
pytest tests/test_metrics_logger.py::test_csv_backend_initialization -v
pytest tests/test_metrics_logger.py::test_csv_backend_log_step_metrics -v
pytest tests/test_metrics_logger.py::test_csv_backend_log_episode_metrics -v
pytest tests/test_metrics_logger.py::test_csv_backend_multiple_writes -v
```

**MetricsLogger Integration:**
```bash
# Test unified logging interface
pytest tests/test_metrics_logger.py -k "metrics_logger" -v

# Test specific features
pytest tests/test_metrics_logger.py -k "flush" -v        # Periodic flushing
pytest tests/test_metrics_logger.py -k "artifact" -v     # W&B artifact uploads
pytest tests/test_metrics_logger.py -k "consistency" -v  # Multi-backend consistency
```

**Verify:**
- TensorBoard event file creation and writing
- W&B graceful degradation when package missing
- CSV file format and multi-write correctness
- Periodic flush mechanism (every 1000 steps)
- W&B artifact upload intervals (every 1M steps)
- Backend failure handling (one backend fails, others continue)
- Standardized metric keys across all backends

### Plotting Tests

**File:** `test_plot_results.py` (20 tests, ~490 lines)

**Run all plotting tests:**
```bash
# Full test suite
pytest tests/test_plot_results.py -v

# Specific component tests
pytest tests/test_plot_results.py -k "load" -v       # Data loading
pytest tests/test_plot_results.py -k "smooth" -v     # Smoothing algorithms
pytest tests/test_plot_results.py -k "plot" -v       # Plot generation
```

**Test coverage:**

**Data Loading:**
```bash
# CSV loading and parsing
pytest tests/test_plot_results.py::test_load_csv_data -v
pytest tests/test_plot_results.py::test_load_csv_data_file_not_found -v
pytest tests/test_plot_results.py::test_load_csv_data_handles_none_values -v
```

**Smoothing:**
```bash
# Smoothing algorithms
pytest tests/test_plot_results.py::test_smooth_curve_moving_average -v
pytest tests/test_plot_results.py::test_smooth_curve_exponential -v
pytest tests/test_plot_results.py::test_smooth_curve_insufficient_data -v
pytest tests/test_plot_results.py::test_smooth_curve_invalid_method -v
```

**Plot Generation:**
```bash
# Individual plot types
pytest tests/test_plot_results.py::test_plot_episode_returns -v
pytest tests/test_plot_results.py::test_plot_training_loss -v
pytest tests/test_plot_results.py::test_plot_evaluation_scores -v
pytest tests/test_plot_results.py::test_plot_epsilon_schedule -v

# Multi-format output
pytest tests/test_plot_results.py::test_plot_multiple_formats -v
```

**Integration:**
```bash
# Full plotting pipeline
pytest tests/test_plot_results.py::test_plot_all_metrics -v
pytest tests/test_plot_results.py::test_plot_all_metrics_partial_data -v

# Edge cases
pytest tests/test_plot_results.py::test_plot_with_nan_values -v
pytest tests/test_plot_results.py::test_plot_empty_data_warning -v
```

**Metadata:**
```bash
# Plot metadata generation and saving
pytest tests/test_plot_results.py::test_plot_metadata_saved -v
pytest tests/test_plot_results.py::test_plot_no_metadata -v
pytest tests/test_plot_results.py::test_plot_deterministic_filenames -v
```

**Verify:**
- CSV data loading with NaN/None value handling
- Moving average and exponential smoothing correctness
- All plot types generate correctly (returns, loss, eval, epsilon)
- Multi-format output (PNG, PDF, SVG)
- Metadata JSON generation with commit hash and timestamp
- Directory auto-creation for output
- Deterministic filename generation
- Empty data and NaN value handling

### Offline/Sandbox Mode

**W&B Tests** run in offline mode by default (no network required):

```bash
# All W&B tests work offline
export WANDB_MODE=offline
pytest tests/test_metrics_logger.py -k "wandb" -v

# Or with W&B completely disabled
export WANDB_DISABLED=true
pytest tests/test_metrics_logger.py -v
```

**No W&B installation required:**
- Tests gracefully handle missing `wandb` package
- W&B backend tests verify graceful degradation
- All other tests pass without W&B

### Synthetic Log Testing

Tests use **synthetic/mock data** (no real training required):

```bash
# Tests create temporary CSV files with synthetic data
pytest tests/test_plot_results.py -v

# Tests use pytest fixtures for temporary directories
# All test files are cleaned up automatically
```

**Synthetic data includes:**
- Random episode returns with realistic distributions
- Training loss with decreasing trend
- Epsilon decay schedules
- Evaluation scores with increasing performance

### Common Test Patterns

**Test with custom data:**
```python
import tempfile
import csv
from pathlib import Path
from scripts.plot_results import load_csv_data, plot_episode_returns

# Create synthetic CSV
temp_dir = Path(tempfile.mkdtemp())
csv_path = temp_dir / 'episodes.csv'

with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['step', 'episode', 'return', 'length'])
    writer.writeheader()
    for i in range(100):
        writer.writerow({
            'step': (i+1) * 1000,
            'episode': i+1,
            'return': 10.0 + i * 0.1,  # Increasing returns
            'length': 500
        })

# Load and plot
data = load_csv_data(csv_path)
plot_episode_returns(
    data,
    output_path=temp_dir / 'plot',
    formats=['png']
)

# Verify plot exists
assert (temp_dir / 'plot.png').exists()
```

### Test Performance

**Typical runtimes:**
- Metrics logger tests: ~1.5 seconds (28 tests)
- Plotting tests: ~6.7 seconds (20 tests)
- Combined: ~8 seconds (48 tests)

**All tests use mocks/fixtures:**
- No real training runs
- No network requests (W&B offline mode)
- Temporary directories (auto-cleanup)
- Fast execution suitable for CI/CD

**Checkpoint Tests** (`test_checkpoint.py`):
```bash
# Run all checkpoint tests
pytest tests/test_checkpoint.py -v

# Test specific features
pytest tests/test_checkpoint.py -k "save" -v          # Save functionality
pytest tests/test_checkpoint.py -k "load" -v          # Load functionality
pytest tests/test_checkpoint.py -k "best" -v          # Best model tracking
pytest tests/test_checkpoint.py -k "atomic" -v        # Atomic writes
```

**Resume Tests** (`test_resume.py`):
```bash
# Run all resume tests
pytest tests/test_resume.py -v

# Test specific features
pytest tests/test_resume.py -k "config" -v            # Config validation
pytest tests/test_resume.py -k "rng" -v               # RNG state restoration
pytest tests/test_resume.py -k "epsilon" -v           # Epsilon scheduler restoration
pytest tests/test_resume.py -k "device" -v            # Device mapping
```

**Seeding Tests** (`test_seeding.py`):
```bash
# Run all seeding tests
pytest tests/test_seeding.py -v

# Test specific RNG sources
pytest tests/test_seeding.py -k "python" -v           # Python random
pytest tests/test_seeding.py -k "numpy" -v            # NumPy random
pytest tests/test_seeding.py -k "torch" -v            # PyTorch random
pytest tests/test_seeding.py -k "env" -v              # Environment seeding
```

**Determinism Tests** (`test_determinism.py`):
```bash
# Run all determinism configuration tests
pytest tests/test_determinism.py -v

# Test specific modes
pytest tests/test_determinism.py -k "basic" -v        # Basic deterministic mode
pytest tests/test_determinism.py -k "strict" -v       # Strict mode
pytest tests/test_determinism.py -k "status" -v       # Status checking
```

**Save/Resume Determinism Smoke Test** (`test_save_resume_determinism.py`):
```bash
# Run end-to-end determinism verification (5000 steps)
pytest tests/test_save_resume_determinism.py -v -s

# Expected output:
# DONE PERFECT DETERMINISM - All metrics match exactly
# Epsilon Matches: 100.0%
# Reward Matches: 100.0%
# Action Matches: 100.0%
# Checksum Match: DONE PASS
```

**Verify:**
- Checkpoint save/load correctness
- Atomic write safety
- Best model tracking
- Config validation on resume
- RNG state capture and restoration
- Epsilon scheduler state restoration
- Deterministic mode configuration
- End-to-end determinism verification

<br><br>

## Adding New Tests

### Test Structure

```python
def test_component_behavior():
    """Test description following numpy docstring style.

    Verifies:
    - Specific behavior 1
    - Specific behavior 2
    - Edge case handling
    """
    # Arrange: Setup test data
    ...

    # Act: Execute the code under test
    ...

    # Assert: Verify expected behavior
    assert result == expected
```

### Test Naming Convention

- `test_<component>_<behavior>` - e.g., `test_epsilon_scheduler_linear_decay`
- `test_<component>_<edge_case>` - e.g., `test_replay_buffer_wrap_around`
- `test_<component>_integration` - e.g., `test_training_step_integration`

### Fixtures

Use pytest fixtures for common setup:

```python
@pytest.fixture
def dummy_model():
    """Create a DQN model for testing."""
    return DQN(num_actions=6)

def test_with_model(dummy_model):
    # Use the fixture
    output = dummy_model(torch.randn(1, 4, 84, 84))
    assert output['q_values'].shape == (1, 6)
```

<br><br>

## Continuous Integration

Tests run automatically on:
- Every commit to pull requests
- Merges to main branch
- Nightly builds

**CI Configuration:** `.github/workflows/tests.yml` (if configured)

**Expected behavior:**
- All tests pass before merge
- Coverage remains > 80%
- No new deprecation warnings

<br><br>

## Troubleshooting

### Common Issues

**Import errors:**
```bash
# Ensure you're in the repository root
cd /path/to/thesis

# Verify PYTHONPATH includes repo root
export PYTHONPATH=.
```

**Missing dependencies:**
```bash
# Install test dependencies
pip install pytest pytest-cov pytest-xdist

# Install project dependencies
pip install -r requirements.txt
```

**Slow tests:**
```bash
# Skip slow tests
pytest tests/ -k "not slow"

# Run in parallel
pytest tests/ -n auto
```

**Test failures:**
```bash
# Run single test with verbose output
pytest tests/test_dqn_trainer.py::test_function_name -vvs

# Check for environment issues
pytest tests/ --collect-only  # Should list all tests
```

<br><br>

## Test Metrics

**Current Coverage:** ~85% (target: >80%)

**Test Counts by Module:**
- `src.models`: 20+ tests
- `src.replay`: 15+ tests
- `src.training`: 163+ tests
- `src.envs`: 20+ tests
- `src.training.checkpoint`: 16 tests
- `src.training.resume`: 15 tests
- `src.utils.seeding`: 17 tests
- `src.utils.determinism`: 20 tests
- Integration (save/resume): 1 test
- `src.training.metrics_logger`: 28 tests
- `scripts.plot_results`: 20 tests

**Total:** 335+ tests

**Average Test Runtime:**
- Unit tests: < 1s each
- Integration tests: 1-5s each
- Full suite: ~2-3 minutes (serial), ~30-60s (parallel)

<br><br>

## Related Documentation

- **Training Loop Tests:** See `docs/design/training_loop_runtime.md` Testing section
- **Smoke Test:** See `experiments/dqn_atari/scripts/README.md`
- **Coverage Reports:** Generated in `htmlcov/` after running `pytest --cov`

<br><br>

**Last Updated:** 2025-11-13
**Subtask:** After Subtask 6 completion (training loop infrastructure)
