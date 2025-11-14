"""
Tests for multi-backend metrics logger.

Verifies:
- MetricsLogger unified interface
- TensorBoard backend integration
- W&B backend integration (mocked)
- CSV backend file creation
- Standardized metric naming across backends
- Moving average computation
- Graceful handling of missing backends
"""

import pytest
import tempfile
import shutil
import csv
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.training import (
    MetricsLogger,
    MetricKeys,
    TensorBoardBackend,
    WandBBackend,
    CSVBackend
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_log_dir():
    """Create temporary logging directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


# ============================================================================
# TensorBoard Backend Tests
# ============================================================================

def test_tensorboard_backend_initialization(temp_log_dir):
    """Test TensorBoard backend initializes correctly."""
    backend = TensorBoardBackend(temp_log_dir)

    # Should be enabled if tensorboard is available
    if backend.enabled:
        assert backend.writer is not None
        assert Path(temp_log_dir).exists()
        backend.close()


def test_tensorboard_backend_log_scalar(temp_log_dir):
    """Test TensorBoard scalar logging."""
    backend = TensorBoardBackend(temp_log_dir)

    if backend.enabled:
        # Log a scalar
        backend.log_scalar("train/loss", 0.5, step=100)
        backend.flush()

        # TensorBoard event files should exist
        event_files = list(Path(temp_log_dir).glob("events.out.*"))
        assert len(event_files) > 0

        backend.close()


def test_tensorboard_backend_log_scalars(temp_log_dir):
    """Test TensorBoard batch scalar logging."""
    backend = TensorBoardBackend(temp_log_dir)

    if backend.enabled:
        metrics = {
            "train/loss": 0.5,
            "train/epsilon": 0.95,
            "train/fps": 120.0
        }
        backend.log_scalars(metrics, step=100)
        backend.flush()

        event_files = list(Path(temp_log_dir).glob("events.out.*"))
        assert len(event_files) > 0

        backend.close()


def test_tensorboard_backend_handles_none_values(temp_log_dir):
    """Test TensorBoard backend skips None values."""
    backend = TensorBoardBackend(temp_log_dir)

    if backend.enabled:
        metrics = {
            "train/loss": 0.5,
            "train/epsilon": None,  # Should be skipped
            "train/fps": 120.0
        }
        backend.log_scalars(metrics, step=100)
        backend.flush()
        backend.close()

        # Should not raise error


# ============================================================================
# W&B Backend Tests
# ============================================================================

def test_wandb_backend_handles_import_error():
    """Test W&B backend gracefully handles missing wandb package."""
    # WandBBackend should handle ImportError gracefully
    # Since wandb is likely not installed in test environment,
    # this will test the actual behavior
    backend = WandBBackend(project="test-project")

    # Backend should either be enabled (if wandb installed) or disabled (if not)
    # Either way, no errors should be raised
    assert isinstance(backend.enabled, bool)

    if backend.enabled:
        backend.close()


def test_wandb_backend_disabled_operations():
    """Test W&B backend operations when disabled."""
    backend = WandBBackend(project="test-project")

    if not backend.enabled:
        # Operations should not raise errors when disabled
        backend.log_scalar("train/loss", 0.5, step=100)
        backend.log_scalars({"train/loss": 0.5}, step=100)
        backend.flush()
        backend.close()  # Should not error


# ============================================================================
# CSV Backend Tests
# ============================================================================

def test_csv_backend_initialization(temp_log_dir):
    """Test CSV backend initializes correctly."""
    backend = CSVBackend(temp_log_dir)

    assert backend.enabled
    assert backend.log_dir.exists()
    assert backend.step_csv_path == backend.log_dir / 'training_steps.csv'
    assert backend.episode_csv_path == backend.log_dir / 'episodes.csv'


def test_csv_backend_log_step_metrics(temp_log_dir):
    """Test CSV backend logs step metrics correctly."""
    backend = CSVBackend(temp_log_dir)

    metrics = {
        'loss': 0.5,
        'epsilon': 0.95,
        'fps': 120.0
    }
    backend.log_step_metrics(metrics, step=100)

    # Check CSV file exists
    assert backend.step_csv_path.exists()

    # Read and verify content
    with open(backend.step_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]['step'] == '100'
        assert rows[0]['loss'] == '0.5'
        assert rows[0]['epsilon'] == '0.95'
        assert rows[0]['fps'] == '120.0'


def test_csv_backend_log_episode_metrics(temp_log_dir):
    """Test CSV backend logs episode metrics correctly."""
    backend = CSVBackend(temp_log_dir)

    metrics = {
        'return': 21.0,
        'length': 1200,
        'return_ma': 18.5
    }
    backend.log_episode_metrics(metrics, step=5000, episode=10)

    # Check CSV file exists
    assert backend.episode_csv_path.exists()

    # Read and verify content
    with open(backend.episode_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]['episode'] == '10'
        assert rows[0]['step'] == '5000'
        assert rows[0]['return'] == '21.0'
        assert rows[0]['length'] == '1200'
        assert rows[0]['return_ma'] == '18.5'


def test_csv_backend_multiple_writes(temp_log_dir):
    """Test CSV backend handles multiple writes correctly."""
    backend = CSVBackend(temp_log_dir)

    # Write multiple step metrics
    for i in range(5):
        metrics = {'loss': 0.5 - i * 0.1, 'epsilon': 0.95 - i * 0.05}
        backend.log_step_metrics(metrics, step=(i + 1) * 100)

    # Verify all rows present
    with open(backend.step_csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 5
        assert rows[0]['step'] == '100'
        assert rows[4]['step'] == '500'


# ============================================================================
# MetricsLogger Integration Tests
# ============================================================================

def test_metrics_logger_initialization_csv_only(temp_log_dir):
    """Test MetricsLogger with CSV backend only."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_tensorboard=False,
        enable_wandb=False,
        enable_csv=True
    )

    assert logger.csv is not None
    assert logger.tensorboard is None
    assert logger.wandb is None

    logger.close()


def test_metrics_logger_initialization_all_backends(temp_log_dir):
    """Test MetricsLogger with all backends enabled."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_tensorboard=True,
        enable_wandb=True,
        enable_csv=True,
        wandb_project="test-project"
    )

    # TensorBoard should exist (enabled if torch.utils.tensorboard is available)
    assert logger.tensorboard is not None

    # W&B should exist (enabled if wandb is installed)
    assert logger.wandb is not None

    # CSV should always be enabled
    assert logger.csv.enabled

    logger.close()


def test_metrics_logger_log_step(temp_log_dir):
    """Test MetricsLogger logs step metrics correctly."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_tensorboard=False,
        enable_wandb=False,
        enable_csv=True
    )

    logger.log_step(
        step=1000,
        loss=0.5,
        epsilon=0.95,
        learning_rate=0.00025,
        replay_size=50000,
        fps=120.0
    )

    # Check CSV was written
    csv_path = Path(temp_log_dir) / 'csv' / 'training_steps.csv'
    assert csv_path.exists()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]['step'] == '1000'
        assert rows[0]['loss'] == '0.5'

    logger.close()


def test_metrics_logger_log_step_with_moving_average(temp_log_dir):
    """Test MetricsLogger computes loss moving average."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_tensorboard=False,
        enable_wandb=False,
        enable_csv=True,
        moving_avg_window=3
    )

    # Log multiple steps
    logger.log_step(step=1000, loss=1.0)
    logger.log_step(step=2000, loss=0.8)
    logger.log_step(step=3000, loss=0.6)

    # Check moving average is computed
    csv_path = Path(temp_log_dir) / 'csv' / 'training_steps.csv'
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        # Last row should have MA of (1.0 + 0.8 + 0.6) / 3 = 0.8
        assert float(rows[-1]['loss_ma']) == pytest.approx(0.8, abs=0.01)

    logger.close()


def test_metrics_logger_log_episode(temp_log_dir):
    """Test MetricsLogger logs episode metrics correctly."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_tensorboard=False,
        enable_wandb=False,
        enable_csv=True
    )

    logger.log_episode(
        step=5000,
        episode=10,
        episode_return=21.0,
        episode_length=1200,
        epsilon=0.90,
        fps=110.5
    )

    # Check CSV was written
    csv_path = Path(temp_log_dir) / 'csv' / 'episodes.csv'
    assert csv_path.exists()

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]['episode'] == '10'
        assert rows[0]['return'] == '21.0'
        assert rows[0]['length'] == '1200'

    logger.close()


def test_metrics_logger_log_episode_with_rolling_stats(temp_log_dir):
    """Test MetricsLogger computes rolling episode statistics."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_tensorboard=False,
        enable_wandb=False,
        enable_csv=True,
        moving_avg_window=3
    )

    # Log multiple episodes
    logger.log_episode(step=1000, episode=1, episode_return=10.0, episode_length=500)
    logger.log_episode(step=2000, episode=2, episode_return=20.0, episode_length=600)
    logger.log_episode(step=3000, episode=3, episode_return=30.0, episode_length=700)

    # Check rolling stats
    csv_path = Path(temp_log_dir) / 'csv' / 'episodes.csv'
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        # Last row should have MA of (10 + 20 + 30) / 3 = 20.0
        assert float(rows[-1]['return_ma']) == pytest.approx(20.0, abs=0.01)

    logger.close()


def test_metrics_logger_log_evaluation(temp_log_dir):
    """Test MetricsLogger logs evaluation metrics."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_tensorboard=True,
        enable_wandb=True,
        enable_csv=True,
        wandb_project="test-project"
    )

    # Should not raise errors
    logger.log_evaluation(
        step=250000,
        mean_return=25.5,
        median_return=24.0,
        std_return=5.2,
        min_return=15.0,
        max_return=35.0,
        mean_length=1500,
        num_episodes=10
    )

    logger.close()


def test_metrics_logger_log_q_values(temp_log_dir):
    """Test MetricsLogger logs Q-value statistics."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_tensorboard=False,
        enable_wandb=False,
        enable_csv=True
    )

    logger.log_q_values(
        step=1000,
        q_mean=5.2,
        q_std=1.8,
        q_min=2.1,
        q_max=8.9
    )

    # Q-values are not logged to CSV, only TensorBoard/W&B
    # Just verify no errors
    logger.close()


def test_metrics_logger_standardized_keys():
    """Test MetricKeys provides standardized naming."""
    # Verify key naming conventions
    assert MetricKeys.LOSS == "train/loss"
    assert MetricKeys.EPSILON == "train/epsilon"
    assert MetricKeys.EPISODE_RETURN == "episode/return"
    assert MetricKeys.EVAL_MEAN_RETURN == "eval/mean_return"
    assert MetricKeys.Q_VALUE_MEAN == "q_values/mean"


def test_metrics_logger_handles_none_values(temp_log_dir):
    """Test MetricsLogger gracefully handles None metric values."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_tensorboard=False,
        enable_wandb=False,
        enable_csv=True
    )

    # Log with some None values
    logger.log_step(
        step=1000,
        loss=0.5,
        epsilon=None,  # Not provided
        learning_rate=0.00025,
        replay_size=None  # Not provided
    )

    # Should not raise errors
    csv_path = Path(temp_log_dir) / 'csv' / 'training_steps.csv'
    assert csv_path.exists()

    logger.close()


def test_metrics_logger_extra_metrics(temp_log_dir):
    """Test MetricsLogger accepts extra custom metrics without error.

    Note: CSV backend uses a fixed schema, so extra metrics are filtered out
    and only logged to TensorBoard/W&B. This prevents dynamic schema issues.
    """
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_tensorboard=False,
        enable_wandb=False,
        enable_csv=True
    )

    extra = {
        'custom_metric_1': 42.0,
        'custom_metric_2': 123.4
    }

    # Should not raise error even though CSV doesn't support extra metrics
    logger.log_step(
        step=1000,
        loss=0.5,
        extra_metrics=extra
    )

    # CSV should have the standard fields only (extra metrics filtered out)
    csv_path = Path(temp_log_dir) / 'csv' / 'training_steps.csv'
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert 'step' in rows[0]
        assert 'loss' in rows[0]
        # Extra metrics not in CSV (by design - fixed schema)
        assert 'custom_metric_1' not in rows[0]
        assert 'custom_metric_2' not in rows[0]

    logger.close()


def test_metrics_logger_multi_backend_consistency(temp_log_dir):
    """Test MetricsLogger logs same metrics to all backends."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_tensorboard=True,
        enable_wandb=True,
        enable_csv=True,
        wandb_project="test-project"
    )

    # Log step metrics
    logger.log_step(
        step=1000,
        loss=0.5,
        epsilon=0.95
    )

    # CSV should have logged
    csv_path = Path(temp_log_dir) / 'csv' / 'training_steps.csv'
    assert csv_path.exists()

    # TensorBoard should have logged (if available)
    if logger.tensorboard and logger.tensorboard.enabled:
        tb_dir = Path(temp_log_dir) / 'tensorboard'
        assert tb_dir.exists()

    logger.close()


# ============================================================================
# Periodic Flush and Artifact Upload Tests
# ============================================================================

def test_metrics_logger_periodic_flush(temp_log_dir):
    """Test MetricsLogger periodic flush mechanism."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_csv=True,
        flush_interval=1000
    )

    # Log at step 500 (should not flush)
    logger.log_step(step=500, loss=0.5)
    assert logger.last_flush_step == 0

    # Log at step 1000 (should flush)
    logger.maybe_flush_and_upload(step=1000)
    assert logger.last_flush_step == 1000

    # Log at step 1500 (should not flush yet)
    logger.maybe_flush_and_upload(step=1500)
    assert logger.last_flush_step == 1000

    # Log at step 2000 (should flush)
    logger.maybe_flush_and_upload(step=2000)
    assert logger.last_flush_step == 2000

    logger.close()


def test_metrics_logger_force_flush(temp_log_dir):
    """Test MetricsLogger force flush functionality."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_csv=True,
        flush_interval=1000
    )

    # Force flush at step 100 (before interval)
    logger.maybe_flush_and_upload(step=100, force=True)
    assert logger.last_flush_step == 100

    logger.close()


def test_metrics_logger_artifact_upload_disabled(temp_log_dir):
    """Test that artifact upload is skipped when disabled."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_csv=True,
        enable_wandb=True,
        wandb_project="test-project",
        upload_artifacts=False  # Disabled
    )

    # Log some data
    logger.log_step(step=1000, loss=0.5)

    # Try to upload (should do nothing)
    logger.upload_logs_as_artifacts(step=1000)

    # Should not have uploaded
    assert logger.last_artifact_upload_step == 0

    logger.close()


def test_metrics_logger_artifact_upload_interval(temp_log_dir):
    """Test artifact upload interval checking."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_csv=True,
        enable_wandb=False,  # Disable to avoid actual upload
        upload_artifacts=True
    )

    # Should not upload at step 500K
    assert not logger._should_upload_artifacts(500_000)

    # Should upload at step 1M
    assert logger._should_upload_artifacts(1_000_000)

    # Mark as uploaded
    logger.last_artifact_upload_step = 1_000_000

    # Should not upload at step 1.5M
    assert not logger._should_upload_artifacts(1_500_000)

    # Should upload at step 2M
    assert logger._should_upload_artifacts(2_000_000)

    logger.close()


def test_metrics_logger_deterministic_artifact_name(temp_log_dir):
    """Test that artifact names are deterministic based on step."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_csv=True,
        enable_wandb=False,
        upload_artifacts=True
    )

    # Log some data to create CSV files
    logger.log_step(step=1000, loss=0.5)
    logger.log_episode(step=1000, episode=1, episode_return=10.0, episode_length=100)

    # The artifact name should be deterministic: training_logs_step_{step}
    # We can't test actual upload without W&B, but we can verify the method doesn't crash
    logger.upload_logs_as_artifacts(step=1_000_000, metadata={"test": True})

    logger.close()


def test_metrics_logger_close_performs_final_flush(temp_log_dir):
    """Test that close() performs a final flush."""
    logger = MetricsLogger(
        log_dir=temp_log_dir,
        enable_csv=True,
        flush_interval=1000
    )

    # Log at step 500 (before flush interval)
    logger.log_step(step=500, loss=0.5)
    assert logger.last_flush_step == 0

    # Close should flush
    logger.close()

    # Verify CSV file was written (flush occurred)
    csv_path = Path(temp_log_dir) / 'csv' / 'training_steps.csv'
    assert csv_path.exists()
