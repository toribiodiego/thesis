"""
Tests for plot_results.py plotting script.

Verifies:
- CSV data loading
- Data smoothing
- Plot generation for all metric types
- Multi-format output (PNG, PDF, SVG)
- CLI argument parsing
"""

import pytest
import tempfile
import shutil
import csv
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch

# Import plotting functions
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.plot_results import (
    load_csv_data,
    smooth_curve,
    plot_episode_returns,
    plot_training_loss,
    plot_evaluation_scores,
    plot_epsilon_schedule,
    plot_all_metrics
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_episodes_csv(temp_dir):
    """Create sample episodes CSV file."""
    csv_path = temp_dir / 'episodes.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['episode', 'step', 'return', 'length'])
        writer.writeheader()

        for i in range(100):
            writer.writerow({
                'episode': i + 1,
                'step': (i + 1) * 1000,
                'return': 10.0 + np.random.randn() * 2.0,
                'length': 500 + int(np.random.randn() * 50)
            })

    return csv_path


@pytest.fixture
def sample_steps_csv(temp_dir):
    """Create sample training steps CSV file."""
    csv_path = temp_dir / 'training_steps.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'loss', 'epsilon', 'fps'])
        writer.writeheader()

        for i in range(100):
            step = (i + 1) * 1000
            # Decreasing epsilon
            epsilon = max(0.1, 1.0 - (i / 100.0) * 0.9)
            writer.writerow({
                'step': step,
                'loss': 0.5 + np.random.randn() * 0.1,
                'epsilon': epsilon,
                'fps': 120.0 + np.random.randn() * 10.0
            })

    return csv_path


@pytest.fixture
def sample_eval_csv(temp_dir):
    """Create sample evaluation CSV file."""
    csv_path = temp_dir / 'evaluations.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'step', 'mean_return', 'std_return', 'num_episodes'
        ])
        writer.writeheader()

        for i in range(10):
            writer.writerow({
                'step': (i + 1) * 250000,
                'mean_return': 15.0 + i * 2.0,
                'std_return': 3.0,
                'num_episodes': 10
            })

    return csv_path


# ============================================================================
# Data Loading Tests
# ============================================================================

def test_load_csv_data(sample_episodes_csv):
    """Test loading data from CSV file."""
    data = load_csv_data(sample_episodes_csv)

    assert 'episode' in data
    assert 'step' in data
    assert 'return' in data
    assert 'length' in data

    assert len(data['episode']) == 100
    assert isinstance(data['step'], np.ndarray)


def test_load_csv_data_file_not_found(temp_dir):
    """Test loading from non-existent file raises error."""
    with pytest.raises(FileNotFoundError):
        load_csv_data(temp_dir / 'nonexistent.csv')


def test_load_csv_data_handles_none_values(temp_dir):
    """Test that CSV loader handles None/empty values."""
    csv_path = temp_dir / 'test.csv'

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['step', 'value'])
        writer.writeheader()
        writer.writerow({'step': '1000', 'value': '1.5'})
        writer.writerow({'step': '2000', 'value': ''})  # Empty value
        writer.writerow({'step': '3000', 'value': 'None'})  # None string

    data = load_csv_data(csv_path)

    assert len(data['step']) == 3
    assert data['value'][0] == 1.5
    assert np.isnan(data['value'][1])
    assert np.isnan(data['value'][2])


# ============================================================================
# Smoothing Tests
# ============================================================================

def test_smooth_curve_moving_average():
    """Test moving average smoothing."""
    x = np.arange(100)
    y = np.random.randn(100)

    x_smooth, y_smooth = smooth_curve(x, y, window=10, method='moving_average')

    # Smoothed array should be shorter
    assert len(y_smooth) == len(y) - 10 + 1
    assert len(x_smooth) == len(y_smooth)


def test_smooth_curve_exponential():
    """Test exponential smoothing."""
    x = np.arange(100)
    y = np.random.randn(100)

    x_smooth, y_smooth = smooth_curve(x, y, window=10, method='exponential')

    # Exponential smoothing preserves length
    assert len(y_smooth) == len(y)
    assert len(x_smooth) == len(x)


def test_smooth_curve_insufficient_data():
    """Test smoothing with insufficient data."""
    x = np.arange(5)
    y = np.random.randn(5)

    x_smooth, y_smooth = smooth_curve(x, y, window=10)

    # Should return original data
    assert len(x_smooth) == len(x)
    assert len(y_smooth) == len(y)


def test_smooth_curve_invalid_method():
    """Test smoothing with invalid method raises error."""
    x = np.arange(100)
    y = np.random.randn(100)

    with pytest.raises(ValueError):
        smooth_curve(x, y, method='invalid_method')


# ============================================================================
# Plotting Tests
# ============================================================================

def test_plot_episode_returns(sample_episodes_csv, temp_dir):
    """Test plotting episode returns."""
    data = load_csv_data(sample_episodes_csv)

    output_path = temp_dir / 'episode_returns'

    plot_episode_returns(
        data,
        output_path,
        smoothing_window=10,
        formats=['png']
    )

    # Check PNG was created
    assert (temp_dir / 'episode_returns.png').exists()


def test_plot_training_loss(sample_steps_csv, temp_dir):
    """Test plotting training loss."""
    data = load_csv_data(sample_steps_csv)

    output_path = temp_dir / 'training_loss'

    plot_training_loss(
        data,
        output_path,
        smoothing_window=10,
        formats=['png']
    )

    # Check PNG was created
    assert (temp_dir / 'training_loss.png').exists()


def test_plot_evaluation_scores(sample_eval_csv, temp_dir):
    """Test plotting evaluation scores."""
    data = load_csv_data(sample_eval_csv)

    output_path = temp_dir / 'eval_scores'

    plot_evaluation_scores(
        data,
        output_path,
        formats=['png']
    )

    # Check PNG was created
    assert (temp_dir / 'eval_scores.png').exists()


def test_plot_epsilon_schedule(sample_steps_csv, temp_dir):
    """Test plotting epsilon schedule."""
    data = load_csv_data(sample_steps_csv)

    output_path = temp_dir / 'epsilon_schedule'

    plot_epsilon_schedule(
        data,
        output_path,
        formats=['png']
    )

    # Check PNG was created
    assert (temp_dir / 'epsilon_schedule.png').exists()


def test_plot_multiple_formats(sample_episodes_csv, temp_dir):
    """Test generating plots in multiple formats."""
    data = load_csv_data(sample_episodes_csv)

    output_path = temp_dir / 'episode_returns'

    plot_episode_returns(
        data,
        output_path,
        formats=['png', 'pdf', 'svg']
    )

    # Check all formats were created
    assert (temp_dir / 'episode_returns.png').exists()
    assert (temp_dir / 'episode_returns.pdf').exists()
    assert (temp_dir / 'episode_returns.svg').exists()


def test_plot_all_metrics(sample_episodes_csv, sample_steps_csv, sample_eval_csv, temp_dir):
    """Test generating all plots at once."""
    episodes_data = load_csv_data(sample_episodes_csv)
    steps_data = load_csv_data(sample_steps_csv)
    eval_data = load_csv_data(sample_eval_csv)

    output_dir = temp_dir / 'plots'

    plot_files, metadata_file = plot_all_metrics(
        episodes_data=episodes_data,
        steps_data=steps_data,
        eval_data=eval_data,
        output_dir=output_dir,
        game_name='pong',
        smoothing_window=10,
        formats=['png']
    )

    # Check all plots were created
    assert (output_dir / 'pong_episode_returns.png').exists()
    assert (output_dir / 'pong_training_loss.png').exists()
    assert (output_dir / 'pong_evaluation_scores.png').exists()
    assert (output_dir / 'pong_epsilon_schedule.png').exists()

    # Check metadata was saved
    assert metadata_file is not None
    assert metadata_file.exists()

    # Check plot files list
    assert len(plot_files) == 4


def test_plot_all_metrics_partial_data(sample_episodes_csv, temp_dir):
    """Test plotting with only partial data available."""
    episodes_data = load_csv_data(sample_episodes_csv)

    output_dir = temp_dir / 'plots'

    # Should not raise errors with missing data
    plot_files, metadata_file = plot_all_metrics(
        episodes_data=episodes_data,
        steps_data=None,
        eval_data=None,
        output_dir=output_dir,
        game_name='pong',
        formats=['png']
    )

    # Only episode returns plot should exist
    assert (output_dir / 'pong_episode_returns.png').exists()
    assert not (output_dir / 'pong_training_loss.png').exists()

    # Should still have metadata
    assert metadata_file is not None


def test_plot_with_nan_values(temp_dir):
    """Test plotting handles NaN values correctly."""
    # Create data with NaN values
    data = {
        'step': np.array([1000, 2000, 3000, 4000, 5000]),
        'return': np.array([10.0, np.nan, 12.0, np.nan, 14.0])
    }

    output_path = temp_dir / 'returns_with_nan'

    # Should not raise errors
    plot_episode_returns(
        data,
        output_path,
        smoothing_window=2,
        formats=['png']
    )

    assert (temp_dir / 'returns_with_nan.png').exists()


def test_plot_empty_data_warning(temp_dir, capsys):
    """Test plotting with empty data prints warning."""
    # Create empty data (all NaN)
    data = {
        'step': np.array([1000, 2000, 3000]),
        'return': np.array([np.nan, np.nan, np.nan])
    }

    output_path = temp_dir / 'empty_returns'

    plot_episode_returns(
        data,
        output_path,
        formats=['png']
    )

    # Should print warning
    captured = capsys.readouterr()
    assert 'Warning' in captured.out or 'Warning' in captured.err


# ============================================================================
# Integration Tests
# ============================================================================

def test_plot_directory_creation(temp_dir):
    """Test that output directory is created automatically."""
    data = {
        'step': np.arange(100) * 1000,
        'return': np.random.randn(100) + 10.0
    }

    output_dir = temp_dir / 'nested' / 'plots'
    assert not output_dir.exists()

    plot_files, metadata_file = plot_all_metrics(
        episodes_data=data,
        steps_data=None,
        eval_data=None,
        output_dir=output_dir,
        game_name='test',
        formats=['png']
    )

    # Directory should be created
    assert output_dir.exists()


def test_plot_deterministic_filenames(sample_episodes_csv, temp_dir):
    """Test that plot filenames are deterministic."""
    data = load_csv_data(sample_episodes_csv)

    output_dir = temp_dir / 'plots'

    plot_files, metadata_file = plot_all_metrics(
        episodes_data=data,
        steps_data=None,
        eval_data=None,
        output_dir=output_dir,
        game_name='breakout',
        formats=['png']
    )

    # Filename should match pattern: {game_name}_{metric_type}.{format}
    expected_file = output_dir / 'breakout_episode_returns.png'
    assert expected_file.exists()


def test_plot_metadata_saved(sample_episodes_csv, temp_dir):
    """Test that plot metadata is saved correctly."""
    data = load_csv_data(sample_episodes_csv)

    output_dir = temp_dir / 'plots'

    plot_files, metadata_file = plot_all_metrics(
        episodes_data=data,
        steps_data=None,
        eval_data=None,
        output_dir=output_dir,
        game_name='pong',
        smoothing_window=50,
        formats=['png'],
        save_metadata=True
    )

    # Metadata file should exist
    assert metadata_file is not None
    assert metadata_file.exists()

    # Load and verify metadata
    import json
    with open(metadata_file) as f:
        metadata = json.load(f)

    assert metadata['game_name'] == 'pong'
    assert metadata['smoothing_window'] == 50
    assert 'commit_hash' in metadata
    assert 'generated_at' in metadata
    assert metadata['formats'] == ['png']


def test_plot_no_metadata(sample_episodes_csv, temp_dir):
    """Test disabling metadata saving."""
    data = load_csv_data(sample_episodes_csv)

    output_dir = temp_dir / 'plots'

    plot_files, metadata_file = plot_all_metrics(
        episodes_data=data,
        steps_data=None,
        eval_data=None,
        output_dir=output_dir,
        game_name='pong',
        formats=['png'],
        save_metadata=False
    )

    # Metadata file should not exist
    assert metadata_file is None
