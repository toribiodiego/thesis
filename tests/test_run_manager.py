"""Tests for run directory management and metadata persistence."""

import json

import pytest
import yaml

from src.config.run_manager import (
    create_run_dir,
    create_run_subdirs,
    get_git_info,
    print_run_info,
    save_config_snapshot,
    save_metadata,
    setup_run_directory,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "experiment": {"name": "test_experiment", "notes": "Test run"},
        "environment": {"env_id": "PongNoFrameskip-v4"},
        "training": {"total_frames": 1000000, "optimizer": {"lr": 0.00025}},
        "seed": {"value": 42},
        "logging": {"base_dir": "test_runs"},
    }


# =============================================================================
# Create Run Dir Tests
# =============================================================================


def test_create_run_dir_with_seed(tmp_path):
    """Test creating run directory with seed."""
    base_dir = tmp_path / "runs"
    run_dir = create_run_dir(
        str(base_dir), "pong", seed=42, timestamp="20250113_120000"
    )

    assert run_dir.exists()
    assert run_dir.name == "pong_42_20250113_120000"
    assert run_dir.parent == base_dir


def test_create_run_dir_without_seed(tmp_path):
    """Test creating run directory without seed."""
    base_dir = tmp_path / "runs"
    run_dir = create_run_dir(
        str(base_dir), "pong", seed=None, timestamp="20250113_120000"
    )

    assert run_dir.exists()
    assert "noseed" in run_dir.name


def test_create_run_dir_auto_timestamp(tmp_path):
    """Test creating run directory with auto-generated timestamp."""
    base_dir = tmp_path / "runs"
    run_dir = create_run_dir(str(base_dir), "pong", seed=42)

    assert run_dir.exists()
    assert run_dir.name.startswith("pong_42_")


def test_create_run_dir_creates_parents(tmp_path):
    """Test that run directory creates parent directories."""
    base_dir = tmp_path / "deep" / "nested" / "runs"
    run_dir = create_run_dir(str(base_dir), "pong", seed=42)

    assert run_dir.exists()
    assert base_dir.exists()


# =============================================================================
# Create Subdirs Tests
# =============================================================================


def test_create_run_subdirs(tmp_path):
    """Test creating standard subdirectories."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    subdirs = create_run_subdirs(run_dir)

    assert "logs" in subdirs
    assert "checkpoints" in subdirs
    assert "artifacts" in subdirs
    assert "eval" in subdirs

    assert subdirs["logs"].exists()
    assert subdirs["checkpoints"].exists()
    assert subdirs["artifacts"].exists()
    assert subdirs["eval"].exists()


def test_create_run_subdirs_paths(tmp_path):
    """Test subdirectory paths are correct."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    subdirs = create_run_subdirs(run_dir)

    assert subdirs["logs"] == run_dir / "logs"
    assert subdirs["checkpoints"] == run_dir / "checkpoints"
    assert subdirs["artifacts"] == run_dir / "artifacts"
    assert subdirs["eval"] == run_dir / "eval"


# =============================================================================
# Save Config Snapshot Tests
# =============================================================================


def test_save_config_snapshot_yaml(tmp_path, sample_config):
    """Test saving config snapshot as YAML."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    config_path = save_config_snapshot(sample_config, run_dir, format="yaml")

    assert config_path.exists()
    assert config_path.name == "config.yaml"

    # Verify content
    with open(config_path) as f:
        loaded = yaml.safe_load(f)

    assert loaded["experiment"]["name"] == "test_experiment"
    assert loaded["seed"]["value"] == 42


def test_save_config_snapshot_json(tmp_path, sample_config):
    """Test saving config snapshot as JSON."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    config_path = save_config_snapshot(sample_config, run_dir, format="json")

    assert config_path.exists()
    assert config_path.name == "config.json"

    # Verify content
    with open(config_path) as f:
        loaded = json.load(f)

    assert loaded["experiment"]["name"] == "test_experiment"
    assert loaded["seed"]["value"] == 42


def test_save_config_snapshot_invalid_format(tmp_path, sample_config):
    """Test error on invalid format."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    with pytest.raises(ValueError, match="Unsupported format"):
        save_config_snapshot(sample_config, run_dir, format="xml")


# =============================================================================
# Get Git Info Tests
# =============================================================================


def test_get_git_info():
    """Test getting git information."""
    git_info = get_git_info()

    assert "commit_hash" in git_info
    assert "commit_hash_full" in git_info
    assert "branch" in git_info
    assert "dirty" in git_info
    assert "available" in git_info

    # Should have git info in a git repo
    assert git_info["available"] is True
    assert git_info["commit_hash"] != "unknown"


def test_get_git_info_has_correct_types():
    """Test git info has correct value types."""
    git_info = get_git_info()

    assert isinstance(git_info["commit_hash"], str)
    assert isinstance(git_info["commit_hash_full"], str)
    assert isinstance(git_info["branch"], str)
    assert isinstance(git_info["dirty"], bool)
    assert isinstance(git_info["available"], bool)


# =============================================================================
# Save Metadata Tests
# =============================================================================


def test_save_metadata(tmp_path, sample_config):
    """Test saving metadata."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    meta_path = save_metadata(sample_config, run_dir)

    assert meta_path.exists()
    assert meta_path.name == "meta.json"

    # Verify content
    with open(meta_path) as f:
        metadata = json.load(f)

    assert "created_at" in metadata
    assert "python_version" in metadata
    assert "git" in metadata
    assert "seed" in metadata
    assert "experiment" in metadata
    assert "environment" in metadata
    assert "training" in metadata

    assert metadata["seed"] == 42
    assert metadata["experiment"]["name"] == "test_experiment"
    assert metadata["environment"]["env_id"] == "PongNoFrameskip-v4"


def test_save_metadata_with_extra(tmp_path, sample_config):
    """Test saving metadata with extra fields."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    extra = {"device": "cuda", "num_gpus": 1}
    meta_path = save_metadata(sample_config, run_dir, extra=extra)

    with open(meta_path) as f:
        metadata = json.load(f)

    assert "extra" in metadata
    assert metadata["extra"]["device"] == "cuda"
    assert metadata["extra"]["num_gpus"] == 1


def test_save_metadata_includes_git_info(tmp_path, sample_config):
    """Test metadata includes git information."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    meta_path = save_metadata(sample_config, run_dir)

    with open(meta_path) as f:
        metadata = json.load(f)

    assert "git" in metadata
    assert "commit_hash" in metadata["git"]
    assert "branch" in metadata["git"]
    assert "dirty" in metadata["git"]


def test_save_metadata_includes_cli_args(tmp_path, sample_config):
    """Test metadata includes CLI arguments if present."""
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    # Add CLI args to config
    sample_config["cli"] = {
        "args": {"config_file": "pong.yaml", "seed": 42, "resume": None}
    }

    meta_path = save_metadata(sample_config, run_dir)

    with open(meta_path) as f:
        metadata = json.load(f)

    assert "cli" in metadata
    assert metadata["cli"]["args"]["seed"] == 42


# =============================================================================
# Setup Run Directory Tests
# =============================================================================


def test_setup_run_directory(tmp_path, sample_config):
    """Test complete run directory setup."""
    # Update base_dir to use tmp_path
    sample_config["logging"]["base_dir"] = str(tmp_path / "runs")

    paths = setup_run_directory(sample_config, timestamp="20250113_120000")

    # Check all paths exist
    assert paths["run_dir"].exists()
    assert paths["logs"].exists()
    assert paths["checkpoints"].exists()
    assert paths["artifacts"].exists()
    assert paths["eval"].exists()
    assert paths["config_file"].exists()
    assert paths["meta_file"].exists()


def test_setup_run_directory_creates_correct_structure(tmp_path, sample_config):
    """Test run directory has correct structure."""
    sample_config["logging"]["base_dir"] = str(tmp_path / "runs")

    paths = setup_run_directory(sample_config, timestamp="20250113_120000")

    # Check directory structure
    assert (paths["run_dir"] / "logs").exists()
    assert (paths["run_dir"] / "checkpoints").exists()
    assert (paths["run_dir"] / "artifacts").exists()
    assert (paths["run_dir"] / "eval").exists()
    assert (paths["run_dir"] / "config.yaml").exists()
    assert (paths["run_dir"] / "meta.json").exists()


def test_setup_run_directory_saves_config(tmp_path, sample_config):
    """Test setup saves configuration correctly."""
    sample_config["logging"]["base_dir"] = str(tmp_path / "runs")

    paths = setup_run_directory(sample_config, timestamp="20250113_120000")

    # Load and verify config
    with open(paths["config_file"]) as f:
        loaded_config = yaml.safe_load(f)

    assert loaded_config["experiment"]["name"] == "test_experiment"
    assert loaded_config["seed"]["value"] == 42


def test_setup_run_directory_saves_metadata(tmp_path, sample_config):
    """Test setup saves metadata correctly."""
    sample_config["logging"]["base_dir"] = str(tmp_path / "runs")

    paths = setup_run_directory(sample_config, timestamp="20250113_120000")

    # Load and verify metadata
    with open(paths["meta_file"]) as f:
        metadata = json.load(f)

    assert metadata["seed"] == 42
    assert metadata["experiment"]["name"] == "test_experiment"
    assert "git" in metadata


def test_setup_run_directory_with_extra_metadata(tmp_path, sample_config):
    """Test setup with extra metadata."""
    sample_config["logging"]["base_dir"] = str(tmp_path / "runs")

    extra = {"custom_field": "custom_value"}
    paths = setup_run_directory(sample_config, extra_metadata=extra)

    with open(paths["meta_file"]) as f:
        metadata = json.load(f)

    assert "extra" in metadata
    assert metadata["extra"]["custom_field"] == "custom_value"


def test_setup_run_directory_naming(tmp_path, sample_config):
    """Test run directory naming convention."""
    sample_config["logging"]["base_dir"] = str(tmp_path / "runs")
    sample_config["experiment"]["name"] = "pong"
    sample_config["seed"]["value"] = 42

    paths = setup_run_directory(sample_config, timestamp="20250113_120000")

    assert paths["run_dir"].name == "pong_42_20250113_120000"


# =============================================================================
# Print Run Info Tests
# =============================================================================


def test_print_run_info(capsys, tmp_path, sample_config):
    """Test printing run info."""
    sample_config["logging"]["base_dir"] = str(tmp_path / "runs")

    paths = setup_run_directory(sample_config, timestamp="20250113_120000")
    print_run_info(paths)

    captured = capsys.readouterr()

    assert "Run Directory Created" in captured.out
    assert "config.yaml" in captured.out
    assert "meta.json" in captured.out
    assert "logs/" in captured.out
    assert "checkpoints/" in captured.out


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_workflow(tmp_path):
    """Test complete workflow from config to run directory."""
    config = {
        "experiment": {"name": "integration_test", "notes": "Full workflow test"},
        "environment": {"env_id": "PongNoFrameskip-v4"},
        "training": {"total_frames": 1000000, "optimizer": {"lr": 0.00025}},
        "seed": {"value": 999},
        "logging": {"base_dir": str(tmp_path / "runs")},
    }

    # Setup run directory
    paths = setup_run_directory(config, timestamp="20250113_120000")

    # Verify everything exists and is correct
    assert paths["run_dir"].name == "integration_test_999_20250113_120000"

    # Check config saved
    with open(paths["config_file"]) as f:
        saved_config = yaml.safe_load(f)
    assert saved_config["seed"]["value"] == 999

    # Check metadata saved
    with open(paths["meta_file"]) as f:
        metadata = json.load(f)
    assert metadata["seed"] == 999
    assert metadata["git"]["available"] is True

    # Check subdirectories
    assert (paths["run_dir"] / "logs").is_dir()
    assert (paths["run_dir"] / "checkpoints").is_dir()
    assert (paths["run_dir"] / "artifacts").is_dir()
    assert (paths["run_dir"] / "eval").is_dir()
