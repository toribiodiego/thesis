"""Tests for CLI argument parsing and setup."""

from argparse import Namespace
from pathlib import Path

import pytest
import yaml

from src.config.cli import (
    create_parser,
    load_config_from_args,
    parse_args,
    print_startup_banner,
    setup_from_args,
    validate_config,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary valid config file."""
    config = {
        "environment": {"env_id": "PongNoFrameskip-v4"},
        "experiment": {"name": "test_experiment"},
        "training": {"total_frames": 1000000, "optimizer": {"lr": 0.00025}},
        "network": {"architecture": "dqn", "device": "cpu"},
        "replay": {"capacity": 100000},
        "seed": {"value": None},
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)

    return str(config_file)


@pytest.fixture
def base_config_file(tmp_path):
    """Create a base config file."""
    base_config = {
        "training": {
            "total_frames": 10000000,
            "optimizer": {"lr": 0.00025, "type": "rmsprop"},
        },
        "network": {"architecture": "dqn", "device": "cpu"},
        "replay": {"capacity": 1000000},
    }

    base_file = tmp_path / "base.yaml"
    with open(base_file, "w") as f:
        yaml.dump(base_config, f)

    return base_file


@pytest.fixture
def game_config_file(tmp_path, base_config_file):
    """Create a game config that inherits from base."""
    game_config = {
        "base_config": str(base_config_file),
        "experiment": {"name": "pong"},
        "environment": {"env_id": "PongNoFrameskip-v4"},
        "training": {"total_frames": 1000000},
        "seed": {"value": None},
    }

    game_file = tmp_path / "pong.yaml"
    with open(game_file, "w") as f:
        yaml.dump(game_config, f)

    return str(game_file)


# =============================================================================
# Parser Tests
# =============================================================================


def test_create_parser():
    """Test parser creation."""
    parser = create_parser()

    assert parser is not None
    assert "Train DQN" in parser.description


def test_parse_args_required_config():
    """Test that --cfg is required."""
    with pytest.raises(SystemExit):
        parse_args([])


def test_parse_args_basic(temp_config_file):
    """Test basic argument parsing."""
    args = parse_args(["--cfg", temp_config_file])

    assert args.config == temp_config_file
    assert args.seed is None
    assert args.resume is None
    assert args.overrides == []


def test_parse_args_with_seed(temp_config_file):
    """Test parsing with --seed."""
    args = parse_args(["--cfg", temp_config_file, "--seed", "42"])

    assert args.config == temp_config_file
    assert args.seed == 42


def test_parse_args_with_resume(temp_config_file):
    """Test parsing with --resume."""
    args = parse_args(["--cfg", temp_config_file, "--resume", "/path/to/checkpoint.pt"])

    assert args.config == temp_config_file
    assert args.resume == "/path/to/checkpoint.pt"


def test_parse_args_with_set_single(temp_config_file):
    """Test parsing with single --set override."""
    args = parse_args(["--cfg", temp_config_file, "--set", "training.lr=0.001"])

    assert args.overrides == ["training.lr=0.001"]


def test_parse_args_with_set_multiple(temp_config_file):
    """Test parsing with multiple --set overrides."""
    args = parse_args(
        [
            "--cfg",
            temp_config_file,
            "--set",
            "training.lr=0.001",
            "--set",
            "training.gamma=0.95",
        ]
    )

    assert args.overrides == ["training.lr=0.001", "training.gamma=0.95"]


def test_parse_args_with_device(temp_config_file):
    """Test parsing with --device."""
    args = parse_args(["--cfg", temp_config_file, "--device", "cuda"])

    assert args.device == "cuda"


def test_parse_args_dry_run(temp_config_file):
    """Test parsing with --dry-run."""
    args = parse_args(["--cfg", temp_config_file, "--dry-run"])

    assert args.dry_run is True


def test_parse_args_print_config(temp_config_file):
    """Test parsing with --print-config."""
    args = parse_args(["--cfg", temp_config_file, "--print-config"])

    assert args.print_config is True


def test_parse_args_verbose(temp_config_file):
    """Test parsing with --verbose."""
    args = parse_args(["--cfg", temp_config_file, "--verbose"])

    assert args.verbose is True


def test_parse_args_quiet(temp_config_file):
    """Test parsing with --quiet."""
    args = parse_args(["--cfg", temp_config_file, "--quiet"])

    assert args.quiet is True


def test_parse_args_all_flags(temp_config_file):
    """Test parsing with all flags combined."""
    args = parse_args(
        [
            "--cfg",
            temp_config_file,
            "--seed",
            "123",
            "--resume",
            "/path/to/checkpoint.pt",
            "--set",
            "training.lr=0.001",
            "--set",
            "training.gamma=0.95",
            "--device",
            "cuda",
            "--verbose",
        ]
    )

    assert args.config == temp_config_file
    assert args.seed == 123
    assert args.resume == "/path/to/checkpoint.pt"
    assert args.overrides == ["training.lr=0.001", "training.gamma=0.95"]
    assert args.device == "cuda"
    assert args.verbose is True


# =============================================================================
# Load Config From Args Tests
# =============================================================================


def test_load_config_from_args_basic(temp_config_file):
    """Test loading config from args."""
    args = Namespace(
        config=temp_config_file,
        seed=None,
        device=None,
        overrides=[],
        dry_run=False,
        print_config=False,
        quiet=True,
        verbose=False,
    )

    config = load_config_from_args(args, print_resolved=False)

    assert config["environment"]["env_id"] == "PongNoFrameskip-v4"
    assert config["training"]["total_frames"] == 1000000


def test_load_config_from_args_with_seed(temp_config_file):
    """Test loading config with seed override."""
    args = Namespace(
        config=temp_config_file,
        seed=42,
        device=None,
        overrides=[],
        dry_run=False,
        print_config=False,
        quiet=True,
        verbose=False,
    )

    config = load_config_from_args(args, print_resolved=False)

    assert config["seed"]["value"] == 42


def test_load_config_from_args_with_device(temp_config_file):
    """Test loading config with device override."""
    args = Namespace(
        config=temp_config_file,
        seed=None,
        device="cuda",
        overrides=[],
        dry_run=False,
        print_config=False,
        quiet=True,
        verbose=False,
    )

    config = load_config_from_args(args, print_resolved=False)

    assert config["network"]["device"] == "cuda"


def test_load_config_from_args_with_overrides(temp_config_file):
    """Test loading config with --set overrides."""
    args = Namespace(
        config=temp_config_file,
        seed=None,
        device=None,
        overrides=["training.optimizer.lr=0.001", "training.total_frames=5000000"],
        dry_run=False,
        print_config=False,
        quiet=True,
        verbose=False,
    )

    config = load_config_from_args(args, print_resolved=False)

    assert config["training"]["optimizer"]["lr"] == 0.001
    assert config["training"]["total_frames"] == 5000000


def test_load_config_from_args_file_not_found():
    """Test error when config file doesn't exist."""
    args = Namespace(
        config="/nonexistent/config.yaml",
        seed=None,
        device=None,
        overrides=[],
        dry_run=False,
        print_config=False,
        quiet=True,
        verbose=False,
    )

    with pytest.raises(FileNotFoundError):
        load_config_from_args(args, print_resolved=False)


def test_load_config_from_args_with_base(game_config_file):
    """Test loading config that inherits from base."""
    args = Namespace(
        config=game_config_file,
        seed=42,
        device=None,
        overrides=[],
        dry_run=False,
        print_config=False,
        quiet=True,
        verbose=False,
    )

    config = load_config_from_args(args, print_resolved=False)

    # Should have merged base + game
    assert config["environment"]["env_id"] == "PongNoFrameskip-v4"
    assert config["training"]["total_frames"] == 1000000  # Overridden
    assert config["training"]["optimizer"]["type"] == "rmsprop"  # From base
    assert config["seed"]["value"] == 42  # From CLI


# =============================================================================
# Validate Config Tests
# =============================================================================


def test_validate_config_success(temp_config_file):
    """Test validating valid config."""
    args = Namespace(
        config=temp_config_file,
        seed=None,
        device=None,
        overrides=[],
        dry_run=False,
        print_config=False,
        quiet=True,
        verbose=False,
    )

    config = load_config_from_args(args, print_resolved=False)

    # Should not raise
    validate_config(config)


def test_validate_config_missing_env_id():
    """Test validation fails when env_id missing."""
    config = {
        "experiment": {"name": "test"},
        "environment": {},
        "training": {"total_frames": 1000, "optimizer": {"lr": 0.001}},
        "network": {"architecture": "dqn"},
        "replay": {"capacity": 1000},
    }

    with pytest.raises(ValueError, match="'env_id' is required"):
        validate_config(config)


def test_validate_config_invalid_gamma():
    """Test validation fails with invalid gamma."""
    config = {
        "experiment": {"name": "test"},
        "environment": {"env_id": "PongNoFrameskip-v4"},
        "network": {"architecture": "dqn"},
        "replay": {"capacity": 1000},
        "training": {"gamma": 1.5},  # Invalid: > 1.0
    }

    with pytest.raises(ValueError, match="gamma.*must be in range"):
        validate_config(config)


def test_validate_config_invalid_optimizer():
    """Test validation fails with invalid optimizer."""
    config = {
        "experiment": {"name": "test"},
        "environment": {"env_id": "PongNoFrameskip-v4"},
        "network": {"architecture": "dqn"},
        "replay": {"capacity": 1000},
        "training": {"optimizer": {"type": "sgd"}},  # Invalid optimizer
    }

    with pytest.raises(ValueError, match="optimizer.*type"):
        validate_config(config)


# =============================================================================
# Setup From Args Tests
# =============================================================================


def test_setup_from_args_success(temp_config_file):
    """Test complete setup from args."""
    args = parse_args(["--cfg", temp_config_file, "--seed", "42", "--quiet"])

    config = setup_from_args(args)

    assert config["seed"]["value"] == 42
    assert "cli" in config
    assert config["cli"]["args"]["seed"] == 42
    assert config["cli"]["args"]["config_file"] == temp_config_file


def test_setup_from_args_with_overrides(temp_config_file):
    """Test setup with CLI overrides."""
    args = parse_args(
        [
            "--cfg",
            temp_config_file,
            "--seed",
            "123",
            "--set",
            "training.optimizer.lr=0.0005",
            "--quiet",
        ]
    )

    config = setup_from_args(args)

    assert config["seed"]["value"] == 123
    assert config["training"]["optimizer"]["lr"] == 0.0005
    assert config["cli"]["args"]["overrides"] == ["training.optimizer.lr=0.0005"]


def test_setup_from_args_validation_error():
    """Test setup fails with invalid config."""
    # Create invalid config (missing required fields)
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump({"environment": {}}, f)
        temp_file = f.name

    try:
        args = parse_args(["--cfg", temp_file, "--quiet"])

        with pytest.raises(ValueError, match="Configuration validation failed"):
            setup_from_args(args)
    finally:
        Path(temp_file).unlink()


# =============================================================================
# Print Startup Banner Tests
# =============================================================================


def test_print_startup_banner(capsys, temp_config_file):
    """Test printing startup banner."""
    args = Namespace(
        config=temp_config_file,
        seed=42,
        device=None,
        overrides=[],
        dry_run=False,
        print_config=False,
        quiet=True,
        verbose=False,
    )

    config = load_config_from_args(args, print_resolved=False)
    print_startup_banner(config)

    captured = capsys.readouterr()
    assert "DQN Training" in captured.out
    assert "PongNoFrameskip-v4" in captured.out
    assert "1,000,000" in captured.out
    assert "42" in captured.out


# =============================================================================
# Integration Tests
# =============================================================================


def test_full_cli_workflow(game_config_file):
    """Test complete CLI workflow."""
    # Simulate command: python train_dqn.py --cfg pong.yaml --seed 42 --set training.lr=0.001
    argv = [
        "--cfg",
        game_config_file,
        "--seed",
        "42",
        "--set",
        "training.optimizer.lr=0.001",
        "--quiet",
    ]

    args = parse_args(argv)
    config = setup_from_args(args)

    # Verify all config sources merged correctly
    assert config["environment"]["env_id"] == "PongNoFrameskip-v4"  # From game config
    assert config["training"]["optimizer"]["type"] == "rmsprop"  # From base
    assert config["training"]["total_frames"] == 1000000  # Overridden in game config
    assert config["training"]["optimizer"]["lr"] == 0.001  # CLI override
    assert config["seed"]["value"] == 42  # CLI flag
    assert config["cli"]["args"]["seed"] == 42  # CLI metadata


def test_cli_with_all_flags(temp_config_file):
    """Test CLI with all available flags."""
    argv = [
        "--cfg",
        temp_config_file,
        "--seed",
        "999",
        "--resume",
        "/path/to/checkpoint.pt",
        "--set",
        "training.optimizer.lr=0.0001",
        "--set",
        "experiment.name=custom",
        "--device",
        "cuda",
        "--quiet",
    ]

    args = parse_args(argv)
    config = setup_from_args(args)

    assert config["seed"]["value"] == 999
    assert config["network"]["device"] == "cuda"
    assert config["training"]["optimizer"]["lr"] == 0.0001
    assert config["experiment"]["name"] == "custom"
    assert config["cli"]["args"]["resume"] == "/path/to/checkpoint.pt"
