"""Tests for scripts/reeval_checkpoints.py.

Verifies checkpoint discovery, model creation, and the full re-evaluation
loop using a small RainbowDQN checkpoint saved to a temp directory.
"""

import csv
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.reeval_checkpoints import (
    append_eval_row,
    create_model,
    discover_checkpoint_steps,
    discover_runs,
    get_existing_eval_steps,
    load_config,
)
from src.models.rainbow import RainbowDQN


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rainbow_config():
    """Minimal config dict matching a Rainbow run."""
    return {
        "environment": {"env_id": "ALE/Pong-v5"},
        "rainbow": {
            "enabled": True,
            "noisy_nets": True,
            "dueling": True,
            "distributional": {
                "num_atoms": 51,
                "v_min": -10.0,
                "v_max": 10.0,
            },
        },
        "network": {"dropout": 0.0},
    }


@pytest.fixture
def dqn_config():
    """Minimal config dict matching a vanilla DQN run."""
    return {
        "environment": {"env_id": "ALE/Pong-v5"},
        "rainbow": {"enabled": False},
        "network": {"dropout": 0.0},
    }


@pytest.fixture
def run_dir_with_rainbow_checkpoint(tmp_path, rainbow_config):
    """Create a temp run directory with config.yaml and a Rainbow checkpoint."""
    run_dir = tmp_path / "runs" / "test_rainbow_run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    # Write config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(rainbow_config, f)

    # Create a small RainbowDQN and save checkpoint
    num_actions = 4
    model = RainbowDQN(
        num_actions=num_actions,
        num_atoms=51,
        v_min=-10.0,
        v_max=10.0,
        noisy=True,
        dueling=True,
    )
    checkpoint = {
        "online_model_state_dict": model.state_dict(),
        "epsilon": 0.05,
        "step": 40000,
    }
    torch.save(checkpoint, checkpoint_dir / "checkpoint_40000.pt")
    torch.save(checkpoint, checkpoint_dir / "checkpoint_80000.pt")

    return run_dir


# =============================================================================
# discover_runs tests
# =============================================================================


def test_discover_runs_finds_valid_runs(run_dir_with_rainbow_checkpoint):
    """discover_runs returns directories with config.yaml + checkpoints/."""
    runs_dir = run_dir_with_rainbow_checkpoint.parent
    result = discover_runs(str(runs_dir))
    assert "test_rainbow_run" in result


def test_discover_runs_skips_dirs_without_checkpoints(tmp_path):
    """Directories without checkpoints/ are not discovered."""
    run_dir = tmp_path / "runs" / "no_checkpoints"
    run_dir.mkdir(parents=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump({"environment": {"env_id": "test"}}, f)

    result = discover_runs(str(tmp_path / "runs"))
    assert "no_checkpoints" not in result


def test_discover_runs_skips_dirs_without_config(tmp_path):
    """Directories without config.yaml are not discovered."""
    run_dir = tmp_path / "runs" / "no_config"
    (run_dir / "checkpoints").mkdir(parents=True)

    result = discover_runs(str(tmp_path / "runs"))
    assert "no_config" not in result


def test_discover_runs_empty_dir(tmp_path):
    """Empty runs directory returns empty list."""
    runs_dir = tmp_path / "empty_runs"
    runs_dir.mkdir()
    assert discover_runs(str(runs_dir)) == []


def test_discover_runs_nonexistent_dir(tmp_path):
    """Non-existent directory returns empty list."""
    assert discover_runs(str(tmp_path / "does_not_exist")) == []


# =============================================================================
# discover_checkpoint_steps tests
# =============================================================================


def test_discover_checkpoint_steps(run_dir_with_rainbow_checkpoint):
    """Finds checkpoint step numbers from filenames."""
    checkpoint_dir = str(run_dir_with_rainbow_checkpoint / "checkpoints")
    steps = discover_checkpoint_steps(checkpoint_dir)
    assert steps == [40000, 80000]


def test_discover_checkpoint_steps_ignores_non_checkpoints(tmp_path):
    """Non-checkpoint files are ignored."""
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    (cp_dir / "best_model.pt").touch()
    (cp_dir / "checkpoint_100000.pt").touch()
    (cp_dir / "notes.txt").touch()

    steps = discover_checkpoint_steps(str(cp_dir))
    assert steps == [100000]


def test_discover_checkpoint_steps_empty_dir(tmp_path):
    """Empty checkpoint directory returns empty list."""
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    assert discover_checkpoint_steps(str(cp_dir)) == []


def test_discover_checkpoint_steps_nonexistent_dir(tmp_path):
    """Non-existent directory returns empty list."""
    assert discover_checkpoint_steps(str(tmp_path / "nope")) == []


# =============================================================================
# get_existing_eval_steps tests
# =============================================================================


def test_get_existing_eval_steps_reads_csv(tmp_path):
    """Reads step numbers from evaluations.csv."""
    eval_dir = tmp_path / "eval_output"
    eval_dir.mkdir()
    # get_existing_eval_steps expects run_dir/eval/evaluations.csv
    run_eval_dir = tmp_path / "run" / "eval"
    run_eval_dir.mkdir(parents=True)
    csv_path = run_eval_dir / "evaluations.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_return"])
        writer.writerow([40000, 10.5])
        writer.writerow([80000, 15.2])

    steps = get_existing_eval_steps(str(tmp_path / "run"))
    assert steps == {40000, 80000}


def test_get_existing_eval_steps_no_csv(tmp_path):
    """Returns empty set when no CSV exists."""
    assert get_existing_eval_steps(str(tmp_path)) == set()


# =============================================================================
# load_config tests
# =============================================================================


def test_load_config(run_dir_with_rainbow_checkpoint, rainbow_config):
    """Loads config.yaml from run directory."""
    config = load_config(str(run_dir_with_rainbow_checkpoint))
    assert config["rainbow"]["enabled"] is True
    assert config["environment"]["env_id"] == rainbow_config["environment"]["env_id"]


# =============================================================================
# create_model tests
# =============================================================================


def test_create_model_rainbow(rainbow_config):
    """Creates RainbowDQN when rainbow.enabled is True."""
    model = create_model(rainbow_config, num_actions=4, device="cpu")
    assert isinstance(model, RainbowDQN)


def test_create_model_dqn(dqn_config):
    """Creates vanilla DQN when rainbow.enabled is False."""
    from src.models.dqn import DQN
    model = create_model(dqn_config, num_actions=4, device="cpu")
    assert isinstance(model, DQN)


# =============================================================================
# append_eval_row tests
# =============================================================================


def test_append_eval_row_creates_csv(tmp_path):
    """Creates evaluations.csv with header when it doesn't exist."""
    results = {
        "mean_return": 10.5,
        "median_return": 9.0,
        "std_return": 3.2,
        "min_return": 5.0,
        "max_return": 18.0,
        "num_episodes": 30,
    }
    append_eval_row(str(tmp_path), step=40000, results=results, training_epsilon=0.05)

    csv_path = tmp_path / "eval" / "evaluations.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert int(rows[0]["step"]) == 40000
    assert float(rows[0]["mean_return"]) == 10.5


def test_append_eval_row_appends(tmp_path):
    """Appends rows without duplicating headers."""
    results = {
        "mean_return": 10.0,
        "median_return": 9.0,
        "std_return": 3.0,
        "min_return": 5.0,
        "max_return": 15.0,
        "num_episodes": 30,
    }
    append_eval_row(str(tmp_path), step=40000, results=results, training_epsilon=0.05)
    append_eval_row(str(tmp_path), step=80000, results=results, training_epsilon=0.05)

    csv_path = tmp_path / "eval" / "evaluations.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


# =============================================================================
# Integration: full checkpoint load and assessment loop
# =============================================================================


def test_rainbow_checkpoint_load_and_assessment(run_dir_with_rainbow_checkpoint):
    """End-to-end: load Rainbow checkpoint, create model, write CSV."""
    run_dir = str(run_dir_with_rainbow_checkpoint)
    config = load_config(run_dir)
    num_actions = 4

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    steps = discover_checkpoint_steps(checkpoint_dir)
    assert len(steps) == 2

    existing = get_existing_eval_steps(run_dir)
    assert len(existing) == 0

    # Load checkpoint and create model
    cp_path = os.path.join(checkpoint_dir, f"checkpoint_{steps[0]}.pt")
    checkpoint = torch.load(cp_path, map_location="cpu", weights_only=False)

    model = create_model(config, num_actions, device="cpu")
    model.load_state_dict(checkpoint["online_model_state_dict"], strict=True)

    # Verify the model produces output with correct shape
    dummy_input = torch.randn(1, 4, 84, 84)
    with torch.no_grad():
        output = model(dummy_input)
    assert output["q_values"].shape == (1, num_actions)

    # Simulate writing results (no real Atari env needed)
    mock_results = {
        "mean_return": 12.5,
        "median_return": 11.0,
        "std_return": 4.1,
        "min_return": 3.0,
        "max_return": 25.0,
        "num_episodes": 30,
    }

    append_eval_row(run_dir, steps[0], mock_results, checkpoint.get("epsilon", 0.1))

    # Verify CSV was written
    csv_path = os.path.join(run_dir, "eval", "evaluations.csv")
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert int(rows[0]["step"]) == steps[0]
    assert float(rows[0]["mean_return"]) == 12.5

    # Verify the step is now in existing assessment results
    existing_after = get_existing_eval_steps(run_dir)
    assert steps[0] in existing_after
    assert steps[1] not in existing_after
