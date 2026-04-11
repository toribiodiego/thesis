"""Tests for scripts/eval_checkpoints.py.

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

from scripts.eval_checkpoints import (
    create_model,
    detect_run_backend,
    discover_checkpoint_steps,
    discover_runs,
    evaluate_jax_checkpoint,
    get_existing_steps,
    get_repeat_action_probability,
    get_run_eval_epsilon,
    load_config,
    load_jax_checkpoint,
    save_results,
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
# get_existing_steps tests
# =============================================================================


def test_get_existing_steps_reads_csv(tmp_path):
    """Reads step numbers from evaluations.csv."""
    eval_dir = tmp_path / "eval_output"
    eval_dir.mkdir()
    # get_existing_steps expects run_dir/eval/evaluations.csv
    run_eval_dir = tmp_path / "run" / "eval"
    run_eval_dir.mkdir(parents=True)
    csv_path = run_eval_dir / "evaluations.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "mean_return"])
        writer.writerow([40000, 10.5])
        writer.writerow([80000, 15.2])

    steps = get_existing_steps(str(tmp_path / "run"))
    assert steps == {40000, 80000}


def test_get_existing_steps_no_csv(tmp_path):
    """Returns empty set when no CSV exists."""
    assert get_existing_steps(str(tmp_path)) == set()


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
# save_results tests
# =============================================================================


def test_save_results_creates_all_outputs(tmp_path):
    """save_results creates CSV, JSONL, per-episode, and detailed JSON."""
    results = {
        "mean_return": 10.5,
        "median_return": 9.0,
        "std_return": 3.2,
        "min_return": 5.0,
        "max_return": 18.0,
        "mean_length": 500.0,
        "num_episodes": 30,
        "episode_returns": [10.0, 11.0],
        "episode_lengths": [450, 550],
    }
    save_results(str(tmp_path), step=40000, results=results,
                 run_eval_epsilon=0.05, training_epsilon=0.05)

    # CSV
    csv_path = tmp_path / "eval" / "evaluations.csv"
    assert csv_path.exists()
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert int(rows[0]["step"]) == 40000
    assert float(rows[0]["mean_return"]) == 10.5

    # JSONL
    jsonl_path = tmp_path / "eval" / "evaluations.jsonl"
    assert jsonl_path.exists()

    # Per-episode returns
    episodes_path = tmp_path / "eval" / "per_episode_returns.jsonl"
    assert episodes_path.exists()
    import json
    with open(episodes_path) as f:
        entry = json.loads(f.readline())
    assert entry["step"] == 40000
    assert entry["episode_returns"] == [10.0, 11.0]

    # Detailed JSON
    detailed_path = tmp_path / "eval" / "detailed" / "eval_step_40000.json"
    assert detailed_path.exists()


def test_save_results_appends(tmp_path):
    """Appends rows without duplicating headers."""
    results = {
        "mean_return": 10.0,
        "median_return": 9.0,
        "std_return": 3.0,
        "min_return": 5.0,
        "max_return": 15.0,
        "mean_length": 400.0,
        "num_episodes": 30,
        "episode_returns": [10.0],
        "episode_lengths": [400],
    }
    save_results(str(tmp_path), step=40000, results=results,
                 run_eval_epsilon=0.05, training_epsilon=0.05)
    save_results(str(tmp_path), step=80000, results=results,
                 run_eval_epsilon=0.05, training_epsilon=0.05)

    csv_path = tmp_path / "eval" / "evaluations.csv"
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


# =============================================================================
# get_run_eval_epsilon tests
# =============================================================================


def test_eval_epsilon_dqn():
    """DQN uses epsilon=0.05."""
    config = {"rainbow": {"enabled": False}}
    assert get_run_eval_epsilon(config) == 0.05


def test_eval_epsilon_rainbow_noisy():
    """Rainbow with NoisyNets uses epsilon=0.0 (greedy)."""
    config = {"rainbow": {"enabled": True, "noisy_nets": True}}
    assert get_run_eval_epsilon(config) == 0.0


def test_eval_epsilon_rainbow_no_noisy():
    """Rainbow without NoisyNets falls back to epsilon=0.05."""
    config = {"rainbow": {"enabled": True, "noisy_nets": False}}
    assert get_run_eval_epsilon(config) == 0.05


# =============================================================================
# get_repeat_action_probability tests
# =============================================================================


def test_repeat_action_probability_present():
    """Reads sticky actions from config."""
    config = {"environment": {"repeat_action_probability": 0.25}}
    assert get_repeat_action_probability(config) == 0.25


def test_repeat_action_probability_missing():
    """Defaults to 0.0 when not in config."""
    config = {"environment": {"env_id": "test"}}
    assert get_repeat_action_probability(config) == 0.0


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

    existing = get_existing_steps(run_dir)
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
        "mean_length": 600.0,
        "num_episodes": 30,
        "episode_returns": [12.0, 13.0],
        "episode_lengths": [550, 650],
    }

    save_results(run_dir, steps[0], mock_results, 0.0, checkpoint.get("epsilon", 0.1))

    # Verify CSV was written
    csv_path = os.path.join(run_dir, "eval", "evaluations.csv")
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert int(rows[0]["step"]) == steps[0]
    assert float(rows[0]["mean_return"]) == 12.5

    # Verify per-episode and detailed files were written
    assert os.path.exists(os.path.join(run_dir, "eval", "per_episode_returns.jsonl"))
    assert os.path.exists(os.path.join(run_dir, "eval", "detailed", f"eval_step_{steps[0]}.json"))

    # Verify the step is now in existing results
    existing_after = get_existing_steps(run_dir)
    assert steps[0] in existing_after
    assert steps[1] not in existing_after


# =============================================================================
# detect_run_backend tests
# =============================================================================


def test_detect_run_backend_pytorch(tmp_path):
    """Detects PyTorch runs by config.yaml."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").touch()
    assert detect_run_backend(str(run_dir)) == "pytorch"


def test_detect_run_backend_jax(tmp_path):
    """Detects JAX runs by config.gin."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.gin").touch()
    assert detect_run_backend(str(run_dir)) == "jax"


def test_detect_run_backend_none(tmp_path):
    """Returns None when no config file exists."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    assert detect_run_backend(str(run_dir)) is None


def test_detect_run_backend_jax_takes_priority(tmp_path):
    """JAX takes priority when both config files exist."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.gin").touch()
    (run_dir / "config.yaml").touch()
    assert detect_run_backend(str(run_dir)) == "jax"


# =============================================================================
# discover_runs with JAX runs tests
# =============================================================================


def test_discover_runs_finds_jax_runs(tmp_path):
    """discover_runs finds JAX runs with config.gin + checkpoints/."""
    run_dir = tmp_path / "runs" / "BBF_boxing_seed42"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.gin").write_text("# gin config")
    result = discover_runs(str(tmp_path / "runs"))
    assert "BBF_boxing_seed42" in result


def test_discover_runs_finds_both_backends(tmp_path):
    """discover_runs finds both PyTorch and JAX runs."""
    runs_dir = tmp_path / "runs"

    # PyTorch run
    pt_dir = runs_dir / "pytorch_run"
    (pt_dir / "checkpoints").mkdir(parents=True)
    (pt_dir / "config.yaml").write_text("{}")

    # JAX run
    jax_dir = runs_dir / "jax_run"
    (jax_dir / "checkpoints").mkdir(parents=True)
    (jax_dir / "config.gin").write_text("# gin")

    result = discover_runs(str(runs_dir))
    assert "pytorch_run" in result
    assert "jax_run" in result


# =============================================================================
# discover_checkpoint_steps with msgpack tests
# =============================================================================


def test_discover_checkpoint_steps_msgpack(tmp_path):
    """Discovers .msgpack checkpoint files."""
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    (cp_dir / "checkpoint_10000.msgpack").touch()
    (cp_dir / "checkpoint_10000.json").touch()
    (cp_dir / "checkpoint_20000.msgpack").touch()
    (cp_dir / "checkpoint_20000.json").touch()

    steps = discover_checkpoint_steps(str(cp_dir))
    assert steps == [10000, 20000]


def test_discover_checkpoint_steps_mixed(tmp_path):
    """Discovers both .pt and .msgpack checkpoints."""
    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()
    (cp_dir / "checkpoint_10000.pt").touch()
    (cp_dir / "checkpoint_20000.msgpack").touch()

    steps = discover_checkpoint_steps(str(cp_dir))
    assert steps == [10000, 20000]


# =============================================================================
# load_jax_checkpoint tests
# =============================================================================


def test_load_jax_checkpoint(tmp_path):
    """Loads msgpack params and JSON metadata."""
    from flax.serialization import msgpack_serialize

    cp_dir = tmp_path / "checkpoints"
    cp_dir.mkdir()

    # Write a small params dict as msgpack
    params = {"encoder": {"conv1": {"kernel": np.ones((3, 3))}}}
    with open(cp_dir / "checkpoint_10000.msgpack", "wb") as f:
        f.write(msgpack_serialize(params))

    # Write metadata JSON
    import json
    meta = {
        "step": 10000,
        "training_steps": 5000,
        "cumulative_resets": 2,
        "cycle_grad_steps": 1000,
        "gin_config": "# resolved config",
    }
    with open(cp_dir / "checkpoint_10000.json", "w") as f:
        json.dump(meta, f)

    loaded_params, loaded_meta = load_jax_checkpoint(str(cp_dir), 10000)

    assert loaded_meta["step"] == 10000
    assert loaded_meta["training_steps"] == 5000
    assert loaded_meta["cumulative_resets"] == 2
    np.testing.assert_array_equal(
        loaded_params["encoder"]["conv1"]["kernel"], np.ones((3, 3))
    )
