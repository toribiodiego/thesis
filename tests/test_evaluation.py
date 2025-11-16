"""
Tests for evaluation harness components.

Verifies:
- Core evaluate() function with various configurations
- EvaluationScheduler (frame-based and wall-clock)
- EvaluationLogger (CSV, JSON, JSONL output formats)
- Train/eval mode switching
- Lives tracking (optional Atari feature)
- Metadata inclusion (seed, step)
"""

import numpy as np

from src.models import DQN
from src.training import EvaluationLogger, EvaluationScheduler, evaluate

# ============================================================================
# Core Evaluation Function Tests
# ============================================================================


def test_evaluate_basic():
    """Test evaluate function runs and returns statistics."""
    from unittest.mock import Mock

    # Mock environment
    env = Mock()
    env.action_space.n = 6
    env.action_space.sample.return_value = 0

    # Mock reset and step
    dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    env.reset.return_value = (dummy_state, {})
    env.step.return_value = (dummy_state, 1.0, False, False, {})

    # Create model
    model = DQN(num_actions=6)

    # Run evaluation (will loop indefinitely without termination, so mock it)
    # Let's fix the mock to terminate after a few steps
    step_count = [0]

    def mock_step(action):
        step_count[0] += 1
        done = step_count[0] >= 10  # Terminate after 10 steps
        return (dummy_state, 1.0, done, False, {})

    env.step.side_effect = mock_step

    # Evaluate
    results = evaluate(env, model, num_episodes=2, eval_epsilon=0.05, device="cpu")

    # Check results structure
    assert "mean_return" in results
    assert "median_return" in results
    assert "std_return" in results
    assert "min_return" in results
    assert "max_return" in results
    assert "mean_length" in results
    assert "episode_returns" in results
    assert "episode_lengths" in results
    assert results["num_episodes"] == 2
    assert results["eval_epsilon"] == 0.05


def test_evaluate_greedy():
    """Test evaluate with greedy policy (epsilon=0)."""
    from unittest.mock import Mock

    env = Mock()
    env.action_space.n = 6
    dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    env.reset.return_value = (dummy_state, {})

    step_count = [0]

    def mock_step(action):
        step_count[0] += 1
        done = step_count[0] >= 5
        if done:
            step_count[0] = 0  # Reset for next episode
        return (dummy_state, 1.0, done, False, {})

    env.step.side_effect = mock_step

    model = DQN(num_actions=6)

    # Greedy evaluation
    results = evaluate(env, model, num_episodes=1, eval_epsilon=0.0, device="cpu")

    assert results["num_episodes"] == 1
    assert len(results["episode_returns"]) == 1


def test_evaluate_with_metadata():
    """Test evaluate includes seed and step in results."""
    from unittest.mock import Mock

    env = Mock()
    env.action_space.n = 6
    dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    env.reset.return_value = (dummy_state, {})

    step_count = [0]

    def mock_step(action):
        step_count[0] += 1
        done = step_count[0] >= 5
        if done:
            step_count[0] = 0
        return (dummy_state, 1.0, done, False, {})

    env.step.side_effect = mock_step
    model = DQN(num_actions=6)

    # Evaluate with metadata
    results = evaluate(
        env,
        model,
        num_episodes=1,
        eval_epsilon=0.05,
        device="cpu",
        seed=42,
        step=250000,
    )

    # Check metadata is included
    assert results["seed"] == 42
    assert results["step"] == 250000
    assert results["eval_epsilon"] == 0.05


def test_evaluate_lives_tracking():
    """Test evaluate tracks lives lost when requested."""
    from unittest.mock import Mock

    env = Mock()
    env.action_space.n = 6
    dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)

    # Mock environment with lives tracking
    step_count = [0]
    initial_lives = 5

    def mock_reset():
        return (dummy_state, {"lives": initial_lives})

    def mock_step(action):
        step_count[0] += 1
        done = step_count[0] >= 10
        # Simulate losing lives during episode
        current_lives = max(0, initial_lives - (step_count[0] // 3))
        if done:
            step_count[0] = 0
        return (dummy_state, 1.0, done, False, {"lives": current_lives})

    env.reset.side_effect = mock_reset
    env.step.side_effect = mock_step
    model = DQN(num_actions=6)

    # Evaluate with lives tracking
    results = evaluate(
        env, model, num_episodes=2, eval_epsilon=0.05, device="cpu", track_lives=True
    )

    # Check lives tracking is included
    assert "episode_lives_lost" in results
    assert len(results["episode_lives_lost"]) == 2
    # Each episode should have lives lost data
    for lives_lost in results["episode_lives_lost"]:
        assert lives_lost is not None


def test_evaluate_without_lives_tracking():
    """Test evaluate omits lives when not requested."""
    from unittest.mock import Mock

    env = Mock()
    env.action_space.n = 6
    dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    env.reset.return_value = (dummy_state, {})

    step_count = [0]

    def mock_step(action):
        step_count[0] += 1
        done = step_count[0] >= 5
        if done:
            step_count[0] = 0
        return (dummy_state, 1.0, done, False, {})

    env.step.side_effect = mock_step
    model = DQN(num_actions=6)

    # Evaluate without lives tracking
    results = evaluate(
        env, model, num_episodes=1, eval_epsilon=0.05, device="cpu", track_lives=False
    )

    # Check lives tracking is NOT included
    assert "episode_lives_lost" not in results


def test_evaluate_train_eval_mode_switching():
    """Test evaluate() properly switches between train and eval modes."""
    from unittest.mock import Mock

    env = Mock()
    env.action_space.n = 6
    dummy_state = np.random.randint(0, 255, (4, 84, 84), dtype=np.uint8)
    env.reset.return_value = (dummy_state, {})

    step_count = [0]

    def mock_step(action):
        step_count[0] += 1
        done = step_count[0] >= 5
        if done:
            step_count[0] = 0
        return (dummy_state, 1.0, done, False, {})

    env.step.side_effect = mock_step
    model = DQN(num_actions=6)

    # Model starts in train mode
    assert model.training

    # Evaluate (should set to eval mode and restore train mode)
    evaluate(env, model, num_episodes=1, eval_epsilon=0.05, device="cpu")

    # Model should be back in train mode
    assert model.training


# ============================================================================
# EvaluationScheduler Tests
# ============================================================================


def test_evaluation_scheduler_interval():
    """Test EvaluationScheduler triggers at correct intervals."""
    scheduler = EvaluationScheduler(eval_interval=250000, num_episodes=10)

    # Should not evaluate at step 0
    assert not scheduler.should_evaluate(0)

    # Should not evaluate before interval
    assert not scheduler.should_evaluate(100000)

    # Should evaluate at interval
    assert scheduler.should_evaluate(250000)

    # Should not evaluate twice at same step
    scheduler.record_evaluation(250000, {"mean_return": 20.0})
    assert not scheduler.should_evaluate(250000)

    # Should evaluate at next interval
    assert scheduler.should_evaluate(500000)


def test_evaluation_scheduler_tracking():
    """Test EvaluationScheduler tracks evaluation history."""
    scheduler = EvaluationScheduler(eval_interval=250000)

    # Record evaluations
    scheduler.record_evaluation(250000, {"mean_return": 15.0})
    scheduler.record_evaluation(500000, {"mean_return": 20.0})
    scheduler.record_evaluation(750000, {"mean_return": 25.0})

    # Check history
    assert len(scheduler.eval_steps) == 3
    assert len(scheduler.eval_returns) == 3
    assert scheduler.get_best_return() == 25.0


def test_evaluation_scheduler_trend():
    """Test EvaluationScheduler detects performance trends."""
    scheduler = EvaluationScheduler()

    # Record improving trend
    scheduler.record_evaluation(250000, {"mean_return": 10.0})
    scheduler.record_evaluation(500000, {"mean_return": 15.0})
    scheduler.record_evaluation(750000, {"mean_return": 20.0})

    assert scheduler.get_recent_trend(n=3) == "improving"

    # Record declining trend
    scheduler.record_evaluation(1000000, {"mean_return": 15.0})

    trend = scheduler.get_recent_trend(n=3)
    assert trend in ["declining", "stable"]


def test_evaluation_scheduler_wall_clock():
    """Test EvaluationScheduler with wall-clock scheduling."""
    import time

    # Create scheduler with 0.1 second interval
    scheduler = EvaluationScheduler(wall_clock_interval=0.1, num_episodes=5)

    # Should trigger first evaluation immediately (after step 0)
    assert scheduler.should_evaluate(100)

    # Record first evaluation
    scheduler.record_evaluation(100, {"mean_return": 10.0})

    # Should not trigger immediately after
    assert not scheduler.should_evaluate(200)

    # Wait for interval to pass
    time.sleep(0.15)

    # Should trigger now
    assert scheduler.should_evaluate(300)


def test_evaluation_scheduler_timestamps():
    """Test EvaluationScheduler tracks timestamps."""
    import time

    scheduler = EvaluationScheduler(eval_interval=100)

    # Record evaluations
    scheduler.record_evaluation(100, {"mean_return": 10.0})
    time.sleep(0.05)
    scheduler.record_evaluation(200, {"mean_return": 15.0})

    # Check timestamps were recorded
    assert len(scheduler.eval_timestamps) == 2
    assert scheduler.last_eval_time is not None
    assert scheduler.eval_timestamps[1] > scheduler.eval_timestamps[0]


def test_evaluation_scheduler_metadata():
    """Test EvaluationScheduler provides schedule metadata."""
    # Frame-based scheduler
    scheduler = EvaluationScheduler(
        eval_interval=250000, num_episodes=10, eval_epsilon=0.05
    )

    scheduler.record_evaluation(250000, {"mean_return": 15.0})
    scheduler.record_evaluation(500000, {"mean_return": 20.0})

    metadata = scheduler.get_schedule_metadata()

    # Check metadata structure
    assert metadata["schedule_type"] == "frame_based"
    assert metadata["eval_interval"] == 250000
    assert metadata["num_episodes"] == 10
    assert metadata["eval_epsilon"] == 0.05
    assert metadata["total_evaluations"] == 2
    assert len(metadata["eval_steps"]) == 2
    assert len(metadata["eval_returns"]) == 2
    assert "eval_timestamps" in metadata
    assert "elapsed_times" in metadata


def test_evaluation_scheduler_wall_clock_metadata():
    """Test EvaluationScheduler metadata for wall-clock scheduling."""
    scheduler = EvaluationScheduler(wall_clock_interval=1800, num_episodes=10)

    scheduler.record_evaluation(100, {"mean_return": 10.0})

    metadata = scheduler.get_schedule_metadata()

    # Check schedule type
    assert metadata["schedule_type"] == "wall_clock"
    assert metadata["wall_clock_interval"] == 1800
    assert "total_elapsed_time" in metadata


# ============================================================================
# EvaluationLogger Tests
# ============================================================================


def test_evaluation_logger_csv():
    """Test EvaluationLogger writes CSV correctly."""
    import csv
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EvaluationLogger(log_dir=tmpdir)

        # Log evaluation
        results = {
            "mean_return": 20.5,
            "median_return": 21.0,
            "std_return": 2.5,
            "min_return": 15.0,
            "max_return": 25.0,
            "mean_length": 1200.5,
            "episode_returns": [15.0, 20.0, 25.0],
            "episode_lengths": [1000, 1200, 1400],
            "num_episodes": 3,
            "eval_epsilon": 0.05,
        }

        logger.log_evaluation(step=250000, results=results, epsilon=0.5)

        # Check CSV exists
        csv_path = os.path.join(tmpdir, "evaluations.csv")
        assert os.path.exists(csv_path)

        # Read CSV
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert int(rows[0]["step"]) == 250000
        assert float(rows[0]["mean_return"]) == 20.5
        assert float(rows[0]["eval_epsilon"]) == 0.05
        assert int(rows[0]["episodes"]) == 3


def test_evaluation_logger_json():
    """Test EvaluationLogger writes detailed JSON."""
    import json
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EvaluationLogger(log_dir=tmpdir)

        results = {
            "mean_return": 20.5,
            "median_return": 21.0,
            "std_return": 2.5,
            "min_return": 15.0,
            "max_return": 25.0,
            "mean_length": 1200.5,
            "episode_returns": [15.0, 20.0, 25.0],
            "episode_lengths": [1000, 1200, 1400],
            "num_episodes": 3,
        }

        logger.log_evaluation(step=250000, results=results)

        # Check JSON exists
        json_path = os.path.join(tmpdir, "detailed", "eval_step_250000.json")
        assert os.path.exists(json_path)

        # Read JSON
        with open(json_path, "r") as f:
            data = json.load(f)

        assert data["step"] == 250000
        assert data["statistics"]["mean_return"] == 20.5
        assert len(data["episode_returns"]) == 3


def test_evaluation_logger_get_all_results():
    """Test EvaluationLogger retrieves all results."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EvaluationLogger(log_dir=tmpdir)

        # Log multiple evaluations
        for step in [250000, 500000, 750000]:
            results = {
                "mean_return": step / 10000,
                "median_return": step / 10000,
                "std_return": 1.0,
                "min_return": 10.0,
                "max_return": 30.0,
                "mean_length": 1000.0,
                "episode_returns": [10.0, 20.0, 30.0],
                "episode_lengths": [1000, 1000, 1000],
                "num_episodes": 3,
            }
            logger.log_evaluation(step=step, results=results)

        # Retrieve all results
        all_results = logger.get_all_results()
        assert len(all_results) == 3


def test_evaluation_logger_jsonl():
    """Test EvaluationLogger writes JSONL format."""
    import json
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EvaluationLogger(log_dir=tmpdir)

        # Log multiple evaluations
        for i, step in enumerate([250000, 500000]):
            results = {
                "mean_return": 10.0 + i * 5,
                "median_return": 10.0 + i * 5,
                "std_return": 2.0,
                "min_return": 5.0,
                "max_return": 15.0,
                "mean_length": 1000.0,
                "episode_returns": [5.0, 10.0, 15.0],
                "episode_lengths": [1000, 1000, 1000],
                "num_episodes": 3,
                "eval_epsilon": 0.05,
            }
            logger.log_evaluation(step=step, results=results)

        # Check JSONL file exists
        jsonl_path = os.path.join(tmpdir, "evaluations.jsonl")
        assert os.path.exists(jsonl_path)

        # Read JSONL (one JSON object per line)
        with open(jsonl_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 2

        # Parse each line
        obj1 = json.loads(lines[0])
        obj2 = json.loads(lines[1])

        assert obj1["step"] == 250000
        assert obj1["mean_return"] == 10.0
        assert obj2["step"] == 500000
        assert obj2["mean_return"] == 15.0


def test_evaluation_logger_per_episode_sidecar():
    """Test EvaluationLogger writes per-episode returns sidecar file."""
    import json
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EvaluationLogger(log_dir=tmpdir)

        results = {
            "mean_return": 20.0,
            "median_return": 20.0,
            "std_return": 5.0,
            "min_return": 10.0,
            "max_return": 30.0,
            "mean_length": 1000.0,
            "episode_returns": [10.0, 20.0, 30.0],
            "episode_lengths": [800, 1000, 1200],
            "num_episodes": 3,
            "eval_epsilon": 0.05,
        }

        logger.log_evaluation(step=250000, results=results)

        # Check sidecar file exists
        episodes_path = os.path.join(tmpdir, "per_episode_returns.jsonl")
        assert os.path.exists(episodes_path)

        # Read sidecar file
        with open(episodes_path, "r") as f:
            line = f.readline()

        data = json.loads(line)
        assert data["step"] == 250000
        assert len(data["episode_returns"]) == 3
        assert data["episode_returns"] == [10.0, 20.0, 30.0]
        assert len(data["episode_lengths"]) == 3
        assert data["episode_lengths"] == [800, 1000, 1200]


def test_evaluation_logger_all_output_files():
    """Test EvaluationLogger creates all required output files."""
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = EvaluationLogger(log_dir=tmpdir)

        results = {
            "mean_return": 20.0,
            "median_return": 20.0,
            "std_return": 5.0,
            "min_return": 10.0,
            "max_return": 30.0,
            "mean_length": 1000.0,
            "episode_returns": [10.0, 20.0, 30.0],
            "episode_lengths": [1000, 1000, 1000],
            "num_episodes": 3,
            "eval_epsilon": 0.05,
        }

        logger.log_evaluation(step=250000, results=results)

        # Check all files exist
        assert os.path.exists(os.path.join(tmpdir, "evaluations.csv"))
        assert os.path.exists(os.path.join(tmpdir, "evaluations.jsonl"))
        assert os.path.exists(os.path.join(tmpdir, "per_episode_returns.jsonl"))
        assert os.path.exists(os.path.join(tmpdir, "detailed", "eval_step_250000.json"))
