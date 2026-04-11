"""Tests for train.py CSV output: verifies new metric columns appear."""

import csv
import io
import json
import os
import sys
import time
import types
from unittest.mock import MagicMock

import pytest

# train.py imports JAX/dopamine/gin at module level. Stub them out
# so this test can run without those heavy dependencies.
_STUB_MODULES = [
    "gin",
    "jax", "jax.tree_util", "jax.random",
    "tensorflow",
    "absl", "absl.logging",
    "dopamine", "dopamine.discrete_domains",
    "dopamine.discrete_domains.atari_lib",
    "bigger_better_faster",
    "bigger_better_faster.bbf",
    "bigger_better_faster.bbf.agents",
    "bigger_better_faster.bbf.agents.metric_agent",
    "bigger_better_faster.bbf.eval_run_experiment",
]
for _mod in _STUB_MODULES:
    if _mod not in sys.modules:
        stub = types.ModuleType(_mod)
        if _mod == "bigger_better_faster.bbf.agents.metric_agent":
            stub.MetricBBFAgent = MagicMock()
        if _mod == "bigger_better_faster.bbf.eval_run_experiment":
            stub.DataEfficientAtariRunner = MagicMock()
        if _mod == "absl":
            stub.logging = types.ModuleType("absl.logging")
        if _mod == "jax":
            tu = types.ModuleType("jax.tree_util")
            tu.tree_leaves = lambda x: list(x) if hasattr(x, '__iter__') else []
            stub.tree_util = tu
        sys.modules[_mod] = stub

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import (  # noqa: E402
    CSV_HEADER, CSV_EXTENSIONS, write_step_row, validate_checkpoint,
    write_progress,
)


NEW_METRIC_COLUMNS = [
    "QValueMean", "QValueMax",
    "GradNorm/encoder", "GradNorm/transition_model",
    "GradNorm/projection", "GradNorm/predictor", "GradNorm/head",
]


def test_csv_header_contains_new_columns():
    """All new metric columns are present in CSV_HEADER."""
    for col in NEW_METRIC_COLUMNS:
        assert col in CSV_HEADER, f"Missing column: {col}"


def test_csv_header_contains_new_columns_in_extensions():
    """New columns are declared in CSV_EXTENSIONS."""
    for col in NEW_METRIC_COLUMNS:
        assert col in CSV_EXTENSIONS, f"Missing in CSV_EXTENSIONS: {col}"


def _make_mock_agent(metrics):
    """Create a minimal mock agent for write_step_row."""
    agent = MagicMock()
    agent._last_metrics = metrics
    agent._noisy = True
    agent.learning_rate = 0.0001
    agent.optimizer_state = []
    agent._replay.add_count = 1000
    return agent


def test_write_step_row_includes_qvalue_columns():
    """write_step_row writes QValueMean and QValueMax from agent metrics."""
    metrics = {
        "TotalLoss": 0.5,
        "DQNLoss": 0.4,
        "TD Error": 0.3,
        "SPRLoss": 0.1,
        "QValueMean": 2.5,
        "QValueMax": 8.1,
        "GradNorm": 1.2,
        "PNorm": 50.0,
    }
    agent = _make_mock_agent(metrics)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_HEADER, extrasaction="ignore")
    writer.writeheader()
    write_step_row(writer, buf, step=1000, fps=100.0, agent=agent)

    buf.seek(0)
    rows = list(csv.DictReader(buf))
    assert len(rows) == 1
    assert float(rows[0]["QValueMean"]) == 2.5
    assert float(rows[0]["QValueMax"]) == 8.1


def test_write_step_row_includes_per_module_gradnorms():
    """write_step_row writes GradNorm/encoder through GradNorm/head."""
    metrics = {
        "TotalLoss": 0.5,
        "GradNorm": 1.2,
        "GradNorm/encoder": 0.8,
        "GradNorm/transition_model": 0.3,
        "GradNorm/projection": 0.15,
        "GradNorm/predictor": 0.1,
        "GradNorm/head": 0.05,
    }
    agent = _make_mock_agent(metrics)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_HEADER, extrasaction="ignore")
    writer.writeheader()
    write_step_row(writer, buf, step=2000, fps=200.0, agent=agent)

    buf.seek(0)
    rows = list(csv.DictReader(buf))
    assert len(rows) == 1
    assert float(rows[0]["GradNorm/encoder"]) == 0.8
    assert float(rows[0]["GradNorm/transition_model"]) == 0.3
    assert float(rows[0]["GradNorm/projection"]) == 0.15
    assert float(rows[0]["GradNorm/predictor"]) == 0.1
    assert float(rows[0]["GradNorm/head"]) == 0.05


def test_write_step_row_missing_metrics_are_empty():
    """Columns with no matching metric get empty string, not KeyError."""
    metrics = {"TotalLoss": 0.5, "GradNorm": 1.0}
    agent = _make_mock_agent(metrics)

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_HEADER, extrasaction="ignore")
    writer.writeheader()
    write_step_row(writer, buf, step=3000, fps=150.0, agent=agent)

    buf.seek(0)
    rows = list(csv.DictReader(buf))
    assert len(rows) == 1
    assert rows[0]["QValueMean"] == ""
    assert rows[0]["GradNorm/encoder"] == ""


# =============================================================================
# validate_checkpoint tests
# =============================================================================


def test_validate_checkpoint_valid(tmp_path):
    """Valid files pass validation."""
    params = tmp_path / "checkpoint_1000.msgpack"
    meta = tmp_path / "checkpoint_1000.json"
    params.write_bytes(b"\x80")
    meta.write_text("{}")

    result = validate_checkpoint(str(params), str(meta))
    assert result["valid"] is True
    assert result["errors"] == []
    assert result["files"]["params"]["exists"] is True
    assert result["files"]["params"]["size"] > 0
    assert result["files"]["metadata"]["exists"] is True


def test_validate_checkpoint_missing_file(tmp_path):
    """Missing file triggers validation failure."""
    params = tmp_path / "checkpoint_1000.msgpack"
    meta = tmp_path / "checkpoint_1000.json"
    params.write_bytes(b"\x80")
    # meta not created

    result = validate_checkpoint(str(params), str(meta))
    assert result["valid"] is False
    assert len(result["errors"]) == 1
    assert "metadata missing" in result["errors"][0]


def test_validate_checkpoint_empty_file(tmp_path):
    """Zero-byte file triggers validation failure."""
    params = tmp_path / "checkpoint_1000.msgpack"
    meta = tmp_path / "checkpoint_1000.json"
    params.write_bytes(b"")  # empty
    meta.write_text("{}")

    result = validate_checkpoint(str(params), str(meta))
    assert result["valid"] is False
    assert len(result["errors"]) == 1
    assert "params empty" in result["errors"][0]


def test_validate_checkpoint_both_missing(tmp_path):
    """Both files missing produces two errors."""
    result = validate_checkpoint(
        str(tmp_path / "nope.msgpack"),
        str(tmp_path / "nope.json"),
    )
    assert result["valid"] is False
    assert len(result["errors"]) == 2


# =============================================================================
# write_progress checkpoint validation tests
# =============================================================================


def test_write_progress_includes_validation(tmp_path):
    """progress.json includes last_checkpoint_validation when provided."""
    validation = {"valid": True, "errors": [], "files": {}}
    write_progress(
        str(tmp_path), step=5000, total_steps=100000,
        episode=10, fps=50.0, start_time=time.time() - 100,
        last_checkpoint_validation=validation,
    )
    with open(tmp_path / "progress.json") as f:
        progress = json.load(f)
    assert "last_checkpoint_validation" in progress
    assert progress["last_checkpoint_validation"]["valid"] is True
    assert progress["last_checkpoint_validation"]["errors"] == []


def test_write_progress_includes_failed_validation(tmp_path):
    """progress.json includes error details on failed validation."""
    validation = {
        "valid": False,
        "errors": ["params missing: /tmp/x.msgpack"],
        "files": {},
    }
    write_progress(
        str(tmp_path), step=5000, total_steps=100000,
        episode=10, fps=50.0, start_time=time.time() - 100,
        last_checkpoint_validation=validation,
    )
    with open(tmp_path / "progress.json") as f:
        progress = json.load(f)
    assert progress["last_checkpoint_validation"]["valid"] is False
    assert len(progress["last_checkpoint_validation"]["errors"]) == 1


def test_write_progress_omits_validation_when_none(tmp_path):
    """progress.json has no validation key when none is provided."""
    write_progress(
        str(tmp_path), step=5000, total_steps=100000,
        episode=10, fps=50.0, start_time=time.time() - 100,
    )
    with open(tmp_path / "progress.json") as f:
        progress = json.load(f)
    assert "last_checkpoint_validation" not in progress
