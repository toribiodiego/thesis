"""Shared pytest configuration and custom markers."""

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False,
        help="run tests marked @pytest.mark.slow",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "jax: tests that require JAX/Flax (BBF agent, gin configs)",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that require GPU or long JIT compilation",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="needs --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
