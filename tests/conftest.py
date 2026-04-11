"""Shared pytest configuration and custom markers."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "jax: tests that require JAX/Flax (BBF agent, gin configs)",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests that require GPU or long JIT compilation",
    )
