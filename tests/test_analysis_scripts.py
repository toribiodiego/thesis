"""Tests for analysis script configuration and setup.

Verifies that all analysis scripts set required environment
variables before importing JAX, and that helper functions in
individual scripts handle edge cases correctly.
"""

import ast
import os

import pytest


SCRIPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "analysis",
)

ANALYSIS_SCRIPTS = [
    "run_all.py",
    "run_cka.py",
    "run_filter_frequency.py",
    "run_inverse_dynamics.py",
    "run_probing.py",
    "run_q_accuracy.py",
    "run_reward_probing.py",
    "run_structural_health.py",
    "run_transition_eval.py",
]


class TestJaxPreallocation:
    """JAX pre-allocates 75% of GPU VRAM by default. All analysis
    scripts must set XLA_PYTHON_CLIENT_PREALLOCATE=false before
    importing JAX to prevent OOM on T4."""

    @pytest.mark.parametrize("script", ANALYSIS_SCRIPTS)
    def test_preallocate_set_before_jax_import(self, script):
        path = os.path.join(SCRIPTS_DIR, script)
        with open(path) as f:
            source = f.read()

        tree = ast.parse(source)

        env_set_line = None
        first_jax_line = None

        for node in ast.walk(tree):
            # Find os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            if (isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Constant)
                    and node.value.value == "false"):
                for target in node.targets:
                    src_segment = ast.get_source_segment(source, target)
                    if src_segment and "XLA_PYTHON_CLIENT_PREALLOCATE" in src_segment:
                        env_set_line = node.lineno

            # Find first import of jax or flax
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("jax", "flax")):
                        if first_jax_line is None or node.lineno < first_jax_line:
                            first_jax_line = node.lineno
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("jax", "flax")):
                    if first_jax_line is None or node.lineno < first_jax_line:
                        first_jax_line = node.lineno

        assert env_set_line is not None, (
            f"{script} does not set XLA_PYTHON_CLIENT_PREALLOCATE"
        )

        # Scripts that don't directly import JAX (they import it
        # indirectly via src.analysis) only need the env var set;
        # the ordering check is for scripts with direct imports.
        if first_jax_line is not None:
            assert env_set_line < first_jax_line, (
                f"{script} sets XLA_PYTHON_CLIENT_PREALLOCATE on "
                f"line {env_set_line} but imports JAX on line "
                f"{first_jax_line} (must be before)"
            )
