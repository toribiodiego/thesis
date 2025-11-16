"""
Unit tests for determinism configuration utilities.

Tests configure_determinism() function with various settings:
- Basic deterministic mode (cuDNN settings)
- Strict deterministic algorithms
- Warn-only mode
- Status checking
- Config integration
"""

import pytest
import torch

from src.utils import configure_determinism, get_determinism_status


class TestConfigureDeterminism:
    """Test configure_determinism() function."""

    def setup_method(self):
        """Reset determinism settings before each test."""
        # Reset to defaults
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)

    def test_enable_basic_determinism(self):
        """Test enabling basic deterministic mode."""
        settings = configure_determinism(enabled=True, strict=False)

        assert settings["cudnn_deterministic"] is True
        assert settings["cudnn_benchmark"] is False
        assert settings["strict_algorithms"] is False

        # Verify actual PyTorch settings
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_disable_determinism(self):
        """Test disabling deterministic mode."""
        # First enable it
        configure_determinism(enabled=True)

        # Then disable it
        settings = configure_determinism(enabled=False)

        assert settings["cudnn_deterministic"] is False
        assert settings["cudnn_benchmark"] is True

        # Verify actual PyTorch settings
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.benchmark is True

    def test_strict_determinism(self):
        """Test strict deterministic algorithms mode."""
        if not hasattr(torch, "use_deterministic_algorithms"):
            pytest.skip("torch.use_deterministic_algorithms not available")

        settings = configure_determinism(enabled=True, strict=True, warn_only=False)

        assert settings["cudnn_deterministic"] is True
        assert settings["cudnn_benchmark"] is False
        assert settings["strict_algorithms"] is True

        # Verify strict algorithms are enabled
        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            assert torch.are_deterministic_algorithms_enabled() is True

    def test_strict_with_warn_only(self):
        """Test strict mode with warn_only parameter."""
        if not hasattr(torch, "use_deterministic_algorithms"):
            pytest.skip("torch.use_deterministic_algorithms not available")

        settings = configure_determinism(enabled=True, strict=True, warn_only=True)

        assert settings["cudnn_deterministic"] is True
        assert settings["strict_algorithms"] is True

        # warn_only may or may not be supported depending on PyTorch version
        assert "warn_only" in settings

    def test_disable_strict_after_enabling(self):
        """Test that strict mode can be disabled after enabling."""
        if not hasattr(torch, "use_deterministic_algorithms"):
            pytest.skip("torch.use_deterministic_algorithms not available")

        # Enable strict
        configure_determinism(enabled=True, strict=True)

        # Disable strict
        settings = configure_determinism(enabled=True, strict=False)

        assert settings["strict_algorithms"] is False

        # Verify strict algorithms are disabled
        if hasattr(torch, "are_deterministic_algorithms_enabled"):
            assert torch.are_deterministic_algorithms_enabled() is False

    def test_default_parameters(self):
        """Test default parameter values."""
        settings = configure_determinism()

        # Default is enabled=True
        assert settings["cudnn_deterministic"] is True
        assert settings["cudnn_benchmark"] is False
        assert settings["strict_algorithms"] is False
        assert settings["warn_only"] is True

    def test_returns_applied_settings(self):
        """Test that function returns dict of applied settings."""
        settings = configure_determinism(enabled=True, strict=False, warn_only=True)

        assert isinstance(settings, dict)
        assert "cudnn_deterministic" in settings
        assert "cudnn_benchmark" in settings
        assert "strict_algorithms" in settings
        assert "warn_only" in settings


class TestGetDeterminismStatus:
    """Test get_determinism_status() function."""

    def setup_method(self):
        """Reset determinism settings before each test."""
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)

    def test_get_status_disabled(self):
        """Test status when determinism is disabled."""
        configure_determinism(enabled=False)
        status = get_determinism_status()

        assert status["cudnn_deterministic"] is False
        assert status["cudnn_benchmark"] is True
        assert status["strict_algorithms"] is False

    def test_get_status_enabled(self):
        """Test status when determinism is enabled."""
        configure_determinism(enabled=True, strict=False)
        status = get_determinism_status()

        assert status["cudnn_deterministic"] is True
        assert status["cudnn_benchmark"] is False
        assert status["strict_algorithms"] is False

    def test_get_status_strict(self):
        """Test status when strict mode is enabled."""
        if not hasattr(torch, "use_deterministic_algorithms"):
            pytest.skip("torch.use_deterministic_algorithms not available")

        configure_determinism(enabled=True, strict=True)
        status = get_determinism_status()

        assert status["cudnn_deterministic"] is True
        assert status["strict_algorithms"] is True

    def test_status_reflects_manual_changes(self):
        """Test that status reflects manual changes to PyTorch settings."""
        # Manually change settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        status = get_determinism_status()

        assert status["cudnn_deterministic"] is True
        assert status["cudnn_benchmark"] is False


class TestDeterminismReproducibility:
    """Test that determinism settings actually affect reproducibility."""

    def setup_method(self):
        """Reset settings before each test."""
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)

    def test_determinism_affects_cudnn_operations(self):
        """Test that deterministic mode affects cuDNN operations."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        # Enable determinism
        configure_determinism(enabled=True)

        # Create input and run multiple times
        torch.manual_seed(42)
        torch.randn(10, 10, device="cuda")
        torch.randn(10, 10, device="cuda")

        results = []
        for _ in range(3):
            torch.manual_seed(42)
            x_test = torch.randn(10, 10, device="cuda")
            y_test = torch.randn(10, 10, device="cuda")
            result = torch.matmul(x_test, y_test)
            results.append(result)

        # All results should be identical with determinism enabled
        for i in range(1, len(results)):
            assert torch.allclose(results[0], results[i], rtol=1e-6, atol=1e-8)

    def test_benchmark_disabled_in_deterministic_mode(self):
        """Test that benchmark is disabled when determinism is enabled."""
        configure_determinism(enabled=True)

        assert torch.backends.cudnn.benchmark is False
        assert torch.backends.cudnn.deterministic is True

    def test_benchmark_enabled_when_determinism_disabled(self):
        """Test that benchmark is enabled when determinism is disabled."""
        configure_determinism(enabled=False)

        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False


class TestConfigIntegration:
    """Test integration with config-based setup."""

    def setup_method(self):
        """Reset settings before each test."""
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)

    def test_config_based_setup_disabled(self):
        """Test config-based setup with determinism disabled."""
        # Simulate config
        config = {
            "experiment": {
                "deterministic": {"enabled": False, "strict": False, "warn_only": True}
            }
        }

        # Apply config
        det_config = config["experiment"]["deterministic"]
        settings = configure_determinism(
            enabled=det_config["enabled"],
            strict=det_config["strict"],
            warn_only=det_config["warn_only"],
        )

        assert settings["cudnn_deterministic"] is False
        assert settings["cudnn_benchmark"] is True

    def test_config_based_setup_enabled(self):
        """Test config-based setup with determinism enabled."""
        # Simulate config
        config = {
            "experiment": {
                "deterministic": {"enabled": True, "strict": False, "warn_only": True}
            }
        }

        # Apply config
        det_config = config["experiment"]["deterministic"]
        settings = configure_determinism(
            enabled=det_config["enabled"],
            strict=det_config["strict"],
            warn_only=det_config["warn_only"],
        )

        assert settings["cudnn_deterministic"] is True
        assert settings["cudnn_benchmark"] is False
        assert settings["strict_algorithms"] is False

    def test_config_based_setup_strict(self):
        """Test config-based setup with strict mode."""
        if not hasattr(torch, "use_deterministic_algorithms"):
            pytest.skip("torch.use_deterministic_algorithms not available")

        # Simulate config
        config = {
            "experiment": {
                "deterministic": {"enabled": True, "strict": True, "warn_only": True}
            }
        }

        # Apply config
        det_config = config["experiment"]["deterministic"]
        settings = configure_determinism(
            enabled=det_config["enabled"],
            strict=det_config["strict"],
            warn_only=det_config["warn_only"],
        )

        assert settings["cudnn_deterministic"] is True
        assert settings["strict_algorithms"] is True


class TestEdgeCases:
    """Test edge cases and error handling."""

    def setup_method(self):
        """Reset settings before each test."""
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(False)

    def test_multiple_calls_are_idempotent(self):
        """Test that calling configure_determinism multiple times is safe."""
        # Call multiple times with same settings
        for _ in range(3):
            settings = configure_determinism(enabled=True, strict=False)
            assert settings["cudnn_deterministic"] is True

        # Settings should still be correct
        status = get_determinism_status()
        assert status["cudnn_deterministic"] is True

    def test_alternating_enable_disable(self):
        """Test alternating between enabled and disabled states."""
        for i in range(3):
            enabled = i % 2 == 0
            settings = configure_determinism(enabled=enabled)

            assert settings["cudnn_deterministic"] == enabled
            assert settings["cudnn_benchmark"] != enabled

    def test_status_dict_has_required_keys(self):
        """Test that status dict always has required keys."""
        status = get_determinism_status()

        required_keys = ["cudnn_deterministic", "cudnn_benchmark", "strict_algorithms"]
        for key in required_keys:
            assert key in status


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
