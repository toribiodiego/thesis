"""
Tests for model utility functions.

Verifies:
- model_summary produces correct information
- print_model_summary runs without errors
- assert_output_shape validates correctly
"""

import torch
from src.models.dqn import DQN
from src.utils.model_utils import (
    model_summary,
    print_model_summary,
    assert_output_shape
)


def test_model_summary():
    """Test model_summary returns correct information."""
    model = DQN(num_actions=6)
    summary = model_summary(model, (4, 84, 84))

    # Check summary contains required keys
    assert 'total_params' in summary
    assert 'trainable_params' in summary
    assert 'input_shape' in summary
    assert 'output_shape' in summary
    assert 'layer_info' in summary

    # Check parameter counts are reasonable
    assert summary['total_params'] > 0
    assert summary['trainable_params'] == summary['total_params']

    # Check input shape
    assert summary['input_shape'] == (4, 84, 84)

    # Check output shape (dict with q_values and features)
    assert isinstance(summary['output_shape'], dict)
    assert 'q_values' in summary['output_shape']
    assert 'features' in summary['output_shape']
    assert summary['output_shape']['q_values'] == (1, 6)
    assert summary['output_shape']['features'] == (1, 256)

    # Check layer info
    assert len(summary['layer_info']) > 0
    for layer in summary['layer_info']:
        assert 'name' in layer
        assert 'type' in layer
        assert 'params' in layer


def test_print_model_summary():
    """Test print_model_summary runs without error."""
    model = DQN(num_actions=4)

    # Should not raise any errors
    print_model_summary(model, (4, 84, 84))


def test_assert_output_shape_valid():
    """Test assert_output_shape with correct shape."""
    model = DQN(num_actions=6)

    # Should not raise for correct shape
    assert_output_shape(model, (4, 84, 84), (6,))


def test_assert_output_shape_invalid():
    """Test assert_output_shape raises for incorrect shape."""
    model = DQN(num_actions=6)

    # Should raise for wrong shape
    try:
        assert_output_shape(model, (4, 84, 84), (4,))  # Wrong action count
        assert False, "Should have raised AssertionError"
    except AssertionError as e:
        assert "Expected output shape" in str(e)


def test_model_validate_output_shape():
    """Test DQN.validate_output_shape method."""
    model = DQN(num_actions=6)

    # Should not raise for valid model
    model.validate_output_shape()

    # Test with different batch size
    model.validate_output_shape(batch_size=4)


def test_model_summary_different_action_sizes():
    """Test model summary with different action space sizes."""
    action_sizes = [4, 6, 9, 18]

    for num_actions in action_sizes:
        model = DQN(num_actions=num_actions)
        summary = model_summary(model, (4, 84, 84))

        # Output shape should match action size
        assert summary['output_shape']['q_values'] == (1, num_actions)


if __name__ == "__main__":
    # Run tests manually
    print("Running model utility tests...")

    test_model_summary()
    print("✓ model_summary test passed")

    test_print_model_summary()
    print("✓ print_model_summary test passed")

    test_assert_output_shape_valid()
    print("✓ assert_output_shape (valid) test passed")

    test_assert_output_shape_invalid()
    print("✓ assert_output_shape (invalid) test passed")

    test_model_validate_output_shape()
    print("✓ model.validate_output_shape test passed")

    test_model_summary_different_action_sizes()
    print("✓ model_summary with different action sizes test passed")

    print("\nAll tests passed! ✓")
