"""
Utilities for model inspection and debugging.

Provides tools for summarizing model architecture, counting parameters,
and validating tensor shapes.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


def model_summary(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu'
) -> dict:
    """
    Generate a summary of model architecture and parameters.

    Args:
        model: PyTorch model to summarize
        input_shape: Shape of input tensor (excluding batch dimension)
                    e.g., (4, 84, 84) for DQN
        device: Device to run forward pass on

    Returns:
        Dict containing:
            - total_params: Total number of parameters
            - trainable_params: Number of trainable parameters
            - input_shape: Input shape used
            - output_shape: Output shape from forward pass
            - layer_info: List of dicts with layer details

    Example:
        >>> model = DQN(num_actions=6)
        >>> summary = model_summary(model, (4, 84, 84))
        >>> print(f"Total params: {summary['total_params']:,}")
    """
    model = model.to(device)
    model.eval()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get layer information
    layer_info = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            num_params = sum(p.numel() for p in module.parameters())
            if num_params > 0:
                layer_info.append({
                    'name': name,
                    'type': module.__class__.__name__,
                    'params': num_params
                })

    # Run forward pass to get output shape
    with torch.no_grad():
        dummy_input = torch.zeros(1, *input_shape, device=device)
        output = model(dummy_input)

        # Handle dict output (like DQN)
        if isinstance(output, dict):
            output_shape = {k: tuple(v.shape) for k, v in output.items()}
        else:
            output_shape = tuple(output.shape)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'layer_info': layer_info
    }


def print_model_summary(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    device: str = 'cpu'
):
    """
    Print a formatted model summary.

    Args:
        model: PyTorch model to summarize
        input_shape: Shape of input tensor (excluding batch)
        device: Device for forward pass
    """
    summary = model_summary(model, input_shape, device)

    print("=" * 70)
    print(f"Model: {model.__class__.__name__}")
    print("=" * 70)
    print(f"Input shape:  {summary['input_shape']}")
    print(f"Output shape: {summary['output_shape']}")
    print("-" * 70)

    print(f"{'Layer':<30} {'Type':<20} {'Params':>15}")
    print("-" * 70)
    for layer in summary['layer_info']:
        print(f"{layer['name']:<30} {layer['type']:<20} {layer['params']:>15,}")

    print("=" * 70)
    print(f"Total params:     {summary['total_params']:>15,}")
    print(f"Trainable params: {summary['trainable_params']:>15,}")
    print("=" * 70)


def assert_output_shape(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    expected_output_shape: Tuple[int, ...],
    batch_size: int = 2
):
    """
    Assert that model produces expected output shape.

    Args:
        model: PyTorch model
        input_shape: Input shape without batch dimension
        expected_output_shape: Expected output shape without batch dimension
        batch_size: Batch size to test with

    Raises:
        AssertionError: If output shape doesn't match expected

    Example:
        >>> model = DQN(num_actions=6)
        >>> assert_output_shape(model, (4, 84, 84), (6,))
    """
    model.eval()
    with torch.no_grad():
        x = torch.randn(batch_size, *input_shape)
        output = model(x)

        # Handle dict output
        if isinstance(output, dict):
            if 'q_values' in output:
                actual_shape = output['q_values'].shape[1:]
            else:
                raise ValueError("Dict output must contain 'q_values' key")
        else:
            actual_shape = output.shape[1:]

        assert actual_shape == expected_output_shape, \
            f"Expected output shape {expected_output_shape}, got {actual_shape}"
