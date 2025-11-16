"""Configuration loading and merging utilities.

Provides functionality for:
- Loading YAML config files
- Merging base + game-specific overrides
- Supporting nested key overrides (e.g., "training.optimizer.lr")
- Printing resolved configuration for traceability
- Converting to dict/dataclass
"""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Recursively merges nested dictionaries. For non-dict values, override
    takes precedence over base.

    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary

    Returns:
        Merged dictionary (deep copy, original dicts unchanged)

    Example:
        >>> base = {'a': 1, 'b': {'c': 2, 'd': 3}}
        >>> override = {'b': {'c': 99}, 'e': 4}
        >>> deep_merge(base, override)
        {'a': 1, 'b': {'c': 99, 'd': 3}, 'e': 4}
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = deep_merge(result[key], value)
        else:
            # Override takes precedence for non-dict values
            result[key] = deepcopy(value)

    return result


def parse_nested_key(key: str, value: Any) -> Dict[str, Any]:
    """
    Parse a nested key string into a nested dictionary.

    Supports dot notation for nested keys (e.g., "training.optimizer.lr").

    Args:
        key: Key string, potentially with dots for nesting
        value: Value to assign to the leaf key

    Returns:
        Nested dictionary structure

    Example:
        >>> parse_nested_key("training.optimizer.lr", 0.001)
        {'training': {'optimizer': {'lr': 0.001}}}
    """
    parts = key.split(".")
    result = {}
    current = result

    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            # Leaf node - assign value
            current[part] = value
        else:
            # Intermediate node - create nested dict
            current[part] = {}
            current = current[part]

    return result


def apply_overrides(
    config: Dict[str, Any], overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply nested key overrides to a configuration.

    Supports both regular dict keys and dot-notation nested keys.

    Args:
        config: Base configuration dictionary
        overrides: Dictionary of overrides (may contain dot-notation keys)

    Returns:
        Configuration with overrides applied

    Example:
        >>> config = {'training': {'lr': 0.001, 'gamma': 0.99}}
        >>> overrides = {'training.lr': 0.0001}
        >>> apply_overrides(config, overrides)
        {'training': {'lr': 0.0001, 'gamma': 0.99}}
    """
    result = deepcopy(config)

    for key, value in overrides.items():
        if "." in key:
            # Nested key - parse and merge
            nested = parse_nested_key(key, value)
            result = deep_merge(result, nested)
        else:
            # Regular key - direct assignment
            result[key] = deepcopy(value)

    return result


def load_yaml(path: str) -> Dict[str, Any]:
    """
    Load a YAML file and return as dictionary.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary containing YAML contents

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If file is not valid YAML
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        try:
            config = yaml.safe_load(f)
            if config is None:
                config = {}
            return config
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file {path}: {e}")


def load_config(
    config_path: str,
    overrides: Optional[Dict[str, Any]] = None,
    resolve_base: bool = True,
) -> Dict[str, Any]:
    """
    Load configuration file with optional base config merging and overrides.

    Supports:
    - Loading YAML config files
    - Automatic base config resolution (if 'base_config' key present)
    - Deep merging of base + game configs
    - Nested key overrides (dot notation)

    Args:
        config_path: Path to main configuration file
        overrides: Optional dictionary of overrides to apply
        resolve_base: If True, automatically load and merge base_config

    Returns:
        Merged and resolved configuration dictionary

    Example:
        >>> # Load pong.yaml (automatically merges base.yaml)
        >>> config = load_config("experiments/dqn_atari/configs/pong.yaml")
        >>>
        >>> # Load with overrides
        >>> config = load_config(
        ...     "experiments/dqn_atari/configs/pong.yaml",
        ...     overrides={"training.lr": 0.0001, "seed.value": 42}
        ... )
    """
    # Load main config
    config = load_yaml(config_path)

    # Check for base config
    if resolve_base and "base_config" in config:
        base_path = config.pop("base_config")

        # Resolve relative path (relative to config file location OR project root)
        base_path_obj = Path(base_path)

        if not base_path_obj.is_absolute():
            # Try relative to config file first
            config_dir = Path(config_path).parent
            candidate = config_dir / base_path

            if not candidate.exists():
                # Try relative to current working directory
                candidate = Path.cwd() / base_path

            base_path = candidate

        # Load base config (recursive - handles nested base configs)
        base_config = load_config(str(base_path), overrides=None, resolve_base=True)

        # Merge base + override (game config overrides base)
        config = deep_merge(base_config, config)

    # Apply additional overrides if provided
    if overrides:
        config = apply_overrides(config, overrides)

    return config


def format_config(
    config: Dict[str, Any], indent: int = 2, max_line_length: int = 100
) -> str:
    """
    Format configuration dictionary as readable YAML string.

    Args:
        config: Configuration dictionary
        indent: Number of spaces for indentation
        max_line_length: Maximum line length before wrapping

    Returns:
        Formatted YAML string
    """
    return yaml.dump(
        config,
        default_flow_style=False,
        sort_keys=False,
        indent=indent,
        width=max_line_length,
    )


def print_config(
    config: Dict[str, Any], title: str = "Resolved Configuration", width: int = 80
) -> None:
    """
    Print configuration in a formatted, readable way.

    Args:
        config: Configuration dictionary to print
        title: Title for the configuration section
        width: Width of separator lines

    Example:
        >>> config = load_config("experiments/dqn_atari/configs/pong.yaml")
        >>> print_config(config, title="Pong Configuration")
    """
    separator = "=" * width

    print(f"\n{separator}")
    print(f"{title:^{width}}")
    print(f"{separator}\n")

    formatted = format_config(config)
    print(formatted)

    print(f"{separator}\n")


def get_nested_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    """
    Get value from nested dictionary using dot notation.

    Args:
        config: Configuration dictionary
        key: Nested key in dot notation (e.g., "training.optimizer.lr")
        default: Default value if key not found

    Returns:
        Value at the nested key, or default if not found

    Example:
        >>> config = {'training': {'optimizer': {'lr': 0.001}}}
        >>> get_nested_value(config, 'training.optimizer.lr')
        0.001
        >>> get_nested_value(config, 'training.batch_size', default=32)
        32
    """
    parts = key.split(".")
    current = config

    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default

    return current


def set_nested_value(config: Dict[str, Any], key: str, value: Any) -> None:
    """
    Set value in nested dictionary using dot notation (in-place).

    Args:
        config: Configuration dictionary (modified in-place)
        key: Nested key in dot notation (e.g., "training.optimizer.lr")
        value: Value to set

    Example:
        >>> config = {'training': {'optimizer': {'lr': 0.001}}}
        >>> set_nested_value(config, 'training.optimizer.lr', 0.0001)
        >>> config
        {'training': {'optimizer': {'lr': 0.0001}}}
    """
    parts = key.split(".")
    current = config

    for i, part in enumerate(parts[:-1]):
        if part not in current:
            current[part] = {}
        current = current[part]

    current[parts[-1]] = value


def validate_config_exists(config_path: str) -> Path:
    """
    Validate that a config file exists and return Path object.

    Args:
        config_path: Path to config file

    Returns:
        Validated Path object

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            f"Absolute path: {path.absolute()}"
        )

    if not path.is_file():
        raise ValueError(f"Configuration path is not a file: {config_path}")

    return path


def merge_cli_overrides(config: Dict[str, Any], cli_args: List[str]) -> Dict[str, Any]:
    """
    Merge CLI override arguments into configuration.

    Parses CLI arguments in the format "key=value" or "key.nested=value"
    and applies them as overrides to the configuration.

    Args:
        config: Base configuration dictionary
        cli_args: List of CLI override strings (e.g., ["training.lr=0.001", "seed.value=42"])

    Returns:
        Configuration with CLI overrides applied

    Example:
        >>> config = {'training': {'lr': 0.0001}}
        >>> cli_args = ["training.lr=0.001", "training.gamma=0.95"]
        >>> merge_cli_overrides(config, cli_args)
        {'training': {'lr': 0.001, 'gamma': 0.95}}
    """
    overrides = {}

    for arg in cli_args:
        if "=" not in arg:
            raise ValueError(
                f"Invalid CLI override format: '{arg}'\n"
                f"Expected format: key=value or key.nested=value"
            )

        key, value_str = arg.split("=", 1)
        key = key.strip()
        value_str = value_str.strip()

        # Try to parse value as YAML (supports int, float, bool, list, etc.)
        try:
            value = yaml.safe_load(value_str)
        except yaml.YAMLError:
            # If parsing fails, treat as string
            value = value_str

        overrides[key] = value

    return apply_overrides(config, overrides)
