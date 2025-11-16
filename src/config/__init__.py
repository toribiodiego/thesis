"""Configuration management for DQN training."""

from .cli import (
    create_parser,
    load_config_from_args,
    main,
    parse_args,
    print_startup_banner,
    setup_from_args,
    validate_config,
)
from .config_loader import (
    apply_overrides,
    deep_merge,
    format_config,
    get_nested_value,
    load_config,
    load_yaml,
    merge_cli_overrides,
    parse_nested_key,
    print_config,
    set_nested_value,
    validate_config_exists,
)
from .run_manager import (
    create_run_dir,
    create_run_subdirs,
    get_git_info,
    print_run_info,
    save_config_snapshot,
    save_metadata,
    setup_run_directory,
)
from .schema_validator import ConfigValidationError
from .schema_validator import validate_config as validate_config_schema
from .schema_validator import validate_config_safe

__all__ = [
    # Config loader
    "load_config",
    "load_yaml",
    "deep_merge",
    "apply_overrides",
    "parse_nested_key",
    "format_config",
    "print_config",
    "get_nested_value",
    "set_nested_value",
    "validate_config_exists",
    "merge_cli_overrides",
    # CLI
    "create_parser",
    "parse_args",
    "load_config_from_args",
    "validate_config",
    "setup_from_args",
    "print_startup_banner",
    "main",
    # Run manager
    "create_run_dir",
    "create_run_subdirs",
    "save_config_snapshot",
    "get_git_info",
    "save_metadata",
    "setup_run_directory",
    "print_run_info",
    # Schema validator
    "validate_config_schema",
    "validate_config_safe",
    "ConfigValidationError",
]
