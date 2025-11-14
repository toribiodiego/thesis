"""Configuration management for DQN training."""

from .config_loader import (
    load_config,
    load_yaml,
    deep_merge,
    apply_overrides,
    parse_nested_key,
    format_config,
    print_config,
    get_nested_value,
    set_nested_value,
    validate_config_exists,
    merge_cli_overrides
)

from .cli import (
    create_parser,
    parse_args,
    load_config_from_args,
    validate_config,
    setup_from_args,
    print_startup_banner,
    main
)

from .run_manager import (
    create_run_dir,
    create_run_subdirs,
    save_config_snapshot,
    get_git_info,
    save_metadata,
    setup_run_directory,
    print_run_info
)

__all__ = [
    # Config loader
    'load_config',
    'load_yaml',
    'deep_merge',
    'apply_overrides',
    'parse_nested_key',
    'format_config',
    'print_config',
    'get_nested_value',
    'set_nested_value',
    'validate_config_exists',
    'merge_cli_overrides',
    # CLI
    'create_parser',
    'parse_args',
    'load_config_from_args',
    'validate_config',
    'setup_from_args',
    'print_startup_banner',
    'main',
    # Run manager
    'create_run_dir',
    'create_run_subdirs',
    'save_config_snapshot',
    'get_git_info',
    'save_metadata',
    'setup_run_directory',
    'print_run_info'
]
