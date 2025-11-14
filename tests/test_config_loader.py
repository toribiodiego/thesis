"""Tests for configuration loading and merging."""

import pytest
import yaml
import tempfile
from pathlib import Path

from src.config import (
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


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def temp_config_dir(tmp_path):
    """Create temporary directory for config files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    return config_dir


@pytest.fixture
def base_config_file(temp_config_dir):
    """Create a temporary base config file."""
    base_config = {
        'experiment': {
            'name': 'base_experiment',
            'seed': 42
        },
        'training': {
            'lr': 0.001,
            'gamma': 0.99,
            'optimizer': {
                'type': 'adam',
                'eps': 1e-8
            }
        },
        'network': {
            'hidden': 256
        }
    }
    
    base_file = temp_config_dir / "base.yaml"
    with open(base_file, 'w') as f:
        yaml.dump(base_config, f)
    
    return base_file


@pytest.fixture
def game_config_file(temp_config_dir, base_config_file):
    """Create a temporary game config file that inherits from base."""
    game_config = {
        'base_config': str(base_config_file),
        'experiment': {
            'name': 'pong'
        },
        'training': {
            'lr': 0.0001
        }
    }
    
    game_file = temp_config_dir / "pong.yaml"
    with open(game_file, 'w') as f:
        yaml.dump(game_config, f)
    
    return game_file


# =============================================================================
# Deep Merge Tests
# =============================================================================

def test_deep_merge_basic():
    """Test basic dictionary merging."""
    base = {'a': 1, 'b': 2}
    override = {'b': 3, 'c': 4}
    
    result = deep_merge(base, override)
    
    assert result == {'a': 1, 'b': 3, 'c': 4}
    # Ensure originals unchanged
    assert base == {'a': 1, 'b': 2}
    assert override == {'b': 3, 'c': 4}


def test_deep_merge_nested():
    """Test nested dictionary merging."""
    base = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3
        }
    }
    override = {
        'b': {
            'c': 99
        },
        'e': 4
    }
    
    result = deep_merge(base, override)
    
    assert result == {
        'a': 1,
        'b': {
            'c': 99,
            'd': 3
        },
        'e': 4
    }


def test_deep_merge_deep_nesting():
    """Test deeply nested dictionary merging."""
    base = {
        'level1': {
            'level2': {
                'level3': {
                    'a': 1,
                    'b': 2
                },
                'x': 10
            }
        }
    }
    override = {
        'level1': {
            'level2': {
                'level3': {
                    'b': 99
                }
            }
        }
    }
    
    result = deep_merge(base, override)
    
    assert result['level1']['level2']['level3'] == {'a': 1, 'b': 99}
    assert result['level1']['level2']['x'] == 10


def test_deep_merge_override_with_dict():
    """Test overriding non-dict with dict."""
    base = {'a': 1}
    override = {'a': {'b': 2}}
    
    result = deep_merge(base, override)
    
    assert result == {'a': {'b': 2}}


def test_deep_merge_override_dict_with_value():
    """Test overriding dict with non-dict value."""
    base = {'a': {'b': 2}}
    override = {'a': 1}
    
    result = deep_merge(base, override)
    
    assert result == {'a': 1}


# =============================================================================
# Nested Key Parsing Tests
# =============================================================================

def test_parse_nested_key_single():
    """Test parsing single key (no nesting)."""
    result = parse_nested_key('key', 'value')
    assert result == {'key': 'value'}


def test_parse_nested_key_two_levels():
    """Test parsing two-level nested key."""
    result = parse_nested_key('a.b', 10)
    assert result == {'a': {'b': 10}}


def test_parse_nested_key_three_levels():
    """Test parsing three-level nested key."""
    result = parse_nested_key('a.b.c', 42)
    assert result == {'a': {'b': {'c': 42}}}


def test_parse_nested_key_with_number():
    """Test parsing nested key with numeric value."""
    result = parse_nested_key('training.lr', 0.001)
    assert result == {'training': {'lr': 0.001}}


# =============================================================================
# Apply Overrides Tests
# =============================================================================

def test_apply_overrides_simple():
    """Test applying simple (non-nested) overrides."""
    config = {'a': 1, 'b': 2}
    overrides = {'b': 99}
    
    result = apply_overrides(config, overrides)
    
    assert result == {'a': 1, 'b': 99}
    assert config == {'a': 1, 'b': 2}  # Original unchanged


def test_apply_overrides_nested_key():
    """Test applying overrides with dot notation."""
    config = {
        'training': {
            'lr': 0.001,
            'gamma': 0.99
        }
    }
    overrides = {
        'training.lr': 0.0001
    }
    
    result = apply_overrides(config, overrides)
    
    assert result['training']['lr'] == 0.0001
    assert result['training']['gamma'] == 0.99


def test_apply_overrides_multiple_nested():
    """Test applying multiple nested overrides."""
    config = {
        'training': {
            'lr': 0.001,
            'optimizer': {
                'type': 'adam',
                'eps': 1e-8
            }
        }
    }
    overrides = {
        'training.lr': 0.0001,
        'training.optimizer.eps': 1e-2
    }
    
    result = apply_overrides(config, overrides)
    
    assert result['training']['lr'] == 0.0001
    assert result['training']['optimizer']['type'] == 'adam'
    assert result['training']['optimizer']['eps'] == 1e-2


def test_apply_overrides_new_keys():
    """Test adding new keys via overrides."""
    config = {'a': 1}
    overrides = {
        'b': 2,
        'c.d': 3
    }
    
    result = apply_overrides(config, overrides)
    
    assert result == {
        'a': 1,
        'b': 2,
        'c': {'d': 3}
    }


# =============================================================================
# Load YAML Tests
# =============================================================================

def test_load_yaml_success(temp_config_dir):
    """Test loading valid YAML file."""
    config_file = temp_config_dir / "test.yaml"
    config_data = {'key': 'value', 'number': 42}
    
    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)
    
    result = load_yaml(str(config_file))
    
    assert result == config_data


def test_load_yaml_file_not_found():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_yaml('/nonexistent/path/config.yaml')


def test_load_yaml_invalid_yaml(temp_config_dir):
    """Test loading invalid YAML."""
    config_file = temp_config_dir / "invalid.yaml"
    
    with open(config_file, 'w') as f:
        f.write("invalid: yaml: content: [")
    
    with pytest.raises(yaml.YAMLError):
        load_yaml(str(config_file))


def test_load_yaml_empty_file(temp_config_dir):
    """Test loading empty YAML file."""
    config_file = temp_config_dir / "empty.yaml"
    
    with open(config_file, 'w') as f:
        f.write("")
    
    result = load_yaml(str(config_file))
    
    assert result == {}


# =============================================================================
# Load Config Tests
# =============================================================================

def test_load_config_simple(base_config_file):
    """Test loading simple config without base."""
    config = load_config(str(base_config_file), resolve_base=False)
    
    assert config['experiment']['name'] == 'base_experiment'
    assert config['training']['lr'] == 0.001


def test_load_config_with_base(game_config_file):
    """Test loading config that inherits from base."""
    config = load_config(str(game_config_file))
    
    # Should have merged base + game
    assert config['experiment']['name'] == 'pong'  # Overridden
    assert config['experiment']['seed'] == 42  # From base
    assert config['training']['lr'] == 0.0001  # Overridden
    assert config['training']['gamma'] == 0.99  # From base
    assert config['network']['hidden'] == 256  # From base


def test_load_config_no_base_key(base_config_file):
    """Test loading config without base_config key."""
    config = load_config(str(base_config_file))
    
    # Should work normally, no base merging
    assert config['experiment']['name'] == 'base_experiment'


def test_load_config_with_overrides(game_config_file):
    """Test loading config with additional overrides."""
    overrides = {
        'experiment.seed': 123,
        'training.gamma': 0.95
    }
    
    config = load_config(str(game_config_file), overrides=overrides)
    
    assert config['experiment']['seed'] == 123
    assert config['training']['gamma'] == 0.95
    assert config['experiment']['name'] == 'pong'


def test_load_config_nested_overrides(game_config_file):
    """Test loading config with deep nested overrides."""
    overrides = {
        'training.optimizer.type': 'rmsprop',
        'training.optimizer.eps': 1e-2
    }
    
    config = load_config(str(game_config_file), overrides=overrides)
    
    assert config['training']['optimizer']['type'] == 'rmsprop'
    assert config['training']['optimizer']['eps'] == 1e-2


# =============================================================================
# Get/Set Nested Value Tests
# =============================================================================

def test_get_nested_value_simple():
    """Test getting simple (non-nested) value."""
    config = {'a': 1, 'b': 2}
    
    assert get_nested_value(config, 'a') == 1


def test_get_nested_value_two_levels():
    """Test getting two-level nested value."""
    config = {'a': {'b': 10}}
    
    assert get_nested_value(config, 'a.b') == 10


def test_get_nested_value_three_levels():
    """Test getting three-level nested value."""
    config = {'a': {'b': {'c': 42}}}
    
    assert get_nested_value(config, 'a.b.c') == 42


def test_get_nested_value_not_found():
    """Test getting non-existent nested value."""
    config = {'a': 1}
    
    assert get_nested_value(config, 'b', default=99) == 99
    assert get_nested_value(config, 'a.b', default=None) is None


def test_set_nested_value_simple():
    """Test setting simple (non-nested) value."""
    config = {'a': 1}
    
    set_nested_value(config, 'a', 99)
    
    assert config['a'] == 99


def test_set_nested_value_two_levels():
    """Test setting two-level nested value."""
    config = {'a': {'b': 10}}
    
    set_nested_value(config, 'a.b', 99)
    
    assert config['a']['b'] == 99


def test_set_nested_value_create_path():
    """Test setting value creates intermediate dicts."""
    config = {}
    
    set_nested_value(config, 'a.b.c', 42)
    
    assert config == {'a': {'b': {'c': 42}}}


# =============================================================================
# Validate Config Exists Tests
# =============================================================================

def test_validate_config_exists_success(base_config_file):
    """Test validating existing config file."""
    path = validate_config_exists(str(base_config_file))
    
    assert path.exists()
    assert path.is_file()


def test_validate_config_exists_not_found():
    """Test validating non-existent config file."""
    with pytest.raises(FileNotFoundError):
        validate_config_exists('/nonexistent/config.yaml')


def test_validate_config_exists_is_directory(temp_config_dir):
    """Test validating directory instead of file."""
    with pytest.raises(ValueError):
        validate_config_exists(str(temp_config_dir))


# =============================================================================
# Merge CLI Overrides Tests
# =============================================================================

def test_merge_cli_overrides_simple():
    """Test merging simple CLI overrides."""
    config = {'a': 1, 'b': 2}
    cli_args = ['a=99']
    
    result = merge_cli_overrides(config, cli_args)
    
    assert result['a'] == 99
    assert result['b'] == 2


def test_merge_cli_overrides_nested():
    """Test merging nested CLI overrides."""
    config = {'training': {'lr': 0.001}}
    cli_args = ['training.lr=0.0001']
    
    result = merge_cli_overrides(config, cli_args)
    
    assert result['training']['lr'] == 0.0001


def test_merge_cli_overrides_multiple():
    """Test merging multiple CLI overrides."""
    config = {'a': 1, 'b': 2}
    cli_args = ['a=99', 'c=3']
    
    result = merge_cli_overrides(config, cli_args)
    
    assert result['a'] == 99
    assert result['b'] == 2
    assert result['c'] == 3


def test_merge_cli_overrides_types():
    """Test CLI overrides with different types."""
    config = {}
    cli_args = [
        'int_val=42',
        'float_val=3.14',
        'bool_val=true',
        'str_val=hello',
        'list_val=[1, 2, 3]'
    ]
    
    result = merge_cli_overrides(config, cli_args)
    
    assert result['int_val'] == 42
    assert result['float_val'] == 3.14
    assert result['bool_val'] is True
    assert result['str_val'] == 'hello'
    assert result['list_val'] == [1, 2, 3]


def test_merge_cli_overrides_invalid_format():
    """Test CLI override with invalid format."""
    config = {}
    cli_args = ['invalid_no_equals']
    
    with pytest.raises(ValueError, match="Invalid CLI override format"):
        merge_cli_overrides(config, cli_args)


# =============================================================================
# Format/Print Config Tests
# =============================================================================

def test_format_config():
    """Test formatting config as YAML string."""
    config = {
        'a': 1,
        'b': {
            'c': 2,
            'd': 3
        }
    }
    
    result = format_config(config)
    
    assert isinstance(result, str)
    assert 'a:' in result
    assert 'b:' in result
    # Verify it's valid YAML
    parsed = yaml.safe_load(result)
    assert parsed == config


def test_print_config(capsys):
    """Test printing config to stdout."""
    config = {'test': 'value'}
    
    print_config(config, title="Test Config")
    
    captured = capsys.readouterr()
    assert "Test Config" in captured.out
    assert "test:" in captured.out
    assert "===" in captured.out


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_workflow_with_real_configs(temp_config_dir):
    """Test complete config loading workflow."""
    # Create base config
    base_config = {
        'experiment': {'name': 'base', 'seed': 42},
        'training': {
            'lr': 0.001,
            'gamma': 0.99,
            'optimizer': {'type': 'adam'}
        }
    }
    base_file = temp_config_dir / "base.yaml"
    with open(base_file, 'w') as f:
        yaml.dump(base_config, f)
    
    # Create game config
    game_config = {
        'base_config': str(base_file),
        'experiment': {'name': 'pong'},
        'training': {'lr': 0.0001}
    }
    game_file = temp_config_dir / "pong.yaml"
    with open(game_file, 'w') as f:
        yaml.dump(game_config, f)
    
    # Load with CLI overrides
    cli_args = ['experiment.seed=123', 'training.gamma=0.95']
    
    config = load_config(str(game_file))
    config = merge_cli_overrides(config, cli_args)
    
    # Verify final merged config
    assert config['experiment']['name'] == 'pong'
    assert config['experiment']['seed'] == 123
    assert config['training']['lr'] == 0.0001
    assert config['training']['gamma'] == 0.95
    assert config['training']['optimizer']['type'] == 'adam'
