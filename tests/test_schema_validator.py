"""Tests for configuration schema validation."""

import pytest
from src.config.schema_validator import (
    validate_config,
    validate_config_safe,
    ConfigValidationError,
    validate_experiment,
    validate_environment,
    validate_network,
    validate_replay,
    validate_training,
    validate_target_network,
    validate_exploration,
    validate_evaluation,
    validate_logging,
    validate_system,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def valid_config():
    """Minimal valid configuration."""
    return {
        'experiment': {
            'name': 'test_experiment',
            'notes': 'Test run'
        },
        'environment': {
            'env_id': 'PongNoFrameskip-v4',
            'action_repeat': 4,
            'preprocessing': {
                'frame_size': 84,
                'frame_stack': 4,
                'grayscale': True
            }
        },
        'network': {
            'architecture': 'dqn',
            'conv1_channels': 16,
            'conv1_kernel': 8,
            'conv1_stride': 4,
            'fc_hidden': 256,
            'dtype': 'float32',
            'device': 'cuda'
        },
        'replay': {
            'capacity': 1000000,
            'batch_size': 32,
            'min_size': 50000,
            'warmup_steps': 50000
        },
        'training': {
            'total_frames': 10000000,
            'train_every': 4,
            'gamma': 0.99,
            'loss': {
                'type': 'mse'
            },
            'optimizer': {
                'type': 'rmsprop',
                'lr': 0.00025
            }
        },
        'target_network': {
            'update_interval': 10000,
            'update_method': 'hard'
        },
        'exploration': {
            'schedule': {
                'type': 'linear',
                'start_epsilon': 1.0,
                'end_epsilon': 0.1,
                'decay_frames': 1000000
            },
            'eval_epsilon': 0.05
        },
        'evaluation': {
            'enabled': True,
            'eval_every': 250000,
            'num_episodes': 10,
            'epsilon': 0.05,
            'video_format': 'mp4'
        },
        'logging': {
            'base_dir': 'runs',
            'log_every_steps': 1000,
            'log_every_episodes': 1,
            'checkpoint': {
                'enabled': True,
                'save_every': 1000000,
                'keep_last_n': 3
            }
        },
        'seed': {
            'value': 42
        },
        'system': {
            'num_workers': 0,
            'empty_cache_every': 1000,
            'progress_bar': True
        }
    }


# =============================================================================
# Experiment Validation Tests
# =============================================================================

def test_experiment_name_required():
    """Test that experiment name is required."""
    config = {'experiment': {}}
    with pytest.raises(ConfigValidationError, match="'name' is required"):
        validate_experiment(config)


def test_experiment_name_non_empty():
    """Test that experiment name must be non-empty."""
    config = {'experiment': {'name': ''}}
    with pytest.raises(ConfigValidationError, match="non-empty string"):
        validate_experiment(config)


def test_experiment_name_string():
    """Test that experiment name must be a string."""
    config = {'experiment': {'name': 123}}
    with pytest.raises(ConfigValidationError, match="non-empty string"):
        validate_experiment(config)


def test_experiment_notes_optional():
    """Test that notes are optional."""
    config = {'experiment': {'name': 'test'}}
    validate_experiment(config)  # Should not raise


def test_experiment_notes_string():
    """Test that notes must be a string if provided."""
    config = {'experiment': {'name': 'test', 'notes': 123}}
    with pytest.raises(ConfigValidationError, match="must be a string"):
        validate_experiment(config)


# =============================================================================
# Environment Validation Tests
# =============================================================================

def test_environment_env_id_required():
    """Test that env_id is required."""
    config = {'environment': {}}
    with pytest.raises(ConfigValidationError, match="'env_id' is required"):
        validate_environment(config)


def test_environment_env_id_null():
    """Test that env_id cannot be null."""
    config = {'environment': {'env_id': None}}
    with pytest.raises(ConfigValidationError, match="'env_id' is required"):
        validate_environment(config)


def test_environment_invalid_env_id():
    """Test that env_id must be a known Atari environment."""
    config = {'environment': {'env_id': 'InvalidGame-v4'}}
    with pytest.raises(ConfigValidationError, match="unknown environment 'InvalidGame-v4'"):
        validate_environment(config)


def test_environment_valid_env_ids():
    """Test that all valid Atari environments are accepted."""
    valid_envs = [
        'PongNoFrameskip-v4',
        'BreakoutNoFrameskip-v4',
        'BeamRiderNoFrameskip-v4',
        'QbertNoFrameskip-v4'
    ]
    for env_id in valid_envs:
        config = {'environment': {'env_id': env_id}}
        validate_environment(config)  # Should not raise


def test_environment_action_repeat_positive():
    """Test that action_repeat must be positive (nonzero frameskip)."""
    config = {'environment': {'env_id': 'PongNoFrameskip-v4', 'action_repeat': 0}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_environment(config)


def test_environment_action_repeat_negative():
    """Test that action_repeat cannot be negative."""
    config = {'environment': {'env_id': 'PongNoFrameskip-v4', 'action_repeat': -1}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_environment(config)


def test_environment_frame_size_positive():
    """Test that frame_size must be positive."""
    config = {
        'environment': {
            'env_id': 'PongNoFrameskip-v4',
            'preprocessing': {'frame_size': 0}
        }
    }
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_environment(config)


def test_environment_frame_stack_positive():
    """Test that frame_stack must be positive."""
    config = {
        'environment': {
            'env_id': 'PongNoFrameskip-v4',
            'preprocessing': {'frame_stack': 0}
        }
    }
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_environment(config)


# =============================================================================
# Network Validation Tests
# =============================================================================

def test_network_architecture_valid():
    """Test that architecture must be a known type."""
    config = {'network': {'architecture': 'unknown'}}
    with pytest.raises(ConfigValidationError, match="must be one of"):
        validate_network(config)


def test_network_conv_channels_positive():
    """Test that conv channels must be positive."""
    config = {'network': {'conv1_channels': 0}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_network(config)


def test_network_fc_hidden_positive():
    """Test that fc_hidden must be positive."""
    config = {'network': {'fc_hidden': -1}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_network(config)


def test_network_dtype_valid():
    """Test that dtype must be a known type."""
    config = {'network': {'dtype': 'float64'}}
    with pytest.raises(ConfigValidationError, match="must be one of"):
        validate_network(config)


def test_network_device_valid():
    """Test that device must be a known type."""
    config = {'network': {'device': 'gpu'}}
    with pytest.raises(ConfigValidationError, match="must be one of"):
        validate_network(config)


def test_network_init_method_valid():
    """Test that init_method must be a known type."""
    config = {'network': {'init_method': 'random'}}
    with pytest.raises(ConfigValidationError, match="must be one of"):
        validate_network(config)


# =============================================================================
# Replay Buffer Validation Tests
# =============================================================================

def test_replay_capacity_positive():
    """Test that capacity must be positive."""
    config = {'replay': {'capacity': 0}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_replay(config)


def test_replay_batch_size_positive():
    """Test that batch_size must be positive."""
    config = {'replay': {'batch_size': 0}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_replay(config)


def test_replay_min_size_non_negative():
    """Test that min_size can be zero."""
    config = {'replay': {'min_size': 0}}
    validate_replay(config)  # Should not raise


def test_replay_min_size_exceeds_capacity():
    """Test that min_size cannot exceed capacity."""
    config = {'replay': {'capacity': 100, 'min_size': 200}}
    with pytest.raises(ConfigValidationError, match="cannot exceed"):
        validate_replay(config)


def test_replay_warmup_steps_non_negative():
    """Test that warmup_steps can be zero."""
    config = {'replay': {'warmup_steps': 0}}
    validate_replay(config)  # Should not raise


# =============================================================================
# Training Validation Tests
# =============================================================================

def test_training_total_frames_positive():
    """Test that total_frames must be positive."""
    config = {'training': {'total_frames': 0}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_training(config)


def test_training_train_every_positive():
    """Test that train_every must be positive."""
    config = {'training': {'train_every': 0}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_training(config)


def test_training_gamma_in_range():
    """Test that gamma must be in [0, 1]."""
    # Test lower bound
    config = {'training': {'gamma': -0.1}}
    with pytest.raises(ConfigValidationError, match="must be in range"):
        validate_training(config)

    # Test upper bound
    config = {'training': {'gamma': 1.5}}
    with pytest.raises(ConfigValidationError, match="must be in range"):
        validate_training(config)

    # Test valid boundaries
    config = {'training': {'gamma': 0.0}}
    validate_training(config)  # Should not raise

    config = {'training': {'gamma': 1.0}}
    validate_training(config)  # Should not raise


def test_training_loss_type_valid():
    """Test that loss type must be known."""
    config = {'training': {'loss': {'type': 'unknown'}}}
    with pytest.raises(ConfigValidationError, match="must be one of"):
        validate_training(config)


def test_training_huber_delta_positive():
    """Test that huber_delta must be positive."""
    config = {'training': {'loss': {'huber_delta': 0}}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_training(config)


def test_training_optimizer_type_valid():
    """Test that optimizer type must be known."""
    config = {'training': {'optimizer': {'type': 'sgd'}}}
    with pytest.raises(ConfigValidationError, match="must be one of"):
        validate_training(config)


def test_training_optimizer_lr_positive():
    """Test that learning rate must be positive."""
    config = {'training': {'optimizer': {'lr': 0}}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_training(config)


def test_training_rmsprop_alpha_in_range():
    """Test that RMSProp alpha must be in [0, 1]."""
    config = {'training': {'optimizer': {'rmsprop': {'alpha': 1.5}}}}
    with pytest.raises(ConfigValidationError, match="must be in range"):
        validate_training(config)


def test_training_rmsprop_eps_positive():
    """Test that RMSProp eps must be positive."""
    config = {'training': {'optimizer': {'rmsprop': {'eps': 0}}}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_training(config)


def test_training_adam_betas_length():
    """Test that Adam betas must have 2 values."""
    config = {'training': {'optimizer': {'adam': {'betas': [0.9]}}}}
    with pytest.raises(ConfigValidationError, match="must be a list of 2 values"):
        validate_training(config)


def test_training_adam_betas_range():
    """Test that Adam betas must be in [0, 1]."""
    config = {'training': {'optimizer': {'adam': {'betas': [0.9, 1.5]}}}}
    with pytest.raises(ConfigValidationError, match="must be in range"):
        validate_training(config)


def test_training_gradient_clip_max_norm_positive():
    """Test that max_norm must be positive."""
    config = {'training': {'gradient_clip': {'max_norm': 0}}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_training(config)


# =============================================================================
# Target Network Validation Tests
# =============================================================================

def test_target_network_update_interval_positive():
    """Test that update_interval must be positive."""
    config = {'target_network': {'update_interval': 0}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_target_network(config)


def test_target_network_update_interval_null():
    """Test that update_interval can be null (disabled)."""
    config = {'target_network': {'update_interval': None}}
    validate_target_network(config)  # Should not raise


def test_target_network_update_method_valid():
    """Test that update_method must be known."""
    config = {'target_network': {'update_method': 'unknown'}}
    with pytest.raises(ConfigValidationError, match="must be one of"):
        validate_target_network(config)


# =============================================================================
# Exploration Validation Tests
# =============================================================================

def test_exploration_schedule_type_valid():
    """Test that schedule type must be known."""
    config = {'exploration': {'schedule': {'type': 'unknown'}}}
    with pytest.raises(ConfigValidationError, match="must be one of"):
        validate_exploration(config)


def test_exploration_start_epsilon_in_range():
    """Test that start_epsilon must be in [0, 1]."""
    config = {'exploration': {'schedule': {'start_epsilon': 1.5}}}
    with pytest.raises(ConfigValidationError, match="must be in range"):
        validate_exploration(config)


def test_exploration_end_epsilon_in_range():
    """Test that end_epsilon must be in [0, 1]."""
    config = {'exploration': {'schedule': {'end_epsilon': -0.1}}}
    with pytest.raises(ConfigValidationError, match="must be in range"):
        validate_exploration(config)


def test_exploration_decay_frames_positive():
    """Test that decay_frames must be positive."""
    config = {'exploration': {'schedule': {'decay_frames': 0}}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_exploration(config)


def test_exploration_eval_epsilon_in_range():
    """Test that eval_epsilon must be in [0, 1]."""
    config = {'exploration': {'eval_epsilon': 2.0}}
    with pytest.raises(ConfigValidationError, match="must be in range"):
        validate_exploration(config)


# =============================================================================
# Evaluation Validation Tests
# =============================================================================

def test_evaluation_eval_every_positive():
    """Test that eval_every must be positive."""
    config = {'evaluation': {'eval_every': 0}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_evaluation(config)


def test_evaluation_num_episodes_positive():
    """Test that num_episodes must be positive."""
    config = {'evaluation': {'num_episodes': 0}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_evaluation(config)


def test_evaluation_epsilon_in_range():
    """Test that epsilon must be in [0, 1]."""
    config = {'evaluation': {'epsilon': 1.5}}
    with pytest.raises(ConfigValidationError, match="must be in range"):
        validate_evaluation(config)


def test_evaluation_video_frequency_non_negative():
    """Test that video_frequency can be zero."""
    config = {'evaluation': {'video_frequency': 0}}
    validate_evaluation(config)  # Should not raise


def test_evaluation_video_format_valid():
    """Test that video_format must be known."""
    config = {'evaluation': {'video_format': 'avi'}}
    with pytest.raises(ConfigValidationError, match="must be one of"):
        validate_evaluation(config)


# =============================================================================
# Logging Validation Tests
# =============================================================================

def test_logging_log_every_steps_positive():
    """Test that log_every_steps must be positive."""
    config = {'logging': {'log_every_steps': 0}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_logging(config)


def test_logging_checkpoint_save_every_positive():
    """Test that save_every must be positive."""
    config = {'logging': {'checkpoint': {'save_every': 0}}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_logging(config)


def test_logging_checkpoint_keep_last_n_positive():
    """Test that keep_last_n must be positive."""
    config = {'logging': {'checkpoint': {'keep_last_n': 0}}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_logging(config)


def test_logging_reference_q_num_states_positive():
    """Test that num_states must be positive."""
    config = {'logging': {'reference_q': {'num_states': 0}}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_logging(config)


# =============================================================================
# System Validation Tests
# =============================================================================

def test_system_num_workers_non_negative():
    """Test that num_workers can be zero."""
    config = {'system': {'num_workers': 0}}
    validate_system(config)  # Should not raise


def test_system_num_workers_negative():
    """Test that num_workers cannot be negative."""
    config = {'system': {'num_workers': -1}}
    with pytest.raises(ConfigValidationError, match="must be non-negative"):
        validate_system(config)


def test_system_empty_cache_every_positive():
    """Test that empty_cache_every must be positive."""
    config = {'system': {'empty_cache_every': 0}}
    with pytest.raises(ConfigValidationError, match="must be positive"):
        validate_system(config)


# =============================================================================
# Unknown Fields Tests
# =============================================================================

def test_unknown_field_at_root():
    """Test that unknown fields at root level are rejected."""
    config = {
        'experiment': {'name': 'test'},
        'environment': {'env_id': 'PongNoFrameskip-v4'},
        'unknown_field': 'value'
    }
    with pytest.raises(ConfigValidationError, match="Unknown fields in root config"):
        validate_config(config, strict=True)


def test_unknown_field_in_section():
    """Test that unknown fields in sections are rejected."""
    config = {
        'experiment': {'name': 'test', 'unknown_param': 123},
        'environment': {'env_id': 'PongNoFrameskip-v4'}
    }
    with pytest.raises(ConfigValidationError, match="Unknown fields in experiment"):
        validate_config(config, strict=True)


def test_unknown_field_nested():
    """Test that unknown fields in nested sections are rejected."""
    config = {
        'experiment': {'name': 'test'},
        'environment': {'env_id': 'PongNoFrameskip-v4'},
        'training': {
            'gamma': 0.99,
            'optimizer': {
                'type': 'rmsprop',
                'unknown_opt': 'value'
            }
        }
    }
    with pytest.raises(ConfigValidationError, match="Unknown fields"):
        validate_config(config, strict=True)


def test_unknown_fields_not_strict():
    """Test that unknown fields are allowed when strict=False."""
    config = {
        'experiment': {'name': 'test'},
        'environment': {'env_id': 'PongNoFrameskip-v4'},
        'unknown_field': 'value'
    }
    # Should not raise when strict=False
    validate_config(config, strict=False)


# =============================================================================
# Full Config Validation Tests
# =============================================================================

def test_validate_valid_config(valid_config):
    """Test that a valid config passes validation."""
    validate_config(valid_config)  # Should not raise


def test_validate_config_safe_valid(valid_config):
    """Test validate_config_safe with valid config."""
    is_valid, error = validate_config_safe(valid_config)
    assert is_valid is True
    assert error is None


def test_validate_config_safe_invalid():
    """Test validate_config_safe with invalid config."""
    config = {
        'experiment': {'name': 'test'},
        'environment': {'env_id': 'PongNoFrameskip-v4'},
        'training': {'gamma': 2.0}  # Invalid gamma
    }
    is_valid, error = validate_config_safe(config)
    assert is_valid is False
    assert error is not None
    assert "gamma" in error
    assert "must be in range" in error


def test_validate_config_not_dict():
    """Test that config must be a dictionary."""
    with pytest.raises(ConfigValidationError, match="must be a dictionary"):
        validate_config("not a dict")


def test_validate_config_helpful_error_messages():
    """Test that error messages are helpful and specific."""
    # Test gamma out of range
    config = {
        'experiment': {'name': 'test'},
        'environment': {'env_id': 'PongNoFrameskip-v4'},
        'training': {'gamma': 1.5}
    }
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    error_msg = str(exc_info.value)
    assert "gamma" in error_msg
    assert "must be in range [0.0, 1.0]" in error_msg
    assert "1.5" in error_msg


def test_validate_config_invalid_optimizer():
    """Test helpful error for invalid optimizer."""
    config = {
        'experiment': {'name': 'test'},
        'environment': {'env_id': 'PongNoFrameskip-v4'},
        'training': {'optimizer': {'type': 'sgd'}}
    }
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    error_msg = str(exc_info.value)
    assert "optimizer" in error_msg
    assert "sgd" in error_msg
    assert "rmsprop" in error_msg or "adam" in error_msg


def test_validate_config_invalid_env_id():
    """Test helpful error for invalid environment."""
    config = {
        'experiment': {'name': 'test'},
        'environment': {'env_id': 'Tetris-v0'}
    }
    with pytest.raises(ConfigValidationError) as exc_info:
        validate_config(config)
    error_msg = str(exc_info.value)
    assert "env_id" in error_msg
    assert "unknown environment" in error_msg
    assert "Tetris-v0" in error_msg
    assert "PongNoFrameskip-v4" in error_msg  # Should suggest valid options


# =============================================================================
# Type Error Tests
# =============================================================================

def test_validate_type_errors():
    """Test that type errors are caught with helpful messages."""
    # String instead of int
    config = {'replay': {'capacity': "one million"}}
    with pytest.raises(ConfigValidationError, match="must be an integer"):
        validate_replay(config)

    # Boolean instead of int
    config = {'replay': {'batch_size': True}}
    with pytest.raises(ConfigValidationError, match="must be an integer"):
        validate_replay(config)

    # Int instead of string
    config = {'network': {'architecture': 123}}
    with pytest.raises(ConfigValidationError, match="must be a string"):
        validate_network(config)


# =============================================================================
# Edge Cases
# =============================================================================

def test_validate_boundary_values():
    """Test validation of boundary values."""
    # Gamma = 0.0 (valid)
    config = {'training': {'gamma': 0.0}}
    validate_training(config)

    # Gamma = 1.0 (valid)
    config = {'training': {'gamma': 1.0}}
    validate_training(config)

    # Epsilon = 0.0 (valid)
    config = {'exploration': {'eval_epsilon': 0.0}}
    validate_exploration(config)

    # Epsilon = 1.0 (valid)
    config = {'exploration': {'eval_epsilon': 1.0}}
    validate_exploration(config)


def test_validate_missing_optional_fields(valid_config):
    """Test that optional fields can be missing."""
    # Remove optional fields
    del valid_config['experiment']['notes']
    del valid_config['seed']

    validate_config(valid_config)  # Should not raise


def test_validate_null_optional_values():
    """Test that null values are handled correctly."""
    config = {
        'experiment': {'name': 'test', 'notes': None},
        'environment': {'env_id': 'PongNoFrameskip-v4'},
        'target_network': {'update_interval': None}  # Null to disable
    }
    # Should not raise - None is allowed for optional fields
    validate_experiment(config)
    validate_target_network(config)
