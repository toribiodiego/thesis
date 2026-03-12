"""Configuration schema validation with strict checking and helpful errors.

Validates DQN configuration against schema constraints:
- Positive integers for buffer sizes, training steps, etc.
- Gamma in [0, 1]
- Known optimizer names
- Valid environment IDs
- Nonzero frameskip
- Rejects unknown fields
"""

from typing import Any, Dict, List, Optional, Set, Tuple


class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""

    pass


# =============================================================================
# Valid Value Sets
# =============================================================================

VALID_OPTIMIZERS = {"rmsprop", "adam"}
VALID_LOSS_TYPES = {"mse", "huber"}
VALID_NETWORK_ARCHITECTURES = {"dqn"}
VALID_EXPLORATION_SCHEDULES = {"linear", "exponential", "constant"}
VALID_TARGET_UPDATE_METHODS = {"hard", "soft"}
VALID_INIT_METHODS = {
    "kaiming_normal",
    "kaiming_uniform",
    "xavier_normal",
    "xavier_uniform",
    "orthogonal",
}
VALID_DEVICES = {"cuda", "cpu", "mps"}
VALID_DTYPES = {"float32", "float16", "bfloat16"}
VALID_VIDEO_FORMATS = {"mp4", "gif"}

# Valid Atari environment IDs (NoFrameskip-v4 variants)
VALID_ENV_IDS = {
    # Atari-100K benchmark (26 games, Schwarzer et al. 2021)
    "AlienNoFrameskip-v4",
    "AmidarNoFrameskip-v4",
    "AssaultNoFrameskip-v4",
    "AsterixNoFrameskip-v4",
    "BankHeistNoFrameskip-v4",
    "BattleZoneNoFrameskip-v4",
    "BoxingNoFrameskip-v4",
    "BreakoutNoFrameskip-v4",
    "ChopperCommandNoFrameskip-v4",
    "CrazyClimberNoFrameskip-v4",
    "DemonAttackNoFrameskip-v4",
    "FreewayNoFrameskip-v4",
    "FrostbiteNoFrameskip-v4",
    "GopherNoFrameskip-v4",
    "HeroNoFrameskip-v4",
    "JamesbondNoFrameskip-v4",
    "KangarooNoFrameskip-v4",
    "KrullNoFrameskip-v4",
    "KungFuMasterNoFrameskip-v4",
    "MsPacmanNoFrameskip-v4",
    "PongNoFrameskip-v4",
    "PrivateEyeNoFrameskip-v4",
    "QbertNoFrameskip-v4",
    "RoadRunnerNoFrameskip-v4",
    "SeaquestNoFrameskip-v4",
    "UpNDownNoFrameskip-v4",
    # Additional Atari games (not in 100K benchmark)
    "BeamRiderNoFrameskip-v4",
    "SpaceInvadersNoFrameskip-v4",
    "EnduroNoFrameskip-v4",
    "AtlantisNoFrameskip-v4",
    "NameThisGameNoFrameskip-v4",
}

# Known configuration structure (for detecting unknown fields)
# Maps config paths to their expected child keys
KNOWN_STRUCTURE = {
    "": {
        "experiment",
        "environment",
        "network",
        "replay",
        "training",
        "target_network",
        "exploration",
        "evaluation",
        "logging",
        "augmentation",
        "spr",
        "ema",
        "rainbow",
        "seed",
        "system",
    },
    "rainbow": {
        "enabled",
        "double_dqn",
        "dueling",
        "noisy_nets",
        "distributional",
        "multi_step",
        "priority",
    },
    "rainbow.distributional": {"num_atoms", "v_min", "v_max"},
    "rainbow.multi_step": {"n"},
    "rainbow.priority": {"alpha", "beta_start", "beta_end", "epsilon"},
    "augmentation": {"enabled", "type", "random_shift"},
    "augmentation.random_shift": {"pad"},
    "spr": {
        "enabled",
        "prediction_steps",
        "loss_weight",
        "projection_dim",
        "transition_channels",
    },
    "ema": {"momentum"},
    "experiment": {"name", "run_id", "notes", "deterministic"},
    "experiment.deterministic": {"enabled", "strict", "warn_only"},
    "environment": {
        "env_id",
        "preprocessing",
        "action_repeat",
        "episode",
        "max_episode_steps",
        "repeat_action_probability",
    },
    "environment.preprocessing": {
        "frame_size",
        "grayscale",
        "frame_stack",
        "frame_max_pool",
        "max_pool_frames",
        "clip_rewards",
        "reward_range",
    },
    "environment.episode": {"episodic_life", "noop_max", "fire_on_reset"},
    "network": {
        "architecture",
        "conv1_channels",
        "conv1_kernel",
        "conv1_stride",
        "conv2_channels",
        "conv2_kernel",
        "conv2_stride",
        "conv3_channels",
        "conv3_kernel",
        "conv3_stride",
        "fc_hidden",
        "dropout",
        "init_method",
        "init_mode",
        "dtype",
        "device",
    },
    "replay": {
        "capacity",
        "batch_size",
        "min_size",
        "warmup_steps",
        "store_uint8",
        "normalize_on_sample",
        "pin_memory",
        "track_episodes",
    },
    "training": {
        "total_frames",
        "train_every",
        "gamma",
        "loss",
        "gradient_clip",
        "optimizer",
    },
    "training.loss": {"type", "huber_delta"},
    "training.gradient_clip": {"enabled", "max_norm", "norm_type"},
    "training.optimizer": {"type", "lr", "rmsprop", "adam"},
    "training.optimizer.rmsprop": {"alpha", "eps", "momentum"},
    "training.optimizer.adam": {"betas", "eps"},
    "target_network": {"update_interval", "update_method"},
    "exploration": {"schedule", "eval_epsilon"},
    "exploration.schedule": {"type", "start_epsilon", "end_epsilon", "decay_frames"},
    "evaluation": {
        "enabled",
        "eval_every",
        "num_episodes",
        "epsilon",
        "deterministic",
        "record_video",
        "video_frequency",
        "video_format",
        "metrics",
    },
    "logging": {
        "base_dir",
        "log_every_steps",
        "log_every_episodes",
        "checkpoint",
        "step_metrics",
        "episode_metrics",
        "reference_q",
        "wandb",
        "tensorboard",
        "csv",
    },
    "logging.checkpoint": {"enabled", "save_every", "keep_last_n", "save_best"},
    "logging.reference_q": {"enabled", "num_states", "log_every"},
    "logging.wandb": {
        "enabled",
        "project",
        "entity",
        "tags",
        "upload_artifacts",
        "artifact_upload_interval",
    },
    "logging.tensorboard": {"enabled", "flush_interval"},
    "logging.csv": {"enabled", "smoothing_window"},
    "seed": {"value", "save_rng_states"},
    "system": {"num_workers", "empty_cache_every", "progress_bar", "verbose"},
}


# =============================================================================
# Validation Functions
# =============================================================================


def _format_path(path: List[str]) -> str:
    """Format config path for error messages."""
    return ".".join(path) if path else "config"


def _validate_positive_int(
    value: Any, path: List[str], field_name: str, allow_zero: bool = False
) -> None:
    """Validate that value is a positive integer."""
    full_path = _format_path(path + [field_name])

    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigValidationError(
            f"{full_path}: must be an integer, got {type(value).__name__}"
        )

    if allow_zero:
        if value < 0:
            raise ConfigValidationError(
                f"{full_path}: must be non-negative, got {value}"
            )
    else:
        if value <= 0:
            raise ConfigValidationError(f"{full_path}: must be positive, got {value}")


def _validate_positive_float(
    value: Any, path: List[str], field_name: str, allow_zero: bool = False
) -> None:
    """Validate that value is a positive float."""
    full_path = _format_path(path + [field_name])

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigValidationError(
            f"{full_path}: must be a number, got {type(value).__name__}"
        )

    if allow_zero:
        if value < 0:
            raise ConfigValidationError(
                f"{full_path}: must be non-negative, got {value}"
            )
    else:
        if value <= 0:
            raise ConfigValidationError(f"{full_path}: must be positive, got {value}")


def _validate_range(
    value: Any,
    path: List[str],
    field_name: str,
    min_val: float,
    max_val: float,
    inclusive: bool = True,
) -> None:
    """Validate that value is in specified range."""
    full_path = _format_path(path + [field_name])

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ConfigValidationError(
            f"{full_path}: must be a number, got {type(value).__name__}"
        )

    if inclusive:
        if not (min_val <= value <= max_val):
            raise ConfigValidationError(
                f"{full_path}: must be in range [{min_val}, {max_val}], got {value}"
            )
    else:
        if not (min_val < value < max_val):
            raise ConfigValidationError(
                f"{full_path}: must be in range ({min_val}, {max_val}), got {value}"
            )


def _validate_choice(
    value: Any, path: List[str], field_name: str, valid_choices: Set[str]
) -> None:
    """Validate that value is one of the valid choices."""
    full_path = _format_path(path + [field_name])

    if not isinstance(value, str):
        raise ConfigValidationError(
            f"{full_path}: must be a string, got {type(value).__name__}"
        )

    if value not in valid_choices:
        choices_str = ", ".join(f"'{c}'" for c in sorted(valid_choices))
        raise ConfigValidationError(
            f"{full_path}: must be one of [{choices_str}], got '{value}'"
        )


def _validate_bool(value: Any, path: List[str], field_name: str) -> None:
    """Validate that value is a boolean."""
    full_path = _format_path(path + [field_name])

    if not isinstance(value, bool):
        raise ConfigValidationError(
            f"{full_path}: must be a boolean (true/false), got {type(value).__name__}"
        )


def _check_unknown_fields(config: Dict[str, Any], path: List[str] = None) -> None:
    """Check for unknown fields in configuration and fail fast."""
    if path is None:
        path = []

    path_str = ".".join(path)

    # Get known fields for this path
    if path_str not in KNOWN_STRUCTURE:
        # Path not in known structure, skip (might be dynamic)
        return

    known_fields = KNOWN_STRUCTURE[path_str]
    actual_fields = set(config.keys())
    unknown_fields = actual_fields - known_fields

    if unknown_fields:
        location = _format_path(path) if path else "root config"
        unknown_list = ", ".join(f"'{f}'" for f in sorted(unknown_fields))
        known_list = ", ".join(f"'{f}'" for f in sorted(known_fields))

        raise ConfigValidationError(
            f"Unknown fields in {location}: {unknown_list}\n"
            f"Valid fields: {known_list}"
        )

    # Recursively check nested dictionaries
    for key, value in config.items():
        if isinstance(value, dict):
            _check_unknown_fields(value, path + [key])


# =============================================================================
# Section Validators
# =============================================================================


def validate_experiment(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate experiment section."""
    if path is None:
        path = []

    exp = config.get("experiment", {})
    if not isinstance(exp, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['experiment'])}: must be a dict"
        )

    # name: required, non-empty string
    if "name" not in exp:
        raise ConfigValidationError(
            f"{_format_path(path + ['experiment'])}: 'name' is required"
        )
    if not isinstance(exp["name"], str) or not exp["name"].strip():
        raise ConfigValidationError(
            f"{_format_path(path + ['experiment', 'name'])}: must be a non-empty string"
        )

    # notes: optional string
    if "notes" in exp and exp["notes"] is not None:
        if not isinstance(exp["notes"], str):
            raise ConfigValidationError(
                f"{_format_path(path + ['experiment', 'notes'])}: must be a string"
            )

    # deterministic: optional dict
    if "deterministic" in exp and exp["deterministic"] is not None:
        det = exp["deterministic"]
        if not isinstance(det, dict):
            raise ConfigValidationError(
                f"{_format_path(path + ['experiment', 'deterministic'])}: must be a dict"
            )
        if "enabled" in det:
            _validate_bool(
                det["enabled"], path + ["experiment", "deterministic"], "enabled"
            )


def validate_environment(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate environment section."""
    if path is None:
        path = []

    env = config.get("environment", {})
    if not isinstance(env, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['environment'])}: must be a dict"
        )

    # env_id: required, must be valid Atari environment
    if "env_id" not in env or env["env_id"] is None:
        raise ConfigValidationError(
            f"{_format_path(path + ['environment'])}: 'env_id' is required"
        )

    env_id = env["env_id"]
    if not isinstance(env_id, str):
        raise ConfigValidationError(
            f"{_format_path(path + ['environment', 'env_id'])}: must be a string"
        )

    if env_id not in VALID_ENV_IDS:
        valid_list = "\n  ".join(sorted(VALID_ENV_IDS))
        raise ConfigValidationError(
            f"{_format_path(path + ['environment', 'env_id'])}: unknown environment '{env_id}'\n"
            f"Valid Atari environments:\n  {valid_list}"
        )

    # action_repeat: must be positive (nonzero frameskip)
    if "action_repeat" in env:
        _validate_positive_int(
            env["action_repeat"], path + ["environment"], "action_repeat"
        )

    # preprocessing validations
    if "preprocessing" in env and env["preprocessing"] is not None:
        prep = env["preprocessing"]
        if not isinstance(prep, dict):
            raise ConfigValidationError(
                f"{_format_path(path + ['environment', 'preprocessing'])}: must be a dict"
            )

        if "frame_size" in prep:
            _validate_positive_int(
                prep["frame_size"],
                path + ["environment", "preprocessing"],
                "frame_size",
            )

        if "frame_stack" in prep:
            _validate_positive_int(
                prep["frame_stack"],
                path + ["environment", "preprocessing"],
                "frame_stack",
            )

        if "max_pool_frames" in prep:
            _validate_positive_int(
                prep["max_pool_frames"],
                path + ["environment", "preprocessing"],
                "max_pool_frames",
            )


def validate_network(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate network section."""
    if path is None:
        path = []

    net = config.get("network", {})
    if not isinstance(net, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['network'])}: must be a dict"
        )

    # architecture: must be valid
    if "architecture" in net:
        _validate_choice(
            net["architecture"],
            path + ["network"],
            "architecture",
            VALID_NETWORK_ARCHITECTURES,
        )

    # conv layer parameters: positive integers
    if "conv1_channels" in net:
        _validate_positive_int(
            net["conv1_channels"], path + ["network"], "conv1_channels"
        )
    if "conv1_kernel" in net:
        _validate_positive_int(net["conv1_kernel"], path + ["network"], "conv1_kernel")
    if "conv1_stride" in net:
        _validate_positive_int(net["conv1_stride"], path + ["network"], "conv1_stride")

    if "conv2_channels" in net:
        _validate_positive_int(
            net["conv2_channels"], path + ["network"], "conv2_channels"
        )
    if "conv2_kernel" in net:
        _validate_positive_int(net["conv2_kernel"], path + ["network"], "conv2_kernel")
    if "conv2_stride" in net:
        _validate_positive_int(net["conv2_stride"], path + ["network"], "conv2_stride")

    if "conv3_channels" in net:
        _validate_positive_int(
            net["conv3_channels"], path + ["network"], "conv3_channels"
        )
    if "conv3_kernel" in net:
        _validate_positive_int(net["conv3_kernel"], path + ["network"], "conv3_kernel")
    if "conv3_stride" in net:
        _validate_positive_int(net["conv3_stride"], path + ["network"], "conv3_stride")

    if "fc_hidden" in net:
        _validate_positive_int(net["fc_hidden"], path + ["network"], "fc_hidden")

    # init_method: must be valid
    if "init_method" in net:
        _validate_choice(
            net["init_method"], path + ["network"], "init_method", VALID_INIT_METHODS
        )

    # dtype: must be valid
    if "dtype" in net:
        _validate_choice(net["dtype"], path + ["network"], "dtype", VALID_DTYPES)

    # device: must be valid
    if "device" in net:
        _validate_choice(net["device"], path + ["network"], "device", VALID_DEVICES)


def validate_replay(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate replay buffer section."""
    if path is None:
        path = []

    replay = config.get("replay", {})
    if not isinstance(replay, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['replay'])}: must be a dict"
        )

    # capacity: must be positive
    if "capacity" in replay:
        _validate_positive_int(replay["capacity"], path + ["replay"], "capacity")

    # batch_size: must be positive
    if "batch_size" in replay:
        _validate_positive_int(replay["batch_size"], path + ["replay"], "batch_size")

    # min_size: must be non-negative
    if "min_size" in replay:
        _validate_positive_int(
            replay["min_size"], path + ["replay"], "min_size", allow_zero=True
        )

    # warmup_steps: must be non-negative
    if "warmup_steps" in replay:
        _validate_positive_int(
            replay["warmup_steps"], path + ["replay"], "warmup_steps", allow_zero=True
        )

    # Check that min_size <= capacity
    if "min_size" in replay and "capacity" in replay:
        if replay["min_size"] > replay["capacity"]:
            raise ConfigValidationError(
                f"replay.min_size ({replay['min_size']}) cannot exceed replay.capacity ({replay['capacity']})"
            )


def validate_training(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate training section."""
    if path is None:
        path = []

    train = config.get("training", {})
    if not isinstance(train, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['training'])}: must be a dict"
        )

    # total_frames: must be positive
    if "total_frames" in train:
        _validate_positive_int(
            train["total_frames"], path + ["training"], "total_frames"
        )

    # train_every: must be positive
    if "train_every" in train:
        _validate_positive_int(train["train_every"], path + ["training"], "train_every")

    # gamma: must be in [0, 1]
    if "gamma" in train:
        _validate_range(train["gamma"], path + ["training"], "gamma", 0.0, 1.0)

    # loss configuration
    if "loss" in train and train["loss"] is not None:
        loss = train["loss"]
        if not isinstance(loss, dict):
            raise ConfigValidationError(
                f"{_format_path(path + ['training', 'loss'])}: must be a dict"
            )

        if "type" in loss:
            _validate_choice(
                loss["type"], path + ["training", "loss"], "type", VALID_LOSS_TYPES
            )

        if "huber_delta" in loss:
            _validate_positive_float(
                loss["huber_delta"], path + ["training", "loss"], "huber_delta"
            )

    # gradient clipping
    if "gradient_clip" in train and train["gradient_clip"] is not None:
        grad_clip = train["gradient_clip"]
        if not isinstance(grad_clip, dict):
            raise ConfigValidationError(
                f"{_format_path(path + ['training', 'gradient_clip'])}: must be a dict"
            )

        if "max_norm" in grad_clip:
            _validate_positive_float(
                grad_clip["max_norm"], path + ["training", "gradient_clip"], "max_norm"
            )

        if "norm_type" in grad_clip:
            _validate_positive_float(
                grad_clip["norm_type"],
                path + ["training", "gradient_clip"],
                "norm_type",
            )

    # optimizer configuration
    if "optimizer" in train and train["optimizer"] is not None:
        opt = train["optimizer"]
        if not isinstance(opt, dict):
            raise ConfigValidationError(
                f"{_format_path(path + ['training', 'optimizer'])}: must be a dict"
            )

        if "type" in opt:
            _validate_choice(
                opt["type"], path + ["training", "optimizer"], "type", VALID_OPTIMIZERS
            )

        if "lr" in opt:
            _validate_positive_float(opt["lr"], path + ["training", "optimizer"], "lr")

        # RMSProp parameters
        if "rmsprop" in opt and opt["rmsprop"] is not None:
            rmsprop = opt["rmsprop"]
            if not isinstance(rmsprop, dict):
                raise ConfigValidationError(
                    f"{_format_path(path + ['training', 'optimizer', 'rmsprop'])}: must be a dict"
                )

            if "alpha" in rmsprop:
                _validate_range(
                    rmsprop["alpha"],
                    path + ["training", "optimizer", "rmsprop"],
                    "alpha",
                    0.0,
                    1.0,
                )

            if "eps" in rmsprop:
                _validate_positive_float(
                    rmsprop["eps"], path + ["training", "optimizer", "rmsprop"], "eps"
                )

            if "momentum" in rmsprop:
                _validate_positive_float(
                    rmsprop["momentum"],
                    path + ["training", "optimizer", "rmsprop"],
                    "momentum",
                    allow_zero=True,
                )

        # Adam parameters
        if "adam" in opt and opt["adam"] is not None:
            adam = opt["adam"]
            if not isinstance(adam, dict):
                raise ConfigValidationError(
                    f"{_format_path(path + ['training', 'optimizer', 'adam'])}: must be a dict"
                )

            if "eps" in adam:
                _validate_positive_float(
                    adam["eps"], path + ["training", "optimizer", "adam"], "eps"
                )

            if "betas" in adam:
                betas = adam["betas"]
                if not isinstance(betas, (list, tuple)) or len(betas) != 2:
                    raise ConfigValidationError(
                        f"{_format_path(path + ['training', 'optimizer', 'adam', 'betas'])}: "
                        f"must be a list of 2 values"
                    )
                for i, beta in enumerate(betas):
                    if not isinstance(beta, (int, float)) or not (0.0 <= beta <= 1.0):
                        raise ConfigValidationError(
                            f"{_format_path(path + ['training', 'optimizer', 'adam', 'betas'])}[{i}]: "
                            f"must be in range [0, 1], got {beta}"
                        )


def validate_target_network(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate target network section."""
    if path is None:
        path = []

    target = config.get("target_network", {})
    if not isinstance(target, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['target_network'])}: must be a dict"
        )

    # update_interval: must be positive (or null to disable)
    if "update_interval" in target and target["update_interval"] is not None:
        _validate_positive_int(
            target["update_interval"], path + ["target_network"], "update_interval"
        )

    # update_method: must be valid
    if "update_method" in target:
        _validate_choice(
            target["update_method"],
            path + ["target_network"],
            "update_method",
            VALID_TARGET_UPDATE_METHODS,
        )


def validate_exploration(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate exploration section."""
    if path is None:
        path = []

    expl = config.get("exploration", {})
    if not isinstance(expl, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['exploration'])}: must be a dict"
        )

    # schedule configuration
    if "schedule" in expl and expl["schedule"] is not None:
        sched = expl["schedule"]
        if not isinstance(sched, dict):
            raise ConfigValidationError(
                f"{_format_path(path + ['exploration', 'schedule'])}: must be a dict"
            )

        if "type" in sched:
            _validate_choice(
                sched["type"],
                path + ["exploration", "schedule"],
                "type",
                VALID_EXPLORATION_SCHEDULES,
            )

        if "start_epsilon" in sched:
            _validate_range(
                sched["start_epsilon"],
                path + ["exploration", "schedule"],
                "start_epsilon",
                0.0,
                1.0,
            )

        if "end_epsilon" in sched:
            _validate_range(
                sched["end_epsilon"],
                path + ["exploration", "schedule"],
                "end_epsilon",
                0.0,
                1.0,
            )

        if "decay_frames" in sched:
            _validate_positive_int(
                sched["decay_frames"],
                path + ["exploration", "schedule"],
                "decay_frames",
            )

    # eval_epsilon: must be in [0, 1]
    if "eval_epsilon" in expl:
        _validate_range(
            expl["eval_epsilon"], path + ["exploration"], "eval_epsilon", 0.0, 1.0
        )


def validate_evaluation(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate evaluation section."""
    if path is None:
        path = []

    eval_cfg = config.get("evaluation", {})
    if not isinstance(eval_cfg, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['evaluation'])}: must be a dict"
        )

    # eval_every: must be positive
    if "eval_every" in eval_cfg:
        _validate_positive_int(
            eval_cfg["eval_every"], path + ["evaluation"], "eval_every"
        )

    # num_episodes: must be positive
    if "num_episodes" in eval_cfg:
        _validate_positive_int(
            eval_cfg["num_episodes"], path + ["evaluation"], "num_episodes"
        )

    # epsilon: must be in [0, 1]
    if "epsilon" in eval_cfg:
        _validate_range(eval_cfg["epsilon"], path + ["evaluation"], "epsilon", 0.0, 1.0)

    # video_frequency: must be non-negative
    if "video_frequency" in eval_cfg:
        _validate_positive_int(
            eval_cfg["video_frequency"],
            path + ["evaluation"],
            "video_frequency",
            allow_zero=True,
        )

    # video_format: must be valid
    if "video_format" in eval_cfg:
        _validate_choice(
            eval_cfg["video_format"],
            path + ["evaluation"],
            "video_format",
            VALID_VIDEO_FORMATS,
        )


def validate_logging(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate logging section."""
    if path is None:
        path = []

    log = config.get("logging", {})
    if not isinstance(log, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['logging'])}: must be a dict"
        )

    # log_every_steps: must be positive
    if "log_every_steps" in log:
        _validate_positive_int(
            log["log_every_steps"], path + ["logging"], "log_every_steps"
        )

    # log_every_episodes: must be positive
    if "log_every_episodes" in log:
        _validate_positive_int(
            log["log_every_episodes"], path + ["logging"], "log_every_episodes"
        )

    # checkpoint configuration
    if "checkpoint" in log and log["checkpoint"] is not None:
        ckpt = log["checkpoint"]
        if not isinstance(ckpt, dict):
            raise ConfigValidationError(
                f"{_format_path(path + ['logging', 'checkpoint'])}: must be a dict"
            )

        if "save_every" in ckpt:
            _validate_positive_int(
                ckpt["save_every"], path + ["logging", "checkpoint"], "save_every"
            )

        if "keep_last_n" in ckpt and ckpt["keep_last_n"] is not None:
            _validate_positive_int(
                ckpt["keep_last_n"], path + ["logging", "checkpoint"], "keep_last_n"
            )

    # reference_q configuration
    if "reference_q" in log and log["reference_q"] is not None:
        ref_q = log["reference_q"]
        if not isinstance(ref_q, dict):
            raise ConfigValidationError(
                f"{_format_path(path + ['logging', 'reference_q'])}: must be a dict"
            )

        if "num_states" in ref_q:
            _validate_positive_int(
                ref_q["num_states"], path + ["logging", "reference_q"], "num_states"
            )

        if "log_every" in ref_q:
            _validate_positive_int(
                ref_q["log_every"], path + ["logging", "reference_q"], "log_every"
            )


def validate_system(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate system section."""
    if path is None:
        path = []

    sys_cfg = config.get("system", {})
    if not isinstance(sys_cfg, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['system'])}: must be a dict"
        )

    # num_workers: must be non-negative
    if "num_workers" in sys_cfg:
        _validate_positive_int(
            sys_cfg["num_workers"], path + ["system"], "num_workers", allow_zero=True
        )

    # empty_cache_every: must be positive
    if "empty_cache_every" in sys_cfg and sys_cfg["empty_cache_every"] is not None:
        _validate_positive_int(
            sys_cfg["empty_cache_every"], path + ["system"], "empty_cache_every"
        )


def validate_spr(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate SPR (self-predictive representations) section."""
    if path is None:
        path = []

    spr = config.get("spr", {})
    if not isinstance(spr, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['spr'])}: must be a dict"
        )

    if "enabled" in spr:
        _validate_bool(spr["enabled"], path + ["spr"], "enabled")

    if "prediction_steps" in spr:
        _validate_positive_int(
            spr["prediction_steps"], path + ["spr"], "prediction_steps"
        )

    if "loss_weight" in spr:
        _validate_positive_float(
            spr["loss_weight"], path + ["spr"], "loss_weight"
        )

    if "projection_dim" in spr:
        _validate_positive_int(
            spr["projection_dim"], path + ["spr"], "projection_dim"
        )

    if "transition_channels" in spr:
        _validate_positive_int(
            spr["transition_channels"], path + ["spr"], "transition_channels"
        )


def validate_ema(config: Dict[str, Any], path: List[str] = None) -> None:
    """Validate EMA (exponential moving average) section."""
    if path is None:
        path = []

    ema = config.get("ema", {})
    if not isinstance(ema, dict):
        raise ConfigValidationError(
            f"{_format_path(path + ['ema'])}: must be a dict"
        )

    if "momentum" in ema:
        _validate_range(
            ema["momentum"], path + ["ema"], "momentum", 0.0, 1.0
        )


# =============================================================================
# Main Validation Entry Point
# =============================================================================


def validate_config(config: Dict[str, Any], strict: bool = True) -> None:
    """
    Validate complete configuration against schema.

    Performs comprehensive validation:
    - Required fields present
    - Positive integers where required
    - Gamma in [0, 1]
    - Known optimizer names
    - Valid environment IDs
    - Nonzero frameskip (action_repeat)
    - Unknown fields rejected (if strict=True)

    Args:
        config: Configuration dictionary to validate
        strict: If True, reject unknown fields. If False, only warn.

    Raises:
        ConfigValidationError: If validation fails with helpful error message

    Example:
        >>> config = load_config("pong.yaml")
        >>> validate_config(config)  # Raises ConfigValidationError if invalid
    """
    if not isinstance(config, dict):
        raise ConfigValidationError("Config must be a dictionary")

    # Check for unknown fields first (fail fast)
    if strict:
        _check_unknown_fields(config)

    # Validate each section
    validate_experiment(config)
    validate_environment(config)
    validate_network(config)
    validate_replay(config)
    validate_training(config)
    validate_target_network(config)
    validate_exploration(config)
    validate_evaluation(config)
    validate_logging(config)
    validate_spr(config)
    validate_ema(config)
    validate_system(config)


def validate_config_safe(
    config: Dict[str, Any], strict: bool = True
) -> Tuple[bool, Optional[str]]:
    """
    Validate configuration and return success status and error message.

    Args:
        config: Configuration dictionary to validate
        strict: If True, reject unknown fields

    Returns:
        Tuple of (is_valid, error_message)
        - is_valid: True if valid, False otherwise
        - error_message: None if valid, error string otherwise

    Example:
        >>> valid, error = validate_config_safe(config)
        >>> if not valid:
        >>>     print(f"Validation failed: {error}")
    """
    try:
        validate_config(config, strict=strict)
        return True, None
    except ConfigValidationError as e:
        return False, str(e)
