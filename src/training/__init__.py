"""Training utilities for DQN."""

# Target network management
# Checkpoint utilities
from .checkpoint_utils import (
    get_rng_states,
    set_rng_states,
    verify_checkpoint_integrity,
)

# Evaluation
from .evaluation import EvaluationLogger, EvaluationScheduler, VideoRecorder, evaluate

# Logging
from .logging import CheckpointManager, EpisodeLogger, StepLogger

# Loss computation
from .loss import (
    compute_dqn_loss,
    compute_td_loss_components,
    compute_td_targets,
    select_q_values,
)

# SPR loss computation
from .spr_loss import compute_spr_forward, compute_spr_loss

# Metadata
from .metadata import MetadataWriter, get_git_commit_hash, get_git_status

# Metrics
from .metrics import EpsilonScheduler, UpdateMetrics, perform_update_step

# Multi-backend metrics logging
from .metrics_logger import (
    CSVBackend,
    MetricKeys,
    MetricsLogger,
    TensorBoardBackend,
    WandBBackend,
)

# Optimization
from .optimization import clip_gradients, configure_optimizer

# Q-value tracking
from .q_tracking import ReferenceQLogger, ReferenceStateQTracker

# Resume functionality
from .resume import (
    add_resume_args,
    check_git_hash_mismatch,
    resume_from_checkpoint,
    validate_config_compatibility,
)

# Schedulers
from .schedulers import TargetNetworkUpdater, TrainingScheduler

# Stability checks
from .stability import (
    detect_nan_inf,
    validate_loss_decrease,
    verify_target_sync_schedule,
)
from .target_network import hard_update_target, init_target_network

# Training loop
from .training_loop import FrameCounter, select_epsilon_greedy_action, training_step

__all__ = [
    # Target network
    "hard_update_target",
    "init_target_network",
    # Loss
    "compute_td_targets",
    "select_q_values",
    "compute_td_loss_components",
    "compute_dqn_loss",
    # SPR loss
    "compute_spr_loss",
    "compute_spr_forward",
    # Optimization
    "configure_optimizer",
    "clip_gradients",
    # Schedulers
    "TargetNetworkUpdater",
    "TrainingScheduler",
    "EpsilonScheduler",
    # Stability
    "detect_nan_inf",
    "validate_loss_decrease",
    "verify_target_sync_schedule",
    # Metrics
    "UpdateMetrics",
    "perform_update_step",
    # Training loop
    "select_epsilon_greedy_action",
    "FrameCounter",
    "training_step",
    # Logging
    "StepLogger",
    "EpisodeLogger",
    "CheckpointManager",
    # Multi-backend metrics logging
    "MetricsLogger",
    "MetricKeys",
    "TensorBoardBackend",
    "WandBBackend",
    "CSVBackend",
    # Evaluation
    "evaluate",
    "EvaluationScheduler",
    "EvaluationLogger",
    "VideoRecorder",
    # Q-tracking
    "ReferenceStateQTracker",
    "ReferenceQLogger",
    # Metadata
    "get_git_commit_hash",
    "get_git_status",
    "MetadataWriter",
    # Checkpoint utilities
    "get_rng_states",
    "set_rng_states",
    "verify_checkpoint_integrity",
    # Resume
    "resume_from_checkpoint",
    "validate_config_compatibility",
    "check_git_hash_mismatch",
    "add_resume_args",
]
