"""Training utilities for DQN."""

# Target network management
from .target_network import (
    hard_update_target,
    init_target_network
)

# Loss computation
from .loss import (
    compute_td_targets,
    select_q_values,
    compute_td_loss_components,
    compute_dqn_loss
)

# Optimization
from .optimization import (
    configure_optimizer,
    clip_gradients
)

# Schedulers
from .schedulers import (
    TargetNetworkUpdater,
    TrainingScheduler
)

# Stability checks
from .stability import (
    detect_nan_inf,
    validate_loss_decrease,
    verify_target_sync_schedule
)

# Metrics
from .metrics import (
    UpdateMetrics,
    perform_update_step,
    EpsilonScheduler
)

# Training loop
from .training_loop import (
    select_epsilon_greedy_action,
    FrameCounter,
    training_step
)

# Logging
from .logging import (
    StepLogger,
    EpisodeLogger,
    CheckpointManager
)

# Evaluation
from .evaluation import (
    evaluate,
    EvaluationScheduler,
    EvaluationLogger
)

# Q-value tracking
from .q_tracking import (
    ReferenceStateQTracker,
    ReferenceQLogger
)

# Metadata
from .metadata import (
    get_git_commit_hash,
    get_git_status,
    MetadataWriter
)

# Checkpoint utilities
from .checkpoint_utils import (
    get_rng_states,
    set_rng_states,
    verify_checkpoint_integrity
)

# Resume functionality
from .resume import (
    resume_from_checkpoint,
    validate_config_compatibility,
    check_git_hash_mismatch,
    add_resume_args
)

__all__ = [
    # Target network
    'hard_update_target',
    'init_target_network',
    # Loss
    'compute_td_targets',
    'select_q_values',
    'compute_td_loss_components',
    'compute_dqn_loss',
    # Optimization
    'configure_optimizer',
    'clip_gradients',
    # Schedulers
    'TargetNetworkUpdater',
    'TrainingScheduler',
    'EpsilonScheduler',
    # Stability
    'detect_nan_inf',
    'validate_loss_decrease',
    'verify_target_sync_schedule',
    # Metrics
    'UpdateMetrics',
    'perform_update_step',
    # Training loop
    'select_epsilon_greedy_action',
    'FrameCounter',
    'training_step',
    # Logging
    'StepLogger',
    'EpisodeLogger',
    'CheckpointManager',
    # Evaluation
    'evaluate',
    'EvaluationScheduler',
    'EvaluationLogger',
    # Q-tracking
    'ReferenceStateQTracker',
    'ReferenceQLogger',
    # Metadata
    'get_git_commit_hash',
    'get_git_status',
    'MetadataWriter',
    # Checkpoint utilities
    'get_rng_states',
    'set_rng_states',
    'verify_checkpoint_integrity',
    # Resume
    'resume_from_checkpoint',
    'validate_config_compatibility',
    'check_git_hash_mismatch',
    'add_resume_args'
]
