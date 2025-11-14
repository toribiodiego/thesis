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
    TrainingScheduler,
    EpsilonScheduler
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
    perform_update_step
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
    'MetadataWriter'
]
