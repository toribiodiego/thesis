"""Multi-backend metrics logging with unified interface.

Supports simultaneous logging to:
- TensorBoard (for rich visualization)
- Weights & Biases (for experiment tracking)
- CSV (for post-hoc analysis)

All backends use standardized metric names for consistency.
"""

import os
import csv
from typing import Dict, Optional, Any, List
from pathlib import Path


# ============================================================================
# Standardized Metric Keys
# ============================================================================

class MetricKeys:
    """Standardized metric names for consistent logging across backends."""

    # Per-step training metrics
    STEP = "step"
    LOSS = "train/loss"
    TD_ERROR = "train/td_error"
    TD_ERROR_STD = "train/td_error_std"
    GRAD_NORM = "train/grad_norm"
    LEARNING_RATE = "train/learning_rate"
    UPDATE_COUNT = "train/update_count"
    EPSILON = "train/epsilon"
    REPLAY_SIZE = "train/replay_size"
    FPS = "train/fps"
    LOSS_MA = "train/loss_moving_avg"

    # Per-episode metrics
    EPISODE = "episode/number"
    EPISODE_RETURN = "episode/return"
    EPISODE_LENGTH = "episode/length"
    EPISODE_RETURN_MA = "episode/return_moving_avg"
    EPISODE_RETURN_STD = "episode/return_std"
    EPISODE_LENGTH_MA = "episode/length_moving_avg"

    # Evaluation metrics
    EVAL_MEAN_RETURN = "eval/mean_return"
    EVAL_MEDIAN_RETURN = "eval/median_return"
    EVAL_STD_RETURN = "eval/std_return"
    EVAL_MIN_RETURN = "eval/min_return"
    EVAL_MAX_RETURN = "eval/max_return"
    EVAL_MEAN_LENGTH = "eval/mean_length"
    EVAL_NUM_EPISODES = "eval/num_episodes"

    # Q-value tracking
    Q_VALUE_MEAN = "q_values/mean"
    Q_VALUE_STD = "q_values/std"
    Q_VALUE_MIN = "q_values/min"
    Q_VALUE_MAX = "q_values/max"


# ============================================================================
# Backend Implementations
# ============================================================================

class TensorBoardBackend:
    """TensorBoard logging backend using torch.utils.tensorboard."""

    def __init__(self, log_dir: str):
        """
        Initialize TensorBoard writer.

        Args:
            log_dir: Directory for TensorBoard logs
        """
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.enabled = True
        except ImportError:
            print("Warning: torch.utils.tensorboard not available. "
                  "TensorBoard logging disabled.")
            self.writer = None
            self.enabled = False

    def log_scalar(self, key: str, value: float, step: int):
        """Log a scalar metric."""
        if self.enabled and value is not None:
            self.writer.add_scalar(key, value, step)

    def log_scalars(self, metrics: Dict[str, float], step: int):
        """Log multiple scalar metrics."""
        if self.enabled:
            for key, value in metrics.items():
                if value is not None:
                    self.writer.add_scalar(key, value, step)

    def flush(self):
        """Flush buffered events to disk."""
        if self.enabled:
            self.writer.flush()

    def close(self):
        """Close the writer."""
        if self.enabled:
            self.writer.close()


class WandBBackend:
    """Weights & Biases logging backend."""

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
        resume: Optional[str] = None,
        id: Optional[str] = None
    ):
        """
        Initialize W&B run.

        Args:
            project: W&B project name
            name: Run name (optional, auto-generated if None)
            config: Configuration dict to log
            tags: List of tags for run
            resume: Resume mode ('allow', 'must', 'never', None)
            id: Unique run ID for resuming
        """
        try:
            import wandb
            self.wandb = wandb

            # Initialize run
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                tags=tags,
                resume=resume,
                id=id
            )
            self.enabled = True

        except ImportError:
            print("Warning: wandb not installed. "
                  "Weights & Biases logging disabled. "
                  "Install with: pip install wandb")
            self.wandb = None
            self.run = None
            self.enabled = False
        except Exception as e:
            print(f"Warning: Failed to initialize W&B: {e}")
            self.wandb = None
            self.run = None
            self.enabled = False

    def upload_artifact(
        self,
        artifact_name: str,
        artifact_type: str,
        file_paths: List[str],
        metadata: Optional[Dict] = None
    ):
        """
        Upload files as a W&B artifact.

        Args:
            artifact_name: Name for the artifact
            artifact_type: Type of artifact (e.g., 'logs', 'checkpoints')
            file_paths: List of file paths to include in artifact
            metadata: Optional metadata dict for artifact
        """
        if not self.enabled:
            return

        try:
            artifact = self.wandb.Artifact(
                name=artifact_name,
                type=artifact_type,
                metadata=metadata
            )

            # Add files to artifact
            for file_path in file_paths:
                if os.path.exists(file_path):
                    artifact.add_file(file_path)

            # Log artifact
            self.run.log_artifact(artifact)

        except Exception as e:
            print(f"Warning: Failed to upload W&B artifact: {e}")

    def log_scalar(self, key: str, value: float, step: int):
        """Log a scalar metric."""
        if self.enabled and value is not None:
            self.wandb.log({key: value}, step=step)

    def log_scalars(self, metrics: Dict[str, float], step: int):
        """Log multiple scalar metrics."""
        if self.enabled:
            # Filter out None values
            metrics_filtered = {k: v for k, v in metrics.items() if v is not None}
            if metrics_filtered:
                self.wandb.log(metrics_filtered, step=step)

    def flush(self):
        """Flush buffered events (W&B handles this automatically)."""
        pass

    def close(self):
        """Finish the W&B run."""
        if self.enabled and self.run is not None:
            self.run.finish()


class CSVBackend:
    """CSV logging backend for per-step and per-episode metrics."""

    def __init__(self, log_dir: str):
        """
        Initialize CSV files for training metrics.

        Args:
            log_dir: Directory for CSV files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # CSV file paths
        self.step_csv_path = self.log_dir / 'training_steps.csv'
        self.episode_csv_path = self.log_dir / 'episodes.csv'

        # Track initialization
        self._step_csv_initialized = False
        self._episode_csv_initialized = False

        # Track fieldnames for each CSV
        self._step_fieldnames = None
        self._episode_fieldnames = None

        self.enabled = True

    def log_step_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log per-step training metrics to CSV.

        Args:
            metrics: Dict of metric_name -> value
            step: Current step number
        """
        if not self.enabled:
            return

        # Add step to metrics
        log_entry = {'step': step}
        log_entry.update(metrics)

        # Initialize CSV if first write
        if not self._step_csv_initialized:
            # Define all expected fields upfront to avoid dynamic schema issues
            self._step_fieldnames = [
                'step', 'epsilon', 'replay_size', 'fps',
                'loss', 'td_error', 'grad_norm', 'learning_rate',
                'loss_ma'  # moving average
            ]
            with open(self.step_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._step_fieldnames)
                writer.writeheader()
            self._step_csv_initialized = True

        # Append to CSV (only write fields that are in fieldnames)
        filtered_entry = {k: v for k, v in log_entry.items() if k in self._step_fieldnames}
        with open(self.step_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._step_fieldnames)
            writer.writerow(filtered_entry)

    def log_episode_metrics(self, metrics: Dict[str, Any], step: int, episode: int):
        """
        Log per-episode metrics to CSV.

        Args:
            metrics: Dict of metric_name -> value
            step: Current step number
            episode: Current episode number
        """
        if not self.enabled:
            return

        # Add step and episode to metrics
        log_entry = {'episode': episode, 'step': step}
        log_entry.update(metrics)

        # Initialize CSV if first write
        if not self._episode_csv_initialized:
            self._episode_fieldnames = list(log_entry.keys())
            with open(self.episode_csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self._episode_fieldnames)
                writer.writeheader()
            self._episode_csv_initialized = True

        # Append to CSV
        with open(self.episode_csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self._episode_fieldnames)
            writer.writerow(log_entry)

    def flush(self):
        """CSV writes are synchronous, no buffering."""
        pass

    def close(self):
        """CSV files are opened/closed per write, nothing to close."""
        pass


# ============================================================================
# Unified Metrics Logger
# ============================================================================

class MetricsLogger:
    """
    Unified multi-backend metrics logger.

    Logs metrics simultaneously to TensorBoard, W&B, and CSV with
    standardized metric names. Gracefully handles missing backends.

    Parameters
    ----------
    log_dir : str
        Base directory for all logging outputs
    enable_tensorboard : bool
        Enable TensorBoard logging (default: True)
    enable_wandb : bool
        Enable W&B logging (default: False)
    enable_csv : bool
        Enable CSV logging (default: True)
    wandb_project : str, optional
        W&B project name (required if enable_wandb=True)
    wandb_name : str, optional
        W&B run name
    wandb_config : dict, optional
        Configuration dict to log to W&B
    wandb_tags : list, optional
        Tags for W&B run
    wandb_resume : str, optional
        W&B resume mode ('allow', 'must', 'never')
    wandb_id : str, optional
        W&B run ID for resuming
    moving_avg_window : int
        Window size for moving averages (default: 100)
    flush_interval : int
        Steps between automatic flushes (default: 1000)
    upload_artifacts : bool
        Enable automatic W&B artifact uploads (default: False)

    Usage
    -----
    >>> logger = MetricsLogger(
    ...     log_dir='runs/pong_123/logs',
    ...     enable_tensorboard=True,
    ...     enable_wandb=True,
    ...     wandb_project='dqn-atari',
    ...     wandb_name='pong_baseline'
    ... )
    >>>
    >>> # Log per-step metrics
    >>> logger.log_step(
    ...     step=1000,
    ...     loss=0.5,
    ...     epsilon=0.95,
    ...     learning_rate=0.00025,
    ...     replay_size=50000,
    ...     fps=120.0
    ... )
    >>>
    >>> # Log per-episode metrics
    >>> logger.log_episode(
    ...     step=5000,
    ...     episode=10,
    ...     episode_return=21.0,
    ...     episode_length=1200
    ... )
    >>>
    >>> logger.close()
    """

    def __init__(
        self,
        log_dir: str,
        enable_tensorboard: bool = True,
        enable_wandb: bool = False,
        enable_csv: bool = True,
        wandb_project: Optional[str] = None,
        wandb_name: Optional[str] = None,
        wandb_config: Optional[Dict] = None,
        wandb_tags: Optional[List[str]] = None,
        wandb_resume: Optional[str] = None,
        wandb_id: Optional[str] = None,
        moving_avg_window: int = 100,
        flush_interval: int = 1000,
        upload_artifacts: bool = False
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.moving_avg_window = moving_avg_window
        self.flush_interval = flush_interval
        self.upload_artifacts = upload_artifacts

        # Initialize backends
        self.backends = []

        if enable_tensorboard:
            tb_dir = self.log_dir / 'tensorboard'
            tb_dir.mkdir(exist_ok=True)
            self.tensorboard = TensorBoardBackend(str(tb_dir))
            if self.tensorboard.enabled:
                self.backends.append(self.tensorboard)
        else:
            self.tensorboard = None

        if enable_wandb:
            if wandb_project is None:
                print("Warning: wandb_project not specified. W&B logging disabled.")
                self.wandb = None
            else:
                self.wandb = WandBBackend(
                    project=wandb_project,
                    name=wandb_name,
                    config=wandb_config,
                    tags=wandb_tags,
                    resume=wandb_resume,
                    id=wandb_id
                )
                if self.wandb.enabled:
                    self.backends.append(self.wandb)
        else:
            self.wandb = None

        if enable_csv:
            csv_dir = self.log_dir / 'csv'
            csv_dir.mkdir(exist_ok=True)
            self.csv = CSVBackend(str(csv_dir))
            self.backends.append(self.csv)
        else:
            self.csv = None

        # Moving average tracking for loss and episode returns
        self.loss_history = []
        self.episode_return_history = []
        self.episode_length_history = []

        # Episode counter
        self.episode_count = 0

        # Track steps for periodic flush
        self.last_flush_step = 0
        self.last_artifact_upload_step = 0

    def log_step(
        self,
        step: int,
        loss: Optional[float] = None,
        td_error: Optional[float] = None,
        td_error_std: Optional[float] = None,
        grad_norm: Optional[float] = None,
        learning_rate: Optional[float] = None,
        update_count: Optional[int] = None,
        epsilon: Optional[float] = None,
        replay_size: Optional[int] = None,
        fps: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log per-step training metrics to all enabled backends.

        Args:
            step: Current environment step
            loss: Training loss
            td_error: Mean TD error
            td_error_std: Standard deviation of TD errors
            grad_norm: Gradient norm after clipping
            learning_rate: Current learning rate
            update_count: Total number of gradient updates
            epsilon: Current exploration rate
            replay_size: Current replay buffer size
            fps: Frames per second
            extra_metrics: Additional custom metrics to log
        """
        import numpy as np

        # Build standardized metrics dict
        metrics = {}

        # Training metrics
        if loss is not None:
            metrics[MetricKeys.LOSS] = loss

            # Update moving average
            self.loss_history.append(loss)
            if len(self.loss_history) > self.moving_avg_window:
                self.loss_history.pop(0)
            metrics[MetricKeys.LOSS_MA] = np.mean(self.loss_history)

        if td_error is not None:
            metrics[MetricKeys.TD_ERROR] = td_error
        if td_error_std is not None:
            metrics[MetricKeys.TD_ERROR_STD] = td_error_std
        if grad_norm is not None:
            metrics[MetricKeys.GRAD_NORM] = grad_norm
        if learning_rate is not None:
            metrics[MetricKeys.LEARNING_RATE] = learning_rate
        if update_count is not None:
            metrics[MetricKeys.UPDATE_COUNT] = update_count
        if epsilon is not None:
            metrics[MetricKeys.EPSILON] = epsilon
        if replay_size is not None:
            metrics[MetricKeys.REPLAY_SIZE] = replay_size
        if fps is not None:
            metrics[MetricKeys.FPS] = fps

        # Add extra metrics
        if extra_metrics is not None:
            metrics.update(extra_metrics)

        # Log to TensorBoard and W&B
        for backend in self.backends:
            if backend != self.csv:
                backend.log_scalars(metrics, step)

        # Log to CSV (uses different format)
        if self.csv is not None:
            # Convert standardized keys to simple names for CSV
            csv_metrics = {
                'loss': loss,
                'td_error': td_error,
                'td_error_std': td_error_std,
                'grad_norm': grad_norm,
                'learning_rate': learning_rate,
                'update_count': update_count,
                'epsilon': epsilon,
                'replay_size': replay_size,
                'fps': fps,
                'loss_ma': metrics.get(MetricKeys.LOSS_MA)
            }
            # Remove None values
            csv_metrics = {k: v for k, v in csv_metrics.items() if v is not None}
            if extra_metrics:
                csv_metrics.update(extra_metrics)
            self.csv.log_step_metrics(csv_metrics, step)

    def log_episode(
        self,
        step: int,
        episode: int,
        episode_return: float,
        episode_length: int,
        epsilon: Optional[float] = None,
        fps: Optional[float] = None,
        extra_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log per-episode metrics to all enabled backends.

        Args:
            step: Current environment step
            episode: Episode number
            episode_return: Total undiscounted return
            episode_length: Number of steps in episode
            epsilon: Current exploration rate
            fps: Frames per second
            extra_metrics: Additional custom metrics to log
        """
        import numpy as np

        self.episode_count = max(self.episode_count, episode)

        # Update histories
        self.episode_return_history.append(episode_return)
        self.episode_length_history.append(episode_length)

        # Compute rolling statistics
        recent_returns = self.episode_return_history[-self.moving_avg_window:]
        recent_lengths = self.episode_length_history[-self.moving_avg_window:]

        # Build standardized metrics dict
        metrics = {
            MetricKeys.EPISODE: episode,
            MetricKeys.EPISODE_RETURN: episode_return,
            MetricKeys.EPISODE_LENGTH: episode_length,
            MetricKeys.EPISODE_RETURN_MA: np.mean(recent_returns),
            MetricKeys.EPISODE_RETURN_STD: np.std(recent_returns),
            MetricKeys.EPISODE_LENGTH_MA: np.mean(recent_lengths),
        }

        if epsilon is not None:
            metrics[MetricKeys.EPSILON] = epsilon
        if fps is not None:
            metrics[MetricKeys.FPS] = fps

        # Add extra metrics
        if extra_metrics is not None:
            metrics.update(extra_metrics)

        # Log to TensorBoard and W&B
        for backend in self.backends:
            if backend != self.csv:
                backend.log_scalars(metrics, step)

        # Log to CSV (uses different format)
        if self.csv is not None:
            csv_metrics = {
                'return': episode_return,
                'length': episode_length,
                'return_ma': metrics[MetricKeys.EPISODE_RETURN_MA],
                'return_std': metrics[MetricKeys.EPISODE_RETURN_STD],
                'length_ma': metrics[MetricKeys.EPISODE_LENGTH_MA],
            }
            if epsilon is not None:
                csv_metrics['epsilon'] = epsilon
            if fps is not None:
                csv_metrics['fps'] = fps
            if extra_metrics:
                csv_metrics.update(extra_metrics)

            self.csv.log_episode_metrics(csv_metrics, step, episode)

    def log_evaluation(
        self,
        step: int,
        mean_return: float,
        median_return: float,
        std_return: float,
        min_return: float,
        max_return: float,
        mean_length: float,
        num_episodes: int,
        extra_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log evaluation metrics to all enabled backends.

        Args:
            step: Current environment step
            mean_return: Mean episode return over evaluation
            median_return: Median episode return
            std_return: Standard deviation of returns
            min_return: Minimum return
            max_return: Maximum return
            mean_length: Mean episode length
            num_episodes: Number of evaluation episodes
            extra_metrics: Additional custom metrics to log
        """
        metrics = {
            MetricKeys.EVAL_MEAN_RETURN: mean_return,
            MetricKeys.EVAL_MEDIAN_RETURN: median_return,
            MetricKeys.EVAL_STD_RETURN: std_return,
            MetricKeys.EVAL_MIN_RETURN: min_return,
            MetricKeys.EVAL_MAX_RETURN: max_return,
            MetricKeys.EVAL_MEAN_LENGTH: mean_length,
            MetricKeys.EVAL_NUM_EPISODES: num_episodes,
        }

        # Add extra metrics
        if extra_metrics is not None:
            metrics.update(extra_metrics)

        # Log to all backends (TensorBoard, W&B, CSV handles eval separately)
        for backend in self.backends:
            if backend != self.csv:
                backend.log_scalars(metrics, step)

    def log_q_values(
        self,
        step: int,
        q_mean: float,
        q_std: float,
        q_min: float,
        q_max: float,
        extra_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Log Q-value statistics to all enabled backends.

        Args:
            step: Current environment step
            q_mean: Mean Q-value
            q_std: Standard deviation of Q-values
            q_min: Minimum Q-value
            q_max: Maximum Q-value
            extra_metrics: Additional custom metrics to log
        """
        metrics = {
            MetricKeys.Q_VALUE_MEAN: q_mean,
            MetricKeys.Q_VALUE_STD: q_std,
            MetricKeys.Q_VALUE_MIN: q_min,
            MetricKeys.Q_VALUE_MAX: q_max,
        }

        # Add extra metrics
        if extra_metrics is not None:
            metrics.update(extra_metrics)

        # Log to all backends (except CSV for Q-values)
        for backend in self.backends:
            if backend != self.csv:
                backend.log_scalars(metrics, step)

    def _should_flush(self, step: int) -> bool:
        """Check if periodic flush is needed."""
        return (step - self.last_flush_step) >= self.flush_interval

    def _should_upload_artifacts(self, step: int) -> bool:
        """Check if artifact upload is needed."""
        # Upload artifacts at checkpoint intervals (default: 1M steps)
        upload_interval = 1_000_000
        return self.upload_artifacts and (step - self.last_artifact_upload_step) >= upload_interval

    def flush(self, step: Optional[int] = None):
        """
        Flush all backends.

        Args:
            step: Current step (optional, for tracking last flush)
        """
        for backend in self.backends:
            backend.flush()

        if step is not None:
            self.last_flush_step = step

    def upload_logs_as_artifacts(self, step: int, metadata: Optional[Dict] = None):
        """
        Upload training logs and run artifacts to W&B.

        Args:
            step: Current training step
            metadata: Optional metadata to attach to artifact
        """
        if not self.upload_artifacts or self.wandb is None or not self.wandb.enabled:
            return

        # Collect artifact files with size check
        artifact_files = []
        total_size_mb = 0.0

        # Add CSV files
        if self.csv is not None:
            if self.csv.step_csv_path.exists():
                artifact_files.append(str(self.csv.step_csv_path))
                total_size_mb += self.csv.step_csv_path.stat().st_size / (1024 * 1024)

            if self.csv.episode_csv_path.exists():
                artifact_files.append(str(self.csv.episode_csv_path))
                total_size_mb += self.csv.episode_csv_path.stat().st_size / (1024 * 1024)

        # Add config and meta files from run directory
        config_path = self.log_dir / 'config.yaml'
        if config_path.exists():
            artifact_files.append(str(config_path))
            total_size_mb += config_path.stat().st_size / (1024 * 1024)

        meta_path = self.log_dir / 'meta.json'
        if meta_path.exists():
            artifact_files.append(str(meta_path))
            total_size_mb += meta_path.stat().st_size / (1024 * 1024)

        # Add evaluation results
        eval_csv = self.log_dir / 'eval' / 'evaluations.csv'
        if eval_csv.exists():
            artifact_files.append(str(eval_csv))
            total_size_mb += eval_csv.stat().st_size / (1024 * 1024)

        eval_jsonl = self.log_dir / 'eval' / 'evaluations.jsonl'
        if eval_jsonl.exists():
            artifact_files.append(str(eval_jsonl))
            total_size_mb += eval_jsonl.stat().st_size / (1024 * 1024)

        if not artifact_files:
            return

        # Warn if uploading large files
        if total_size_mb > 100.0:
            print(f"Warning: Uploading large artifact ({total_size_mb:.1f} MB). "
                  f"This may take some time.")

        # Create artifact name with step
        artifact_name = f"training_logs_step_{step}"

        # Add step to metadata
        artifact_metadata = {"step": step}
        if metadata:
            artifact_metadata.update(metadata)

        # Upload artifact
        self.wandb.upload_artifact(
            artifact_name=artifact_name,
            artifact_type="logs",
            file_paths=artifact_files,
            metadata=artifact_metadata
        )

        self.last_artifact_upload_step = step

    def maybe_flush_and_upload(self, step: int, force: bool = False):
        """
        Conditionally flush and upload artifacts based on step intervals.

        Args:
            step: Current training step
            force: If True, flush and upload regardless of interval
        """
        # Periodic flush
        if force or self._should_flush(step):
            self.flush(step)

        # Periodic artifact upload
        if force or self._should_upload_artifacts(step):
            self.upload_logs_as_artifacts(step)

    def close(self):
        """Close all backends and perform final flush."""
        # Final flush
        self.flush()

        # Close all backends
        for backend in self.backends:
            backend.close()
