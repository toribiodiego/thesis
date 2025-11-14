"""Evaluation system for periodic performance assessment.
"""

import os
import csv
import json
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Any, List, Callable
from pathlib import Path


class VideoRecorder:
    """
    Records video frames during episode execution.

    Saves frames to MP4 format and optionally exports to GIF.

    Parameters
    ----------
    output_path : str
        Path where video will be saved (e.g., 'videos/pong_250000.mp4')
    fps : int
        Frames per second for video (default: 30)
    export_gif : bool
        Also export video as GIF (default: False)
    """

    def __init__(self, output_path: str, fps: int = 30, export_gif: bool = False):
        self.output_path = Path(output_path)
        self.fps = fps
        self.export_gif = export_gif
        self.frames = []

        # Create output directory
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def capture_frame(self, frame: np.ndarray):
        """Capture a single frame."""
        # Store as RGB uint8
        if frame.dtype == np.float32 or frame.dtype == np.float64:
            frame = (frame * 255).astype(np.uint8)
        self.frames.append(frame.copy())

    def save(self):
        """Save recorded frames to MP4 (and optionally GIF)."""
        if not self.frames:
            return None

        # Save MP4 using OpenCV
        import cv2

        # Get frame dimensions
        height, width = self.frames[0].shape[:2]
        if len(self.frames[0].shape) == 2:
            # Grayscale - convert to RGB
            self.frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2RGB) for f in self.frames]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (width, height)
        )

        # Write frames
        for frame in self.frames:
            # OpenCV expects BGR
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        video_writer.release()

        # Optionally export GIF
        gif_path = None
        if self.export_gif:
            gif_path = self.output_path.with_suffix('.gif')
            self._export_gif(gif_path)

        return {
            'video_path': str(self.output_path),
            'gif_path': str(gif_path) if gif_path else None,
            'num_frames': len(self.frames),
            'fps': self.fps
        }

    def _export_gif(self, gif_path: Path):
        """Export frames to GIF format."""
        try:
            from PIL import Image

            # Convert frames to PIL Images
            pil_frames = [Image.fromarray(f) for f in self.frames]

            # Save as GIF
            pil_frames[0].save(
                str(gif_path),
                save_all=True,
                append_images=pil_frames[1:],
                duration=int(1000 / self.fps),  # Duration in milliseconds
                loop=0
            )
        except ImportError:
            # PIL/Pillow not available, skip GIF export
            pass


def evaluate(
    env,
    model: torch.nn.Module,
    num_episodes: int = 10,
    eval_epsilon: float = 0.05,
    num_actions: int = None,
    device: str = 'cpu',
    seed: int = None,
    step: int = None,
    track_lives: bool = False,
    record_video: bool = False,
    video_dir: str = None,
    video_fps: int = 30,
    export_gif: bool = False,
    render_mode: str = 'rgb_array'
) -> dict:
    """
    Evaluate agent over multiple episodes with low/greedy epsilon.

    Runs the agent in evaluation mode (no learning) and computes
    performance statistics over multiple episodes.

    Parameters
    ----------
    env : gym.Env
        Evaluation environment (should NOT have EpisodicLifeEnv wrapper)
    model : torch.nn.Module
        Q-network to evaluate
    num_episodes : int
        Number of episodes to run (default: 10)
    eval_epsilon : float
        Exploration rate during evaluation (default: 0.05, use 0.0 for greedy)
    num_actions : int
        Number of available actions (if None, inferred from env)
    device : str
        Device for model inference (default: 'cpu')
    seed : int
        Random seed for reproducibility (optional)
    step : int
        Training step when evaluation occurred (optional, included in results)
    track_lives : bool
        Track lives lost per episode (optional, default: False)
    record_video : bool
        Record video of first evaluation episode (default: False)
    video_dir : str
        Directory to save videos (default: 'results/videos')
    video_fps : int
        Frames per second for video (default: 30)
    export_gif : bool
        Also export video as GIF (default: False)
    render_mode : str
        Render mode for capturing frames (default: 'rgb_array')

    Returns
    -------
    dict
        Evaluation results containing:
        - mean_return: Average episode return
        - median_return: Median episode return
        - std_return: Standard deviation of returns
        - min_return: Minimum episode return
        - max_return: Maximum episode return
        - mean_length: Average episode length
        - episode_returns: List of individual episode returns
        - episode_lengths: List of individual episode lengths
        - episode_lives_lost: List of lives lost per episode (if track_lives=True)
        - num_episodes: Number of episodes evaluated
        - seed: Random seed used (if provided)
        - step: Training step when evaluation occurred (if provided)
        - eval_epsilon: Epsilon value used for evaluation
        - video_info: Video recording metadata (if record_video=True)

    Example
    -------
    >>> results = evaluate(eval_env, model, num_episodes=10, eval_epsilon=0.05)
    >>> print(f"Mean return: {results['mean_return']:.2f}")
    """
    import numpy as np

    # Set model to eval mode
    model.eval()

    # Infer num_actions if not provided
    if num_actions is None:
        num_actions = env.action_space.n

    # Set seed if provided
    if seed is not None:
        env.reset(seed=seed)

    episode_returns = []
    episode_lengths = []
    episode_lives_lost = [] if track_lives else None

    # Setup video recording for first episode if requested
    video_recorder = None
    video_info = None
    if record_video:
        # Determine video filename
        if video_dir is None:
            video_dir = 'results/videos'

        # Get environment name for filename
        env_name = getattr(env.unwrapped, 'spec', None)
        if env_name is not None and hasattr(env_name, 'id'):
            env_name = env_name.id.replace('/', '_').replace('NoFrameskip-v4', '')
        else:
            env_name = 'env'

        # Create filename with step if provided
        if step is not None:
            video_filename = f"{env_name}_step_{step}.mp4"
        else:
            video_filename = f"{env_name}_eval.mp4"

        video_path = os.path.join(video_dir, video_filename)
        video_recorder = VideoRecorder(video_path, fps=video_fps, export_gif=export_gif)

    for ep in range(num_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False

        # Track initial lives if requested
        if track_lives:
            initial_lives = info.get('lives', None)
            if initial_lives is None and hasattr(env.unwrapped, 'ale'):
                initial_lives = env.unwrapped.ale.lives()

        while not done:
            # Convert observation to tensor
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float().to(device)
                # Normalize if needed
                if obs_tensor.max() > 1.0:
                    obs_tensor = obs_tensor / 255.0
            else:
                obs_tensor = obs.float().to(device)

            # Select action with eval epsilon
            with torch.no_grad():
                if torch.rand(1).item() < eval_epsilon:
                    # Random action
                    action = env.action_space.sample()
                else:
                    # Greedy action
                    if obs_tensor.dim() == 3:
                        obs_tensor = obs_tensor.unsqueeze(0)
                    output = model(obs_tensor)
                    q_values = output['q_values']
                    action = q_values.argmax(dim=1).item()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_length += 1
            done = terminated or truncated

            # Capture video frame for first episode
            if video_recorder is not None and ep == 0:
                try:
                    frame = env.render()
                    if frame is not None:
                        video_recorder.capture_frame(frame)
                except Exception:
                    # Rendering not supported or failed, skip
                    pass

        # Record episode statistics
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)

        # Track lives lost if requested
        if track_lives:
            final_lives = info.get('lives', None)
            if final_lives is None and hasattr(env.unwrapped, 'ale'):
                final_lives = env.unwrapped.ale.lives()

            if initial_lives is not None and final_lives is not None:
                lives_lost = initial_lives - final_lives
            else:
                lives_lost = None
            episode_lives_lost.append(lives_lost)

        # Save video after first episode
        if video_recorder is not None and ep == 0:
            video_info = video_recorder.save()

    # Compute statistics
    results = {
        'mean_return': np.mean(episode_returns),
        'median_return': np.median(episode_returns),
        'std_return': np.std(episode_returns),
        'min_return': np.min(episode_returns),
        'max_return': np.max(episode_returns),
        'mean_length': np.mean(episode_lengths),
        'episode_returns': episode_returns,
        'episode_lengths': episode_lengths,
        'num_episodes': num_episodes,
        'eval_epsilon': eval_epsilon
    }

    # Add optional metadata
    if seed is not None:
        results['seed'] = seed

    if step is not None:
        results['step'] = step

    if track_lives and episode_lives_lost is not None:
        results['episode_lives_lost'] = episode_lives_lost

    if video_info is not None:
        results['video_info'] = video_info

    # Set model back to train mode
    model.train()

    return results


class EvaluationScheduler:
    """
    Scheduler for periodic evaluation during training.

    Triggers evaluation at regular intervals (frame-based or wall-clock)
    and tracks evaluation history with metadata.

    Parameters
    ----------
    eval_interval : int
        Steps between evaluations (default: 250,000)
    num_episodes : int
        Number of episodes per evaluation (default: 10)
    eval_epsilon : float
        Exploration rate during evaluation (default: 0.05)
    wall_clock_interval : float
        Optional wall-clock time between evaluations in seconds (default: None)
        If set, evaluations trigger based on elapsed time instead of steps

    Usage
    -----
    >>> # Frame-based scheduling (default)
    >>> scheduler = EvaluationScheduler(eval_interval=250000, num_episodes=10)
    >>> if scheduler.should_evaluate(current_step):
    ...     results = evaluate(eval_env, model, num_episodes=scheduler.num_episodes,
    ...                        eval_epsilon=scheduler.eval_epsilon)
    ...     scheduler.record_evaluation(current_step, results)
    >>>
    >>> # Wall-clock scheduling (every 30 minutes)
    >>> scheduler = EvaluationScheduler(wall_clock_interval=1800, num_episodes=10)
    """

    def __init__(
        self,
        eval_interval: int = 250_000,
        num_episodes: int = 10,
        eval_epsilon: float = 0.05,
        wall_clock_interval: float = None
    ):
        self.eval_interval = eval_interval
        self.num_episodes = num_episodes
        self.eval_epsilon = eval_epsilon
        self.wall_clock_interval = wall_clock_interval

        # Track evaluation history
        self.eval_steps = []
        self.eval_returns = []
        self.eval_timestamps = []
        self.last_eval_step = 0
        self.last_eval_time = None

        # Metadata tracking
        import time
        self.start_time = time.time()

    def should_evaluate(self, step: int) -> bool:
        """
        Check if evaluation should be performed at this step.

        Supports both frame-based and wall-clock scheduling:
        - Frame-based: Triggers every eval_interval steps
        - Wall-clock: Triggers every wall_clock_interval seconds

        Args:
            step: Current environment step

        Returns:
            True if should evaluate, False otherwise
        """
        if step == 0:
            return False

        # Wall-clock based scheduling
        if self.wall_clock_interval is not None:
            import time
            current_time = time.time()

            # First evaluation
            if self.last_eval_time is None:
                return True

            # Check if enough time has elapsed
            elapsed = current_time - self.last_eval_time
            if elapsed >= self.wall_clock_interval:
                return True

            return False

        # Frame-based scheduling (default)
        if step >= self.eval_interval and step % self.eval_interval == 0:
            # Avoid duplicate evaluations
            return step != self.last_eval_step

        return False

    def record_evaluation(self, step: int, results: dict):
        """
        Record evaluation results with timestamp metadata.

        Args:
            step: Environment step when evaluation occurred
            results: Dictionary returned by evaluate()
        """
        import time
        current_time = time.time()

        self.last_eval_step = step
        self.last_eval_time = current_time

        self.eval_steps.append(step)
        self.eval_returns.append(results['mean_return'])
        self.eval_timestamps.append(current_time)

    def get_best_return(self) -> float:
        """Get best mean return across all evaluations."""
        if not self.eval_returns:
            return float('-inf')
        return max(self.eval_returns)

    def get_recent_trend(self, n: int = 3) -> str:
        """
        Get recent performance trend.

        Args:
            n: Number of recent evaluations to consider

        Returns:
            'improving', 'declining', or 'stable'
        """
        if len(self.eval_returns) < n:
            return 'insufficient_data'

        recent = self.eval_returns[-n:]
        if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
            return 'improving'
        elif all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
            return 'declining'
        else:
            return 'stable'

    def get_schedule_metadata(self) -> dict:
        """
        Get evaluation schedule metadata for logging.

        Returns:
            dict: Schedule configuration and history
        """
        import time

        metadata = {
            'schedule_type': 'wall_clock' if self.wall_clock_interval is not None else 'frame_based',
            'eval_interval': self.eval_interval,
            'wall_clock_interval': self.wall_clock_interval,
            'num_episodes': self.num_episodes,
            'eval_epsilon': self.eval_epsilon,
            'total_evaluations': len(self.eval_steps),
            'eval_steps': self.eval_steps.copy(),
            'eval_returns': self.eval_returns.copy(),
        }

        # Add timing information
        if self.eval_timestamps:
            elapsed_times = [t - self.start_time for t in self.eval_timestamps]
            metadata['eval_timestamps'] = self.eval_timestamps.copy()
            metadata['elapsed_times'] = elapsed_times
            metadata['total_elapsed_time'] = time.time() - self.start_time

        return metadata


class EvaluationLogger:
    """
    Logger for evaluation results.

    Saves evaluation statistics to CSV and JSON files.

    Parameters
    ----------
    log_dir : str
        Directory to save evaluation logs

    Usage
    -----
    >>> logger = EvaluationLogger(log_dir='runs/pong_123/eval')
    >>> logger.log_evaluation(step=250000, results=eval_results)
    """

    def __init__(self, log_dir: str):
        import os
        self.log_dir = log_dir

        # Create log directory
        os.makedirs(log_dir, exist_ok=True)

        # CSV file for evaluation summary
        self.csv_path = os.path.join(log_dir, 'evaluations.csv')
        self._csv_initialized = False

        # JSON directory for detailed per-eval results
        self.json_dir = os.path.join(log_dir, 'detailed')
        os.makedirs(self.json_dir, exist_ok=True)

    def log_evaluation(self, step: int, results: dict, epsilon: float = None):
        """
        Log evaluation results to CSV and JSON.

        Args:
            step: Environment step when evaluation occurred
            results: Dictionary returned by evaluate()
            epsilon: Current training epsilon (optional)
        """
        import csv
        import json
        import os

        # Prepare CSV entry (summary statistics)
        csv_entry = {
            'step': step,
            'mean_return': results['mean_return'],
            'median_return': results['median_return'],
            'std_return': results['std_return'],
            'min_return': results['min_return'],
            'max_return': results['max_return'],
            'mean_length': results['mean_length'],
            'num_episodes': results['num_episodes']
        }

        if epsilon is not None:
            csv_entry['training_epsilon'] = epsilon

        # Write CSV
        if not self._csv_initialized:
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=csv_entry.keys())
                writer.writeheader()
            self._csv_initialized = True

        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=csv_entry.keys())
            writer.writerow(csv_entry)

        # Save detailed results to JSON
        json_path = os.path.join(self.json_dir, f'eval_step_{step}.json')
        detailed_results = {
            'step': step,
            'statistics': {
                'mean_return': float(results['mean_return']),
                'median_return': float(results['median_return']),
                'std_return': float(results['std_return']),
                'min_return': float(results['min_return']),
                'max_return': float(results['max_return']),
                'mean_length': float(results['mean_length'])
            },
            'episode_returns': [float(r) for r in results['episode_returns']],
            'episode_lengths': [int(l) for l in results['episode_lengths']],
            'num_episodes': results['num_episodes']
        }

        if epsilon is not None:
            detailed_results['training_epsilon'] = epsilon

        with open(json_path, 'w') as f:
            json.dump(detailed_results, f, indent=2)

    def get_all_results(self) -> list:
        """
        Load all evaluation results from CSV.

        Returns:
            List of dictionaries with evaluation statistics
        """
        import csv
        import os

        if not os.path.exists(self.csv_path):
            return []

        with open(self.csv_path, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)


# ============================================================================
# Reference-State Q Tracking
# ============================================================================
