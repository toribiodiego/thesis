"""
Atari environment wrappers for DQN preprocessing.

Implements frame preprocessing (grayscale, resize) and frame stacking
following the DQN 2013 paper specification.
"""

from collections import deque
from pathlib import Path
from typing import Optional, Tuple

import cv2
import gymnasium as gym
import numpy as np


class MaxAndSkipEnv(gym.Wrapper):
    """
    Action repeat with max-pooling over last 2 frames to reduce flicker.

    Repeat each action for K steps (default: 4) and return the max-pooled
    observation from the last two frames. This reduces flickering artifacts
    common in Atari games.

    Args:
        env: Gymnasium environment
        skip: Number of times to repeat action (default: 4)

    Returns:
        Max-pooled observation from last 2 frames, accumulated reward
    """

    def __init__(self, env: gym.Env, skip: int = 4):
        super().__init__(env)
        self._skip = skip
        # Buffer to store last 2 frames for max-pooling
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """
        Repeat action for K steps, accumulate reward, and max-pool last 2 frames.

        Args:
            action: Action to repeat

        Returns:
            obs: Max-pooled observation from last 2 frames
            total_reward: Accumulated reward over all steps
            terminated: Whether episode terminated
            truncated: Whether episode was truncated
            info: Info dict from last step
        """
        total_reward = 0.0
        terminated = truncated = False
        info = {}

        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward

            # Store last 2 frames in buffer for max-pooling
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            elif i == self._skip - 1:
                self._obs_buffer[1] = obs

            # Exit early if episode ends
            if terminated or truncated:
                break

        # Max-pool over last 2 frames (element-wise maximum)
        # This reduces flickering by taking the brightest pixel at each position
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and clear observation buffer."""
        obs, info = self.env.reset(**kwargs)
        # Fill buffer with initial observation
        self._obs_buffer[0] = obs
        self._obs_buffer[1] = obs
        return obs, info


class AtariPreprocessing(gym.ObservationWrapper):
    """
    Preprocess Atari frames: RGB (210×160×3) → grayscale 84×84.

    Following DQN 2013 paper:
    - Convert RGB to grayscale using luminance formula
    - Resize to 84×84 using bilinear interpolation
    - Return uint8 [0, 255] range (converted to float32 later)

    Args:
        env: Gymnasium environment
        frame_size: Output frame size (default: 84)
        grayscale: Whether to convert to grayscale (default: True)

    Returns:
        Preprocessed observation of shape (84, 84) in uint8
    """

    def __init__(self, env: gym.Env, frame_size: int = 84, grayscale: bool = True):
        super().__init__(env)
        self.frame_size = frame_size
        self.grayscale = grayscale

        # Update observation space
        if grayscale:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(frame_size, frame_size),
                dtype=np.uint8
            )
        else:
            self.observation_space = gym.spaces.Box(
                low=0,
                high=255,
                shape=(frame_size, frame_size, 3),
                dtype=np.uint8
            )

    def observation(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess a single frame.

        Args:
            frame: RGB frame of shape (210, 160, 3) in uint8

        Returns:
            Preprocessed frame of shape (84, 84) in uint8
        """
        if self.grayscale:
            # Convert RGB to grayscale using OpenCV (uses standard luminance formula)
            # Y = 0.299*R + 0.587*G + 0.114*B
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize to target size using bilinear interpolation
        frame = cv2.resize(
            frame,
            (self.frame_size, self.frame_size),
            interpolation=cv2.INTER_LINEAR
        )

        return frame


class FrameStack(gym.Wrapper):
    """
    Stack the last K frames as observation.

    Stores frames in uint8 to save memory, converts to float32 on retrieval.
    Implements channels-first format: (K, H, W) for PyTorch compatibility.

    Args:
        env: Gymnasium environment (should already be preprocessed to grayscale)
        num_stack: Number of frames to stack (default: 4)
        save_samples: If True, saves sample stacks to disk
        sample_dir: Directory to save sample stacks
        max_samples: Maximum number of sample stacks to save

    Returns:
        Stacked observation of shape (num_stack, 84, 84) in uint8
    """

    def __init__(
        self,
        env: gym.Env,
        num_stack: int = 4,
        save_samples: bool = False,
        sample_dir: Optional[Path] = None,
        max_samples: int = 5
    ):
        super().__init__(env)
        self.num_stack = num_stack
        self.save_samples = save_samples
        self.sample_dir = Path(sample_dir) if sample_dir else None
        self.max_samples = max_samples
        self.num_saved = 0

        # Create sample directory if needed
        if self.save_samples and self.sample_dir:
            self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Initialize frame buffer (stores uint8 frames)
        self.frames = deque(maxlen=num_stack)

        # Update observation space to channels-first (K, H, W)
        low = np.repeat(self.observation_space.low[np.newaxis, ...], num_stack, axis=0)
        high = np.repeat(self.observation_space.high[np.newaxis, ...], num_stack, axis=0)

        self.observation_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.uint8
        )

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        """Reset environment and initialize frame stack."""
        obs, info = self.env.reset(**kwargs)

        # Fill frame buffer with initial observation
        for _ in range(self.num_stack):
            self.frames.append(obs)

        stacked_obs = self._get_observation()

        # Save sample if requested
        if self.save_samples and self.num_saved < self.max_samples:
            self._save_sample_stack(stacked_obs, f"reset_{self.num_saved}")
            self.num_saved += 1

        return stacked_obs, info

    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action and return stacked observation."""
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Add new frame to buffer (oldest is automatically dropped)
        self.frames.append(obs)

        stacked_obs = self._get_observation()

        return stacked_obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """
        Stack frames in channels-first format.

        Returns:
            Stacked frames of shape (num_stack, H, W) in uint8
        """
        # Stack frames along first axis (channels-first for PyTorch)
        # Keeps uint8 dtype for memory efficiency
        return np.stack(self.frames, axis=0)

    def _save_sample_stack(self, stacked_obs: np.ndarray, prefix: str) -> None:
        """
        Save individual frames from a stack as PNG images.

        Args:
            stacked_obs: Stacked observation of shape (num_stack, H, W)
            prefix: Filename prefix for saved images
        """
        if self.sample_dir is None:
            return

        for i in range(self.num_stack):
            frame = stacked_obs[i]  # Shape: (H, W)

            # Save as PNG using OpenCV
            filename = self.sample_dir / f"{prefix}_frame_{i}.png"
            cv2.imwrite(str(filename), frame)

        print(f"Saved sample stack to {self.sample_dir}/{prefix}_frame_*.png")

    @staticmethod
    def to_float32(obs: np.ndarray) -> np.ndarray:
        """
        Convert uint8 observation to float32 in [0, 1] range.

        Args:
            obs: Observation in uint8 [0, 255]

        Returns:
            Observation in float32 [0.0, 1.0]
        """
        return obs.astype(np.float32) / 255.0


def make_atari_env(
    env_id: str,
    frame_size: int = 84,
    num_stack: int = 4,
    frame_skip: int = 4,
    save_samples: bool = False,
    sample_dir: Optional[Path] = None,
    **env_kwargs
) -> gym.Env:
    """
    Create Atari environment with preprocessing and frame stacking.

    Applies wrappers in order:
    1. MaxAndSkipEnv - Action repeat (4x) with max-pooling over last 2 frames
    2. AtariPreprocessing - Grayscale conversion and resize to 84×84
    3. FrameStack - Stack last 4 frames in channels-first format

    Args:
        env_id: Gymnasium environment ID (e.g., "ALE/Pong-v5")
        frame_size: Target frame size (default: 84)
        num_stack: Number of frames to stack (default: 4)
        frame_skip: Action repeat count (default: 4)
        save_samples: Whether to save sample frame stacks
        sample_dir: Directory to save samples
        **env_kwargs: Additional arguments for gym.make()

    Returns:
        Wrapped environment with preprocessing and frame stacking
    """
    # Create base environment
    env = gym.make(env_id, **env_kwargs)

    # Apply action repeat with max-pooling wrapper (reduces flicker)
    env = MaxAndSkipEnv(env, skip=frame_skip)

    # Apply preprocessing wrapper (grayscale + resize)
    env = AtariPreprocessing(env, frame_size=frame_size, grayscale=True)

    # Apply frame stacking wrapper
    env = FrameStack(
        env,
        num_stack=num_stack,
        save_samples=save_samples,
        sample_dir=sample_dir
    )

    return env
