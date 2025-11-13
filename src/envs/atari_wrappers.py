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
    save_samples: bool = False,
    sample_dir: Optional[Path] = None,
    **env_kwargs
) -> gym.Env:
    """
    Create Atari environment with preprocessing and frame stacking.

    Args:
        env_id: Gymnasium environment ID (e.g., "ALE/Pong-v5")
        frame_size: Target frame size (default: 84)
        num_stack: Number of frames to stack (default: 4)
        save_samples: Whether to save sample frame stacks
        sample_dir: Directory to save samples
        **env_kwargs: Additional arguments for gym.make()

    Returns:
        Wrapped environment with preprocessing and frame stacking
    """
    # Create base environment
    env = gym.make(env_id, **env_kwargs)

    # Apply preprocessing wrapper
    env = AtariPreprocessing(env, frame_size=frame_size, grayscale=True)

    # Apply frame stacking wrapper
    env = FrameStack(
        env,
        num_stack=num_stack,
        save_samples=save_samples,
        sample_dir=sample_dir
    )

    return env
