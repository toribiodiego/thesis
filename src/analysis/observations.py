"""Collect observations from Atari environments for offline analysis.

Runs a trained agent's greedy policy (or random policy) in an Atari
environment, collecting frame-stacked observations, actions, rewards,
and episode boundaries. Observations are stored in the network's
expected HWC format (84, 84, 4) as uint8.
"""

import os
from dataclasses import dataclass
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from src.analysis.checkpoint import CheckpointData
from src.envs import make_atari_env


@dataclass
class CollectedData:
    """Observations and metadata collected from environment rollouts.

    Attributes:
        observations: (N, 84, 84, 4) uint8 stacked frames in HWC format.
        actions: (N,) int32 actions taken.
        rewards: (N,) float32 rewards received (clipped).
        terminals: (N,) bool, True when episode ended.
        episode_returns: list of float, total return per episode.
    """

    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray
    episode_returns: list


def _game_to_env_id(game: str) -> str:
    """Convert a game name to an ALE Gymnasium environment ID.

    Accepts CamelCase ('CrazyClimber') or snake_case ('crazy_climber').
    Returns 'ALE/CrazyClimber-v5'.
    """
    if "_" in game or game[0].islower():
        # snake_case or lowercase -> CamelCase
        game = "".join(word.capitalize() for word in game.split("_"))
    return f"ALE/{game}-v5"


def _build_q_fn(checkpoint: CheckpointData):
    """Build a JIT-compiled function that returns Q-values for an observation.

    Args:
        checkpoint: Loaded checkpoint data.

    Returns:
        A function (obs_hwc_uint8, rng) -> q_values array of shape
        (num_actions,).
    """
    net = checkpoint.network_def
    params = {"params": checkpoint.online_params}
    support = checkpoint.support

    def _forward(obs_batch, rng):
        """obs_batch: (1, 84, 84, 4) float32 in [0, 1]."""
        result = net.apply(
            params,
            obs_batch,
            support=support,
            eval_mode=True,
            key=rng,
            rngs={"dropout": rng},
            mutable=["batch_stats"],
        )
        output = result[0]
        return output.q_values

    return jax.jit(_forward)


def collect_greedy(
    checkpoint: CheckpointData,
    game: str,
    num_steps: int = 10_000,
    epsilon: float = 0.05,
    seed: int = 0,
    noop_max: int = 30,
) -> CollectedData:
    """Collect observations using an epsilon-greedy policy.

    Runs the trained network in the Atari environment, selecting
    argmax-Q actions with probability (1 - epsilon) and random
    actions with probability epsilon. This matches the standard
    Atari-100K evaluation protocol (epsilon_eval = 0.001) but
    uses a slightly higher epsilon (0.05) to increase state
    diversity for analysis.

    Args:
        checkpoint: Loaded checkpoint (from load_checkpoint).
        game: Game name, CamelCase ('CrazyClimber') or
            snake_case ('crazy_climber').
        num_steps: Number of environment steps to collect.
        epsilon: Exploration rate (default 0.05).
        seed: Random seed for environment and action selection.
        noop_max: Maximum no-ops on reset (default 30).

    Returns:
        CollectedData with observations, actions, rewards, terminals.
    """
    env_id = _game_to_env_id(game)
    env = make_atari_env(
        env_id,
        noop_max=noop_max,
        clip_rewards=True,
        episode_life=False,
        repeat_action_probability=0.0,
    )

    q_fn = _build_q_fn(checkpoint)
    rng = jax.random.PRNGKey(seed)
    np_rng = np.random.RandomState(seed)

    observations = np.empty((num_steps, 84, 84, 4), dtype=np.uint8)
    actions = np.empty(num_steps, dtype=np.int32)
    rewards = np.empty(num_steps, dtype=np.float32)
    terminals = np.empty(num_steps, dtype=bool)
    episode_returns = []

    obs_chw, _ = env.reset(seed=seed)
    obs_hwc = np.transpose(obs_chw, (1, 2, 0))  # CHW -> HWC
    episode_return = 0.0

    for t in range(num_steps):
        observations[t] = obs_hwc

        # Epsilon-greedy action selection
        if np_rng.random() < epsilon:
            action = np_rng.randint(checkpoint.num_actions)
        else:
            rng, rng_act = jax.random.split(rng)
            obs_batch = obs_hwc[np.newaxis].astype(np.float32) / 255.0
            q_values = q_fn(obs_batch, rng_act)
            action = int(jnp.argmax(q_values))

        obs_chw_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        actions[t] = action
        rewards[t] = reward
        terminals[t] = done
        episode_return += reward

        if done:
            episode_returns.append(episode_return)
            episode_return = 0.0
            obs_chw, _ = env.reset()
            obs_hwc = np.transpose(obs_chw, (1, 2, 0))
        else:
            obs_hwc = np.transpose(obs_chw_next, (1, 2, 0))

    env.close()

    return CollectedData(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminals=terminals,
        episode_returns=episode_returns,
    )
