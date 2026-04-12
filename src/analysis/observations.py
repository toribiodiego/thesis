"""Collect observations from Atari environments for offline analysis.

Runs a trained agent's greedy policy (or random policy) in an Atari
environment, collecting frame-stacked observations, actions, rewards,
and episode boundaries. Observations are stored in the network's
expected HWC format (84, 84, 4) as uint8.
"""

from dataclasses import dataclass

import numpy as np

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


def _run_collection_loop(env, num_steps, action_fn, seed):
    """Shared environment loop for both greedy and random collection.

    Args:
        env: Wrapped Atari environment (outputs CHW observations).
        num_steps: Number of steps to collect.
        action_fn: Callable (obs_hwc, step) -> int action.
        seed: Seed for env.reset().

    Returns:
        CollectedData.
    """
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
        action = action_fn(obs_hwc, t)

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


def collect_greedy(
    checkpoint,
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
        checkpoint: CheckpointData from load_checkpoint.
        game: Game name, CamelCase ('CrazyClimber') or
            snake_case ('crazy_climber').
        num_steps: Number of environment steps to collect.
        epsilon: Exploration rate (default 0.05).
        seed: Random seed for environment and action selection.
        noop_max: Maximum no-ops on reset (default 30).

    Returns:
        CollectedData with observations, actions, rewards, terminals.
    """
    import jax
    import jax.numpy as jnp

    env_id = _game_to_env_id(game)
    env = make_atari_env(
        env_id,
        noop_max=noop_max,
        clip_rewards=True,
        episode_life=False,
        repeat_action_probability=0.0,
    )

    # Build JIT-compiled Q-value function
    net = checkpoint.network_def
    params = {"params": checkpoint.online_params}
    support = checkpoint.support

    @jax.jit
    def q_fn(obs_batch, rng):
        result = net.apply(
            params, obs_batch, support=support, eval_mode=True,
            key=rng, rngs={"dropout": rng}, mutable=["batch_stats"],
        )
        return result[0].q_values

    rng = jax.random.PRNGKey(seed)
    np_rng = np.random.RandomState(seed)

    def action_fn(obs_hwc, t):
        nonlocal rng
        if np_rng.random() < epsilon:
            return int(np_rng.randint(checkpoint.num_actions))
        rng, rng_act = jax.random.split(rng)
        obs_batch = obs_hwc[np.newaxis].astype(np.float32) / 255.0
        q_values = q_fn(obs_batch, rng_act)
        return int(jnp.argmax(q_values))

    return _run_collection_loop(env, num_steps, action_fn, seed)


def collect_random(
    game: str,
    num_actions: int,
    num_steps: int = 10_000,
    seed: int = 0,
    noop_max: int = 30,
) -> CollectedData:
    """Collect observations using a uniform random policy.

    Provides a baseline observation set for probing analysis.
    Random-policy observations show what a representation
    encodes about states the agent visits without any learned
    behavior, serving as a control against greedy-policy
    observations.

    Args:
        game: Game name, CamelCase ('CrazyClimber') or
            snake_case ('crazy_climber').
        num_actions: Number of discrete actions for the game.
        num_steps: Number of environment steps to collect.
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

    np_rng = np.random.RandomState(seed)

    def action_fn(obs_hwc, t):
        return int(np_rng.randint(num_actions))

    return _run_collection_loop(env, num_steps, action_fn, seed)
