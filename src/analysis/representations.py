"""Extract representations and Q-values from loaded checkpoints.

Provides functions to run observations through specific parts of
a loaded RainbowDQNNetwork:
- FeatureLayer representations (encode + flatten + project) for
  probing methods M9, M10, M14
- Full-network Q-values (Task 28 checklist items 2-3, added later)
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.analysis.checkpoint import CheckpointData


def extract_representations(
    checkpoint: CheckpointData,
    observations: np.ndarray,
    batch_size: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Extract FeatureLayer representations from observations.

    Runs each observation through encode -> flatten -> project
    (the network's encode_project method) to produce the hidden
    representation used by probing methods. Output dimension is
    hidden_dim: 512 for Nature CNN, 2048 for IMPALA.

    Args:
        checkpoint: Loaded checkpoint from load_checkpoint.
        observations: (N, 84, 84, 4) uint8 HWC stacked frames.
        batch_size: Number of observations per forward pass chunk.
        seed: RNG seed for dropout keys (deterministic in eval mode).

    Returns:
        (N, hidden_dim) float32 representation array.
    """
    net = checkpoint.network_def
    params = {"params": checkpoint.online_params}

    @jax.jit
    def _extract_batch(obs_batch, keys):
        """Process a batch through encode_project with vmap."""
        def _single(obs, key):
            # obs: (84, 84, 4) float32 in [0, 1]
            return net.apply(
                params, obs, key, True,
                method=net.encode_project,
                rngs={"dropout": key},
            )
        return jax.vmap(_single)(obs_batch, keys)

    rng = jax.random.PRNGKey(seed)
    n = len(observations)
    results = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = observations[start:end]

        # Preprocess: uint8 -> float32 [0, 1]
        obs_f32 = chunk.astype(np.float32) / 255.0

        rng, rng_batch = jax.random.split(rng)
        keys = jax.random.split(rng_batch, len(chunk))

        reps = _extract_batch(obs_f32, keys)
        results.append(np.asarray(reps))

    return np.concatenate(results, axis=0)


def extract_representations_target(
    checkpoint: CheckpointData,
    observations: np.ndarray,
    batch_size: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Extract FeatureLayer representations using target network params.

    Same as extract_representations but uses target_params instead
    of online_params. Useful for CKA comparison between online and
    target encoders (M13). Returns None if target params are not
    available.

    Args:
        checkpoint: Loaded checkpoint from load_checkpoint.
        observations: (N, 84, 84, 4) uint8 HWC stacked frames.
        batch_size: Number of observations per forward pass chunk.
        seed: RNG seed for dropout keys.

    Returns:
        (N, hidden_dim) float32 representation array, or None if
        target params are not available.
    """
    if checkpoint.target_params is None:
        return None

    net = checkpoint.network_def
    params = {"params": checkpoint.target_params}

    @jax.jit
    def _extract_batch(obs_batch, keys):
        def _single(obs, key):
            return net.apply(
                params, obs, key, True,
                method=net.encode_project,
                rngs={"dropout": key},
            )
        return jax.vmap(_single)(obs_batch, keys)

    rng = jax.random.PRNGKey(seed)
    n = len(observations)
    results = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = observations[start:end]
        obs_f32 = chunk.astype(np.float32) / 255.0

        rng, rng_batch = jax.random.split(rng)
        keys = jax.random.split(rng_batch, len(chunk))

        reps = _extract_batch(obs_f32, keys)
        results.append(np.asarray(reps))

    return np.concatenate(results, axis=0)
