"""Extract representations and Q-values from loaded checkpoints.

Provides functions to run observations through specific parts of
a loaded RainbowDQNNetwork:
- FeatureLayer representations (encode + flatten + project) for
  probing methods M9, M10, M14
- Full-network Q-values via C51 expected value for value
  accuracy analysis (M15)
- Transition model predictions in projected space with cosine
  similarity against encode_project targets (M16)
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


def extract_q_values(
    checkpoint: CheckpointData,
    observations: np.ndarray,
    batch_size: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Extract Q-values via the full network forward pass.

    Runs each observation through the complete network (encoder,
    projection, ReLU, distributional head) and returns the C51
    expected Q-values: sum(support * softmax(logits)) per action.
    Used for value accuracy analysis (M15).

    Args:
        checkpoint: Loaded checkpoint from load_checkpoint.
        observations: (N, 84, 84, 4) uint8 HWC stacked frames.
        batch_size: Number of observations per forward pass chunk.
        seed: RNG seed for dropout keys.

    Returns:
        (N, num_actions) float32 Q-value array.
    """
    net = checkpoint.network_def
    params = {"params": checkpoint.online_params}
    support = checkpoint.support

    @jax.jit
    def _q_batch(obs_batch, keys):
        def _single(obs, key):
            output = net.apply(
                params, obs, support=support,
                eval_mode=True, key=key,
                rngs={"dropout": key},
            )
            return output.q_values
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

        q_vals = _q_batch(obs_f32, keys)
        results.append(np.asarray(q_vals))

    return np.concatenate(results, axis=0)


# ---------------------------------------------------------------------------
# Transition model prediction function (used by method= in apply)
# ---------------------------------------------------------------------------

def _transition_predict(self, obs, action, key):
    """Predict one step forward: encode, transition model, flatten, project, predict."""
    spatial = self.encode(obs, eval_mode=True)
    actions_seq = jnp.expand_dims(action, 0)  # (1,) for single-step scan
    _, pred_latents = self.transition_model(spatial, actions_seq)
    pred_flat = self.flatten_spatial_latent(pred_latents[0])
    pred_proj = self.project(pred_flat, key, eval_mode=True)
    return self.predictor(pred_proj)


def evaluate_transition_model(
    checkpoint: CheckpointData,
    obs: np.ndarray,
    actions: np.ndarray,
    obs_next: np.ndarray,
    batch_size: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Evaluate transition model prediction accuracy via cosine similarity.

    For each transition (obs[t], action[t], obs[t+1]):
    - Prediction: encode obs[t] with online params, run transition
      model with action[t], flatten, project, predict.
    - Target: encode_project obs[t+1] with target params (or online
      params if target unavailable).
    - Metric: cosine similarity after L2 normalization, matching
      the SPR loss formulation (N4 in notes_16.md).

    Args:
        checkpoint: Loaded checkpoint from load_checkpoint.
        obs: (N, 84, 84, 4) uint8 HWC stacked frames at time t.
        actions: (N,) int32 actions taken at time t.
        obs_next: (N, 84, 84, 4) uint8 HWC stacked frames at time t+1.
        batch_size: Number of transitions per forward pass chunk.
        seed: RNG seed for dropout keys.

    Returns:
        (N,) float32 cosine similarities in [-1, 1]. Values near 1
        indicate accurate transition model predictions.
    """
    net = checkpoint.network_def
    online_params = {"params": checkpoint.online_params}
    target_params_dict = checkpoint.target_params
    if target_params_dict is not None:
        target_params = {"params": target_params_dict}
    else:
        target_params = online_params

    @jax.jit
    def _eval_batch(obs_batch, acts_batch, obs_next_batch, keys):
        def _single(ob, act, ob_next, key):
            # Prediction: online params, full SPR pipeline
            prediction = net.apply(
                online_params, ob, act, key,
                method=_transition_predict,
                rngs={"dropout": key},
            )
            # Target: target (or online) params, encode_project
            target = net.apply(
                target_params, ob_next, key, True,
                method=net.encode_project,
                rngs={"dropout": key},
            )
            # L2-normalize and compute cosine similarity
            pred_norm = prediction / (jnp.linalg.norm(prediction) + 1e-8)
            tgt_norm = target / (jnp.linalg.norm(target) + 1e-8)
            return jnp.dot(pred_norm, tgt_norm)

        return jax.vmap(_single)(obs_batch, acts_batch, obs_next_batch, keys)

    rng = jax.random.PRNGKey(seed)
    n = len(obs)
    results = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        obs_chunk = obs[start:end].astype(np.float32) / 255.0
        acts_chunk = actions[start:end]
        obs_next_chunk = obs_next[start:end].astype(np.float32) / 255.0

        rng, rng_batch = jax.random.split(rng)
        keys = jax.random.split(rng_batch, end - start)

        sims = _eval_batch(obs_chunk, acts_chunk, obs_next_chunk, keys)
        results.append(np.asarray(sims))

    return np.concatenate(results, axis=0)
