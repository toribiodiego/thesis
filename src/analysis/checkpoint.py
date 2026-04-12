"""Load JAX training checkpoints for offline analysis.

Reads msgpack parameter files and JSON metadata sidecars produced by
train.py's save_checkpoint(), reconstructs the RainbowDQNNetwork with
correct architecture, and returns everything needed for downstream
representation extraction and Q-value computation.
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import jax.numpy as jnp
from flax.serialization import msgpack_restore


@dataclass
class CheckpointData:
    """Everything needed to run a loaded checkpoint through the network.

    Attributes:
        online_params: Deserialized online network parameters.
        target_params: Deserialized target network parameters, or None
            if the target checkpoint file does not exist (older runs).
        network_def: A RainbowDQNNetwork instance matching the
            checkpoint's architecture.
        support: C51 support vector, shape (num_atoms,).
        num_actions: Number of discrete actions for this environment.
        num_atoms: Number of distributional atoms (typically 51).
        encoder_type: 'dqn' (Nature CNN) or 'impala' (IMPALA ResNet).
        hidden_dim: FeatureLayer output dimension (512 or 2048).
        metadata: Dict from the JSON sidecar (step, training_steps,
            cumulative_resets, cycle_grad_steps).
    """

    online_params: dict
    target_params: Optional[dict]
    network_def: Any  # RainbowDQNNetwork (lazy-imported to avoid gin conflict)
    support: Any  # jnp.ndarray
    num_actions: int
    num_atoms: int
    encoder_type: str
    hidden_dim: int
    metadata: dict


def load_checkpoint(
    run_dir: str,
    step: int,
    vmin: float = -10.0,
    vmax: float = 10.0,
) -> CheckpointData:
    """Load a training checkpoint for analysis.

    Args:
        run_dir: Path to the run directory (e.g.,
            experiments/dqn_atari/runs/bbf_crazy_climber_seed13).
        step: Checkpoint step number (e.g., 10000).
        vmin: Minimum value for C51 support (default -10.0).
        vmax: Maximum value for C51 support (default 10.0).

    Returns:
        CheckpointData with params, network definition, and metadata.

    Raises:
        FileNotFoundError: If online params or metadata file is missing.
        ValueError: If required network parameters cannot be parsed
            from the gin config.
    """
    ckpt_dir = os.path.join(run_dir, "checkpoints")

    # -- Load online params --------------------------------------------------
    online_path = os.path.join(ckpt_dir, f"checkpoint_{step}.msgpack")
    if not os.path.isfile(online_path):
        raise FileNotFoundError(f"Online params not found: {online_path}")
    with open(online_path, "rb") as f:
        online_params = _unwrap_params(msgpack_restore(f.read()))

    # -- Load target params (optional -- older runs lack this file) ----------
    target_path = os.path.join(ckpt_dir, f"target_{step}.msgpack")
    target_params = None
    if os.path.isfile(target_path):
        with open(target_path, "rb") as f:
            target_params = _unwrap_params(msgpack_restore(f.read()))

    # -- Load metadata and gin config ----------------------------------------
    meta_path = os.path.join(ckpt_dir, f"checkpoint_{step}.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata not found: {meta_path}")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    gin_config = meta["gin_config"]

    # -- Parse network architecture from gin config --------------------------
    net_params = _parse_gin_network_params(gin_config)
    num_atoms = net_params["num_atoms"]

    # -- Infer num_actions from the params tree ------------------------------
    num_actions = _infer_num_actions(online_params, num_atoms)

    # -- Reconstruct network -------------------------------------------------
    RainbowDQNNetwork = _import_network_class()
    network_def = RainbowDQNNetwork(
        num_actions=num_actions,
        num_atoms=num_atoms,
        noisy=net_params["noisy"],
        dueling=net_params["dueling"],
        distributional=net_params["distributional"],
        renormalize=net_params["renormalize"],
        encoder_type=net_params["encoder_type"],
        hidden_dim=net_params["hidden_dim"],
        width_scale=net_params["width_scale"],
    )

    # -- C51 support vector --------------------------------------------------
    support = jnp.linspace(vmin, vmax, num_atoms)

    # -- Strip gin_config from metadata (large, already parsed) --------------
    metadata = {k: v for k, v in meta.items() if k != "gin_config"}

    return CheckpointData(
        online_params=online_params,
        target_params=target_params,
        network_def=network_def,
        support=support,
        num_actions=num_actions,
        num_atoms=num_atoms,
        encoder_type=net_params["encoder_type"],
        hidden_dim=net_params["hidden_dim"],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _import_network_class():
    """Lazy-import RainbowDQNNetwork to avoid gin double-registration.

    The module is importable as both 'src.bigger_better_faster.bbf...'
    and 'bigger_better_faster.bbf...' depending on sys.path. Importing
    at module level via one path while other code uses the other path
    causes gin to register the @gin.configurable class twice, breaking
    partial-selector resolution. Deferring the import to call time
    avoids this.
    """
    try:
        from bigger_better_faster.bbf.spr_networks import RainbowDQNNetwork
    except ImportError:
        from src.bigger_better_faster.bbf.spr_networks import RainbowDQNNetwork
    return RainbowDQNNetwork


def _unwrap_params(params: dict) -> dict:
    """Strip the outer 'params' key if present.

    Flax's to_state_dict wraps the param tree in {'params': ...}.
    Downstream code expects the inner dict with keys like 'encoder',
    'head', etc.
    """
    if "params" in params and len(params) == 1:
        return params["params"]
    return params


def _parse_gin_value(gin_config: str, cls: str, param: str) -> Optional[str]:
    """Extract a single parameter value from a gin config string.

    Matches lines like ``ClassName.param_name = value`` and returns
    the raw value string, or None if not found.
    """
    pattern = rf"^{re.escape(cls)}\.{re.escape(param)}\s*=\s*(.+)$"
    match = re.search(pattern, gin_config, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def _parse_gin_bool(gin_config: str, cls: str, param: str) -> Optional[bool]:
    raw = _parse_gin_value(gin_config, cls, param)
    if raw is None:
        return None
    return raw == "True"


def _parse_gin_int(gin_config: str, cls: str, param: str) -> Optional[int]:
    raw = _parse_gin_value(gin_config, cls, param)
    if raw is None:
        return None
    return int(raw)


def _parse_gin_float(gin_config: str, cls: str, param: str) -> Optional[float]:
    raw = _parse_gin_value(gin_config, cls, param)
    if raw is None:
        return None
    return float(raw)


def _parse_gin_str(gin_config: str, cls: str, param: str) -> Optional[str]:
    raw = _parse_gin_value(gin_config, cls, param)
    if raw is None:
        return None
    return raw.strip("'\"")


def _parse_gin_network_params(gin_config: str) -> dict:
    """Extract all parameters needed to reconstruct RainbowDQNNetwork.

    Parses from two gin sections:
    - RainbowDQNNetwork: encoder_type, hidden_dim, width_scale, renormalize
    - BBFAgent: noisy, dueling, distributional, num_atoms
    """
    encoder_type = _parse_gin_str(gin_config, "RainbowDQNNetwork", "encoder_type")
    if encoder_type is None:
        raise ValueError("Cannot parse RainbowDQNNetwork.encoder_type from gin config")

    hidden_dim = _parse_gin_int(gin_config, "RainbowDQNNetwork", "hidden_dim")
    if hidden_dim is None:
        raise ValueError("Cannot parse RainbowDQNNetwork.hidden_dim from gin config")

    width_scale = _parse_gin_float(gin_config, "RainbowDQNNetwork", "width_scale")
    if width_scale is None:
        raise ValueError("Cannot parse RainbowDQNNetwork.width_scale from gin config")

    renormalize = _parse_gin_bool(gin_config, "RainbowDQNNetwork", "renormalize")
    if renormalize is None:
        renormalize = False  # default

    noisy = _parse_gin_bool(gin_config, "BBFAgent", "noisy")
    if noisy is None:
        noisy = False  # default

    dueling = _parse_gin_bool(gin_config, "BBFAgent", "dueling")
    if dueling is None:
        dueling = False  # default

    distributional = _parse_gin_bool(gin_config, "BBFAgent", "distributional")
    if distributional is None:
        distributional = True  # default

    num_atoms = _parse_gin_int(gin_config, "BBFAgent", "num_atoms")
    if num_atoms is None:
        num_atoms = 51  # default

    return {
        "encoder_type": encoder_type,
        "hidden_dim": hidden_dim,
        "width_scale": width_scale,
        "renormalize": renormalize,
        "noisy": noisy,
        "dueling": dueling,
        "distributional": distributional,
        "num_atoms": num_atoms,
    }


def _infer_num_actions(params: dict, num_atoms: int) -> int:
    """Infer num_actions from the head's advantage layer kernel shape.

    The advantage FeatureLayer has output features = num_actions * num_atoms.
    Its kernel shape is (hidden_dim, num_actions * num_atoms).
    """
    try:
        kernel = params["head"]["advantage"]["net"]["kernel"]
        out_features = kernel.shape[-1]
    except (KeyError, IndexError) as e:
        raise ValueError(
            f"Cannot infer num_actions from params: {e}. "
            "Expected params['head']['advantage']['net']['kernel']."
        ) from e

    if out_features % num_atoms != 0:
        raise ValueError(
            f"Advantage kernel output dim ({out_features}) is not "
            f"divisible by num_atoms ({num_atoms})."
        )

    return out_features // num_atoms
