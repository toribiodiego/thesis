"""Analysis infrastructure for checkpoint loading and representation extraction."""

from src.analysis.checkpoint import CheckpointData, discover_checkpoints, load_checkpoint
from src.analysis.observations import CollectedData, collect_greedy, collect_random
from src.analysis.replay_buffer import (
    ReplayData,
    TransitionData,
    get_valid_transitions,
    load_replay_buffer,
)
from src.analysis.probing import ProbeResult, train_probe, train_probes_multi
from src.analysis.returns import compute_returns
from src.analysis.sequences import SequenceData, get_multi_step_sequences
from src.analysis.representations import (
    evaluate_transition_model,
    extract_q_values,
    extract_representations,
    extract_representations_target,
)

__all__ = [
    "CheckpointData",
    "discover_checkpoints",
    "load_checkpoint",
    "CollectedData",
    "collect_greedy",
    "collect_random",
    "ReplayData",
    "TransitionData",
    "load_replay_buffer",
    "get_valid_transitions",
    "evaluate_transition_model",
    "extract_q_values",
    "extract_representations",
    "extract_representations_target",
    "ProbeResult",
    "train_probe",
    "train_probes_multi",
    "compute_returns",
    "SequenceData",
    "get_multi_step_sequences",
]
