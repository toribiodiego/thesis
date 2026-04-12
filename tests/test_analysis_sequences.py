"""Tests for multi-step sequence extraction.

Uses synthetic replay data with hand-verified boundary filtering.
No checkpoint or GPU needed.
"""

import numpy as np
import pytest

from src.analysis.replay_buffer import ReplayData
from src.analysis.sequences import SequenceData, get_multi_step_sequences


def _make_replay(n, terminal_indices):
    """Create replay with sequential observations for easy verification."""
    return ReplayData(
        observations=np.arange(n).reshape(n, 1, 1).astype(np.uint8) * np.ones((1, 84, 84), dtype=np.uint8),
        actions=np.arange(n, dtype=np.int32),
        rewards=np.zeros(n, dtype=np.float32),
        terminals=np.array([i in terminal_indices for i in range(n)]),
        add_count=n,
    )


class TestGetMultiStepSequences:

    def test_basic_shapes(self):
        replay = _make_replay(20, terminal_indices={19})
        seqs = get_multi_step_sequences(replay, K=5)

        assert isinstance(seqs, SequenceData)
        assert seqs.obs.ndim == 4  # (M, K+1, 84, 84)
        assert seqs.obs.shape[1] == 6  # K+1
        assert seqs.actions.ndim == 2  # (M, K)
        assert seqs.actions.shape[1] == 5  # K

    def test_no_terminals_gives_all_windows(self):
        """10 frames, no terminals, K=3 -> 7 valid windows."""
        replay = _make_replay(10, terminal_indices=set())
        seqs = get_multi_step_sequences(replay, K=3)
        assert len(seqs.obs) == 7  # 10 - 3

    def test_terminal_filters_crossing_windows(self):
        """Terminal at index 5, K=3. Windows [3,6], [4,7], [5,8] are blocked."""
        replay = _make_replay(10, terminal_indices={5})
        seqs = get_multi_step_sequences(replay, K=3)
        # Valid starting indices: 0,1,2 (window ends before 5)
        # and 6 (starts after terminal)
        # Index 3: window [3,4,5,6], terminal at 5 in first K=3 positions -> blocked
        # Index 4: window [4,5,6,7], terminal at 5 -> blocked
        # Index 5: window [5,6,7,8], terminal at 5 -> blocked
        assert len(seqs.obs) == 4  # indices 0,1,2,6

    def test_actions_match_indices(self):
        """Actions in sequences should be sequential from the start index."""
        replay = _make_replay(10, terminal_indices=set())
        seqs = get_multi_step_sequences(replay, K=2)
        # First sequence starts at index 0: actions [0, 1]
        np.testing.assert_array_equal(seqs.actions[0], [0, 1])
        # Last sequence starts at index 7: actions [7, 8]
        np.testing.assert_array_equal(seqs.actions[-1], [7, 8])

    def test_terminal_at_end_of_window_is_allowed(self):
        """Terminal at the K+1-th position (last frame) is OK."""
        # K=2, n=7, terminal at index 2.
        # i=0: terms[0,1]=F,F -> valid (window [0,1,2], term at last pos OK)
        # i=1: terms[1,2]=F,T -> blocked
        # i=2: terms[2,3]=T,F -> blocked
        # i=3: terms[3,4]=F,F -> valid
        # i=4: terms[4,5]=F,F -> valid
        replay = _make_replay(7, terminal_indices={2})
        seqs = get_multi_step_sequences(replay, K=2)
        assert len(seqs.obs) == 3

    def test_two_episodes(self):
        """Two complete episodes: [0..4] and [5..9]."""
        replay = _make_replay(10, terminal_indices={4, 9})
        seqs = get_multi_step_sequences(replay, K=2)
        # n=10, K=2, valid indices 0..7.
        # i=0: terms[0,1]=F,F -> valid
        # i=1: terms[1,2]=F,F -> valid
        # i=2: terms[2,3]=F,F -> valid
        # i=3: terms[3,4]=F,T -> blocked
        # i=4: terms[4,5]=T,F -> blocked
        # i=5: terms[5,6]=F,F -> valid
        # i=6: terms[6,7]=F,F -> valid
        # i=7: terms[7,8]=F,F -> valid
        assert len(seqs.obs) == 6

    def test_all_terminals(self):
        """Every step is terminal -> no valid K>0 sequences."""
        replay = _make_replay(5, terminal_indices={0, 1, 2, 3, 4})
        seqs = get_multi_step_sequences(replay, K=2)
        assert len(seqs.obs) == 0

    def test_too_short_buffer(self):
        replay = _make_replay(3, terminal_indices=set())
        seqs = get_multi_step_sequences(replay, K=5)
        assert len(seqs.obs) == 0

    def test_k_equals_1(self):
        """K=1 gives pairs like get_valid_transitions."""
        replay = _make_replay(5, terminal_indices={2})
        seqs = get_multi_step_sequences(replay, K=1)
        # Valid: 0 (term at 2 is pos 2>K=1), wait:
        # Window [i, i+1], terminal[i] must be False.
        # i=0: term[0]=F -> valid
        # i=1: term[1]=F -> valid
        # i=2: term[2]=T -> blocked
        # i=3: term[3]=F -> valid
        assert len(seqs.obs) == 3
        assert seqs.obs.shape[1] == 2  # K+1
        assert seqs.actions.shape[1] == 1  # K
