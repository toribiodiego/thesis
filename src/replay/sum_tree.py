"""
Sum-tree data structure for O(log N) priority-based sampling.

Array-based binary tree where each leaf stores a priority value and
internal nodes store the sum of their children. Supports efficient
proportional sampling and priority updates, both in O(log N) time.

Used by PrioritizedReplayBuffer to implement the proportional
variant of prioritized experience replay (Schaul et al. 2016,
Appendix B.2.1).

Tree layout (capacity=4 example):
    Index 0 is unused. Index 1 is the root (total sum).
    Internal nodes: indices [1, capacity).
    Leaf nodes: indices [capacity, 2*capacity).

             [1] sum=10
            /         \\
        [2] sum=6    [3] sum=4
        /    \\       /    \\
    [4] 3  [5] 3  [6] 1  [7] 3
"""

import numpy as np


class SumTree:
    """
    Array-based binary sum-tree for priority sampling.

    Args:
        capacity: Maximum number of leaf nodes (must be positive).
            Rounded up to the next power of 2 internally for a
            balanced tree structure.

    Attributes:
        capacity: Number of leaf nodes.
        tree: Array of size 2*capacity storing node values.
        max_priority: Current maximum leaf priority (for new items).
        count: Number of items that have been written at least once.
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")

        self.capacity = capacity
        # tree[0] is unused; tree[1] is root; leaves at [capacity, 2*capacity)
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self._max_priority = 1.0
        self._count = 0

    @property
    def total(self) -> float:
        """Total sum of all priorities (root node value)."""
        return float(self.tree[1])

    @property
    def max_priority(self) -> float:
        """Maximum priority across all leaves."""
        return self._max_priority

    @property
    def count(self) -> int:
        """Number of leaf positions that have been written."""
        return self._count

    def update(self, leaf_index: int, priority: float) -> None:
        """
        Set the priority of a leaf and propagate the change to root.

        Args:
            leaf_index: Index in [0, capacity). Identifies which leaf
                to update (not the raw tree array index).
            priority: New priority value (must be non-negative).
        """
        if not (0 <= leaf_index < self.capacity):
            raise IndexError(
                f"leaf_index {leaf_index} out of range [0, {self.capacity})"
            )
        if priority < 0:
            raise ValueError(f"priority must be non-negative, got {priority}")

        tree_index = leaf_index + self.capacity
        delta = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # Propagate delta up to root
        tree_index //= 2
        while tree_index >= 1:
            self.tree[tree_index] += delta
            tree_index //= 2

        # Track max priority
        if priority > self._max_priority:
            self._max_priority = priority

        # Track count of written positions
        if self._count < self.capacity:
            self._count = min(self._count + 1, self.capacity)

    def sample(self, value: float) -> int:
        """
        Sample a leaf index proportional to priorities.

        Traverses the tree from root to leaf, choosing left or right
        child based on cumulative sums.

        Args:
            value: Uniform random value in [0, total). Determines
                which leaf is selected.

        Returns:
            Leaf index in [0, capacity).
        """
        tree_index = 1  # start at root

        while tree_index < self.capacity:
            left = 2 * tree_index
            right = left + 1

            if value <= self.tree[left]:
                tree_index = left
            else:
                value -= self.tree[left]
                tree_index = right

        return tree_index - self.capacity

    def get(self, leaf_index: int) -> float:
        """
        Get the priority of a specific leaf.

        Args:
            leaf_index: Index in [0, capacity).

        Returns:
            Priority value at the leaf.
        """
        if not (0 <= leaf_index < self.capacity):
            raise IndexError(
                f"leaf_index {leaf_index} out of range [0, {self.capacity})"
            )
        return float(self.tree[leaf_index + self.capacity])

    def batch_sample(self, batch_size: int) -> np.ndarray:
        """
        Sample a batch of leaf indices using stratified sampling.

        Divides [0, total) into batch_size equal segments and draws
        one uniform sample per segment. This reduces variance compared
        to independent sampling.

        Args:
            batch_size: Number of indices to sample.

        Returns:
            Array of leaf indices, shape (batch_size,).
        """
        indices = np.empty(batch_size, dtype=np.int64)
        segment = self.total / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = np.random.uniform(low, high)
            indices[i] = self.sample(value)

        return indices
