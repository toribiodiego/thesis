"""Linear probe training for representation analysis.

Trains multinomial logistic regression on frozen encoder
representations to measure what information is linearly decodable.
Shared probing core used by M9 (AtariARI), M10 (reward), and
M14 (inverse dynamics).

Follows the AtariARI protocol (Anand et al. 2019, Section 5.3):
- StandardScaler on features (fit on train only)
- L2-regularized logistic regression (C=1.0, L-BFGS solver)
- Macro-averaged F1 on held-out test set
- Entropy filter: skip variables with normalized entropy < 0.6
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Default split seed shared across all probing analyses so that
# learning-dynamics comparisons across checkpoints use the same
# held-out test set (see 9-atariari-probing.md Issue 1 Solution).
PROBE_SPLIT_SEED = 13

# Entropy filter threshold from Anand et al. 2019 Section 5.3.
ENTROPY_THRESHOLD = 0.6


@dataclass
class ProbeResult:
    """Result of training a single linear probe.

    Attributes:
        variable: Name of the target variable.
        f1_test: Macro-averaged F1 on the held-out test set.
        f1_train: Macro-averaged F1 on the training set.
        accuracy_test: Raw accuracy on the test set.
        n_classes: Number of distinct classes in training labels.
        normalized_entropy: Normalized entropy of the training
            label distribution.
        skipped: True if the variable was skipped due to low
            entropy or constant value.
        skip_reason: Reason for skipping, or None.
    """

    variable: str
    f1_test: float
    f1_train: float
    accuracy_test: float
    n_classes: int
    normalized_entropy: float
    skipped: bool = False
    skip_reason: Optional[str] = None


def _normalized_entropy(labels: np.ndarray) -> float:
    """Compute normalized entropy of a discrete label distribution."""
    counts = np.bincount(labels)
    probs = counts[counts > 0] / counts.sum()
    n_classes = len(probs)
    if n_classes <= 1:
        return 0.0
    entropy = -(probs * np.log2(probs)).sum()
    max_entropy = np.log2(n_classes)
    return entropy / max_entropy


def train_probe(
    representations: np.ndarray,
    labels: np.ndarray,
    variable_name: str = "target",
    test_size: float = 0.2,
    seed: int = PROBE_SPLIT_SEED,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    max_iter: int = 1000,
) -> ProbeResult:
    """Train a linear probe on frozen representations.

    Standardizes features (zero mean, unit variance fit on train),
    splits data, checks entropy filter, and trains multinomial
    logistic regression.

    Args:
        representations: (N, D) float32 feature matrix.
        labels: (N,) int categorical target labels.
        variable_name: Name for reporting.
        test_size: Fraction of data for test set (default 0.2).
        seed: Random seed for train/test split (default 13).
        entropy_threshold: Skip variables with normalized entropy
            below this value (default 0.6).
        max_iter: Maximum L-BFGS iterations (default 1000).

    Returns:
        ProbeResult with F1 scores and metadata.
    """
    n_classes = len(np.unique(labels))

    if n_classes <= 1:
        return ProbeResult(
            variable=variable_name,
            f1_test=0.0, f1_train=0.0, accuracy_test=0.0,
            n_classes=n_classes, normalized_entropy=0.0,
            skipped=True, skip_reason="constant value",
        )

    # Split (stratify preserves class proportions in both sets,
    # preventing empty-class splits on imbalanced targets like
    # binary reward probing at early checkpoints)
    X_train, X_test, y_train, y_test = train_test_split(
        representations, labels, test_size=test_size, random_state=seed,
        stratify=labels,
    )

    # Entropy filter on training labels
    norm_ent = _normalized_entropy(y_train)
    if norm_ent < entropy_threshold:
        return ProbeResult(
            variable=variable_name,
            f1_test=0.0, f1_train=0.0, accuracy_test=0.0,
            n_classes=n_classes, normalized_entropy=norm_ent,
            skipped=True, skip_reason=f"low entropy ({norm_ent:.3f})",
        )

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train logistic regression
    clf = LogisticRegression(
        max_iter=max_iter, random_state=seed, solver="lbfgs", C=1.0,
    )
    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)

    return ProbeResult(
        variable=variable_name,
        f1_test=f1_score(y_test, y_test_pred, average="macro", zero_division=0),
        f1_train=f1_score(y_train, y_train_pred, average="macro", zero_division=0),
        accuracy_test=clf.score(X_test, y_test),
        n_classes=n_classes,
        normalized_entropy=norm_ent,
    )


def train_probes_multi(
    representations: np.ndarray,
    labels_dict: dict,
    test_size: float = 0.2,
    seed: int = PROBE_SPLIT_SEED,
    entropy_threshold: float = ENTROPY_THRESHOLD,
    max_iter: int = 1000,
) -> list:
    """Train one linear probe per variable in a label dict.

    Convenience wrapper around train_probe for AtariARI-style
    multi-variable probing.

    Args:
        representations: (N, D) float32 feature matrix.
        labels_dict: Dict mapping variable name to (N,) int labels.
        test_size: Fraction of data for test set.
        seed: Random seed for train/test split.
        entropy_threshold: Skip variables below this threshold.
        max_iter: Maximum L-BFGS iterations.

    Returns:
        List of ProbeResult, one per variable (including skipped).
    """
    results = []
    for var_name, var_labels in sorted(labels_dict.items()):
        result = train_probe(
            representations, var_labels, variable_name=var_name,
            test_size=test_size, seed=seed,
            entropy_threshold=entropy_threshold, max_iter=max_iter,
        )
        results.append(result)
    return results
