"""Data contracts, MAT loading and leakage-safe preprocessing."""

from .schema import (
    BINARY_TASK_ALIASES,
    CANONICAL_NAMES,
    DatasetBundle,
    LABEL_NAMES,
    canonicalize_labels,
    is_binary_task,
    raw_label_names,
)
from .mat_loader import load_ai4ad_mat
from .synthetic import make_synthetic_bundle

__all__ = [
    "BINARY_TASK_ALIASES",
    "CANONICAL_NAMES",
    "DatasetBundle",
    "LABEL_NAMES",
    "canonicalize_labels",
    "is_binary_task",
    "load_ai4ad_mat",
    "make_synthetic_bundle",
    "raw_label_names",
]
