"""Data contracts, MAT loading and leakage-safe preprocessing."""

from .schema import DatasetBundle, LABEL_NAMES, canonicalize_labels
from .mat_loader import load_ai4ad_mat
from .synthetic import make_synthetic_bundle

__all__ = [
    "DatasetBundle",
    "LABEL_NAMES",
    "canonicalize_labels",
    "load_ai4ad_mat",
    "make_synthetic_bundle",
]
