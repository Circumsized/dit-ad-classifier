"""Explainability for AFQ classifiers.

The competition's post-hoc criterion is the consistency of estimated
predictors, which only makes sense if the model's inputs can be addressed by
anatomical name.  Everything here maps coefficients or importance scores back
to tract, node and metric through :class:`FeatureLayout`.
"""

from .importance import block_permutation_importance, node_importance_from_coefficients
from .heatmap import (
    literature_hits,
    node_matrix,
    render_metric_panel,
    render_node_heatmap,
)

__all__ = [
    "block_permutation_importance",
    "literature_hits",
    "node_importance_from_coefficients",
    "node_matrix",
    "render_metric_panel",
    "render_node_heatmap",
]
