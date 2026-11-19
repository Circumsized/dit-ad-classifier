"""Serializable outer-CV result shared by the runners and the report assembler.

The evaluation loops themselves live in :mod:`dit.evaluation.experiment`
(``run_experiment`` / ``run_ensemble``); this module keeps only the result
contract they and the reporting layer exchange.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dit.evaluation.threshold import ThresholdPolicy


@dataclass(frozen=True)
class EvaluationResult:
    """Serializable outer-CV result and optional deployment decision policy."""

    model: str
    task: str
    split_strategy: str
    feature_view: str
    aggregate: dict[str, object]
    folds: tuple[dict[str, object], ...]
    predictions: np.ndarray
    probabilities: np.ndarray
    threshold: ThresholdPolicy | None = None
    thresholded_selection_metrics: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self.model,
            "task": self.task,
            "split_strategy": self.split_strategy,
            "feature_view": self.feature_view,
            "aggregate": self.aggregate,
            "folds": list(self.folds),
            "predictions": self.predictions.tolist(),
            "probabilities": self.probabilities.tolist(),
        }
        if self.threshold is not None:
            payload["threshold"] = self.threshold.parameters()
        if self.thresholded_selection_metrics is not None:
            payload["thresholded_selection_metrics"] = self.thresholded_selection_metrics
        return payload
