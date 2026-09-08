"""Metric regression tests for class-support weighting."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score

from dit.evaluation.metrics import classification_metrics


def test_weighted_f1_uses_true_class_support() -> None:
    y_true = np.array([0] * 90 + [1] * 8 + [2] * 2)
    y_pred = y_true.copy()
    y_pred[90:] = 0

    result = classification_metrics(y_true, y_pred)

    assert result["weighted_f1"] == pytest.approx(
        f1_score(y_true, y_pred, average="weighted")
    )
    assert result["weighted_f1"] > result["macro_f1"]
