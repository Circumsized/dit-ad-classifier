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


def test_two_class_multiclass_fold_gets_no_binary_keys() -> None:
    """A multiclass fold that lost a class must not publish sensitivity/specificity.

    LOSO can hold out a site with no AD subject; the surviving labels are
    NC/MCI, so class 1 there is MCI and the binary convention would mislabel
    MCI recall as disease sensitivity.
    """

    y = np.array([0] * 30 + [1] * 20)
    result = classification_metrics(y, y)
    assert "sensitivity" not in result
    assert "specificity" not in result
    assert "f1_disease" not in result
    assert result["class_recall"]["1"] == 1.0


def test_binary_view_keeps_the_disease_convention() -> None:
    y = np.array([0] * 30 + [1] * 20)
    probabilities = np.zeros((y.size, 2))
    probabilities[np.arange(y.size), y] = 0.9
    probabilities[np.arange(y.size), 1 - y] = 0.1
    result = classification_metrics(y, y, probabilities, binary_view=True)
    assert result["sensitivity"] == 1.0
    assert result["specificity"] == 1.0
    assert result["f1_disease"] == 1.0
    assert result["auc_disease"] == 1.0


@pytest.mark.parametrize("alias", ["binary", "ad_nc", "ad_vs_nc", "Binary", "AD-VS-NC"])
def test_binary_task_aliases_are_recognised(alias) -> None:
    from dit.data.schema import is_binary_task

    assert is_binary_task(alias)
    assert not is_binary_task("multiclass")
    assert not is_binary_task("three_class")


def test_binary_view_single_class_fold_emits_no_disease_keys() -> None:
    """A fold whose labels are single-class must not publish NaN conventions.

    ``classes`` unions y_true and y_pred, so a single-class fold whose
    predictions spill into a second class used to take the two-class branch
    and report NaN sensitivity/auc_disease (rescued only downstream by
    _json_safe).  The gate is on the labels present in y_true.
    """

    y = np.zeros(5, dtype=int)
    pred = np.array([0, 0, 1, 0, 0])
    probabilities = np.array([[0.9, 0.1]] * 4 + [[0.2, 0.8]])
    result = classification_metrics(y, pred, probabilities, binary_view=True)
    assert "sensitivity" not in result
    assert "specificity" not in result
    assert "auc_disease" not in result
    assert result["class_recall"]["0"] == pytest.approx(0.8)
