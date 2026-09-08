"""Decision-policy tests: adaptive selection stays separate from evaluation."""

from __future__ import annotations

import numpy as np

from dit.evaluation.metrics import classification_metrics
from dit.evaluation.threshold import fit_threshold, threshold_sweep


def test_fixed_policy_is_plain_argmax() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.9, 0.1], [0.6, 0.4], [0.4, 0.6], [0.1, 0.9]])
    policy = fit_threshold(y, probabilities, criterion="fixed")

    assert policy.thresholds == (0.5, 0.5)
    assert np.array_equal(policy.predict(probabilities), np.argmax(probabilities, axis=1))


def test_adaptive_policy_score_matches_its_selection_data_only() -> None:
    y = np.array([0] * 6 + [1] * 6)
    probabilities = np.array(
        [[0.55, 0.45]] * 6 + [[0.45, 0.55]] * 6,
        dtype=float,
    )
    policy = fit_threshold(y, probabilities, criterion="balanced")
    selected = policy.predict(probabilities)
    score = classification_metrics(y, selected)["balanced_accuracy"]

    assert policy.score == score
    assert policy.thresholds != (0.05, 0.05)


def test_threshold_sweep_changes_the_disease_operating_point() -> None:
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.4, 0.6], [0.1, 0.9]])
    rows = threshold_sweep(y, probabilities, n_points=11)

    assert len({row["sensitivity"] for row in rows}) > 1
    assert len({row["specificity"] for row in rows}) > 1
