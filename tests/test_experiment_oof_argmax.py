"""The main path's headline metrics are the outer-fold argmax evaluation.

``run_ensemble`` pins this contract in ``test_ensemble.py``; the solo
``run_experiment`` path -- the one the CLI uses for every classical model --
did not.  The assertions recompute the argmax metrics independently from
``view.y``, ``result.predictions`` and ``result.probabilities`` instead of
trusting ``result.aggregate``, so replacing the aggregate with the adaptive
threshold policy's numbers fails here.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from dit.data.synthetic import make_synthetic_bundle
from dit.evaluation.experiment import ExperimentConfig, run_experiment
from dit.evaluation.metrics import classification_metrics

CORE_METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "roc_auc")


def _bundle():
    """A fixture whose balanced policy is not the argmax rule.

    With this seed the fitted thresholds change at least one prediction, so the
    argmax evaluation and the deployment policy are genuinely different
    results and the two can be told apart.
    """

    return make_synthetic_bundle(n_samples=40, n_tracts=3, n_points=10, n_metrics=2, seed=1)


def _config(**overrides: object) -> ExperimentConfig:
    base = ExperimentConfig(
        model="logistic",
        task="binary",
        split_strategy="stratified",
        feature_view="summary",
        n_splits=2,
        inner_splits=2,
        seed=1,
    )
    return replace(base, **overrides)


def test_predictions_are_the_argmax_of_the_out_of_fold_probabilities() -> None:
    bundle = _bundle()
    result = run_experiment(bundle, _config())
    view = bundle.task_view("binary")
    assert np.array_equal(result.predictions, np.argmax(result.probabilities, axis=1))
    assert np.isfinite(result.probabilities).all()
    # Every subject of the task view received exactly one probability row.
    assert result.probabilities.shape == (view.n_samples, 2)
    assert np.allclose(result.probabilities.sum(axis=1), 1.0)


def test_aggregate_equals_the_independently_recomputed_argmax_metrics() -> None:
    bundle = _bundle()
    result = run_experiment(bundle, _config())
    view = bundle.task_view("binary")
    expected = classification_metrics(view.y, result.predictions, result.probabilities)
    for metric in CORE_METRICS:
        assert result.aggregate[metric] == pytest.approx(expected[metric]), metric
        assert result.oof_argmax is not None
        assert result.oof_argmax[metric] == pytest.approx(expected[metric]), metric
    assert result.aggregate["threshold_evaluation"] == "outer_oof_argmax"
    assert result.aggregate["prediction_rule"] == "predict_proba_argmax"
    assert result.aggregate["selection_metric"] == "balanced_accuracy"
    assert result.aggregate["threshold_criterion"] == "f1"
    assert result.aggregate["class_names"] == ["NC", "AD"]
    assert result.aggregate["aggregate_argmax_accuracy"] == pytest.approx(
        float(np.mean(result.predictions == view.y))
    )


def test_thresholded_selection_metrics_are_labelled_deployment_only() -> None:
    bundle = _bundle()
    result = run_experiment(bundle, _config(threshold_criterion="balanced"))
    view = bundle.task_view("binary")
    policy_predictions = result.threshold.predict(result.probabilities)
    expected = classification_metrics(view.y, policy_predictions, result.probabilities)
    block = result.thresholded_selection_metrics
    assert block is not None
    assert block["scope"] == "all_outer_oof_used_for_threshold_fit"
    assert block["not_for_performance_comparison"] is True
    for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
        assert block[metric] == pytest.approx(expected[metric]), metric


def test_the_adaptive_policy_does_not_replace_the_argmax_aggregate() -> None:
    # On this fixture the fitted policy changes at least one prediction, so
    # the argmax and thresholded rules produce different numbers.  The
    # aggregate must stay the argmax one; the last two assertions prove the
    # fixture really discriminates, so the equality above cannot pass by
    # coincidence.
    bundle = _bundle()
    result = run_experiment(bundle, _config(threshold_criterion="balanced"))
    view = bundle.task_view("binary")
    policy_predictions = result.threshold.predict(result.probabilities)
    assert result.aggregate["threshold_criterion"] == "balanced"
    assert np.array_equal(result.predictions, np.argmax(result.probabilities, axis=1))
    assert not np.array_equal(policy_predictions, result.predictions)
    argmax_metrics = classification_metrics(view.y, result.predictions, result.probabilities)
    policy_metrics = classification_metrics(view.y, policy_predictions, result.probabilities)
    for metric in CORE_METRICS:
        assert result.aggregate[metric] == pytest.approx(argmax_metrics[metric]), metric
    assert any(result.aggregate[m] != policy_metrics[m] for m in ("accuracy", "balanced_accuracy"))


def test_fixed_criterion_keeps_half_thresholds_and_the_argmax_rule() -> None:
    bundle = _bundle()
    result = run_experiment(bundle, _config(threshold_criterion="fixed"))
    assert result.threshold.thresholds == (0.5, 0.5)
    assert np.array_equal(result.predictions, np.argmax(result.probabilities, axis=1))
    view = bundle.task_view("binary")
    expected = classification_metrics(view.y, result.predictions, result.probabilities)
    assert result.aggregate["threshold_evaluation"] == "outer_oof_argmax"
    for metric in CORE_METRICS:
        assert result.aggregate[metric] == pytest.approx(expected[metric]), metric


def test_the_contract_holds_for_loso_folds() -> None:
    bundle = _bundle()
    result = run_experiment(bundle, _config(split_strategy="loso"))
    view = bundle.task_view("binary")
    expected = classification_metrics(view.y, result.predictions, result.probabilities)
    assert len(result.folds) >= 2
    assert np.array_equal(result.predictions, np.argmax(result.probabilities, axis=1))
    for metric in CORE_METRICS:
        assert result.aggregate[metric] == pytest.approx(expected[metric]), metric
    assert result.aggregate["threshold_evaluation"] == "outer_oof_argmax"
    assert result.thresholded_selection_metrics["not_for_performance_comparison"] is True


def test_per_fold_reports_score_the_argmax_predictions() -> None:
    bundle = _bundle()
    result = run_experiment(bundle, _config())
    view = bundle.task_view("binary")
    for fold in result.folds:
        assert fold["n_test"] > 0
        assert "threshold" not in fold
        assert fold["balanced_accuracy"] <= 1.0
    # Fold accuracies must sit on the same argmax predictions the aggregate
    # uses: the pooled accuracy equals the support-weighted mean of folds.
    weighted = sum(fold["accuracy"] * fold["n_test"] for fold in result.folds) / view.n_samples
    assert result.aggregate["accuracy"] == pytest.approx(weighted)
