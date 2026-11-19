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

from dit.data.schema import DatasetBundle
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


def test_binary_run_carries_the_disease_metrics_end_to_end() -> None:
    # The wiring run_experiment -> is_binary_task -> binary_view is what makes
    # the disease convention appear at all. Without it every binary report
    # would silently lose sensitivity/specificity/f1_disease/auc_disease, and
    # only the ensemble path used to notice.
    bundle = _bundle()
    result = run_experiment(bundle, _config())
    view = bundle.task_view("binary")
    expected = classification_metrics(
        view.y, result.predictions, result.probabilities, binary_view=True
    )
    for key in ("sensitivity", "specificity", "f1_disease", "auc_disease"):
        assert result.aggregate[key] == pytest.approx(expected[key]), key
        assert result.oof_argmax is not None
        assert result.oof_argmax[key] == pytest.approx(expected[key]), key
    for fold in result.folds:
        assert "sensitivity" in fold and "specificity" in fold


def test_loso_multiclass_fold_without_ad_publishes_no_binary_keys() -> None:
    # Production-path equivalent of the compatibility runner's deleted test:
    # a held-out site with no AD subject leaves NC/MCI, and the fold must be
    # flagged incomparable instead of gaining disease-convention keys.
    rng = np.random.default_rng(5)
    n = 60
    X = rng.normal(size=(n, 3, 5, 2)).astype(np.float32)
    y = np.tile([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], 6)
    site = np.repeat(np.arange(6), 10)
    y[50:] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # site 5: NC and MCI only
    X = (X * 0.2 + np.array([-3.0, 0.0, 3.0])[y][:, None, None, None]).astype(np.float32)
    bundle = DatasetBundle(X=X, y=y, site=site)
    result = run_experiment(
        bundle,
        _config(task="multiclass", split_strategy="loso", n_splits=3, inner_splits=3),
    )
    last = result.folds[-1]
    assert last["held_out_site"] == "5"
    assert last["test_class_counts"]["2"] == 0
    assert last["fold_comparable"] is False
    assert "sensitivity" not in last
    assert "specificity" not in last
    # The fold is excluded from the macro summary, not dropped from the report.
    assert len(result.folds) == 6
    assert "5" in result.aggregate["fold_macro"]["incomparable_folds"]


def test_incomplete_outer_folds_fail_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    # Scoring only the covered subset would hide a fold that failed to write
    # its probabilities; the guard is the production path's only defence.
    from dit.evaluation import experiment

    bundle = _bundle()
    real = experiment.split_indices

    def skipping(y, site=None, **kwargs):
        folds = list(real(y, site, **kwargs))
        train, test, fold_id = folds[0]
        return [(train, test[:-1], fold_id), *folds[1:]]

    monkeypatch.setattr(experiment, "split_indices", skipping)
    with pytest.raises(RuntimeError, match="did not predict every subject"):
        run_experiment(bundle, _config())
