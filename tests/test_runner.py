"""Compatibility runners must emit the same fold provenance as the main path."""

from __future__ import annotations

import numpy as np

from dit.data.schema import DatasetBundle
from dit.data.synthetic import make_synthetic_bundle
from dit.evaluation.runner import evaluate_classical, evaluate_metric_ensemble


def _assert_manifest(folds, *, strategy: str) -> None:
    assert folds
    for fold in folds:
        assert fold["task"] == "binary"
        assert fold["split_strategy"] == strategy
        assert len(fold["test_index_digest"]) == 64
        assert sum(fold["test_class_counts"].values()) == fold["n_test"]
        if strategy == "loso":
            assert fold["held_out_site"] == fold["fold"]
            assert fold["inner_cv"] == "site_grouped"
        else:
            assert "held_out_site" not in fold
            assert fold["inner_cv"] == "class_stratified"


def test_classical_runner_emits_stratified_fold_manifest() -> None:
    dataset = make_synthetic_bundle(
        n_samples=60, n_sites=3, n_tracts=3, n_points=8, n_metrics=2, seed=12
    )
    result = evaluate_classical(
        dataset,
        task="binary",
        model_name="linear_svm",
        split_strategy="stratified",
        feature_view="summary",
        n_splits=2,
        inner_splits=2,
        seed=12,
    )
    _assert_manifest(result.folds, strategy="stratified")


def test_metric_ensemble_runner_emits_loso_fold_manifest() -> None:
    dataset = make_synthetic_bundle(
        n_samples=60, n_sites=3, n_tracts=3, n_points=8, n_metrics=2, seed=13
    )
    result = evaluate_metric_ensemble(
        dataset,
        task="binary",
        split_strategy="loso",
        n_splits=2,
        inner_splits=2,
        seed=13,
    )
    _assert_manifest(result.folds, strategy="loso")


def test_loso_multiclass_fold_without_ad_has_no_binary_keys() -> None:
    """A held-out site with no AD subject leaves NC/MCI: class 1 is MCI there."""

    rng = np.random.default_rng(5)
    n = 60
    X = rng.normal(size=(n, 3, 5, 2)).astype(np.float32)
    y = np.tile([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], 6)
    site = np.repeat(np.arange(6), 10)
    y[50:] = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # site 5: NC and MCI only
    X = (X * 0.2 + np.array([-3.0, 0.0, 3.0])[y][:, None, None, None]).astype(np.float32)
    bundle = DatasetBundle(X=X, y=y, site=site)
    result = evaluate_classical(
        bundle,
        task="multiclass",
        split_strategy="loso",
        feature_view="summary",
        n_splits=3,
        inner_splits=3,
        seed=1,
    )
    last = result.folds[-1]
    assert last["held_out_site"] == "5"
    assert last["test_class_counts"]["2"] == 0
    assert last["fold_comparable"] is False
    assert "sensitivity" not in last
    assert "specificity" not in last
