"""Compatibility runners must emit the same fold provenance as the main path."""

from __future__ import annotations

import numpy as np

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
