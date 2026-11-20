"""The block selector is refit inside every inner CV fold.

Mirrors the ``ResidualizeCovariates`` spy in ``test_p0_hardening.py``: patch
``SparseBlockSelector.fit`` and record the shape of every call.  If the
selector were hoisted in front of ``GridSearchCV.fit`` -- or fit once on the
whole task view -- the recorded row counts would show it, because the inner
search would then score columns chosen with the inner validation labels in
hand.
"""

from __future__ import annotations

from unittest import mock

import numpy as np

from dit.data.selection import SparseBlockSelector
from dit.data.synthetic import make_synthetic_bundle
from dit.evaluation.experiment import ExperimentConfig, run_experiment
from dit.models.classical import describe_feature_layout


def _run_with_spy(*, residualize: bool = False):
    """Run one selection-enabled experiment, recording every selector fit."""

    bundle = make_synthetic_bundle(n_samples=40, n_tracts=3, n_points=10, n_metrics=2, seed=3)
    config = ExperimentConfig(
        model="logistic",
        task="binary",
        split_strategy="stratified",
        feature_view="summary",
        n_splits=2,
        inner_splits=2,
        seed=3,
        enable_selection=True,
        max_features=12,
        covariate_strategy="residualize" if residualize else "feature",
    )
    seen: list[tuple[int, int]] = []
    original_fit = SparseBlockSelector.fit

    def spy(self: SparseBlockSelector, X, y=None):
        values = np.asarray(X)
        seen.append((int(values.shape[0]), int(values.shape[1])))
        return original_fit(self, X, y)

    with mock.patch.object(SparseBlockSelector, "fit", spy):
        result = run_experiment(bundle, config)
    return bundle, result, seen


def test_selector_is_fit_on_inner_fold_subsets_and_refit_on_the_outer_fold() -> None:
    _, result, seen = _run_with_spy()
    outer_train = [fold["n_train"] for fold in result.folds]
    assert seen, "the selector was never fit"
    # Inner-search fits see strictly fewer rows than the outer training fold,
    # which is only possible if the selector rides inside the search.
    assert any(rows < outer_train[-1] for rows, _ in seen)
    assert min(rows for rows, _ in seen) < max(rows for rows, _ in seen)
    # The final call is the outer refit on the last fold's full training set.
    assert seen[-1][0] == outer_train[-1]
    # And no fit ever sees more rows than an outer training fold.
    assert max(rows for rows, _ in seen) == max(outer_train)


def test_selector_never_sees_rows_outside_the_outer_training_fold() -> None:
    bundle, result, seen = _run_with_spy()
    view = bundle.task_view("binary")
    outer_train = max(fold["n_train"] for fold in result.folds)
    # A selector fit once on the whole task view would record view.n_samples
    # rows; the largest fit here is the outer training fold, which excludes
    # the held-out rows.
    assert max(rows for rows, _ in seen) <= outer_train < view.n_samples


def test_selector_fit_count_matches_the_reported_inner_budget() -> None:
    _, result, seen = _run_with_spy()
    # One selector fit per pipeline fit: n_candidates x n_inner_folds inner
    # fits plus one outer refit, per outer fold -- exactly the budget each
    # fold report publishes.
    expected = sum(fold["inner_budget"]["n_fits"] for fold in result.folds)
    assert len(seen) == expected


def test_selector_sees_the_residualized_matrix_without_covariates() -> None:
    # Pipeline order is residualizer -> selector, so under ``residualize`` the
    # selector ranks residuals.  Its own layout is the covariate-free model
    # layout, so its input is narrower than the covariate-inclusive matrix:
    # demographics never reach the ranking model.
    bundle, result, seen = _run_with_spy(residualize=True)
    view = bundle.task_view("binary")
    full_layout = describe_feature_layout(view, view="summary", include_covariates=True)
    selector_width = result.folds[0]["selection"]["n_total"]
    assert seen, "the selector was never fit"
    assert {width for _, width in seen} == {selector_width}
    assert selector_width < full_layout.total_features
