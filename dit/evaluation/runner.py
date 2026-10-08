"""Cross-validation runners with fold-local preprocessing and tuning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from dit.data.schema import DatasetBundle
from dit.data.splits import split_indices
from dit.evaluation.metrics import classification_metrics
from dit.evaluation.provenance import fold_provenance
from dit.evaluation.threshold import ThresholdPolicy
from dit.models.classical import align_probabilities, build_feature_matrix, make_search_estimator


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


def evaluate_classical(
    dataset: DatasetBundle,
    *,
    task: str = "binary",
    model_name: str = "linear_svm",
    split_strategy: str = "loso",
    feature_view: str = "profile",
    smooth_window: int = 5,
    include_covariates: bool = True,
    n_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 42,
    n_jobs: int = 1,
) -> EvaluationResult:
    """Evaluate a classical model using out-of-fold probabilities."""

    view = dataset.task_view(task)
    if view.y is None:
        raise ValueError("evaluation requires labels")
    features = build_feature_matrix(
        view,
        view=feature_view,
        smooth_window=smooth_window,
        include_covariates=include_covariates,
    )
    n_classes = int(np.max(view.y)) + 1
    oof_prob = np.full((view.n_samples, n_classes), np.nan, dtype=float)
    oof_pred = np.full(view.n_samples, -1, dtype=int)
    fold_reports: list[dict[str, object]] = []

    for train_idx, test_idx, fold_id in split_indices(
        view.y,
        view.site,
        strategy=split_strategy,
        n_splits=n_splits,
        seed=seed,
    ):
        provenance = fold_provenance(
            test_idx,
            view.y,
            fold_id=fold_id,
            task=task,
            split_strategy=split_strategy,
            site=view.site,
            data_digest=view.data_digest,
        )
        cv, groups = _inner_cv(
            view.y[train_idx],
            None if view.site is None else view.site[train_idx],
            strategy=split_strategy,
            n_splits=inner_splits,
            seed=seed,
        )
        search = make_search_estimator(
            model_name,
            cv=cv,
            seed=seed,
            scoring="balanced_accuracy",
            n_jobs=n_jobs,
        )
        fit_kwargs: dict[str, Any] = {}
        if groups is not None:
            fit_kwargs["groups"] = groups
        search.fit(features[train_idx], view.y[train_idx], **fit_kwargs)
        probabilities = align_probabilities(
            search.predict_proba(features[test_idx]), search.classes_, n_classes
        )
        predictions = np.argmax(probabilities, axis=1)
        oof_prob[test_idx] = probabilities
        oof_pred[test_idx] = predictions
        report = classification_metrics(view.y[test_idx], predictions, probabilities)
        report.update(
            {
                "fold": str(fold_id),
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "inner_cv": "site_grouped" if groups is not None else "class_stratified",
                "best_params": search.best_params_,
                "inner_best_score": float(search.best_score_),
                **provenance,
            }
        )
        fold_reports.append(report)

    _ensure_complete_oof(oof_pred, oof_prob)
    aggregate = classification_metrics(view.y, oof_pred, oof_prob)
    return EvaluationResult(
        model=model_name,
        task=task,
        split_strategy=split_strategy,
        feature_view=feature_view,
        aggregate=aggregate,
        folds=tuple(fold_reports),
        predictions=oof_pred,
        probabilities=oof_prob,
    )


def evaluate_metric_ensemble(
    dataset: DatasetBundle,
    *,
    task: str = "binary",
    split_strategy: str = "loso",
    smooth_window: int = 5,
    include_covariates: bool = True,
    n_splits: int = 5,
    inner_splits: int = 3,
    seed: int = 42,
    n_jobs: int = 1,
) -> EvaluationResult:
    """Soft-vote linear SVMs trained independently per diffusion metric."""

    view = dataset.task_view(task)
    if view.y is None:
        raise ValueError("evaluation requires labels")
    matrices = [
        build_feature_matrix(
            view,
            view=metric,
            smooth_window=smooth_window,
            include_covariates=include_covariates,
        )
        for metric in view.metric_names
    ]
    n_classes = int(np.max(view.y)) + 1
    oof_prob = np.full((view.n_samples, n_classes), np.nan, dtype=float)
    oof_pred = np.full(view.n_samples, -1, dtype=int)
    fold_reports = []

    for train_idx, test_idx, fold_id in split_indices(
        view.y, view.site, strategy=split_strategy, n_splits=n_splits, seed=seed
    ):
        provenance = fold_provenance(
            test_idx,
            view.y,
            fold_id=fold_id,
            task=task,
            split_strategy=split_strategy,
            site=view.site,
            data_digest=view.data_digest,
        )
        cv, groups = _inner_cv(
            view.y[train_idx],
            None if view.site is None else view.site[train_idx],
            strategy=split_strategy,
            n_splits=inner_splits,
            seed=seed,
        )
        metric_probabilities = []
        best_params = {}
        for metric, features in zip(view.metric_names, matrices):
            search = make_search_estimator(
                "linear_svm", cv=cv, seed=seed, n_jobs=n_jobs
            )
            fit_kwargs = {"groups": groups} if groups is not None else {}
            search.fit(features[train_idx], view.y[train_idx], **fit_kwargs)
            metric_probabilities.append(
                align_probabilities(
                    search.predict_proba(features[test_idx]), search.classes_, n_classes
                )
            )
            best_params[metric] = search.best_params_
        probabilities = np.mean(metric_probabilities, axis=0)
        predictions = np.argmax(probabilities, axis=1)
        oof_prob[test_idx] = probabilities
        oof_pred[test_idx] = predictions
        report = classification_metrics(view.y[test_idx], predictions, probabilities)
        report.update(
            {
                "fold": str(fold_id),
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "inner_cv": "site_grouped" if groups is not None else "class_stratified",
                "best_params": best_params,
                **provenance,
            }
        )
        fold_reports.append(report)

    _ensure_complete_oof(oof_pred, oof_prob)
    aggregate = classification_metrics(view.y, oof_pred, oof_prob)
    return EvaluationResult(
        model="metric_ensemble_linear_svm",
        task=task,
        split_strategy=split_strategy,
        feature_view="per_metric_soft_vote",
        aggregate=aggregate,
        folds=tuple(fold_reports),
        predictions=oof_pred,
        probabilities=oof_prob,
    )


def _ensure_complete_oof(
    predictions: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Fail if any subject was never predicted by an outer fold.

    ``split_indices`` yields a partition, so a gap means a fold failed to
    write its probabilities; scoring the covered subset would hide that.
    """

    valid = (predictions >= 0) & np.all(np.isfinite(probabilities), axis=1)
    if not np.all(valid):
        missing = np.flatnonzero(~valid).tolist()
        raise RuntimeError(f"cross-validation did not predict every subject: {missing[:10]}")


def _inner_cv(y, site, *, strategy: str, n_splits: int, seed: int):
    try:
        from sklearn.model_selection import GroupKFold, StratifiedKFold
    except (ImportError, ValueError) as exc:
        raise RuntimeError("scikit-learn is required for evaluation") from exc

    name = strategy.lower().replace("-", "_")
    if name in {"loso", "leave_one_site_out", "group"} and site is not None:
        unique_sites = np.unique(site)
        if unique_sites.size >= 2:
            return GroupKFold(n_splits=min(n_splits, unique_sites.size)), site
    counts = np.unique(y, return_counts=True)[1]
    usable = min(n_splits, int(np.min(counts)))
    if usable < 2:
        raise ValueError("not enough samples per class for inner cross-validation")
    return StratifiedKFold(n_splits=usable, shuffle=True, random_state=seed), None
