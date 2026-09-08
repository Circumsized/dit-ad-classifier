"""Unified fold-local experiment runner.

This is the entry point the CLI uses.  Every learned component — imputation,
scaling, covariate residualisation, feature selection, hyperparameter search —
is instantiated inside a single outer fold so that no statistic from the held-out
folds can reach the estimator.  ``runner.py`` retains the earlier leaner path;
this module adds the covariate and selection policy on top of it.

``covariate_strategy`` is the one knob that changes what the model is allowed to
see:

* ``feature``      age and sex enter as ordinary inputs;
* ``residualize``  the covariate component is regressed out per fold and the
  residuals are classified, isolating white matter structure from demographics;
* ``none``         demographics are dropped entirely.

The gap between the three is a finding about this dataset, not a bug, so all
three are run and reported together rather than picking one silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from dit.data.covariates import (
    COVARIATE_STRATEGIES,
    Residualizer,
    covariate_matrix,
    strip_covariate_columns,
)
from dit.data.schema import DatasetBundle
from dit.data.preprocessing import smooth_profiles
from dit.data.selection import SparseBlockSelector
from dit.data.splits import split_indices
from dit.evaluation.metrics import classification_metrics
from dit.evaluation.provenance import fold_provenance
from dit.evaluation.site_balance import fold_macro_summary
from dit.evaluation.threshold import ThresholdPolicy, fit_threshold
from dit.models.classical import (
    align_probabilities,
    build_feature_matrix,
    describe_feature_layout,
    make_search_estimator,
)
from dit.models import ALIGNMENTS
from dit.models.calibration import CALIBRATIONS


def _load_deep_modules() -> tuple[Any, Any]:
    """Import the torch-dependent training loop only when a deep model is used.

    PyTorch is an optional extra, so the classical path, the CLI and the config
    loader must import without it.  This is the single place that crosses into
    torch, and it fails with the install hint rather than a bare ImportError.
    """

    try:
        from dit.models.domain_train import DomainTrainConfig, search_domain_classifier
    except ImportError as exc:
        raise RuntimeError(
            "the profile Transformer requires PyTorch, which is an optional extra. "
            "Install it with: pip install -e '.[torch]'"
        ) from exc
    return DomainTrainConfig, search_domain_classifier


# Models that consume the raw [N,T,P,M] profiles instead of the flattened
# table the classical path builds.  Both aliases are accepted so the CLI can be
# read as "the transformer" or "the transformer with domain alignment".
DEEP_MODELS: frozenset[str] = frozenset({"tract_transformer", "domain_transformer", "transformer"})

# Base lineup for cross-model soft voting.  Kept to cheap sklearn models:
# naming the Transformer explicitly is supported, and any deep member is given
# temperature calibration automatically so its probabilities are comparable.
DEFAULT_ENSEMBLE_MODELS: tuple[str, ...] = (
    "linear_svm",
    "logistic",
    "random_forest",
    "hist_gradient_boosting",
)

ENSEMBLE_WEIGHTINGS: tuple[str, ...] = ("equal", "inner_score")
ENSEMBLE_MODEL_NAME: str = "ensemble"


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything that defines one reproducible run."""

    model: str = "linear_svm"
    task: str = "binary"
    split_strategy: str = "stratified"
    feature_view: str = "summary"
    smooth_window: int = 5
    covariate_strategy: str = "feature"
    include_missing_pattern: bool = False
    top_blocks: int | None = None
    keep_fraction: float = 0.05
    max_features: int | None = 300
    threshold_criterion: str = "f1"
    n_splits: int = 5
    inner_splits: int = 3
    seed: int = 42
    n_jobs: int = 1
    enable_selection: bool = False
    alignment: str = "none"
    # ``None`` means "not chosen".  A literal ``"none"`` would be indistinguishable
    # from the default, and then the ensemble could not tell an explicit opt-out
    # apart from an unset value when deciding whether to calibrate a deep base.
    deep_calibration: str | None = None
    deep_epochs: int = 120
    deep_batch_size: int = 16
    deep_patience: int = 20
    deep_d_model: int = 64
    deep_layers: int = 3
    device: str = "cpu"
    ensemble_models: tuple[str, ...] = DEFAULT_ENSEMBLE_MODELS
    ensemble_weighting: str = "inner_score"

    def __post_init__(self) -> None:
        if self.covariate_strategy not in COVARIATE_STRATEGIES:
            raise ValueError(
                f"unknown covariate_strategy {self.covariate_strategy!r}; "
                f"use {COVARIATE_STRATEGIES}"
            )
        if self.threshold_criterion not in {"f1", "balanced", "fixed"}:
            raise ValueError(f"unknown threshold_criterion {self.threshold_criterion!r}")
        if self.n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not (0.0 < self.keep_fraction <= 1.0):
            raise ValueError("keep_fraction must be in (0, 1]")
        if self.alignment not in ALIGNMENTS:
            raise ValueError(f"unknown alignment {self.alignment!r}; use {ALIGNMENTS}")
        if self.deep_epochs < 1:
            raise ValueError("deep_epochs must be >= 1")
        if self.deep_batch_size < 2:
            raise ValueError("deep_batch_size must be >= 2")
        if self.deep_patience < 1:
            raise ValueError("deep_patience must be >= 1")
        if self.deep_d_model % 4:
            raise ValueError("deep_d_model must be divisible by the head count")
        if self.ensemble_weighting not in ENSEMBLE_WEIGHTINGS:
            raise ValueError(
                f"unknown ensemble_weighting {self.ensemble_weighting!r}; use {ENSEMBLE_WEIGHTINGS}"
            )
        if not self.ensemble_models:
            raise ValueError("ensemble_models must name at least one base model")
        if self.deep_calibration is not None and self.deep_calibration not in CALIBRATIONS:
            raise ValueError(f"unknown deep_calibration {self.deep_calibration!r}; use {CALIBRATIONS}")
        if len(set(self.ensemble_models)) != len(self.ensemble_models):
            raise ValueError("ensemble_models must not repeat a model")

    def describe(self) -> dict[str, object]:
        return {
            key: value
            for key, value in sorted(self.__dict__.items())
            if value is not None and value != () and value != [] and value != tuple()
        }

    @property
    def covariates_as_features(self) -> bool:
        return self.covariate_strategy == "feature"


@dataclass(frozen=True)
class ExperimentResult:
    """Outer-CV metrics, a deployment policy and selection diagnostics.

    ``predictions`` and ``aggregate`` are the honest outer-fold argmax
    evaluation.  ``threshold`` is fitted after that evaluation for future
    deployment, so applying it to ``probabilities`` is intentionally not the
    same operation as recreating ``predictions``.
    """

    config: ExperimentConfig
    model_name: str
    aggregate: dict[str, object]
    folds: tuple[dict[str, object], ...]
    predictions: np.ndarray
    probabilities: np.ndarray
    threshold: ThresholdPolicy
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    selection_report: dict[str, object] | None = None
    oof_argmax: dict[str, object] | None = None
    thresholded_selection_metrics: dict[str, object] | None = None

    @property
    def key(self) -> str:
        cfg = self.config
        return (
            f"{self.model_name}|{cfg.task}|{cfg.split_strategy}|"
            f"{cfg.feature_view}|{cfg.covariate_strategy}|"
            f"{cfg.smooth_window}"
        )

    def to_fold_dicts(self) -> tuple[dict[str, object], ...]:
        return self.folds

    def as_evaluation_result(self):
        """Adapt to the shape :mod:`dit.evaluation.runner` expects."""

        from dit.evaluation.runner import EvaluationResult

        # The deep path always consumes the raw 4-D profiles regardless of the
        # requested feature_view, so a report must not claim it used, say,
        # ``summary``; label it ``profiles`` to stay truthful.
        is_deep = self.model_name.lower() in DEEP_MODELS
        view_label = "profiles" if is_deep else self.config.feature_view
        return EvaluationResult(
            model=self.model_name,
            task=self.config.task,
            split_strategy=self.config.split_strategy,
            feature_view=f"{view_label}/{self.config.covariate_strategy}",
            aggregate=self.aggregate,
            folds=self.folds,
            predictions=self.predictions,
            probabilities=self.probabilities,
            threshold=self.threshold,
            thresholded_selection_metrics=self.thresholded_selection_metrics,
        )


def prepare_matrix(
    view: DatasetBundle,
    config: ExperimentConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Build the full feature matrix and its column map for one task view.

    Returns the matrix, the covariate block used for residualisation, and the
    metadata describing which columns are anatomical.
    """

    include_covariates = config.covariates_as_features
    features = build_feature_matrix(
        view,
        view=config.feature_view,
        smooth_window=config.smooth_window,
        include_covariates=include_covariates,
        include_missing_pattern=config.include_missing_pattern,
    )
    layout = describe_feature_layout(
        view,
        view=config.feature_view,
        include_covariates=include_covariates,
        include_missing_pattern=config.include_missing_pattern,
    )
    layout.validate(features.shape[1])
    covariates = covariate_matrix(view.age, view.sex, view.n_samples)
    return features, covariates, {
        "layout": layout,
        "n_core_columns": layout.n_feature_columns,
    }


def _apply_residualizer(
    train: np.ndarray,
    test: np.ndarray,
    train_cov: np.ndarray,
    test_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Residualizer]:
    fitted = Residualizer().fit(train, train_cov)
    return (
        fitted.transform(train, train_cov),
        fitted.transform(test, test_cov),
        fitted,
    )


def _deep_train_config(config: ExperimentConfig) -> "DomainTrainConfig":
    """Map the experiment flags onto one :class:`DomainTrainConfig`."""

    DomainTrainConfig, _ = _load_deep_modules()
    return DomainTrainConfig(
        d_model=config.deep_d_model,
        n_heads=4,
        n_layers=config.deep_layers,
        feedforward_dim=config.deep_d_model * 3,
        epochs=config.deep_epochs,
        batch_size=config.deep_batch_size,
        patience=config.deep_patience,
        alignment=config.alignment,
        calibration=config.deep_calibration or "none",
        seed=config.seed,
    )


def _fit_deep_fold(
    train_profiles: np.ndarray,
    test_profiles: np.ndarray,
    y_train: np.ndarray,
    train_covariates: np.ndarray | None,
    test_covariates: np.ndarray | None,
    train_site: np.ndarray | None,
    config: ExperimentConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Tune, fit and score the profile Transformer on one outer fold.

    The grid search is inner-fold only, exactly like the sklearn path: the
    tuning and validation slices come from ``train_profiles`` and the
    returned probabilities are produced by a refit on the whole outer training
    fold.
    """

    _, search_domain_classifier = _load_deep_modules()
    classifier, search_report = search_domain_classifier(
        train_profiles,
        y_train,
        config=_deep_train_config(config),
        covariates=train_covariates,
        site=train_site,
        device=config.device,
    )
    probabilities = align_probabilities(
        classifier.predict_proba(test_profiles, test_covariates), classifier.classes_, int(np.max(y_train)) + 1
    )
    return probabilities, {
        "best_params": search_report["best_params"],
        "tuning_score": float(search_report["tuning_score"]),
        "training": classifier.parameters(),
        "alignment_grid": search_report["grid"],
    }


def _inner_cv_for(y: np.ndarray, site: np.ndarray | None, config: ExperimentConfig):
    try:
        from sklearn.model_selection import GroupKFold, StratifiedKFold
    except ImportError as exc:
        raise RuntimeError("scikit-learn is required for evaluation") from exc

    name = config.split_strategy.lower().replace("-", "_")
    if name in {"loso", "leave_one_site_out", "group"} and site is not None:
        unique_sites = np.unique(site)
        if unique_sites.size >= 2:
            return GroupKFold(n_splits=min(config.inner_splits, unique_sites.size)), site
    counts = np.unique(y, return_counts=True)[1]
    usable = min(config.inner_splits, int(np.min(counts)))
    if usable < 2:
        raise ValueError("not enough samples per class for inner cross-validation")
    return StratifiedKFold(n_splits=usable, shuffle=True, random_state=config.seed), None


def run_experiment(
    dataset: DatasetBundle,
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Execute one configuration and return out-of-fold scores."""

    config = config or ExperimentConfig()
    view = dataset.task_view(config.task)
    if view.y is None:
        raise ValueError("evaluation requires labels")
    if int(np.max(view.y)) >= 2 and config.task == "binary":
        raise ValueError("binary view must not contain MCI labels")

    features, covariates, meta = prepare_matrix(view, config)
    layout = meta["layout"]
    n_classes = int(np.max(view.y)) + 1
    oof_prob = np.full((view.n_samples, n_classes), np.nan, dtype=float)
    oof_pred = np.full(view.n_samples, -1, dtype=int)
    fold_reports: list[dict[str, object]] = []
    last_selection: dict[str, object] | None = None

    is_deep = config.model.lower() in DEEP_MODELS
    if not is_deep and config.alignment != "none":
        # Domain alignment is a term in the Transformer's training loss; a
        # classical sklearn model has no such loss, so silently ignoring the
        # flag would misreport what was run.  Fail loudly instead.
        raise ValueError(
            f"alignment={config.alignment!r} only applies to the profile Transformer; "
            f"classical model {config.model!r} has no alignment loss"
        )
    if is_deep and config.enable_selection:
        raise ValueError(
            "feature selection is not defined for the profile Transformer, "
            "which never flattens the profiles into columns"
        )
    if is_deep and config.covariate_strategy == "residualize":
        raise ValueError(
            "covariate residualisation is not defined for the profile Transformer; "
            "use 'feature' or 'none'"
        )
    if is_deep:
        # Cross into torch once, up front, so a missing optional extra fails with
        # the install hint before any fold runs rather than mid-evaluation.
        _load_deep_modules()
    # The deep path keeps the 4-D profiles; the classical path uses the table.
    profiles = smooth_profiles(view.X, window=config.smooth_window) if is_deep else None
    deep_covariates = None if config.covariate_strategy == "none" else covariates

    for train_idx, test_idx, fold_id in split_indices(
        view.y,
        view.site,
        strategy=config.split_strategy,
        n_splits=config.n_splits,
        seed=config.seed,
    ):
        train_idx = np.asarray(train_idx, dtype=int)
        test_idx = np.asarray(test_idx, dtype=int)
        provenance = fold_provenance(
            test_idx,
            view.y,
            fold_id=fold_id,
            task=config.task,
            split_strategy=config.split_strategy,
            site=view.site,
        )

        if is_deep:
            probabilities, extra = _fit_deep_fold(
                profiles[train_idx],
                profiles[test_idx],
                view.y[train_idx],
                deep_covariates[train_idx] if deep_covariates is not None else None,
                deep_covariates[test_idx] if deep_covariates is not None else None,
                None if view.site is None else view.site[train_idx],
                config,
            )
            oof_prob[test_idx] = probabilities
            oof_pred[test_idx] = np.argmax(probabilities, axis=1)
            fold_reports.append(
                {
                    **classification_metrics(view.y[test_idx], oof_pred[test_idx], probabilities),
                    "fold": str(fold_id),
                    "n_train": int(train_idx.size),
                    "n_test": int(test_idx.size),
                    "n_features_used": int(profiles[0].size),
                    **provenance,
                    **extra,
                }
            )
            continue

        train_x = features[train_idx]
        test_x = features[test_idx]

        if config.covariate_strategy == "residualize":
            train_x, test_x, residualizer = _apply_residualizer(
                train_x,
                test_x,
                covariates[train_idx],
                covariates[test_idx],
            )
        else:
            residualizer = None

        # The selector is handed to the search so it is fit inside every inner
        # CV fold, not once on the whole outer-training fold.  Fitting it before
        # GridSearchCV would let inner-validation labels pick the columns the
        # search then scores, an optimistic model/hyperparameter choice.
        selector: SparseBlockSelector | None = None
        if config.enable_selection:
            selector = SparseBlockSelector(
                layout,
                top_blocks=config.top_blocks,
                keep_fraction=config.keep_fraction,
                max_features=config.max_features,
                seed=config.seed,
            )

        cv, groups = _inner_cv_for(
            view.y[train_idx],
            None if view.site is None else view.site[train_idx],
            config,
        )
        search = make_search_estimator(
            config.model,
            cv=cv,
            seed=config.seed,
            scoring="balanced_accuracy",
            n_jobs=config.n_jobs,
            selector=selector,
        )
        fit_kwargs: dict[str, Any] = {}
        if groups is not None:
            fit_kwargs["groups"] = groups
        search.fit(train_x, view.y[train_idx], **fit_kwargs)

        probabilities = align_probabilities(
            search.predict_proba(test_x), search.classes_, n_classes
        )
        predictions = np.argmax(probabilities, axis=1)
        oof_prob[test_idx] = probabilities
        oof_pred[test_idx] = predictions

        # Read the selector that was refit on the full outer-training fold, so
        # the per-fold diagnostics describe the model actually scored on the
        # held-out rows.
        fitted_selector = None
        n_features_used = int(train_x.shape[1])
        if selector is not None:
            fitted_selector = search.best_estimator_.named_steps["selector"]
            n_features_used = int(fitted_selector.n_selected)

        report = classification_metrics(view.y[test_idx], predictions, probabilities)
        report.update(
            {
                "fold": str(fold_id),
                "n_train": int(train_idx.size),
                "n_test": int(test_idx.size),
                "best_params": search.best_params_,
                "inner_best_score": float(search.best_score_),
                "n_features_used": n_features_used,
                **provenance,
            }
        )
        if residualizer is not None:
            report["residualizer"] = residualizer.parameters()
        if fitted_selector is not None:
            fold_selection = fitted_selector.report()
            report["selection"] = fold_selection
            last_selection = fold_selection
        fold_reports.append(report)

    if not np.all((oof_pred >= 0) & np.all(np.isfinite(oof_prob), axis=1)):
        missing = np.flatnonzero(
            (oof_pred < 0) | ~np.all(np.isfinite(oof_prob), axis=1)
        ).tolist()
        raise RuntimeError(f"cross-validation did not predict every subject: {missing[:10]}")

    # The outer-fold argmax is the honest performance estimate: each row was
    # predicted by a model that did not train on that row, and no later
    # decision rule has seen its label yet.
    oof_argmax_metrics = classification_metrics(view.y, oof_pred, oof_prob)
    aggregate = dict(oof_argmax_metrics)
    aggregate["threshold_criterion"] = config.threshold_criterion
    aggregate["threshold_evaluation"] = "outer_oof_argmax"
    aggregate["aggregate_argmax_accuracy"] = float(np.mean(oof_pred == view.y))
    # Class names must come from the view's own mapping: a binary view
    # renumbers the labels (1=AD), so the global canonical table would
    # misname the disease class as MCI.
    aggregate["class_names"] = [
        view.label_map[int(c)] for c in sorted(set(view.y.tolist()))
    ]
    # Pooled OOF weights subjects equally; the macro summary weights folds
    # equally and drops incomparable folds, which matters most for LOSO.
    aggregate["fold_macro"] = fold_macro_summary(fold_reports)

    # Fit the requested adaptive rule only as a final deployment policy.  Its
    # score is deliberately kept separate because this same OOF label set was
    # used to choose the policy and therefore cannot evaluate it independently.
    policy = fit_threshold(
        view.y,
        oof_prob,
        criterion=config.threshold_criterion,
    )
    thresholded_selection_metrics = classification_metrics(view.y, policy.predict(oof_prob), oof_prob)
    thresholded_selection_metrics["scope"] = "all_outer_oof_used_for_threshold_fit"
    thresholded_selection_metrics["not_for_performance_comparison"] = True

    return ExperimentResult(
        config=config,
        model_name=config.model,
        aggregate=aggregate,
        folds=tuple(fold_reports),
        predictions=oof_pred,
        probabilities=oof_prob,
        threshold=policy,
        feature_names=tuple(layout.feature_names()),
        selection_report=last_selection,
        oof_argmax=oof_argmax_metrics,
        thresholded_selection_metrics=thresholded_selection_metrics,
    )


def _fold_weights(scores: list[float], weighting: str) -> np.ndarray:
    """Normalised per-fold weights for soft voting.

    ``inner_score`` weights each base model by the balanced accuracy its nested
    search achieved on this fold.  That number comes from inside the same outer
    fold, so weighting on it is not leakage.  A base model that scored zero or
    worse gets weight zero instead of dragging the average down, and if every
    base model is unusable on a fold the weights fall back to equal.
    """

    if weighting == "equal":
        return np.full(len(scores), 1.0 / len(scores))
    values = np.where(np.isfinite(scores) & (np.asarray(scores) > 0), scores, 0.0)
    total = values.sum()
    if total <= 0:
        return np.full(len(scores), 1.0 / len(scores))
    return values / total


def _base_config_for_ensemble(config: ExperimentConfig, model: str) -> ExperimentConfig:
    """Per-base configuration, enabling calibration where a base is deep.

    Soft voting averages probabilities, and only those are comparable across
    models.  A class-weighted Transformer emits systematically inflated
    confidences, so leaving it uncalibrated would let its vote dominate on
    scale rather than on information.  The ensemble switches on temperature
    calibration for a deep base itself instead of trusting the caller to
    remember, and never overrides an explicit user choice.
    """

    base = replace(config, model=model)
    if model.lower() in DEEP_MODELS and base.deep_calibration is None:
        base = replace(base, deep_calibration="temperature")
    return base


def run_ensemble(
    dataset: DatasetBundle,
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Soft-vote across the base lineup on out-of-fold probabilities.

    Each base model is run through :func:`run_experiment` with its own nested
    search, so every averaged probability remains out of fold for the subject
    it describes.  Weights are recomputed per fold rather than once globally:
    which model is informative differs by fold, and a single global weighting
    would silently assume otherwise.
    """

    config = config or ExperimentConfig()
    view = dataset.task_view(config.task)
    if view.y is None:
        raise ValueError("evaluation requires labels")

    base_configs = [_base_config_for_ensemble(config, model) for model in config.ensemble_models]
    base_results = [run_experiment(dataset, base_config) for base_config in base_configs]
    n_classes = int(np.max(view.y)) + 1
    oof_prob = np.zeros((view.n_samples, n_classes), dtype=float)
    fold_reports: list[dict[str, object]] = []

    def _fold_tuning_score(fold: dict[str, object]) -> float:
        # Deep folds report their nested-search score as ``tuning_score``
        # (there is no sklearn GridSearchCV to name ``inner_best_score``
        # after); reading only the classical key would silently weight every
        # deep member of the lineup to zero.
        raw = fold.get("inner_best_score", fold.get("tuning_score", 0.0))
        return float(raw) if raw is not None else 0.0

    for train_idx, test_idx, fold_id in split_indices(
        view.y,
        view.site,
        strategy=config.split_strategy,
        n_splits=config.n_splits,
        seed=config.seed,
    ):
        test_idx = np.asarray(test_idx, dtype=int)
        fold_id = str(fold_id)
        indexed: list[dict[str, object]] = []
        for result in base_results:
            match = next(
                (fold for fold in result.folds if str(fold.get("fold")) == fold_id), None
            )
            if match is None:
                raise RuntimeError(
                    f"base model {result.model_name} reported no fold {fold_id!r}; "
                    "base runs must share the outer fold structure"
                )
            indexed.append(match)

        scores = [_fold_tuning_score(match) for match in indexed]
        weights = _fold_weights(scores, config.ensemble_weighting)
        averaged = sum(
            weight * result.probabilities[test_idx]
            for weight, result in zip(weights, base_results)
        )
        oof_prob[test_idx] = averaged

        average_report = classification_metrics(
            view.y[test_idx], np.argmax(averaged, axis=1), averaged
        )
        provenance = fold_provenance(
            test_idx,
            view.y,
            fold_id=fold_id,
            task=config.task,
            split_strategy=config.split_strategy,
            site=view.site,
        )
        fold_reports.append(
            {
                **average_report,
                "fold": fold_id,
                "n_train": int(np.asarray(train_idx, dtype=int).size),
                "n_test": int(test_idx.size),
                **provenance,
                "weights": {
                    model: float(weight)
                    for model, weight in zip(config.ensemble_models, weights)
                },
                "base_models": [
                    {
                        "model": base_config.model,
                        # Recorded here because the effective value for a deep
                        # base is decided in _base_config_for_ensemble, which is
                        # local to this function.  Without it a reader cannot
                        # tell whether the probabilities being averaged were
                        # calibrated at all.
                        "deep_calibration": base_config.deep_calibration,
                        "inner_best_score": _fold_tuning_score(match),
                        "fold_balanced_accuracy": float(match.get("balanced_accuracy", 0.0)),
                    }
                    for base_config, result, match in zip(base_configs, base_results, indexed)
                ],
            }
        )

    if not np.all(np.isfinite(oof_prob)):
        missing = np.flatnonzero(~np.isfinite(oof_prob)).tolist()
        raise RuntimeError(f"ensemble did not predict every subject: {missing[:10]}")

    oof_argmax = np.argmax(oof_prob, axis=1)
    oof_argmax_metrics = classification_metrics(view.y, oof_argmax, oof_prob)
    aggregate = dict(oof_argmax_metrics)
    aggregate["threshold_criterion"] = config.threshold_criterion
    aggregate["threshold_evaluation"] = "outer_oof_argmax"
    aggregate["aggregate_argmax_accuracy"] = float(np.mean(oof_argmax == view.y))
    # Same view-local naming as run_experiment: the ensemble inherits the
    # same binary-view renumbering, so the global table would misname here too.
    aggregate["class_names"] = [
        view.label_map[int(c)] for c in sorted(set(view.y.tolist()))
    ]
    aggregate["ensemble_models"] = list(config.ensemble_models)
    aggregate["ensemble_weighting"] = config.ensemble_weighting
    aggregate["fold_macro"] = fold_macro_summary(fold_reports)
    aggregate["base_model_scores"] = {
        result.model_name: result.aggregate
        for result in base_results
    }

    # This policy is fit for future deployment only.  It is not used for the
    # outer-CV score because the full OOF labels selected it.
    policy = fit_threshold(view.y, oof_prob, criterion=config.threshold_criterion)
    thresholded_selection_metrics = classification_metrics(view.y, policy.predict(oof_prob), oof_prob)
    thresholded_selection_metrics["scope"] = "all_outer_oof_used_for_threshold_fit"
    thresholded_selection_metrics["not_for_performance_comparison"] = True

    return ExperimentResult(
        config=config,
        model_name=ENSEMBLE_MODEL_NAME,
        aggregate=aggregate,
        folds=tuple(fold_reports),
        predictions=oof_argmax,
        probabilities=oof_prob,
        threshold=policy,
        feature_names=base_results[0].feature_names,
        selection_report=base_results[0].selection_report,
        oof_argmax=oof_argmax_metrics,
        thresholded_selection_metrics=thresholded_selection_metrics,
    )


def run_configured_experiment(
    dataset: DatasetBundle,
    config: ExperimentConfig | None = None,
) -> ExperimentResult:
    """Dispatch to the ensemble path when the model name selects it."""

    config = config or ExperimentConfig()
    if config.model.lower() == ENSEMBLE_MODEL_NAME:
        return run_ensemble(dataset, config)
    return run_experiment(dataset, config)
