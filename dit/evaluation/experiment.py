"""Unified fold-local experiment runner.

The CLI enters through this module. Every learned component (imputation,
scaling, covariate residualisation, feature selection, and hyperparameter
search) is instantiated within an outer fold, keeping held-out statistics out
of the estimator. ``runner.py`` retains the leaner path; this module adds the
covariate and selection policy.

``covariate_strategy`` controls what the model can use:

* ``feature``      age and sex enter as ordinary inputs;
* ``residualize``  the covariate component is regressed out per fold and the
  residuals are classified, isolating white matter structure from demographics;
* ``none``         demographics are dropped entirely.

All three strategies are reported because their gap is a dataset result.
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
    CONTROL_VIEWS,
    align_probabilities,
    build_control_matrix,
    build_feature_matrix,
    describe_feature_layout,
    fitted_pipeline_step,
    make_search_estimator,
)
from dit.models import ALIGNMENTS
from dit.models.calibration import CALIBRATIONS


def _load_deep_modules() -> tuple[Any, Any]:
    """Import the torch training loop when a deep model is selected.

    PyTorch is optional, so the classical path, CLI, and config loader import
    without it. This is the only boundary into torch and provides the install
    hint when the extra is missing.
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
    # Opt-in deterministic deep-training kernels (plan P0.6); reported, not
    # assumed — see DomainTrainConfig.deterministic.
    deep_deterministic: bool = False
    ensemble_models: tuple[str, ...] = DEFAULT_ENSEMBLE_MODELS
    ensemble_weighting: str = "inner_score"
    # The inner search optimises this scorer on estimator.predict(); the outer
    # report scores argmax(predict_proba()).  Declaring both makes the contract
    # explicit: where they can disagree (see make_search_estimator), the report
    # names which rule produced its numbers.
    selection_metric: str = "balanced_accuracy"
    # Negative-control input: demographics alone or missingness alone, run
    # through the same folds and metrics as the imaging views. ``None`` keeps
    # the configured anatomical feature view.
    control_view: str | None = None

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
        if not isinstance(self.selection_metric, str) or not self.selection_metric:
            raise ValueError("selection_metric must be a non-empty scorer name")
        if self.control_view is not None and self.control_view not in CONTROL_VIEWS:
            raise ValueError(
                f"unknown control_view {self.control_view!r}; use one of {CONTROL_VIEWS} or None"
            )

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
    """Outer-CV metrics, a deployment policy, and selection diagnostics.

    ``predictions`` and ``aggregate`` are the outer-fold argmax evaluation.
    ``threshold`` is fitted afterward for deployment, so applying it to
    ``probabilities`` does not recreate ``predictions``.
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
        base = (
            f"{self.model_name}|{cfg.task}|{cfg.split_strategy}|"
            f"{cfg.feature_view}|{cfg.covariate_strategy}|"
            f"{cfg.smooth_window}"
        )
        # Appended only when set, so keys for ordinary runs stay identical to
        # earlier reports.
        return f"{base}|{cfg.control_view}" if cfg.control_view else base

    def to_fold_dicts(self) -> tuple[dict[str, object], ...]:
        return self.folds

    def as_evaluation_result(self):
        """Adapt to the shape :mod:`dit.evaluation.runner` expects."""

        from dit.evaluation.runner import EvaluationResult

        # The deep path always consumes raw 4-D profiles. Label it ``profiles``
        # instead of using the configured view, which the deep path ignores.
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

    Under ``residualize`` the covariates ride along as the matrix's trailing
    columns so the pipeline's :class:`ResidualizeCovariates` step can strip and
    regress them out fold-internally; ``model_layout`` then describes the
    columns the classifier actually sees (covariates removed). Under a
    negative-control view the matrix *is* the control input and no layout
    applies. Returns the matrix, the standalone covariate block used by the
    deep path, and the metadata describing which columns are anatomical.
    """

    if config.control_view is not None:
        features, names = build_control_matrix(view, config.control_view)
        covariates = covariate_matrix(view.age, view.sex, view.n_samples)
        return features, covariates, {
            "layout": None,
            "model_layout": None,
            "control_names": tuple(names),
            "n_core_columns": int(features.shape[1]),
        }

    include_covariates = config.covariates_as_features
    residualize = config.covariate_strategy == "residualize"
    features = build_feature_matrix(
        view,
        view=config.feature_view,
        smooth_window=config.smooth_window,
        include_covariates=include_covariates or residualize,
        include_missing_pattern=config.include_missing_pattern,
    )
    layout = describe_feature_layout(
        view,
        view=config.feature_view,
        include_covariates=include_covariates or residualize,
        include_missing_pattern=config.include_missing_pattern,
    )
    layout.validate(features.shape[1])
    model_layout = (
        replace(layout, has_covariates=False) if residualize and not include_covariates else layout
    )
    covariates = covariate_matrix(view.age, view.sex, view.n_samples)
    return features, covariates, {
        "layout": layout,
        "model_layout": model_layout,
        "n_core_columns": layout.n_feature_columns,
    }


def _apply_residualizer(
    train: np.ndarray,
    test: np.ndarray,
    train_cov: np.ndarray,
    test_cov: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, Residualizer]:
    """Standalone residualization kept for direct callers and tests.

    The experiment path no longer uses it: residualization there lives inside
    the sklearn pipeline (see :func:`make_search_estimator`), which is what
    keeps the regression fold-local under nested search.
    """

    fitted = Residualizer().fit(train, train_cov)
    return (
        fitted.transform(train, train_cov),
        fitted.transform(test, test_cov),
        fitted,
    )


def _require_inner_class_support(cv: Any, y_train: np.ndarray, groups: np.ndarray | None) -> None:
    """Fail before the search when an inner validation slice misses a class.

    Balanced accuracy averages per-class recall over the classes present in the
    scored rows, so a candidate judged on a slice without every class would win
    grid points on a partial — and therefore incomparable — score.
    """

    y = np.asarray(y_train).reshape(-1)
    expected = set(np.unique(y).tolist())
    stub = np.zeros((y.size, 1))
    for position, (_, validation) in enumerate(cv.split(stub, y, groups)):
        present = set(np.unique(y[validation]).tolist())
        missing = sorted(expected - present)
        if missing:
            raise ValueError(
                f"inner fold {position} holds out rows without classes {missing}; "
                "the search would score candidates on a partial class set. "
                "Reduce inner_splits or use a split strategy that keeps every "
                "class inside each inner validation slice."
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
        deterministic=config.deep_deterministic,
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
    n_classes: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Tune, fit and score the profile Transformer on one outer fold.

    The grid search is inner-fold only, exactly like the sklearn path: the
    tuning and validation slices come from ``train_profiles`` and the
    returned probabilities are produced by a refit on the whole outer training
    fold. ``n_classes`` is the task-view class count, not the training fold's
    observed maximum: aligning to the training fold would silently narrow the
    probability matrix when the outer training rows miss the highest class.
    """

    _, search_domain_classifier = _load_deep_modules()
    classifier, search_report = search_domain_classifier(
        train_profiles,
        y_train,
        config=_deep_train_config(config),
        covariates=train_covariates,
        site=train_site,
        split_strategy=config.split_strategy,
        device=config.device,
    )
    probabilities = align_probabilities(
        classifier.predict_proba(test_profiles, test_covariates),
        classifier.classes_,
        n_classes,
    )
    return probabilities, {
        "best_params": search_report["best_params"],
        # search_domain_classifier always records how candidates were scored;
        # indexing strictly instead of .get(...) keeps a silently missing
        # field from masquerading as the stratified default in reports.
        "inner_cv": search_report["inner_cv"],
        "tuning_score": float(search_report["tuning_score"]),
        "epochs_requested": search_report.get("epochs_requested"),
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
    layout = meta["model_layout"]
    control_names: tuple[str, ...] | None = meta.get("control_names")
    n_classes = int(np.max(view.y)) + 1
    oof_prob = np.full((view.n_samples, n_classes), np.nan, dtype=float)
    oof_pred = np.full(view.n_samples, -1, dtype=int)
    fold_reports: list[dict[str, object]] = []
    last_selection: dict[str, object] | None = None

    is_deep = config.model.lower() in DEEP_MODELS
    if config.control_view is not None:
        # A control isolates one input channel; combining it with a deep
        # network, feature selection, residualization or an ensemble changes
        # the question it answers.
        if is_deep:
            raise ValueError("control views are defined for classical models only")
        if config.enable_selection:
            raise ValueError("feature selection does not apply to control views")
        if config.covariate_strategy == "residualize":
            raise ValueError("residualization does not apply to control views")
    if not is_deep and config.alignment != "none":
        # Domain alignment is part of the Transformer loss; classical sklearn
        # models have no corresponding term.
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
        # Import torch before folds start so a missing optional extra reports
        # the install hint at startup.
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
            data_digest=view.data_digest,
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
                n_classes,
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
        residualize = config.covariate_strategy == "residualize"

        # The selector is handed to the search so it is fit inside every inner
        # CV fold, not once on the whole outer-training fold.  Fitting it before
        # GridSearchCV would let inner-validation labels pick the columns the
        # search then scores, an optimistic model/hyperparameter choice.  Under
        # residualization the selector sees the covariate-free residuals, and
        # the regression itself is refit inside each inner fold as well.
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
        _require_inner_class_support(cv, view.y[train_idx], groups)
        search = make_search_estimator(
            config.model,
            cv=cv,
            seed=config.seed,
            scoring=config.selection_metric,
            n_jobs=config.n_jobs,
            selector=selector,
            residualize=residualize,
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
        fitted_selector = fitted_pipeline_step(search, "selector")
        n_features_used = layout.total_features if residualize else int(train_x.shape[1])
        if fitted_selector is not None:
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
        # Actual inner-search accounting, so reports state what was computed
        # rather than leaving the reader to infer it from the grid.
        n_candidates = len(search.cv_results_["params"])
        report["inner_budget"] = {
            "n_candidates": n_candidates,
            "n_inner_folds": int(cv.get_n_splits(groups=groups)),
            "n_fits": n_candidates * int(cv.get_n_splits(groups=groups)) + 1,
            "candidate_fit_seconds": float(
                np.sum(search.cv_results_.get("mean_fit_time", np.zeros(n_candidates)))
            ),
        }
        if residualize:
            fitted_residualizer = fitted_pipeline_step(search, "residualizer")
            report["residualizer"] = fitted_residualizer.residualizer_.parameters()
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

    # The outer-fold argmax is the performance estimate: each row came from a
    # model that did not train on it, before any rule observed its label.
    oof_argmax_metrics = classification_metrics(view.y, oof_pred, oof_prob)
    aggregate = dict(oof_argmax_metrics)
    aggregate["threshold_criterion"] = config.threshold_criterion
    aggregate["threshold_evaluation"] = "outer_oof_argmax"
    aggregate["aggregate_argmax_accuracy"] = float(np.mean(oof_pred == view.y))
    # Declared alongside the numbers: the inner search optimised
    # selection_metric on estimator.predict(), the rows below are argmax over
    # predict_proba(). For the calibrated-SVM path the two coincide by
    # construction; the field keeps the contract checkable rather than assumed.
    aggregate["selection_metric"] = config.selection_metric
    aggregate["prediction_rule"] = "predict_proba_argmax"
    if config.control_view is not None:
        aggregate["control_view"] = config.control_view
    # Use the task view's label mapping. Binary views renumber AD to class 1;
    # the global canonical table would name it MCI.
    aggregate["class_names"] = [
        view.label_map[int(c)] for c in sorted(set(view.y.tolist()))
    ]
    # Pooled OOF weights subjects equally; the macro summary weights folds
    # equally and drops incomparable folds, which matters most for LOSO.
    aggregate["fold_macro"] = fold_macro_summary(fold_reports)

    # Fit the adaptive rule as a deployment policy. Its score stays separate
    # because the same OOF labels selected the policy.
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
        feature_names=control_names if control_names is not None else tuple(layout.feature_names()),
        selection_report=last_selection,
        oof_argmax=oof_argmax_metrics,
        thresholded_selection_metrics=thresholded_selection_metrics,
    )


def _fold_weights(scores: list[float], weighting: str) -> np.ndarray:
    """Normalised per-fold weights for soft voting.

    ``inner_score`` uses each base model's balanced accuracy from nested search
    within the same outer fold, avoiding leakage. Non-positive scores receive
    zero weight; if all scores are unusable, weights fall back to equal.
    """

    if weighting == "equal":
        return np.full(len(scores), 1.0 / len(scores))
    values = np.where(np.isfinite(scores) & (np.asarray(scores) > 0), scores, 0.0)
    total = values.sum()
    if total <= 0:
        return np.full(len(scores), 1.0 / len(scores))
    return values / total


def _base_config_for_ensemble(config: ExperimentConfig, model: str) -> ExperimentConfig:
    """Per-base configuration, enabling calibration for deep members.

    Soft voting averages probabilities, so their scales must be comparable.
    Class weighting can inflate Transformer confidence; temperature calibration
    keeps that vote from dominating by scale. Explicit user choices are kept.
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

    Each base model runs through :func:`run_experiment` with its own nested
    search, keeping every averaged probability out of fold for its subject.
    Weights are recomputed per fold because model informativeness can vary
    across folds.
    """

    config = config or ExperimentConfig()
    view = dataset.task_view(config.task)
    if view.y is None:
        raise ValueError("evaluation requires labels")
    if config.control_view is not None:
        raise ValueError("control views cannot enter an ensemble; run them standalone")

    base_configs = [_base_config_for_ensemble(config, model) for model in config.ensemble_models]
    base_results = [run_experiment(dataset, base_config) for base_config in base_configs]
    n_classes = int(np.max(view.y)) + 1
    oof_prob = np.zeros((view.n_samples, n_classes), dtype=float)
    fold_reports: list[dict[str, object]] = []

    def _fold_tuning_score(fold: dict[str, object]) -> float:
        # Deep folds use ``tuning_score`` because they do not use sklearn
        # GridSearchCV. Falling back only to the classical key would give deep
        # members zero weight.
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
            data_digest=view.data_digest,
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
