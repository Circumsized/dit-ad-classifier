"""Leakage-safe classical baselines for AFQ profiles."""

from __future__ import annotations

from typing import Any

import numpy as np

from dit.data.covariates import ResidualizeCovariates, covariate_matrix
from dit.data.layout import FeatureLayout, view_for_layout
from dit.data.preprocessing import smooth_profiles
from dit.data.schema import DatasetBundle
from dit.data.sklearn_compat import l1_ratio_kwargs

# Negative-control inputs: each answers "what does the model learn from
# demographics alone / missingness alone?" and is run through the same
# folds, metrics and threshold machinery as the imaging views.
CONTROL_VIEWS = ("demographics", "missingness")


def build_control_matrix(dataset: DatasetBundle, kind: str) -> tuple[np.ndarray, list[str]]:
    """Build a negative-control input matrix and its column names.

    ``demographics``  the [age, sex] matrix alone, missing values as NaN.
    ``missingness``   one column per tract: the fraction of nodes AFQ failed
                      to identify. If a model scores as well on this as on
                      imaging, the signal is acquisition/quality artifacts,
                      not biology.
    """

    normalized = str(kind).lower()
    if normalized == "demographics":
        matrix = covariate_matrix(dataset.age, dataset.sex, dataset.n_samples)
        names = ["age", "sex"]
    elif normalized == "missingness":
        valid = dataset.mask if dataset.mask is not None else np.isfinite(dataset.X)
        matrix = (~valid).any(axis=-1).astype(np.float64).mean(axis=-1)
        names = [f"missing|{tract}" for tract in dataset.tract_names]
    else:
        raise ValueError(
            f"unknown control view {kind!r}; use one of {CONTROL_VIEWS}"
        )
    return matrix.astype(np.float64, copy=False), names


def build_feature_matrix(
    dataset: DatasetBundle,
    *,
    view: str = "profile",
    smooth_window: int = 1,
    include_covariates: bool = True,
    include_missing_pattern: bool = False,
) -> np.ndarray:
    """Build raw features; learned transforms belong in the sklearn pipeline.

    ``profile`` keeps every along-tract node, laid out tract-major then metric
    so each (tract, metric) pair is a contiguous block of ``n_points``
    columns. Contiguity makes block selection and the tract x node heatmaps
    addressable; see :func:`describe_feature_layout`.

    ``summary`` emits mean, standard deviation, linear slope and trapezoidal
    area per tract/metric. A metric name such as ``FA`` selects one channel
    while retaining all tract nodes.

    ``include_missing_pattern`` appends the per-tract fraction of nodes AFQ
    failed to identify, after the profile columns so the block accounting is
    unchanged.
    """

    values = smooth_profiles(dataset.X, smooth_window)
    normalized = view.upper()
    if normalized == "PROFILE":
        features = values.transpose(0, 1, 3, 2).reshape(values.shape[0], -1)
    elif normalized == "SUMMARY":
        features = _profile_summary(values)
    elif normalized in {name.upper() for name in dataset.metric_names}:
        index = [name.upper() for name in dataset.metric_names].index(normalized)
        features = values[..., index].reshape(values.shape[0], -1)
    else:
        raise ValueError(
            f"unknown feature view {view!r}; use profile, summary, or one of "
            f"{dataset.metric_names}"
        )

    if include_missing_pattern:
        features = np.column_stack((features, _missing_pattern(dataset)))

    if include_covariates:
        age = np.full(dataset.n_samples, np.nan) if dataset.age is None else dataset.age
        sex = np.full(dataset.n_samples, np.nan) if dataset.sex is None else dataset.sex
        features = np.column_stack((features, age, sex))
    return features.astype(np.float64, copy=False)


def describe_feature_layout(
    dataset: DatasetBundle,
    *,
    view: str,
    include_covariates: bool = True,
    include_missing_pattern: bool = False,
) -> FeatureLayout:
    """Return the column map matching :func:`build_feature_matrix`.

    Derived from the same view string and dataset metadata as the matrix, so a
    mismatch raises instead of mislabeling features.
    """

    return view_for_layout(
        view,
        dataset.metric_names,
        dataset.tract_names,
        dataset.n_points,
        has_covariates=include_covariates,
        has_missing_pattern=include_missing_pattern,
    )


def _missing_pattern(dataset: DatasetBundle) -> np.ndarray:
    """Per-tract fraction of nodes that are missing, shape [N, T]."""

    valid = dataset.mask if dataset.mask is not None else np.isfinite(dataset.X)
    return (~valid).any(axis=-1).astype(np.float64).mean(axis=-1)


def make_search_estimator(
    model_name: str,
    *,
    cv: Any,
    seed: int = 42,
    scoring: str = "balanced_accuracy",
    n_jobs: int = 1,
    selector: Any | None = None,
    residualize: bool = False,
):
    """Create a tuned sklearn pipeline with split-local preprocessing.

    When ``selector`` is given it is inserted as the first pipeline step, so it
    is fit inside every inner cross-validation fold rather than once on the
    whole outer training fold. Inner-validation labels then stay out of the
    features the search scores on.

    ``residualize`` prepends :class:`ResidualizeCovariates`, which reads age
    and sex from the matrix's trailing columns and regresses them out. Inside
    the pipeline, the regression is refit on every inner training fold — the
    covariates of inner-validation rows never touch the coefficients.

    For SVMs on sklearn >= 1.9 the probability calibrator wraps the *entire*
    preprocessing pipeline, not just the estimator: the Platt scalars are then
    fit on folds where imputation, scaling and selection are refit as well, so
    a supervised selector cannot leak its fold's labels into the calibration
    rows. Grid keys address the inner estimator as
    ``model__estimator__model__<param>``.
    """

    try:
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
    except (ImportError, ValueError) as exc:
        raise RuntimeError(
            "scikit-learn could not be imported. Install the versions pinned "
            "in pyproject.toml inside an isolated environment."
        ) from exc

    name = model_name.lower().replace("-", "_")
    calibrated_svm = False
    if name in {"linear_svm", "svm", "official_svm"}:
        kernel = "linear"
        parameters = None
        calibrated_svm = True
    elif name in {"rbf_svm", "rbf"}:
        kernel = "rbf"
        parameters = None
        calibrated_svm = True
    elif name in {"logistic", "elastic_net", "lr"}:
        # ``_l1_ratio_kwargs`` already carries the solver it needs.
        estimator = LogisticRegression(
            **_l1_ratio_kwargs(0.5, "saga"),
            class_weight="balanced",
            # SAGA with an elastic net stalls well short of 5 000 iterations on
            # the 7 200-column profile view and returns coefficients from an
            # unconverged solution.
            max_iter=20000,
            random_state=seed,
        )
        parameters = {
            "model__C": np.logspace(-3, 2, 6),
            "model__l1_ratio": [0.0, 0.5, 1.0],
        }
    elif name in {"random_forest", "rf"}:
        estimator = RandomForestClassifier(
            class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed
        )
        parameters = {
            "model__n_estimators": [200, 500],
            "model__max_features": ["sqrt", 0.25],
            "model__min_samples_leaf": [1, 3, 5],
        }
    elif name in {"hist_gradient_boosting", "histgb", "hgb"}:
        estimator = HistGradientBoostingClassifier(random_state=seed)
        parameters = {
            "model__learning_rate": [0.03, 0.1],
            "model__max_leaf_nodes": [7, 15, 31],
            "model__l2_regularization": [0.0, 1.0],
        }
    else:
        raise ValueError(f"unknown classical model: {model_name}")

    # ``keep_empty_features`` preserves the column count when a fold contains
    # an all-NaN feature: dropping it would shift every later coefficient away
    # from its anatomical column and mislabel interpretation output.
    preprocessing: list[tuple[str, Any]] = []
    if residualize:
        preprocessing.append(("residualizer", ResidualizeCovariates()))
    if selector is not None:
        preprocessing.append(("selector", selector))
    preprocessing.append(
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))
    )
    preprocessing.append(("scaler", StandardScaler()))

    if calibrated_svm:
        from sklearn import __version__ as sklearn_version

        parts = tuple(int(part) for part in sklearn_version.split(".")[:2])
        svc = SVC(kernel=kernel, class_weight="balanced", random_state=seed)
        if parts < (1, 9):
            # Older sklearn has no calibration wrapper API: Platt scaling is
            # solved jointly with the margin inside SVC(probability=True).
            # Its internal cross-fit cannot refit preprocessing either; the
            # maintained path is the >= 1.9 branch above.
            pipeline = Pipeline(
                preprocessing
                + [
                    (
                        "model",
                        SVC(
                            kernel=kernel,
                            probability=True,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    )
                ]
            )
            prefix = "model__"
        else:
            from sklearn.calibration import CalibratedClassifierCV

            inner = Pipeline(preprocessing + [("model", svc)])
            wrapper = Pipeline(
                [
                    (
                        "model",
                        CalibratedClassifierCV(
                            inner,
                            cv=3,
                            ensemble=False,
                            method="sigmoid",
                        ),
                    )
                ]
            )
            pipeline = wrapper
            prefix = "model__estimator__model__"
        if name in {"linear_svm", "svm", "official_svm"}:
            parameters = {f"{prefix}C": np.logspace(-4, 0, 9)}
        else:
            parameters = {
                f"{prefix}C": np.logspace(-2, 2, 5),
                f"{prefix}gamma": ["scale", 1e-3, 1e-2],
            }
    else:
        pipeline = Pipeline(preprocessing + [("model", estimator)])

    return GridSearchCV(
        pipeline,
        parameters,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )


def fitted_pipeline_step(search: Any, step_name: str) -> Any | None:
    """Locate a named step inside a fitted search's best pipeline.

    The calibrated-SVM layout nests the working pipeline one level deeper, so
    consumers ask here instead of hard-coding a ``named_steps`` path that only
    matches one estimator family. On sklearn >= 1.9 the wrapper's ``.estimator``
    attribute is the unfitted template; the pipeline fitted on the full outer
    training fold lives in ``calibrated_classifiers_[0].estimator``.
    """

    estimator = search.best_estimator_
    named = getattr(estimator, "named_steps", {})
    if step_name in named:
        return named[step_name]
    model = named.get("model")
    calibrated = getattr(model, "calibrated_classifiers_", None)
    if calibrated:
        inner_candidates = [getattr(entry, "estimator", None) for entry in calibrated]
    else:
        inner_candidates = [getattr(model, "estimator", None)]
    for inner in inner_candidates:
        inner_named = getattr(inner, "named_steps", {})
        if step_name in inner_named:
            return inner_named[step_name]
    return None


def align_probabilities(probabilities: np.ndarray, model_classes, n_classes: int) -> np.ndarray:
    """Place estimator probabilities into canonical class columns.

    Fails closed when the estimator's class set does not cover the task: a
    zero-filled column would silently present "the model never saw this
    class" as a calibrated probability of 0 and let argmax pick a winner on
    noise.
    """

    labels = [int(label) for label in np.asarray(model_classes, dtype=int)]
    seen = set(labels)
    if len(seen) != len(labels):
        raise ValueError("model reports a duplicated class")
    outside = sorted(seen - set(range(n_classes)))
    if outside:
        raise ValueError(
            f"model reports classes {outside} outside the task range 0..{n_classes - 1}"
        )
    missing = sorted(set(range(n_classes)) - seen)
    if missing:
        raise ValueError(
            f"training fold never saw class(es) {missing}; padding their "
            "probabilities with zeros would hide that they cannot be scored"
        )
    output = np.zeros((probabilities.shape[0], n_classes), dtype=float)
    for source, label in enumerate(labels):
        output[:, label] = probabilities[:, source]
    return output


def _l1_ratio_kwargs(l1_ratio: float, solver: str) -> dict:
    """LogisticRegression regularisation kwargs spanning sklearn 1.3 to 1.10.

    See :func:`dit.data.sklearn_compat.l1_ratio_kwargs`; kept as a local alias
    so the call sites in this module read the same way.
    """

    return l1_ratio_kwargs(l1_ratio, solver)


def _profile_summary(values: np.ndarray) -> np.ndarray:
    """Per-tract/metric mean, standard deviation, linear slope and area.

    ``np.trapz`` is not NaN-aware: a single missing node makes the whole area
    NaN, turning a 1% missing rate into a 16% missing feature rate and dropping
    a quarter of the columns. Slope and area are computed over the measured
    nodes only.
    """

    safe = np.where(np.isfinite(values), values, np.nan)
    positions = np.linspace(-1.0, 1.0, values.shape[2], dtype=np.float64)
    mean = np.nanmean(safe, axis=2)
    std = np.nanstd(safe, axis=2)
    slope = _nan_slope(safe, positions)
    area = _nan_area(safe, positions)
    return np.concatenate(
        [item.reshape(values.shape[0], -1) for item in (mean, std, slope, area)], axis=1
    )


def _nan_slope(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Least-squares slope of ``values`` against ``x``, ignoring NaN pairs.

    Each (tract, metric) profile is fit independently, so every sum is taken
    along the point axis with the finite mask applied per metric.
    """

    finite = np.isfinite(values)
    x4 = x.reshape(1, 1, x.shape[0], 1)
    weighted = np.where(finite, values, 0.0)
    n = finite.sum(axis=2).astype(np.float64)
    sum_x = np.sum(np.where(finite, x4, 0.0), axis=2)
    sum_y = weighted.sum(axis=2)
    sum_xx = np.sum(np.where(finite, x4 * x4, 0.0), axis=2)
    sum_xy = np.sum(weighted * np.where(finite, x4, 0.0), axis=2)
    denominator = n * sum_xx - sum_x * sum_x
    slope = np.divide(
        n * sum_xy - sum_x * sum_y,
        denominator,
        out=np.full_like(sum_y, np.nan, dtype=np.float64),
        where=denominator > 1e-12,
    )
    return np.where(n >= 2, slope, np.nan)


def _nan_area(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Trapezoidal area over consecutive finite nodes, in true x-extent.

    Missing nodes break the curve into segments; each segment contributes the
    area of the measured span rather than being stretched across the full
    range, so a partially measured tract is not rescaled upward.
    """

    left = values[..., :-1, :]
    right = values[..., 1:, :]
    both = np.isfinite(left) & np.isfinite(right)
    steps = np.where(both, 0.5 * (left + right), 0.0) * np.diff(x)[None, None, :, None]
    area = steps.sum(axis=2)
    # A profile with no measurable span is unknown rather than zero.
    return np.where(both.any(axis=2), area, np.nan)
