"""Leakage-safe classical baselines for AFQ profiles."""

from __future__ import annotations

from typing import Any

import numpy as np

from dit.data.layout import FeatureLayout, view_for_layout
from dit.data.preprocessing import smooth_profiles
from dit.data.schema import DatasetBundle
from dit.data.sklearn_compat import l1_ratio_kwargs


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
    columns.  Contiguity is what makes block selection and the tract x node
    heatmaps addressable; see :func:`describe_feature_layout`.

    ``summary`` emits mean, standard deviation, linear slope and trapezoidal
    area per tract/metric.  A metric name such as ``FA`` selects one channel
    while retaining all tract nodes.

    ``include_missing_pattern`` appends the per-tract fraction of nodes AFQ
    failed to identify.  Tract identification failure is informative, so it is
    appended after the profile columns to avoid disturbing the block accounting.
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
    mismatch raises instead of silently mislabeling features.
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
):
    """Create a tuned sklearn pipeline with split-local preprocessing.

    When ``selector`` is given it is inserted as the first pipeline step so it
    is fit inside every inner cross-validation fold, not once on the whole outer
    training fold.  That keeps hyperparameter selection honest: inner-validation
    labels never influence the features the search scores on.
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
    if name in {"linear_svm", "svm", "official_svm"}:
        estimator, prefix = _probability_svc("linear", seed)
        parameters = {f"model__{prefix}C": np.logspace(-4, 0, 9)}
    elif name in {"rbf_svm", "rbf"}:
        estimator, prefix = _probability_svc("rbf", seed)
        parameters = {
            f"model__{prefix}C": np.logspace(-2, 2, 5),
            f"model__{prefix}gamma": ["scale", 1e-3, 1e-2],
        }
    elif name in {"logistic", "elastic_net", "lr"}:
        # ``_l1_ratio_kwargs`` already carries the solver it needs.
        estimator = LogisticRegression(
            **_l1_ratio_kwargs(0.5, "saga"),
            class_weight="balanced",
            # SAGA with an elastic net stalls well short of 5 000 iterations on
            # the 7 200-column profile view, so it returned coefficients from an
            # unconverged solution and reported them as if they were fitted.
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

    steps = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
        ("model", estimator),
    ]
    if selector is not None:
        steps.insert(0, ("selector", selector))
    pipeline = Pipeline(steps)
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


def align_probabilities(probabilities: np.ndarray, model_classes, n_classes: int) -> np.ndarray:
    """Place estimator probabilities into canonical class columns."""

    output = np.zeros((probabilities.shape[0], n_classes), dtype=float)
    for source, label in enumerate(np.asarray(model_classes, dtype=int)):
        output[:, int(label)] = probabilities[:, source]
    return output


def _l1_ratio_kwargs(l1_ratio: float, solver: str) -> dict:
    """LogisticRegression regularisation kwargs spanning sklearn 1.3 to 1.10.

    See :func:`dit.data.sklearn_compat.l1_ratio_kwargs`; kept as a local alias
    so the call sites in this module read the same way.
    """

    return l1_ratio_kwargs(l1_ratio, solver)


def _probability_svc(
    kernel: str, seed: int, calibration_folds: int = 3
) -> tuple[object, str]:
    """Return an SVC exposing ``predict_proba``, plus its grid parameter prefix.

    ``SVC(probability=True)`` was deprecated in sklearn 1.9 in favour of an
    explicit calibration wrapper, which is also the more honest construction:
    the Platt scalars are fitted on calibration folds instead of being solved
    jointly with the margin.  ``ensemble=False`` keeps a single model, so the
    reported decisions stay those of the plain SVM.

    The wrapper nests the SVM, so grid keys must address it as
    ``model__estimator__C`` rather than ``model__C``.  The prefix is returned
    with the estimator so the grid is built for whichever form was chosen.
    """

    from sklearn import __version__ as sklearn_version
    from sklearn.svm import SVC

    parts = tuple(int(part) for part in sklearn_version.split(".")[:2])
    if parts < (1, 9):
        return (
            SVC(
                kernel=kernel,
                probability=True,
                class_weight="balanced",
                random_state=seed,
            ),
            "",
        )
    from sklearn.calibration import CalibratedClassifierCV

    return (
        CalibratedClassifierCV(
            SVC(kernel=kernel, class_weight="balanced", random_state=seed),
            cv=max(calibration_folds, 2),
            ensemble=False,
            method="sigmoid",
        ),
        "estimator__",
    )


def _profile_summary(values: np.ndarray) -> np.ndarray:
    """Per-tract/metric mean, standard deviation, linear slope and area.

    ``np.trapz`` is not NaN-aware: a single missing node makes the whole area
    NaN, which turns a 1% missing rate into a 16% missing feature rate and
    quietly removes a quarter of the columns from every model.  Slope and area
    are therefore computed over the measured nodes only.
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
    range, so a partially measured tract is not silently rescaled upward.
    """

    left = values[..., :-1, :]
    right = values[..., 1:, :]
    both = np.isfinite(left) & np.isfinite(right)
    steps = np.where(both, 0.5 * (left + right), 0.0) * np.diff(x)[None, None, :, None]
    area = steps.sum(axis=2)
    # A profile with no measurable span is unknown, not zero.
    return np.where(both.any(axis=2), area, np.nan)
