"""Covariate handling strategies for AFQ features.

Age is the dominant confounder in this dataset: AD subjects are substantially
older than controls, so a model can classify age instead of white matter. The
competition does not state whether demographic variables may be used, so the
three strategies below are run side by side and their gap is reported.

``none``         drop covariates entirely (upper bound on disease signal).
``feature``      append age and sex as ordinary inputs (default baseline).
``residualize``  regress each feature on covariates within the fold and feed
                 the residuals to the classifier, separating white matter
                 structure from demographic variation.

Residualization is stateful and exposes an explicit fit/transform pair rather
than an sklearn estimator: ``transform`` needs the covariates of the rows being
transformed, which sklearn's ``transform(X)`` signature cannot express. Both
calls happen inside a single fold. :class:`ResidualizeCovariates` wraps that
pair for use *inside* an sklearn pipeline: the covariates ride along as the
trailing feature columns, so GridSearchCV refits the regression on every inner
training fold and inner-validation rows never touch the coefficients.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

COVARIATE_STRATEGIES = ("none", "feature", "residualize")


@dataclass
class Residualizer:
    """Fits a design matrix on training covariates and removes it from features.

    Missing feature values follow one predefined scheme: training-row medians
    are imputed *for the regression only*, so a column with scattered gaps is
    still de-confounded instead of silently keeping its raw values. Columns
    that are entirely NaN impute to zero, which yields zero coefficients —
    nothing measurable to residualize — and stay NaN on output.
    """

    design_: np.ndarray | None = None
    cov_mean_: np.ndarray | None = None
    cov_scale_: np.ndarray | None = None
    include_intercept_: bool = True
    feature_median_: np.ndarray | None = None
    n_train_rows_: int | None = None
    n_imputed_columns_: int | None = None

    def fit(self, X: np.ndarray, covariates: np.ndarray) -> "Residualizer":
        """Fit per-feature regression coefficients on training data only."""

        values = np.asarray(X, dtype=np.float64)
        cov = np.asarray(covariates, dtype=np.float64)
        if values.ndim != 2 or cov.ndim != 2:
            raise ValueError("X must be [N,F] and covariates [N,K]")
        if values.shape[0] != cov.shape[0]:
            raise ValueError("X and covariates must have the same row count")
        if cov.shape[0] < cov.shape[1] + 1:
            raise ValueError(
                "not enough rows to fit a covariate model; use a different strategy"
            )
        if not np.isfinite(values).any():
            raise ValueError("no finite feature values to residualize")

        self.cov_mean_ = np.nanmean(cov, axis=0)
        self.cov_mean_ = np.where(np.isfinite(self.cov_mean_), self.cov_mean_, 0.0)
        filled = np.where(np.isfinite(cov), cov, self.cov_mean_)
        self.cov_scale_ = np.std(filled, axis=0)
        self.cov_scale_ = np.where(self.cov_scale_ > 1e-10, self.cov_scale_, 1.0)
        cov_centered = (filled - self.cov_mean_) / self.cov_scale_

        finite = np.isfinite(values)
        with warnings.catch_warnings():
            # An all-NaN column has no median; the guarded fallback below
            # defines it as 0 rather than failing the whole fit.
            warnings.simplefilter("ignore", category=RuntimeWarning)
            medians = np.nanmedian(values, axis=0)
        self.feature_median_ = np.where(np.isfinite(medians), medians, 0.0)
        self.n_train_rows_ = int(values.shape[0])
        self.n_imputed_columns_ = int((~finite.all(axis=0)).sum())
        imputed = np.where(finite, values, self.feature_median_)

        design = cov_centered
        if self.include_intercept_:
            design = np.column_stack((np.ones(cov.shape[0]), design))

        # Solves design @ w = imputed, so ``w`` maps one row of covariates to
        # the predicted value of every feature column. lstsq returns the
        # minimum-norm solution for a rank-deficient design, which keeps the
        # fit defined instead of crashing on constant covariates.
        solution = np.linalg.lstsq(design, imputed, rcond=None)[0]
        self.design_ = solution
        return self

    def transform(self, X: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        """Remove the fitted covariate component from ``X``."""

        if self.design_ is None:
            raise RuntimeError("residualizer must be fit before transform")
        values = np.asarray(X, dtype=np.float64)
        cov = np.asarray(covariates, dtype=np.float64)
        if values.ndim != 2 or cov.ndim != 2:
            raise ValueError("X must be [N,F] and covariates [N,K]")
        if values.shape[1] != self.design_.shape[1]:
            raise ValueError("X has a different number of columns than at fit time")
        if cov.shape[1] != (self.design_.shape[0] - int(self.include_intercept_)):
            raise ValueError("covariates have a different width than at fit time")

        cov_filled = np.where(np.isfinite(cov), cov, self.cov_mean_)
        cov_centered = (cov_filled - self.cov_mean_) / self.cov_scale_
        design = cov_centered
        if self.include_intercept_:
            design = np.column_stack((np.ones(cov.shape[0]), design))

        clean = np.where(np.isfinite(values), values, 0.0)
        out = clean - design @ self.design_
        return np.where(np.isfinite(values), out, np.nan).astype(np.float64)

    def fit_transform(self, X: np.ndarray, covariates: np.ndarray) -> np.ndarray:
        return self.fit(X, covariates).transform(X, covariates)

    def parameters(self) -> dict[str, object]:
        return {
            "include_intercept": self.include_intercept_,
            "cov_mean": None if self.cov_mean_ is None else self.cov_mean_.tolist(),
            "cov_scale": None if self.cov_scale_ is None else self.cov_scale_.tolist(),
            "n_fitted_features": None if self.design_ is None else int(self.design_.shape[1]),
            "n_train_rows": self.n_train_rows_,
            "n_columns_with_imputation": self.n_imputed_columns_,
        }


class ResidualizeCovariates:
    """Pipeline step that residualizes features against trailing covariate columns.

    The last ``n_covariates`` columns of ``X`` are treated as demographics:
    :class:`Residualizer` is fitted on the training rows, the residuals of the
    remaining columns are returned, and the covariate columns themselves are
    dropped, so downstream steps never see age or sex. Living inside the
    pipeline is what keeps the regression honest under nested search —
    GridSearchCV refits this step on every inner training fold, so
    inner-validation rows cannot influence the coefficients, and the refit on
    the full outer training fold happens through the same code path.
    """

    def __init__(self, n_covariates: int = 2) -> None:
        self.n_covariates = n_covariates

    def get_params(self, deep: bool = True) -> dict[str, object]:
        return {"n_covariates": self.n_covariates}

    def set_params(self, **params: object) -> "ResidualizeCovariates":
        for name, value in params.items():
            if not hasattr(self, name):
                raise ValueError(f"unknown parameter {name!r}")
            setattr(self, name, value)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ResidualizeCovariates":
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] <= self.n_covariates:
            raise ValueError(
                "residualization needs more feature columns than covariates"
            )
        self.residualizer_ = Residualizer().fit(
            values[:, : -self.n_covariates], values[:, -self.n_covariates :]
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "residualizer_", None) is None:
            raise RuntimeError("residualizer step must be fit before transform")
        values = np.asarray(X, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] <= self.n_covariates:
            raise ValueError(
                "residualization needs more feature columns than covariates"
            )
        residuals = self.residualizer_.transform(
            values[:, : -self.n_covariates], values[:, -self.n_covariates :]
        )
        return residuals

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


def covariate_matrix(age: np.ndarray | None, sex: np.ndarray | None, n: int) -> np.ndarray:
    """Return [age, sex] with missing entries marked NaN, shape [N, 2]."""

    age_arr = np.full(n, np.nan, dtype=np.float64) if age is None else np.asarray(age, dtype=np.float64).reshape(-1)
    sex_arr = np.full(n, np.nan, dtype=np.float64) if sex is None else np.asarray(sex, dtype=np.float64).reshape(-1)
    if age_arr.shape[0] != n or sex_arr.shape[0] != n:
        raise ValueError("age and sex must have one value per sample")
    return np.column_stack((age_arr, sex_arr))


def strip_covariate_columns(
    features: np.ndarray, n_covariates: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Split trailing covariate columns off a feature matrix."""

    if features.shape[1] <= n_covariates:
        raise ValueError("feature matrix has no non-covariate columns")
    return features[:, :-n_covariates], features[:, -n_covariates:]
