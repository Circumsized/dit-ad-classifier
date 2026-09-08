"""Covariate handling strategies for AFQ features.

Age is the dominant confounder in this dataset: AD subjects are substantially
older than controls, so a model can classify age instead of white matter.  The
competition does not state whether demographic variables may be used, so the
three strategies below are run side by side and their gap is itself a result.

``none``         drop covariates entirely (upper bound on disease signal).
``feature``      append age and sex as ordinary inputs (default baseline).
``residualize``  regress each feature on covariates within the fold and feed
                 the residuals to the classifier, separating white matter
                 structure from demographic variation.

Residualization is stateful, so it exposes an explicit fit/transform pair
instead of an sklearn estimator: ``transform`` must receive the covariates of
the rows being transformed, which sklearn's ``transform(X)`` signature cannot
express.  Both calls therefore happen inside a single fold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

COVARIATE_STRATEGIES = ("none", "feature", "residualize")


@dataclass
class Residualizer:
    """Fits a design matrix on training covariates and removes it from features."""

    design_: np.ndarray | None = None
    cov_mean_: np.ndarray | None = None
    cov_scale_: np.ndarray | None = None
    include_intercept_: bool = True

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

        self.cov_mean_ = np.nanmean(cov, axis=0)
        self.cov_mean_ = np.where(np.isfinite(self.cov_mean_), self.cov_mean_, 0.0)
        filled = np.where(np.isfinite(cov), cov, self.cov_mean_)
        self.cov_scale_ = np.std(filled, axis=0)
        self.cov_scale_ = np.where(self.cov_scale_ > 1e-10, self.cov_scale_, 1.0)
        cov_centered = (filled - self.cov_mean_) / self.cov_scale_

        clean = np.where(np.isfinite(values), values, 0.0)
        present = np.isfinite(values).all(axis=0)
        columns = np.flatnonzero(present)
        if columns.size == 0:
            raise ValueError("no finite feature columns to residualize")

        design = cov_centered
        if self.include_intercept_:
            design = np.column_stack((np.ones(cov.shape[0]), design))

        # Solves design @ w = clean[:, columns], so ``w`` maps one row of
        # covariates to the predicted value of every feature column.
        solution = np.linalg.lstsq(design, clean[:, columns], rcond=None)[0]
        self.design_ = np.zeros((design.shape[1], values.shape[1]), dtype=np.float64)
        self.design_[:, columns] = solution
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
        }


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
