"""Leakage-safe preprocessing for AFQ profiles."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def smooth_profiles(X: np.ndarray, window: int = 1) -> np.ndarray:
    """Average along the point axis while preserving shape.

    NaNs are ignored.  A window of one is a no-op.  The operation is applied
    before flattening and must therefore be fit/applied inside each split.
    """

    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 4:
        raise ValueError(f"expected [N,T,P,M], got {values.shape}")
    if window < 1:
        raise ValueError("smoothing window must be >= 1")
    if window == 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("smoothing window must be odd")
    if window > values.shape[2]:
        raise ValueError(
            f"smoothing window {window} exceeds the number of nodes {values.shape[2]}"
        )
    radius = window // 2
    padded = np.pad(values, ((0, 0), (0, 0), (radius, radius), (0, 0)), mode="edge")
    finite = np.isfinite(padded)
    safe = np.where(finite, padded, 0.0)
    kernel = np.ones(window, dtype=np.float32)
    sums = np.zeros_like(values, dtype=np.float32)
    counts = np.zeros_like(values, dtype=np.float32)
    for offset in range(window):
        sums += safe[:, :, offset : offset + values.shape[2], :]
        counts += finite[:, :, offset : offset + values.shape[2], :]
    return np.divide(sums, counts, out=np.full_like(values, np.nan), where=counts > 0)


@dataclass
class TractFeaturePreprocessor:
    """Fit imputation and standardization using training data only."""

    smooth_window: int = 1
    include_covariates: bool = True
    robust_scale: bool = False

    def __post_init__(self) -> None:
        self.fitted = False

    def fit(
        self,
        X: np.ndarray,
        age: np.ndarray | None = None,
        sex: np.ndarray | None = None,
    ) -> "TractFeaturePreprocessor":
        values = smooth_profiles(X, self.smooth_window)
        flat = values.reshape(values.shape[0], -1).astype(np.float64)
        self.feature_mean_ = np.nanmean(flat, axis=0)
        self.feature_mean_ = np.where(np.isfinite(self.feature_mean_), self.feature_mean_, 0.0)
        filled = np.where(np.isfinite(flat), flat, self.feature_mean_)
        if self.robust_scale:
            self.center_ = np.median(filled, axis=0)
            q75, q25 = np.percentile(filled, [75, 25], axis=0)
            self.scale_ = q75 - q25
        else:
            self.center_ = np.mean(filled, axis=0)
            self.scale_ = np.std(filled, axis=0)
        self.scale_ = np.where(self.scale_ > 1e-8, self.scale_, 1.0)
        self.cov_center_ = np.zeros(2, dtype=np.float64)
        self.cov_scale_ = np.ones(2, dtype=np.float64)
        if self.include_covariates:
            cov = _covariates(age, sex, flat.shape[0])
            self.cov_center_ = np.nanmean(cov, axis=0)
            self.cov_center_ = np.where(np.isfinite(self.cov_center_), self.cov_center_, 0.0)
            cov_filled = np.where(np.isfinite(cov), cov, self.cov_center_)
            self.cov_scale_ = np.std(cov_filled, axis=0)
            self.cov_scale_ = np.where(self.cov_scale_ > 1e-8, self.cov_scale_, 1.0)
        self.n_profile_features_ = flat.shape[1]
        self.fitted = True
        return self

    def transform(
        self,
        X: np.ndarray,
        age: np.ndarray | None = None,
        sex: np.ndarray | None = None,
    ) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("preprocessor must be fit before transform")
        values = smooth_profiles(X, self.smooth_window)
        flat = values.reshape(values.shape[0], -1).astype(np.float64)
        filled = np.where(np.isfinite(flat), flat, self.feature_mean_)
        output = (filled - self.center_) / self.scale_
        if self.include_covariates:
            cov = _covariates(age, sex, flat.shape[0])
            cov = np.where(np.isfinite(cov), cov, self.cov_center_)
            output = np.hstack((output, (cov - self.cov_center_) / self.cov_scale_))
        return output.astype(np.float32)

    def fit_transform(self, X: np.ndarray, age=None, sex=None) -> np.ndarray:
        return self.fit(X, age=age, sex=sex).transform(X, age=age, sex=sex)


def _covariates(age, sex, n_samples: int) -> np.ndarray:
    age_array = np.full(n_samples, np.nan, dtype=np.float64) if age is None else np.asarray(age, dtype=np.float64).reshape(-1)
    sex_array = np.full(n_samples, np.nan, dtype=np.float64) if sex is None else np.asarray(sex, dtype=np.float64).reshape(-1)
    if age_array.shape[0] != n_samples or sex_array.shape[0] != n_samples:
        raise ValueError("age and sex must have one value per sample")
    return np.column_stack((age_array, sex_array))
