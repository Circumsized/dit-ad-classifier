"""Leakage-safe preprocessing for AFQ profiles."""

from __future__ import annotations

import numpy as np


def smooth_profiles(X: np.ndarray, window: int = 1) -> np.ndarray:
    """Average along the point axis while preserving shape.

    NaNs are ignored.  A window of one is a no-op.  The operation is strictly
    row-local -- each subject's profile is averaged along its own point axis
    with edge padding, and no statistic crosses subjects -- so applying it
    before the split cannot leak.  It must not be confused with the fold-fitted
    transforms (imputation, scaling), which is why it is safe to run once on
    the whole view.
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
    sums = np.zeros_like(values, dtype=np.float32)
    counts = np.zeros_like(values, dtype=np.float32)
    for offset in range(window):
        sums += safe[:, :, offset : offset + values.shape[2], :]
        counts += finite[:, :, offset : offset + values.shape[2], :]
    return np.divide(sums, counts, out=np.full_like(values, np.nan), where=counts > 0)
