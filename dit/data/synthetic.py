"""Deterministic synthetic AFQ data for tests and smoke runs."""

from __future__ import annotations

import numpy as np

from .schema import DatasetBundle


def make_synthetic_bundle(
    n_samples: int = 84,
    n_tracts: int = 18,
    n_points: int = 100,
    n_metrics: int = 4,
    n_sites: int = 7,
    n_classes: int = 3,
    seed: int = 42,
) -> DatasetBundle:
    """Generate a small dataset with disease and site effects.

    It is not a scientific simulator.  It exists to exercise parsing,
    splitting, training and inference without redistributing AI4AD data.

    Two couplings are deliberate and matter when interpreting results: age is
    constructed as ``62 + 7 * disease + noise``, so age is a *causal proxy* for
    the label here rather than the confounder it is in the real cohort.  On
    this data the ``residualize`` covariate strategy therefore destroys signal
    and lands near chance, whereas on AI4AD it isolates the white matter
    biomarker.  Reading the strategy gap requires knowing which dataset is in
    play.  The site offset is also correlated with class position before the
    shuffle, which is why plain accuracy saturates quickly.
    """

    if n_samples < n_sites * n_classes:
        raise ValueError("n_samples must cover every site/class combination")
    rng = np.random.default_rng(seed)
    site = np.arange(n_samples) % n_sites
    y = (np.arange(n_samples) // n_sites) % n_classes
    rng.shuffle(y)
    age = 62 + y * 7 + rng.normal(0, 4, n_samples)
    sex = rng.integers(0, 2, n_samples)
    X = rng.normal(0, 0.5, (n_samples, n_tracts, n_points, n_metrics)).astype(np.float32)
    positions = np.linspace(-1, 1, n_points, dtype=np.float32)
    for index in range(n_samples):
        X[index] += site[index] * 0.06
        X[index, : min(4, n_tracts), :, 0] += y[index] * (0.18 + 0.08 * positions)
        if n_metrics > 1:
            X[index, -min(3, n_tracts) :, :, 1] += y[index] * 0.12
    missing = rng.random(X.shape) < 0.01
    X[missing] = np.nan
    metrics = ("FA", "MD", "RD", "AD", "CL", "VOLUME")[:n_metrics]
    class_labels = {0: "NC", 1: "MCI", 2: "AD"} if n_classes == 3 else {0: "NC", 1: "AD"}
    return DatasetBundle(
        X=X,
        y=y.astype(np.int64),
        age=age.astype(np.float32),
        sex=sex.astype(np.float32),
        site=site.astype(np.int64),
        subject_id=np.asarray([f"synthetic-{i:04d}" for i in range(n_samples)]),
        tract_names=tuple(f"tract_{i:02d}" for i in range(n_tracts)),
        metric_names=tuple(metrics),
        mask=np.isfinite(X),
        raw_label_map={label: label for label in range(n_classes)},
        class_labels=class_labels,
    )
