"""Deterministic subject-level and site-level splitters."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def _validate_labels(y: np.ndarray) -> np.ndarray:
    values = np.asarray(y).reshape(-1)
    if values.size == 0:
        raise ValueError("y cannot be empty")
    return values


def _validate_site(site: np.ndarray) -> np.ndarray:
    """Shared site checks: integer-coded labels, -1 for missing, no NaN."""
    groups = np.asarray(site).reshape(-1)
    if groups.size == 0:
        raise ValueError("site cannot be empty")
    if groups.dtype.kind in "fc" and bool(np.isnan(groups).any()):
        raise ValueError("site contains NaN/missing values; sites must be integer-coded")
    if np.any(groups == -1):
        raise ValueError("site contains -1/missing values; cannot split by site")
    return groups


def stratified_kfold_indices(
    y: np.ndarray, n_splits: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield deterministic stratified folds without relying on sklearn."""

    labels = _validate_labels(y)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    # A fold missing a class cannot be stratified, and its accuracy would not
    # be comparable to the other folds.
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    smallest = int(np.min(counts[counts > 0])) if counts.any() else 0
    if n_splits > smallest:
        raise ValueError(
            f"n_splits={n_splits} exceeds the smallest class count ({smallest}); "
            "reduce n_splits so that every fold contains every class"
        )

    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            buckets[position % n_splits].append(int(index))
    all_indices = np.arange(labels.shape[0])
    for fold in range(n_splits):
        test = np.asarray(sorted(buckets[fold]), dtype=int)
        train_mask = np.ones(labels.shape[0], dtype=bool)
        train_mask[test] = False
        yield all_indices[train_mask], test


def leave_one_site_out(site: np.ndarray) -> Iterator[tuple[np.ndarray, np.ndarray, object]]:
    """Yield train/test indices for each site, sorted for stable reports."""

    groups = _validate_site(site)
    all_indices = np.arange(groups.shape[0])
    for held_out in sorted(np.unique(groups).tolist(), key=str):
        test = np.flatnonzero(groups == held_out)
        train = all_indices[groups != held_out]
        if train.size == 0 or test.size == 0:
            continue
        yield train, test, held_out


def site_stratified_kfold_indices(
    site: np.ndarray, n_splits: int = 5, seed: int = 42
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield deterministic folds that hold out 1/n of every site.

    Ported from the 2020 upstream ``data_division.py`` protocol: every fold's
    test slice draws a proportional chunk from each site, so site prevalence
    is balanced across folds (unlike LOSO, which removes a site entirely).
    Remainder subjects are dealt round-robin instead of the upstream's
    last-chunk padding. Expects integer-coded site labels (the MAT loader
    emits int64 with -1 for missing); other dtypes are outside the data
    contract.
    """

    groups = _validate_site(site)
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")

    values = np.unique(groups)
    if values.size == 1:
        raise ValueError(
            "site contains a single site value; site-stratified folds need at least two sites"
        )

    sizes = [int(np.sum(groups == value)) for value in values]
    smallest = min(sizes)
    if n_splits > smallest:
        raise ValueError(
            f"n_splits={n_splits} exceeds the smallest site count ({smallest}); "
            "reduce n_splits so every fold holds out part of every site"
        )

    rng = np.random.default_rng(seed)
    buckets: list[list[int]] = [[] for _ in range(n_splits)]
    for value in values:
        indices = np.flatnonzero(groups == value)
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            buckets[position % n_splits].append(int(index))
    all_indices = np.arange(groups.shape[0])
    for fold in range(n_splits):
        test = np.asarray(sorted(buckets[fold]), dtype=int)
        train_mask = np.ones(groups.shape[0], dtype=bool)
        train_mask[test] = False
        yield all_indices[train_mask], test


def split_indices(
    y: np.ndarray,
    site: np.ndarray | None = None,
    strategy: str = "stratified",
    n_splits: int = 5,
    seed: int = 42,
) -> Iterator[tuple[np.ndarray, np.ndarray, object]]:
    """Unified split interface used by benchmark runners."""

    name = strategy.lower().replace("-", "_")
    if name in {"loso", "leave_one_site_out", "group"}:
        if site is None:
            raise ValueError("site metadata is required for LOSO")
        yield from leave_one_site_out(site)
        return
    if name in {"site_stratified", "site_stratified_kfold"}:
        if site is None:
            raise ValueError("site metadata is required for site-stratified folds")
        for fold, (train, test) in enumerate(site_stratified_kfold_indices(site, n_splits, seed)):
            yield train, test, fold
        return
    if name not in {"stratified", "stratified_kfold", "competition"}:
        raise ValueError(f"unknown split strategy: {strategy}")
    for fold, (train, test) in enumerate(stratified_kfold_indices(y, n_splits, seed)):
        yield train, test, fold
