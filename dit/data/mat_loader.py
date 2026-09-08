"""Readers for AI4AD MATLAB files.

AI4AD files have appeared in multiple layouts: a structured array with one
record per tract, a MATLAB cell array, and pre-flattened numeric tensors.  The
loader accepts these layouts and converts them to one explicit representation:
``[subject, tract, point, metric]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .schema import LABEL_NAMES, DatasetBundle, canonicalize_labels

OFFICIAL_REPOSITORY = "https://github.com/YongLiuLab/AI4AD_AFQ"
KNOWN_METRICS = ("FA", "MD", "RD", "AD", "CL", "CURVATURE", "TORSION", "VOLUME")


def load_ai4ad_mat(
    path: str | Path,
    *,
    split: str = "auto",
    metrics: Iterable[str] | None = None,
) -> DatasetBundle:
    """Load an AI4AD training or private-test MAT file.

    Parameters
    ----------
    path:
        Path to ``MCAD_AFQ_competition.mat``, ``MCAD_AFQ_test.mat`` or a
        compatible export.
    split:
        ``auto``, ``train`` or ``test``.  ``auto`` prefers ``train_set`` when
        the file has one and otherwise falls back to ``test_set``; it does not
        switch based on whether labels are present, because silently reading the
        wrong half of a file is worse than reading the one that exists.
    metrics:
        Optional metric names to retain.  Names are case-insensitive.
    """

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"AI4AD data file not found: {source}. The data are not bundled "
            f"with this repository; see {OFFICIAL_REPOSITORY} for access details."
        )
    try:
        from scipy.io import loadmat
    except ImportError as exc:
        raise RuntimeError("scipy is required to read MATLAB files") from exc

    payload = loadmat(source, mat_dtype=True, struct_as_record=True, squeeze_me=False)
    mode = split.lower()
    if mode not in {"auto", "train", "test"}:
        raise ValueError("split must be auto, train or test")

    label_value = _first(payload, "train_diagnose", "label", "diagnose")
    population_value = _first(payload, "train_population", "population", "popu")
    site_value = _first(payload, "train_sites", "sites", "site", "center")

    if mode == "test":
        feature_value = _first(payload, "test_set", "MCAD_AFQ_test")
        label_value = _first(payload, "test_diagnose", "test_label")
        population_value = _first(payload, "test_population", "test_popu")
        site_value = _first(payload, "test_sites", "test_site", "test_center")
    elif mode == "train":
        feature_value = _first(payload, "train_set", "MCAD_AFQ_data")
    else:
        feature_value = _first(payload, "train_set", "MCAD_AFQ_data")
        if feature_value is None:
            feature_value = _first(payload, "test_set", "MCAD_AFQ_test")
            label_value = _first(payload, "test_diagnose", "test_label")
            population_value = _first(payload, "test_population", "test_popu")
            site_value = _first(payload, "test_sites", "test_site", "test_center")

    if feature_value is None:
        public_keys = sorted(key for key in payload if not key.startswith("__"))
        raise ValueError(
            "no supported AFQ feature key found; expected train_set, test_set "
            f"or MCAD_AFQ_data, found {public_keys}"
        )

    raw_labels = _parse_labels(label_value) if label_value is not None else None
    n_hint = None if raw_labels is None else raw_labels.shape[0]
    if n_hint is None:
        n_hint = _metadata_length(population_value, site_value)

    X, metric_names = _parse_features(feature_value, n_hint=n_hint, metrics=metrics)
    n_samples = X.shape[0]

    y = None
    raw_label_map: dict[int, int] = {}
    if raw_labels is not None:
        y, raw_label_map = canonicalize_labels(raw_labels)
        if y.shape[0] != n_samples:
            raise ValueError(f"diagnosis has {y.shape[0]} rows but features have {n_samples}")

    age, sex = _parse_population(population_value, n_samples)
    site = _parse_vector(site_value, n_samples, "site", default=-1, dtype=np.int64)
    tract_names = _parse_names(_first(payload, "fgnames", "tract_names"))
    if len(tract_names) != X.shape[1]:
        tract_names = tuple(f"tract_{index:02d}" for index in range(X.shape[1]))

    subject_value = _first(payload, "subject_id", "subject_ids", "subjects")
    subject_names = _parse_names(subject_value)
    if len(subject_names) == n_samples:
        subject_id = np.asarray(subject_names, dtype=object)
    else:
        subject_id = np.arange(n_samples, dtype=np.int64)

    class_labels: dict[int, str] = {}
    for raw, canonical in raw_label_map.items():
        if int(raw) in LABEL_NAMES:
            class_labels[int(canonical)] = LABEL_NAMES[int(raw)]

    return DatasetBundle(
        X=X,
        y=y,
        age=age,
        sex=sex,
        site=site,
        subject_id=subject_id,
        tract_names=tract_names,
        metric_names=metric_names,
        mask=np.isfinite(X),
        raw_label_map=raw_label_map,
        class_labels=class_labels,
    )


def _first(payload: dict[str, Any], *names: str) -> Any | None:
    lower = {key.casefold(): key for key in payload}
    for name in names:
        key = lower.get(name.casefold())
        if key is not None:
            return payload[key]
    return None


def _parse_labels(value: Any) -> np.ndarray:
    labels = np.asarray(value)
    labels = np.squeeze(labels)
    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = np.argmax(labels, axis=1) + 1
    labels = labels.reshape(-1)
    if not np.issubdtype(labels.dtype, np.number):
        raise ValueError("diagnosis labels must be numeric")
    # Reject before the cast: astype(int) would silently truncate 1.5 into a
    # different, valid label, and canonicalize_labels never sees the corruption.
    if not np.all(np.equal(labels, np.floor(labels))):
        raise ValueError("diagnosis labels must be integer-valued")
    return labels.astype(np.int64)


def _metadata_length(*values: Any | None) -> int | None:
    for value in values:
        if value is None:
            continue
        arr = np.squeeze(np.asarray(value))
        if arr.ndim == 1:
            return int(arr.shape[0])
        if arr.ndim >= 2:
            return int(max(arr.shape[0], arr.shape[1]))
    return None


def _parse_features(
    value: Any,
    *,
    n_hint: int | None,
    metrics: Iterable[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    raw = np.asarray(value)
    requested = None if metrics is None else {str(name).upper() for name in metrics}

    if raw.dtype.names:
        fields = list(raw.dtype.names)
        chosen = _ordered_metrics(fields, requested)
        if not chosen:
            raise ValueError(f"none of requested metrics found; available fields: {fields}")
        records = list(raw.reshape(-1))
        by_metric: list[np.ndarray] = []
        for metric in chosen:
            tracts = [_as_subject_point_matrix(record[metric], n_hint) for record in records]
            _check_profile_shapes(tracts, metric)
            by_metric.append(np.stack(tracts, axis=1))
        return np.stack(by_metric, axis=-1).astype(np.float32), tuple(chosen)

    if raw.dtype == object:
        unwrapped = _unwrap_singleton(raw)
        if isinstance(unwrapped, np.ndarray) and unwrapped.dtype.names:
            return _parse_features(unwrapped, n_hint=n_hint, metrics=metrics)
        grid = np.squeeze(raw)
        if grid.ndim == 2 and _looks_like_tract_metric_grid(grid.shape):
            if grid.shape[1] in {18, 20} and grid.shape[0] not in {18, 20}:
                grid = grid.T
            available = KNOWN_METRICS[: grid.shape[1]]
            if requested is None:
                chosen_indices = list(range(grid.shape[1]))
            else:
                lookup = {name: index for index, name in enumerate(available)}
                missing = requested - set(lookup)
                if missing:
                    raise ValueError(f"requested metrics not present: {sorted(missing)}")
                chosen_indices = [lookup[name] for name in KNOWN_METRICS if name in requested]
            by_metric = []
            for metric_index in chosen_indices:
                tracts = [
                    _as_subject_point_matrix(grid[tract_index, metric_index], n_hint)
                    for tract_index in range(grid.shape[0])
                ]
                _check_profile_shapes(tracts, available[metric_index])
                by_metric.append(np.stack(tracts, axis=1))
            names = tuple(available[index] for index in chosen_indices)
            return np.stack(by_metric, axis=-1).astype(np.float32), names

        cells = list(raw.reshape(-1))
        matrices = []
        for cell in cells:
            candidate = _unwrap_singleton(cell)
            if isinstance(candidate, np.void) and candidate.dtype.names:
                structured = np.asarray([candidate], dtype=candidate.dtype)
                return _parse_features(structured, n_hint=n_hint, metrics=metrics)
            matrices.append(_as_subject_point_matrix(candidate, n_hint))
        _check_profile_shapes(matrices, "cell")
        X = np.stack(matrices, axis=1)[..., np.newaxis]
        return X.astype(np.float32), ("FA",)

    return _parse_numeric_tensor(raw, n_hint=n_hint, requested=requested)


def _looks_like_tract_metric_grid(shape: tuple[int, int]) -> bool:
    return (
        shape[0] in {18, 20} and 1 < shape[1] <= len(KNOWN_METRICS)
    ) or (
        shape[1] in {18, 20} and 1 < shape[0] <= len(KNOWN_METRICS)
    )


def _ordered_metrics(fields: list[str], requested: set[str] | None) -> list[str]:
    lookup = {field.upper(): field for field in fields}
    if requested is not None:
        missing = requested - set(lookup)
        if missing:
            raise ValueError(f"requested metrics not present: {sorted(missing)}")
    wanted = set(lookup) if requested is None else requested
    ordered = [lookup[name] for name in KNOWN_METRICS if name in lookup and name in wanted]
    ordered.extend(field for field in fields if field.upper() in wanted and field not in ordered)
    return ordered


def _unwrap_singleton(value: Any) -> Any:
    current = value
    for _ in range(8):
        if isinstance(current, np.ndarray) and current.dtype == object and current.size == 1:
            current = current.reshape(-1)[0]
            continue
        break
    return current


def _as_subject_point_matrix(value: Any, n_hint: int | None) -> np.ndarray:
    arr = np.asarray(_unwrap_singleton(value))
    while arr.ndim > 2 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr[:, np.newaxis]
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"tract metric must be a numeric 2D matrix, got {arr.shape}/{arr.dtype}")
    if n_hint is not None:
        if arr.shape[0] == n_hint:
            return arr.astype(np.float32)
        if arr.shape[1] == n_hint:
            return arr.T.astype(np.float32)
        raise ValueError(f"tract matrix {arr.shape} has no subject axis of length {n_hint}")
    if arr.shape[0] < arr.shape[1] and arr.shape[1] > 100:
        arr = arr.T
    return arr.astype(np.float32)


def _check_profile_shapes(values: list[np.ndarray], metric: str) -> None:
    if not values:
        raise ValueError("AFQ feature collection is empty")
    expected = values[0].shape
    mismatched = [value.shape for value in values if value.shape != expected]
    if mismatched:
        raise ValueError(f"inconsistent profile shapes for {metric}: expected {expected}, got {mismatched[:3]}")


def _parse_numeric_tensor(
    value: np.ndarray,
    *,
    n_hint: int | None,
    requested: set[str] | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    arr = np.squeeze(np.asarray(value))
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError("AFQ numeric tensor must contain numeric values")
    if arr.ndim not in {3, 4}:
        raise ValueError(f"numeric AFQ tensor must have 3 or 4 dimensions, got {arr.shape}")

    # The metric axis is the smallest one: AFQ carries 1-8 metrics per tract
    # while a profile holds 20-100 nodes.  It is excluded before the subject
    # axis is located, so a mis-sized label vector cannot claim it.
    metric_axis = int(np.argmin(arr.shape))
    sample_axis = _choose_axis(
        arr.shape,
        n_hint,
        prefer=None,
        exclude={metric_axis},
        require_hint=n_hint is not None,
    )
    arr = np.moveaxis(arr, sample_axis, 0)
    remaining = arr.shape[1:]
    point_axis_local = 1 + _choose_axis(remaining, 100, prefer=int(np.argmax(remaining)))
    arr = np.moveaxis(arr, point_axis_local, 2)

    if arr.ndim == 3:
        X = arr[..., np.newaxis]
        names = ("FA",)
    else:
        # After moving sample and point, the smaller remaining dimension is the
        # metric axis unless the requested count identifies it exactly.
        candidates = [1, 3]
        metric_axis = None
        if requested is not None:
            metric_axis = next(
                (axis for axis in candidates if arr.shape[axis] == len(requested)), None
            )
        if metric_axis is None:
            metric_axis = min(candidates, key=lambda axis: arr.shape[axis])
        if metric_axis != 3:
            arr = np.moveaxis(arr, metric_axis, 3)
        names = list(KNOWN_METRICS[: arr.shape[3]])
        if requested is not None:
            missing = requested - set(names)
            if missing:
                raise ValueError(f"requested metrics not present: {sorted(missing)}")
            keep = [index for index, name in enumerate(names) if name in requested]
            if not keep:
                raise ValueError(f"none of requested metrics found; available fields: {names}")
            arr = np.take(arr, keep, axis=3)
            names = [names[index] for index in keep]
        X = arr
        names = tuple(names)
    return X.astype(np.float32), names


def _choose_axis(
    shape: tuple[int, ...],
    target: int | None,
    prefer: int | None,
    *,
    exclude: frozenset[int] = frozenset(),
    require_hint: bool = False,
) -> int:
    """Locate one axis by size, refusing to guess.

    ``target`` is normally the expected subject count taken from the label
    vector.  A match is only trusted when it is the sole match outside
    ``exclude``; when ``require_hint`` is set and nothing matches, the loader
    raises instead of picking an axis.  Without that guard a mis-sized label
    vector whose length happened to equal the metric count silently redefined
    which axis held subjects, and the downstream row-count check passed on the
    reinterpretation instead of failing.
    """

    if target is not None:
        matches = [
            index for index, size in enumerate(shape) if size == target and index not in exclude
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            return matches[0]
        if require_hint:
            raise ValueError(
                f"expected {target} subjects but no axis of the tensor {shape} "
                "matches once the metric axis is excluded; the label vector "
                "probably has the wrong row count"
            )
    if prefer is not None and prefer not in exclude:
        return prefer
    candidates = [index for index, _ in enumerate(shape) if index not in exclude]
    return max(candidates, key=lambda index: shape[index])


def _parse_population(value: Any | None, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    if value is None:
        missing = np.full(n_samples, np.nan, dtype=np.float32)
        return missing.copy(), missing.copy()
    arr = np.squeeze(np.asarray(value))
    if arr.ndim == 1:
        if arr.size != n_samples:
            raise ValueError("population vector length does not match features")
        return arr.astype(np.float32), np.full(n_samples, np.nan, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"population must be a 2D matrix, got {arr.shape}")
    if arr.shape[0] != n_samples and arr.shape[1] == n_samples:
        arr = arr.T
    if arr.shape[0] != n_samples or arr.shape[1] < 2:
        raise ValueError(f"population must have shape [N,>=2], got {arr.shape}")
    # Competition MAT convention documented by AI4AD: sex first, then age.
    return arr[:, 1].astype(np.float32), arr[:, 0].astype(np.float32)


def _parse_vector(value: Any | None, n_samples: int, name: str, default, dtype) -> np.ndarray:
    if value is None:
        return np.full(n_samples, default, dtype=dtype)
    arr = np.squeeze(np.asarray(value)).reshape(-1)
    if arr.shape[0] != n_samples:
        raise ValueError(f"{name} has {arr.shape[0]} rows but features have {n_samples}")
    return arr.astype(dtype)


def _parse_names(value: Any | None) -> tuple[str, ...]:
    if value is None:
        return tuple()
    names: list[str] = []
    for item in np.asarray(value, dtype=object).reshape(-1):
        candidate = _unwrap_singleton(item)
        if isinstance(candidate, bytes):
            names.append(candidate.decode("utf-8", errors="replace"))
        elif isinstance(candidate, str):
            names.append(candidate)
        elif isinstance(candidate, np.ndarray):
            flat = candidate.reshape(-1)
            if flat.size and all(isinstance(x, str) for x in flat):
                names.append("".join(flat.tolist()))
            elif flat.size == 1:
                names.append(str(flat[0]))
        else:
            names.append(str(candidate))
    return tuple(name.strip() for name in names if name.strip())
