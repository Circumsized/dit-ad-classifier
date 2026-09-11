"""Explicit data contracts for AI4AD/AFQ features.

Shapes and label encodings are declared here and validated at the project
boundary, rather than inferred from filenames and array positions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

# Official AI4AD encoding: 1=NC, 2=MCI, 3=AD.
LABEL_NAMES: Mapping[int, str] = {1: "NC", 2: "MCI", 3: "AD"}
CANONICAL_NAMES: Mapping[int, str] = {0: "NC", 1: "MCI", 2: "AD"}


def canonicalize_labels(labels: Iterable[int]) -> tuple[np.ndarray, dict[int, int]]:
    """Convert official 1/2/3 labels to contiguous 0/1/2 labels.

    Already-canonical labels are accepted only when they are a subset of
    ``{0, 1, 2}`` and no official label is present. The returned mapping lets
    prediction reports restore the original coding.
    """

    values = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels)
    values = values.reshape(-1)
    if values.size == 0:
        raise ValueError("labels cannot be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("labels contain NaN or infinite values")
    # A label of 1.5 is corrupt data: rounding it would produce a different
    # but valid label.
    if not np.all(np.equal(values, np.floor(values))):
        raise ValueError("labels must be integer-valued")
    values = values.astype(int)
    unique = sorted(set(values.tolist()))
    if not (set(unique) <= {1, 2, 3} or set(unique) <= {0, 1, 2}):
        raise ValueError(
            f"unsupported labels {unique}; expected official 1/2/3 "
            "or canonical 0/1/2"
        )
    # Contiguous renumbering, so callers do not have to handle gaps. The
    # returned mapping restores the original coding, and
    # :class:`DatasetBundle` carries class names separately because a
    # two-class subset renumbers the disease class.
    mapping = {raw: index for index, raw in enumerate(unique)}
    return np.asarray([mapping[int(value)] for value in values], dtype=np.int64), mapping


@dataclass(frozen=True)
class DatasetBundle:
    """Validated AFQ dataset.

    Parameters
    ----------
    X:
        Float array with shape ``[subjects, tracts, points, metrics]``.
    y:
        Canonical integer labels ``0=NC, 1=MCI, 2=AD``. May be ``None`` for
        the private competition test set.
    age, sex, site:
        Subject-level metadata. Missing metadata is NaN for numeric values and
        ``-1`` for sites.
    mask:
        Boolean validity mask with the same shape as ``X``, so models can
        distinguish missing measurements from true zero values.
    raw_label_map:
        Mapping from labels found in the source file to canonical labels.
    class_labels:
        Explicit name per canonical label present in this view. When omitted,
        names come from :data:`CANONICAL_NAMES`, which is correct only for the
        three-class view: a binary view renumbers the labels, so ``class 1``
        means AD rather than MCI and must be named explicitly.
    """

    X: np.ndarray
    y: np.ndarray | None = None
    age: np.ndarray | None = None
    sex: np.ndarray | None = None
    site: np.ndarray | None = None
    subject_id: np.ndarray | None = None
    tract_names: tuple[str, ...] = field(default_factory=tuple)
    metric_names: tuple[str, ...] = field(default_factory=tuple)
    mask: np.ndarray | None = None
    raw_label_map: Mapping[int, int] = field(default_factory=dict)
    class_labels: Mapping[int, str] | None = None

    def __post_init__(self) -> None:
        x = np.asarray(self.X)
        if x.ndim != 4:
            raise ValueError(f"X must have shape [N, T, P, M], got {x.shape}")
        if x.shape[0] == 0 or x.shape[1] == 0 or x.shape[2] == 0 or x.shape[3] == 0:
            raise ValueError(f"X dimensions must be non-zero, got {x.shape}")
        if not np.issubdtype(x.dtype, np.number):
            raise TypeError("X must be numeric")
        object.__setattr__(self, "X", x.astype(np.float32, copy=False))

        n = x.shape[0]
        if self.y is not None:
            y = np.asarray(self.y).reshape(-1)
            if y.shape[0] != n:
                raise ValueError(f"y has {y.shape[0]} rows but X has {n}")
            if not np.issubdtype(y.dtype, np.integer):
                if not np.all(np.isfinite(y)):
                    raise ValueError("y contains NaN or infinite values")
                if not np.all(np.equal(y, np.floor(y))):
                    raise ValueError("y must be integer-valued")
                y = y.astype(np.int64)
            object.__setattr__(self, "y", y.astype(np.int64, copy=False))
            if set(np.unique(y).tolist()) - {0, 1, 2}:
                raise ValueError("y must use canonical labels 0=NC, 1=MCI, 2=AD")

        for name in ("age", "sex", "site", "subject_id"):
            value = getattr(self, name)
            if value is None:
                continue
            arr = np.asarray(value).reshape(-1)
            if arr.shape[0] != n:
                raise ValueError(f"{name} has {arr.shape[0]} rows but X has {n}")
            object.__setattr__(self, name, arr)

        if self.mask is None:
            object.__setattr__(self, "mask", np.isfinite(x))
        else:
            mask = np.asarray(self.mask, dtype=bool)
            if mask.shape != x.shape:
                raise ValueError(f"mask shape {mask.shape} does not match X {x.shape}")
            object.__setattr__(self, "mask", mask)

        if self.tract_names and len(self.tract_names) != x.shape[1]:
            raise ValueError("tract_names length does not match X.shape[1]")
        if self.metric_names and len(self.metric_names) != x.shape[3]:
            raise ValueError("metric_names length does not match X.shape[3]")
        if not self.tract_names:
            object.__setattr__(self, "tract_names", tuple(f"tract_{i:02d}" for i in range(x.shape[1])))
        if not self.metric_names:
            object.__setattr__(self, "metric_names", tuple(f"metric_{i:02d}" for i in range(x.shape[3])))

    @property
    def n_samples(self) -> int:
        return int(self.X.shape[0])

    @property
    def n_tracts(self) -> int:
        return int(self.X.shape[1])

    @property
    def n_points(self) -> int:
        return int(self.X.shape[2])

    @property
    def n_metrics(self) -> int:
        return int(self.X.shape[3])

    @property
    def class_names(self) -> tuple[str, ...]:
        if self.y is None:
            return tuple()
        names: list[str] = []
        for label in sorted(np.unique(self.y).tolist()):
            label = int(label)
            if self.class_labels and label in self.class_labels:
                names.append(str(self.class_labels[label]))
            else:
                names.append(CANONICAL_NAMES.get(label, str(label)))
        return tuple(names)

    @property
    def label_map(self) -> dict[int, str]:
        """Canonical label to display name, for every present class."""

        result: dict[int, str] = {}
        if self.y is None:
            return result
        for label in sorted(np.unique(self.y).tolist()):
            label = int(label)
            if self.class_labels and label in self.class_labels:
                result[label] = str(self.class_labels[label])
            else:
                result[label] = CANONICAL_NAMES.get(label, str(label))
        return result

    def select(self, indices: Sequence[int]) -> "DatasetBundle":
        """Return a subject subset without changing feature semantics."""

        idx = np.asarray(indices, dtype=int)
        kwargs = {}
        for name in ("y", "age", "sex", "site", "subject_id", "mask"):
            value = getattr(self, name)
            kwargs[name] = None if value is None else value[idx] if name != "mask" else value[idx]
        return DatasetBundle(
            X=self.X[idx],
            tract_names=self.tract_names,
            metric_names=self.metric_names,
            raw_label_map=self.raw_label_map,
            class_labels=self.class_labels,
            **kwargs,
        )

    def task_view(self, task: str = "binary") -> "DatasetBundle":
        """Create an explicit binary or three-class view.

        Classes are resolved by name rather than by index. A bundle built from
        a two-class source has NC at 0 and AD at 1, so a fixed ``y == 2`` test
        would drop every AD subject.
        """

        if self.y is None:
            raise ValueError("task_view requires labels")
        normalized = task.lower().replace("-", "_")
        if normalized in {"multiclass", "three_class", "3class", "all"}:
            return self
        if normalized not in {"binary", "ad_nc", "ad_vs_nc"}:
            raise ValueError("task must be binary/ad_nc or multiclass")

        reference = self._label_for("NC")
        disease = self._label_for("AD")
        if reference is None or disease is None:
            raise ValueError(
                "binary view needs both an NC and an AD class; found "
                f"{sorted(set(self.y.tolist()))}"
            )
        keep = np.isin(self.y, [reference, disease])
        selected = self.select(np.flatnonzero(keep))
        object.__setattr__(selected, "y", (selected.y == disease).astype(np.int64))
        object.__setattr__(selected, "class_labels", {0: "NC", 1: "AD"})
        object.__setattr__(
            selected, "raw_label_map", {reference: 0, disease: 1}
        )
        return selected

    def _label_for(self, name: str) -> int | None:
        """Find the canonical label carrying a given class name."""

        if self.class_labels:
            for label, label_name in self.class_labels.items():
                if str(label_name) == name:
                    return int(label)
        for raw, canonical in self.raw_label_map.items():
            if LABEL_NAMES.get(int(raw)) == name:
                return int(canonical)
        for label, label_name in CANONICAL_NAMES.items():
            if label_name == name:
                return int(label)
        return None

    @property
    def data_digest(self) -> str:
        """Order-sensitive, non-PHI SHA-256 over this view's arrays.

        Identifies the exact data snapshot a result was produced on: same
        shape with different values, or the same rows in a different order,
        produce different digests. Fold manifests carry it so paired
        comparisons can reject results that came from different snapshots.
        Computed once and cached; hashing ~50 MB takes well under a second.
        """

        cached = self.__dict__.get("_data_digest")
        if cached is not None:
            return str(cached)
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(self.X, dtype=np.float32).tobytes())
        for name in ("y", "site", "age", "sex"):
            value = getattr(self, name)
            if value is None:
                digest.update(b"\x00|none")
            else:
                digest.update(np.ascontiguousarray(value).tobytes())
        digest.update(np.ascontiguousarray(self.mask, dtype=bool).tobytes())
        digest.update("|".join(self.tract_names).encode("utf-8"))
        digest.update("|".join(self.metric_names).encode("utf-8"))
        value = digest.hexdigest()
        object.__setattr__(self, "_data_digest", value)
        return value

    def summary(self) -> dict[str, object]:
        """Return JSON-friendly schema and label counts."""

        return {
            "n_samples": self.n_samples,
            "n_tracts": self.n_tracts,
            "n_points": self.n_points,
            "n_metrics": self.n_metrics,
            "tract_names": list(self.tract_names),
            "metric_names": list(self.metric_names),
            "class_labels": self.label_map,
            "label_counts": (
                {self.label_map[int(k)]: int(v) for k, v in zip(*np.unique(self.y, return_counts=True))}
                if self.y is not None
                else None
            ),
            "site_counts": (
                {str(k): int(v) for k, v in zip(*np.unique(self.site, return_counts=True))}
                if self.site is not None
                else None
            ),
            "missing_fraction": float(np.mean(~self.mask)),
        }
