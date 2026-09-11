"""Non-identifying provenance for outer cross-validation folds.

Paired fold comparisons are valid only when both results were evaluated on
the same subjects. Reports carry a digest of sorted indices within the task
view rather than source subject identifiers or imaging data.
"""

from __future__ import annotations

import hashlib

import numpy as np


_LOSO_NAMES = frozenset({"loso", "leave_one_site_out", "group"})


def fold_provenance(
    test_indices: np.ndarray,
    labels: np.ndarray,
    *,
    fold_id: object,
    task: str,
    split_strategy: str,
    site: np.ndarray | None = None,
    data_digest: str | None = None,
) -> dict[str, object]:
    """Return an auditable, non-identifying manifest for one outer test fold.

    ``test_index_digest`` is an SHA-256 digest of sorted, little-endian int64
    task-view positions, used to reject paired comparisons whose folds do not
    hold out the same rows. ``data_digest`` — when the caller supplies it —
    identifies the underlying data snapshot, so results from different
    snapshots cannot be paired even when their fold structure matches.
    """

    indices = np.asarray(test_indices, dtype=np.int64).reshape(-1)
    values = np.asarray(labels, dtype=int).reshape(-1)
    if indices.size == 0:
        raise ValueError("test fold cannot be empty")
    if np.any(indices < 0) or np.any(indices >= values.size):
        raise ValueError("test indices must address the label vector")

    ordered = np.sort(indices).astype("<i8", copy=False)
    test_labels = values[indices]
    expected_labels = np.unique(values)
    normalized_strategy = split_strategy.lower().replace("-", "_")
    class_counts = {
        str(int(label)): int(np.sum(test_labels == label))
        for label in expected_labels
    }
    # A fold whose test set misses a class present in the task view scores on
    # a different label set than full folds, so flag it rather than pool it.
    fold_comparable = all(count > 0 for count in class_counts.values())
    manifest: dict[str, object] = {
        "task": str(task),
        "split_strategy": normalized_strategy,
        "test_index_digest": hashlib.sha256(ordered.tobytes()).hexdigest(),
        "test_class_counts": class_counts,
        "fold_comparable": fold_comparable,
    }
    if data_digest:
        manifest["data_digest"] = str(data_digest)

    if normalized_strategy in _LOSO_NAMES:
        if site is None:
            raise ValueError("LOSO fold provenance requires site metadata")
        sites = np.asarray(site).reshape(-1)
        if sites.size != values.size:
            raise ValueError("site must have one value per label")
        held_out = np.unique(sites[indices])
        if held_out.size != 1:
            raise RuntimeError(
                f"LOSO fold {fold_id!r} contains {held_out.size} held-out sites"
            )
        manifest["held_out_site"] = str(held_out[0])

    return manifest
