"""Direct tests for fold provenance manifests.

Coverage until now was transitive (via runner/site_balance/reporting tests);
these pin the digest construction, the non-identifying contract, and the
fail-closed validations that paired comparisons depend on.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from dit.evaluation.provenance import fold_provenance


def _manifest(indices, labels, **kwargs):
    return fold_provenance(
        np.asarray(indices, dtype=np.int64),
        np.asarray(labels, dtype=int),
        **kwargs,
    )


def test_digest_is_sha256_of_sorted_little_endian_int64():
    labels = [0, 1, 0, 1]
    forward = _manifest([1, 3, 0], labels, fold_id="a", task="binary", split_strategy="stratified")
    # Same rows in a different input order must produce the same digest...
    reordered = _manifest([3, 0, 1], labels, fold_id="a", task="binary", split_strategy="stratified")
    assert forward["test_index_digest"] == reordered["test_index_digest"]
    # ...and the digest must equal the documented construction exactly.
    expected = hashlib.sha256(np.sort(np.asarray([1, 3, 0], dtype="<i8")).tobytes()).hexdigest()
    assert forward["test_index_digest"] == expected


def test_manifest_is_non_identifying():
    labels = [0, 1, 0, 1]
    manifest = _manifest([0, 2], labels, fold_id="a", task="binary", split_strategy="stratified")
    assert set(manifest) == {
        "task",
        "split_strategy",
        "test_index_digest",
        "test_class_counts",
        "fold_comparable",
    }


def test_class_counts_and_comparability_flag():
    labels = [0, 1, 1, 1]
    manifest = _manifest([0, 1], labels, fold_id="a", task="binary", split_strategy="stratified")
    assert manifest["test_class_counts"] == {"0": 1, "1": 1}
    assert manifest["fold_comparable"] is True

    missing_class = _manifest([2, 3], labels, fold_id="b", task="binary", split_strategy="stratified")
    assert missing_class["test_class_counts"] == {"0": 0, "1": 2}
    assert missing_class["fold_comparable"] is False


def test_strategy_is_normalized_for_comparison():
    labels = [0, 1, 0, 1]
    plain = _manifest([0], labels, fold_id="a", task="binary", split_strategy="loso", site=[0, 0, 1, 1])
    dashed = _manifest([0], labels, fold_id="a", task="binary", split_strategy="leave-one-site-out", site=[0, 0, 1, 1])
    assert plain["split_strategy"] == "loso"
    assert dashed["split_strategy"] == "leave_one_site_out"
    # Both normalize into the LOSO family, so both demand site metadata.


def test_rejects_empty_and_out_of_range_folds():
    labels = [0, 1, 0, 1]
    with pytest.raises(ValueError, match="cannot be empty"):
        _manifest([], labels, fold_id="a", task="binary", split_strategy="stratified")
    with pytest.raises(ValueError, match="address the label vector"):
        _manifest([0, 4], labels, fold_id="a", task="binary", split_strategy="stratified")
    with pytest.raises(ValueError, match="address the label vector"):
        _manifest([-1], labels, fold_id="a", task="binary", split_strategy="stratified")


def test_loso_requires_site_metadata():
    with pytest.raises(ValueError, match="requires site metadata"):
        _manifest([0, 1], [0, 1], fold_id="a", task="binary", split_strategy="loso")


def test_loso_rejects_multi_site_fold():
    labels = [0, 1, 0, 1]
    with pytest.raises(RuntimeError, match="2 held-out sites"):
        _manifest([0, 2], labels, fold_id="a", task="binary", split_strategy="loso", site=[0, 0, 1, 1])


def test_loso_records_held_out_site():
    labels = [0, 1, 0, 1]
    manifest = _manifest(
        [2, 3], labels, fold_id="site-1", task="binary", split_strategy="loso", site=[0, 0, 1, 1]
    )
    assert manifest["held_out_site"] == "1"
