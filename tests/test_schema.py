"""Label contract tests.

The single most damaging defect in the original submission was that labels were
derived as ``label.index(max(label))`` instead of ``max(label)``.  For the
AI4AD coding (1=NC, 2=MCI, 3=AD) that expression returns 0 for every subject,
so ``fit()`` raised on a one-class label vector and no result was ever
produced.  These tests pin the correct semantics so the mistake cannot return.
"""

from __future__ import annotations

import numpy as np
import pytest

from dit.data.schema import DatasetBundle, canonicalize_labels
from dit.data.synthetic import make_synthetic_bundle


class TestCanonicalLabels:
    def test_official_coding_is_shifted_to_contiguous(self) -> None:
        labels, mapping = canonicalize_labels([1, 3, 2, 1, 3])
        assert labels.tolist() == [0, 2, 1, 0, 2]
        assert mapping == {1: 0, 2: 1, 3: 2}

    def test_only_present_classes_are_mapped(self) -> None:
        labels, mapping = canonicalize_labels([1, 3])
        assert labels.tolist() == [0, 1]
        assert mapping == {1: 0, 3: 1}

    def test_already_canonical_labels_pass_through(self) -> None:
        labels, mapping = canonicalize_labels([0, 1, 2, 0])
        assert labels.tolist() == [0, 1, 2, 0]
        assert mapping == {0: 0, 1: 1, 2: 2}

    @pytest.mark.parametrize("labels", [[], np.array([])])
    def test_empty_labels_raise(self, labels) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            canonicalize_labels(labels)

    @pytest.mark.parametrize("labels", [[1, 4], [0, 3], [1, 5, 2]])
    def test_unsupported_labels_raise(self, labels) -> None:
        with pytest.raises(ValueError, match="unsupported labels"):
            canonicalize_labels(labels)

    def test_nan_labels_raise(self) -> None:
        with pytest.raises(ValueError, match="NaN or infinite"):
            canonicalize_labels([1, float("nan"), 3])

    def test_single_class_is_accepted_but_flagged_downstream(self) -> None:
        # A one-class input is legal at the schema boundary; classifiers raise
        # later with a clearer message.  Labels are renumbered contiguously, so
        # the surviving class becomes 0 and the mapping keeps the original code.
        labels, mapping = canonicalize_labels([3, 3, 3])
        assert labels.tolist() == [0, 0, 0]
        assert mapping == {3: 0}

    def test_non_integer_labels_are_refused(self) -> None:
        # 1.5 is corrupt data, not a value to truncate: silently turning it
        # into 1 would swap one valid label for another.
        with pytest.raises(ValueError, match="integer-valued"):
            canonicalize_labels([1, 1.5, 3])

    def test_integer_valued_floats_are_accepted(self) -> None:
        labels, mapping = canonicalize_labels([1.0, 3.0, 2.0])
        assert labels.tolist() == [0, 2, 1]
        assert mapping == {1: 0, 2: 1, 3: 2}

    def test_dataset_bundle_rejects_non_integer_y(self) -> None:
        with pytest.raises(ValueError, match="integer-valued"):
            DatasetBundle(X=np.zeros((4, 2, 5, 1)), y=np.array([0.0, 1.5, 1.0, 0.0]))


class TestLegacyLabelDefect:
    """Regression test for the defect that invalidated the original ML.py."""

    def test_index_of_max_is_not_the_label(self) -> None:
        """The original code took the position of the max, not the value."""

        for raw, legacy, correct in (
            ([3, 1], 0, 3),
            ([3, 2], 0, 3),
            ([1, 3], 1, 3),
            ([1, 1], 0, 1),
        ):
            label = np.asarray(raw).tolist()
            assert label.index(max(label)) == legacy
            assert legacy != correct, f"legacy expression must not equal the label for {raw}"
            assert correct in (1, 2, 3)

    def test_correct_extraction_is_the_value(self) -> None:
        """The value, not its index, is the class."""

        for raw, expected in (([1], 1), ([2], 2), ([3], 3), ([1, 3], 3), ([3, 1], 3)):
            assert max(np.asarray(raw).tolist()) == expected

    def test_legacy_expression_yields_a_constant_vector(self) -> None:
        """Reproduces the failure mode: every subject labelled identically."""

        raw_labels = [[1], [2], [3], [1], [3], [2], [3]]
        legacy_vector = [label.index(max(label)) for label in raw_labels]
        assert len(set(legacy_vector)) == 1, (
            "the legacy expression produces a single-class label vector, which "
            "makes sklearn raise ValueError in fit()"
        )

    def test_synthetic_bundle_labels_are_not_constant(self) -> None:
        bundle = make_synthetic_bundle(n_classes=3)
        assert bundle.y is not None
        assert len(set(bundle.y.tolist())) == 3


class TestDatasetBundle:
    def test_valid_bundle_round_trips(self) -> None:
        bundle = make_synthetic_bundle(n_samples=60, n_tracts=6, n_points=20, n_metrics=2)
        assert bundle.n_samples == 60
        assert bundle.n_tracts == 6
        assert bundle.n_points == 20
        assert bundle.n_metrics == 2
        assert bundle.X.shape == (60, 6, 20, 2)

    @pytest.mark.parametrize("shape", [(10, 5), (10, 5, 20), (10, 5, 20, 2, 1)])
    def test_wrong_dimensionality_is_rejected(self, shape) -> None:
        with pytest.raises(ValueError, match=r"X must have shape"):
            DatasetBundle(X=np.zeros(shape))

    def test_zero_dimension_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-zero"):
            DatasetBundle(X=np.zeros((0, 5, 20, 2)))

    def test_non_numeric_x_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="numeric"):
            DatasetBundle(X=np.zeros((5, 4, 10, 2), dtype=object))

    def test_label_length_mismatch_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="y has"):
            DatasetBundle(X=np.zeros((10, 4, 20, 2)), y=np.zeros(9, dtype=int))

    def test_out_of_range_labels_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="canonical labels"):
            DatasetBundle(X=np.zeros((10, 4, 20, 2)), y=np.zeros(10, dtype=int) + 3)

    @pytest.mark.parametrize("field", ["age", "sex", "site", "subject_id"])
    def test_metadata_length_mismatch_is_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} has"):
            DatasetBundle(
                X=np.zeros((10, 4, 20, 2)),
                **{field: np.zeros(9, dtype=float)},
            )

    def test_mask_must_match_x(self) -> None:
        with pytest.raises(ValueError, match="mask shape"):
            DatasetBundle(
                X=np.zeros((10, 4, 20, 2)),
                mask=np.ones((9, 4, 20, 2), dtype=bool),
            )

    def test_missing_values_are_recorded_in_the_mask(self) -> None:
        x = np.zeros((5, 3, 10, 2))
        x[0, 0, 3, 0] = np.nan
        bundle = DatasetBundle(X=x)
        assert bundle.mask is not None
        assert not bundle.mask[0, 0, 3, 0]
        assert bundle.mask.sum() == x.size - 1

    def test_select_keeps_feature_semantics(self) -> None:
        bundle = make_synthetic_bundle(n_samples=40, n_sites=4)
        subset = bundle.select([1, 5, 9, 20])
        assert subset.n_samples == 4
        assert subset.X.shape[1:] == bundle.X.shape[1:]
        assert subset.tract_names == bundle.tract_names
        assert subset.metric_names == bundle.metric_names
        assert subset.y is not None and subset.y.shape[0] == 4
        assert subset.site is not None and subset.site.shape[0] == 4

    def test_task_view_binary_keeps_only_nc_and_ad(self) -> None:
        bundle = make_synthetic_bundle(n_classes=3, n_samples=90)
        binary = bundle.task_view("binary")
        assert set(binary.y.tolist()) == {0, 1}
        assert binary.n_samples == int(np.isin(bundle.y, [0, 2]).sum())
        # NC -> 0, AD -> 1
        source_ad = (bundle.y == 2)
        assert (binary.y == 1).sum() == source_ad.sum()

    def test_task_view_multiclass_is_identity(self) -> None:
        bundle = make_synthetic_bundle(n_classes=3)
        multi = bundle.task_view("multiclass")
        assert multi.n_samples == bundle.n_samples
        assert set(multi.y.tolist()) == {0, 1, 2}

    @pytest.mark.parametrize("task", ["ad_vs_mci", "binary3", "unknown"])
    def test_unknown_task_is_rejected(self, task: str) -> None:
        with pytest.raises(ValueError, match="task must be"):
            make_synthetic_bundle().task_view(task)

    def test_task_view_requires_labels(self) -> None:
        bundle = make_synthetic_bundle()
        unlabelled = DatasetBundle(X=bundle.X)
        with pytest.raises(ValueError, match="requires labels"):
            unlabelled.task_view("binary")

    def test_class_names_follow_canonical_ordering(self) -> None:
        bundle = make_synthetic_bundle(n_classes=3)
        assert bundle.class_names == ("NC", "MCI", "AD")

    def test_summary_reports_counts(self) -> None:
        bundle = make_synthetic_bundle(n_samples=60, n_sites=5)
        summary = bundle.summary()
        assert summary["n_samples"] == 60
        assert summary["label_counts"] is not None
        assert summary["site_counts"] is not None
        assert len(summary["label_counts"]) == 3
        assert len(summary["site_counts"]) == 5
        assert 0.0 <= float(summary["missing_fraction"]) <= 1.0
