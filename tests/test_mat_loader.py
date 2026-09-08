"""Loader tests built from real MAT files.

This module is the only code path that touches the real competition data, and
it has never been exercised because the .mat is not bundled with the repo.
Every fixture here is written with ``scipy.io.savemat``, so the assertions run
against a genuine MATLAB container rather than a hand-rolled dict.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.io import savemat

from dit.data.mat_loader import KNOWN_METRICS, load_ai4ad_mat
from dit.data.schema import LABEL_NAMES


def _write(payload: dict[str, np.ndarray], path) -> None:
    savemat(path, payload)


def _tensor_bundle(
    tmp_path,
    *,
    n_samples: int = 30,
    n_tracts: int = 6,
    n_points: int = 40,
    n_metrics: int = 3,
) -> dict:
    """The flattened numeric layout: [subjects, tracts, points, metrics]."""

    rng = np.random.default_rng(0)
    tensor = rng.normal(0.5, 0.2, (n_samples, n_tracts, n_points, n_metrics))
    labels = np.array([1, 3, 2] * (n_samples // 3), dtype=np.int64)[:n_samples]
    population = np.stack((rng.integers(0, 2, n_samples), rng.normal(65, 6, n_samples)), axis=1)
    return {
        "train_set": tensor,
        "train_diagnose": labels,
        "train_population": population,
        "train_sites": np.arange(n_samples) % 5,
    }


def _structured_bundle(
    tmp_path,
    *,
    n_samples: int = 24,
    n_tracts: int = 4,
    n_points: int = 30,
    metrics=("FA", "MD", "RD"),
) -> dict:
    """One structured record per tract, each field a [subjects, points] matrix."""

    rng = np.random.default_rng(1)
    dtypes = [(metric, np.float64, (n_samples, n_points)) for metric in metrics]
    records = np.zeros(n_tracts, dtype=dtypes)
    for index in range(n_tracts):
        for metric in metrics:
            records[index][metric] = rng.normal(0.5, 0.2, (n_samples, n_points))
    labels = np.array([1, 3] * (n_samples // 2), dtype=np.int64)[:n_samples]
    return {
        "train_set": records,
        "train_diagnose": labels,
        "train_population": np.stack(
            (np.array([0, 1] * (n_samples // 2)), rng.normal(65, 6, n_samples)), axis=1
        ),
    }


def _cell_bundle(
    tmp_path,
    *,
    n_samples: int = 20,
    n_tracts: int = 5,
    n_points: int = 25,
) -> dict:
    """A cell array holding one [subjects, points] matrix per tract."""

    rng = np.random.default_rng(2)
    cells = np.empty((n_tracts, 1), dtype=object)
    for index in range(n_tracts):
        cells[index, 0] = rng.normal(0.5, 0.2, (n_samples, n_points))
    labels = np.array([1, 3] * (n_samples // 2), dtype=np.int64)[:n_samples]
    return {
        "train_set": cells,
        "train_diagnose": labels,
        "train_population": np.stack((np.zeros(n_samples), rng.normal(65, 6, n_samples)), axis=1),
    }


class TestNumericTensor:
    def test_four_dimensional_tensor_is_reconstructed(self, tmp_path) -> None:
        _write(_tensor_bundle(tmp_path), tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.X.shape == (30, 6, 40, 3)
        assert bundle.metric_names == KNOWN_METRICS[:3]
        assert np.isfinite(bundle.X).all()

    def test_output_is_float32(self, tmp_path) -> None:
        _write(_tensor_bundle(tmp_path), tmp_path / "data.mat")
        assert load_ai4ad_mat(tmp_path / "data.mat").X.dtype == np.float32

    def test_labels_follow_the_official_coding(self, tmp_path) -> None:
        _write(_tensor_bundle(tmp_path), tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.y is not None
        assert set(bundle.y.tolist()) <= {0, 1, 2}
        assert bundle.raw_label_map == {1: 0, 2: 1, 3: 2}
        assert bundle.class_labels == {
            bundle.raw_label_map[raw]: LABEL_NAMES[raw] for raw in LABEL_NAMES
        }

    def test_population_is_split_with_sex_first(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        _write(payload, tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.sex.shape == (30,)
        assert bundle.age.shape == (30,)
        assert np.allclose(bundle.age, payload["train_population"][:, 1])
        assert np.allclose(bundle.sex, payload["train_population"][:, 0])

    def test_site_is_parsed(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        _write(payload, tmp_path / "data.mat")
        assert np.array_equal(
            load_ai4ad_mat(tmp_path / "data.mat").site, payload["train_sites"]
        )

    def test_metric_subset_is_case_insensitive(self, tmp_path) -> None:
        _write(_tensor_bundle(tmp_path), tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat", metrics=["md", "FA"])
        assert bundle.metric_names == ("FA", "MD")
        assert bundle.X.shape[-1] == 2

    def test_unknown_metric_name_is_refused(self, tmp_path) -> None:
        _write(_tensor_bundle(tmp_path), tmp_path / "data.mat")
        with pytest.raises(ValueError, match="requested metrics not present"):
            load_ai4ad_mat(tmp_path / "data.mat", metrics=("PEAK_FLOW",))

    def test_transposed_tensor_is_realigned(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        payload["train_set"] = np.transpose(payload["train_set"], (1, 3, 2, 0))
        _write(payload, tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.X.shape[0] == 30

    def test_three_dimensional_tensor_becomes_a_single_metric(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        payload["train_set"] = payload["train_set"][..., 0]
        _write(payload, tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.metric_names == ("FA",)
        assert bundle.X.shape == (30, 6, 40, 1)


class TestStructuredLayout:
    def test_one_record_per_tract(self, tmp_path) -> None:
        _write(_structured_bundle(tmp_path), tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.X.shape == (24, 4, 30, 3)
        assert bundle.metric_names == ("FA", "MD", "RD")

    def test_metric_order_follows_the_known_catalog(self, tmp_path) -> None:
        payload = _structured_bundle(tmp_path, metrics=("RD", "FA"))
        _write(payload, tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.metric_names == ("FA", "RD")

    def test_metric_selection_respects_the_request_order_of_the_catalog(self, tmp_path) -> None:
        _write(_structured_bundle(tmp_path), tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat", metrics=("RD",))
        assert bundle.metric_names == ("RD",)
        assert bundle.X.shape == (24, 4, 30, 1)

    def test_subject_axis_is_inferred_from_the_labels(self, tmp_path) -> None:
        # Fields stored as [points, subjects] instead of [subjects, points].
        rng = np.random.default_rng(5)
        transposed = np.zeros(4, dtype=[("FA", np.float64, (30, 24)), ("MD", np.float64, (30, 24))])
        for index in range(4):
            transposed[index]["FA"] = rng.normal(0.5, 0.2, (30, 24))
            transposed[index]["MD"] = rng.normal(0.5, 0.2, (30, 24))
        _write(
            {
                "train_set": transposed,
                "train_diagnose": np.array([1, 3] * 12, dtype=np.int64),
                "train_population": np.stack((np.zeros(24), rng.normal(65, 6, 24)), axis=1),
            },
            tmp_path / "data.mat",
        )
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.X.shape == (24, 4, 30, 2)

    def test_inconsistent_field_shapes_are_refused(self, tmp_path) -> None:
        # A numpy structured dtype forces every field to share one shape, so the
        # only layout that can carry a truncated tract is a cell array.
        rng = np.random.default_rng(6)
        cells = np.empty((4, 1), dtype=object)
        cells[0, 0] = rng.normal(0.5, 0.2, (24, 30))
        cells[1, 0] = rng.normal(0.5, 0.2, (24, 12))
        cells[2, 0] = rng.normal(0.5, 0.2, (24, 30))
        cells[3, 0] = rng.normal(0.5, 0.2, (24, 30))
        _write(
            {
                "train_set": cells,
                "train_diagnose": np.array([1, 3] * 12, dtype=np.int64),
            },
            tmp_path / "data.mat",
        )
        with pytest.raises(ValueError, match="inconsistent profile shapes"):
            load_ai4ad_mat(tmp_path / "data.mat")

    def test_tract_names_fall_back_to_generated_ids(self, tmp_path) -> None:
        _write(_structured_bundle(tmp_path), tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.tract_names == ("tract_00", "tract_01", "tract_02", "tract_03")

    def test_explicit_tract_names_are_kept(self, tmp_path) -> None:
        payload = _structured_bundle(tmp_path)
        payload["fgnames"] = np.array(["AF", "CST", "IFO", "SLF"], dtype=object)
        _write(payload, tmp_path / "data.mat")
        assert load_ai4ad_mat(tmp_path / "data.mat").tract_names == ("AF", "CST", "IFO", "SLF")

    def test_missing_metric_fields_are_refused(self, tmp_path) -> None:
        payload = _structured_bundle(tmp_path, metrics=("FA",))
        _write(payload, tmp_path / "data.mat")
        with pytest.raises(ValueError, match="requested metrics not present"):
            load_ai4ad_mat(tmp_path / "data.mat", metrics=("MD",))


class TestCellLayout:
    def test_one_matrix_per_tract(self, tmp_path) -> None:
        _write(_cell_bundle(tmp_path), tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.X.shape == (20, 5, 25, 1)
        assert bundle.metric_names == ("FA",)


class TestSplitSelection:
    def test_auto_prefers_the_training_set(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        payload["test_set"] = np.zeros((7, 6, 40, 3))
        payload["test_diagnose"] = np.array([1, 1, 1, 1, 1, 1, 1])
        _write(payload, tmp_path / "data.mat")
        assert load_ai4ad_mat(tmp_path / "data.mat").X.shape[0] == 30

    def test_train_mode_ignores_the_test_set(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        payload["test_set"] = np.zeros((7, 6, 40, 3))
        _write(payload, tmp_path / "data.mat")
        assert load_ai4ad_mat(tmp_path / "data.mat", split="train").X.shape[0] == 30

    def test_test_mode_uses_the_test_set(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        payload["test_set"] = np.random.default_rng(3).normal(size=(9, 6, 40, 3))
        payload["test_diagnose"] = np.array([1, 3, 2, 1, 3, 2, 1, 3, 2])
        payload["test_population"] = np.stack(
            (np.zeros(9), np.full(9, 65.0)), axis=1
        )
        _write(payload, tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat", split="test")
        assert bundle.X.shape[0] == 9
        assert set(bundle.y.tolist()) <= {0, 1, 2}

    def test_auto_falls_back_to_the_test_set_when_train_is_absent(self, tmp_path) -> None:
        rng = np.random.default_rng(4)
        _write(
            {
                "test_set": rng.normal(size=(8, 6, 40, 3)),
                "test_diagnose": np.array([1, 3, 1, 3, 1, 3, 1, 3]),
                "test_population": np.stack((np.zeros(8), rng.normal(65, 6, 8)), axis=1),
            },
            tmp_path / "data.mat",
        )
        assert load_ai4ad_mat(tmp_path / "data.mat").X.shape[0] == 8

    def test_auto_keeps_the_train_set_even_when_labels_are_missing(self, tmp_path) -> None:
        # Auto does not switch on the presence of labels: silently reading the
        # wrong half of a file would be worse than reading the one that exists.
        payload = _tensor_bundle(tmp_path)
        del payload["train_diagnose"]
        del payload["train_sites"]
        payload["test_set"] = np.random.default_rng(7).normal(size=(8, 6, 40, 3))
        payload["test_diagnose"] = np.array([1, 3, 1, 3, 1, 3, 1, 3])
        _write(payload, tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.X.shape[0] == 30
        assert bundle.y is None
        assert np.isfinite(bundle.age).all()
        assert np.all(bundle.site == -1)

    def test_unknown_split_is_refused(self, tmp_path) -> None:
        _write(_tensor_bundle(tmp_path), tmp_path / "data.mat")
        with pytest.raises(ValueError, match="split must be"):
            load_ai4ad_mat(tmp_path / "data.mat", split="validation")


class TestOneHotLabels:
    def test_one_hot_diagnosis_is_reduced_to_a_value(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        one_hot = np.zeros((30, 3))
        one_hot[:, 0] = (payload["train_diagnose"] == 1).astype(np.float64)
        one_hot[:, 1] = (payload["train_diagnose"] == 2).astype(np.float64)
        one_hot[:, 2] = (payload["train_diagnose"] == 3).astype(np.float64)
        payload["train_diagnose"] = one_hot
        _write(payload, tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.y.shape == (30,)
        assert set(bundle.y.tolist()) <= {0, 1, 2}


class TestMissingMetadata:
    def test_missing_population_is_represented_as_nan(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        del payload["train_population"]
        del payload["train_sites"]
        _write(payload, tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert np.isnan(bundle.age).all()
        assert np.isnan(bundle.sex).all()
        assert np.all(bundle.site == -1)

    def test_population_without_labels_is_accepted(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        del payload["train_diagnose"]
        _write(payload, tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert bundle.y is None
        assert bundle.age.shape == (30,)
        assert np.isfinite(bundle.age).all()

    def test_private_test_has_no_labels(self, tmp_path) -> None:
        rng = np.random.default_rng(8)
        _write(
            {"test_set": rng.normal(size=(8, 6, 40, 3))},
            tmp_path / "data.mat",
        )
        bundle = load_ai4ad_mat(tmp_path / "data.mat", split="test")
        assert bundle.y is None
        assert np.isnan(bundle.age).all()
        assert np.all(bundle.site == -1)


class TestContractErrors:
    def test_missing_file_points_at_the_repository(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError, match="not bundled"):
            load_ai4ad_mat(tmp_path / "does_not_exist.mat")

    def test_no_supported_feature_key_is_refused(self, tmp_path) -> None:
        _write({"something_else": np.zeros(4)}, tmp_path / "data.mat")
        with pytest.raises(ValueError, match="no supported AFQ feature key"):
            load_ai4ad_mat(tmp_path / "data.mat")

    def test_mis_sized_label_vector_is_not_silently_accepted(self, tmp_path) -> None:
        # Regression: a 3-row label vector matched the 3-metric axis, so the
        # tensor was reinterpreted as having 3 subjects and the row-count
        # check passed.  The mismatch must fail loudly instead.
        payload = _tensor_bundle(tmp_path)
        payload["train_diagnose"] = np.array([1, 3, 1])
        _write(payload, tmp_path / "data.mat")
        with pytest.raises(ValueError, match="wrong row count"):
            load_ai4ad_mat(tmp_path / "data.mat")

    def test_population_row_count_must_agree(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        payload["train_population"] = np.zeros((3, 2))
        _write(payload, tmp_path / "data.mat")
        with pytest.raises(ValueError, match="population must have shape"):
            load_ai4ad_mat(tmp_path / "data.mat")

    def test_site_row_count_must_agree(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        payload["train_sites"] = np.array([1, 2])
        _write(payload, tmp_path / "data.mat")
        with pytest.raises(ValueError, match="rows but features have"):
            load_ai4ad_mat(tmp_path / "data.mat")

    def test_non_numeric_labels_are_refused(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        payload["train_diagnose"] = np.array(["AD", "NC", "MCI"])
        _write(payload, tmp_path / "data.mat")
        with pytest.raises(ValueError, match="numeric"):
            load_ai4ad_mat(tmp_path / "data.mat")

    def test_non_integer_labels_are_refused(self, tmp_path) -> None:
        # A fractional diagnosis must be rejected before the int cast, which
        # would otherwise truncate 1.5 into a different, valid label.
        payload = _tensor_bundle(tmp_path)
        payload["train_diagnose"] = np.array([1.0, 1.5, 3.0])
        _write(payload, tmp_path / "data.mat")
        with pytest.raises(ValueError, match="integer-valued"):
            load_ai4ad_mat(tmp_path / "data.mat")

    def test_too_few_tensor_dimensions_are_refused(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        payload["train_set"] = np.zeros((30, 6))
        _write(payload, tmp_path / "data.mat")
        with pytest.raises(ValueError, match="3 or 4 dimensions"):
            load_ai4ad_mat(tmp_path / "data.mat")

    def test_subject_ids_are_generated_when_absent(self, tmp_path) -> None:
        _write(_tensor_bundle(tmp_path), tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert len(bundle.subject_id) == 30

    def test_validity_mask_tracks_finite_values(self, tmp_path) -> None:
        payload = _tensor_bundle(tmp_path)
        tensor = payload["train_set"].copy()
        tensor[0, 0, 0, 0] = np.nan
        payload["train_set"] = tensor
        _write(payload, tmp_path / "data.mat")
        bundle = load_ai4ad_mat(tmp_path / "data.mat")
        assert not bundle.mask[0, 0, 0, 0]
        assert bundle.mask.sum() == bundle.X.size - 1
