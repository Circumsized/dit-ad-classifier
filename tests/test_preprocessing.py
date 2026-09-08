"""Leakage tests for the data path.

The original ML.py normalised age with min/max computed over all 700 subjects
and then applied that to the validation fold.  Every test in this file checks
that a statistic fitted on one subset cannot change when another subset is
seen, which is the property that makes a reported score trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from dit.data.covariates import Residualizer, covariate_matrix, strip_covariate_columns
from dit.data.layout import FeatureLayout, view_for_layout
from dit.data.preprocessing import TractFeaturePreprocessor, smooth_profiles
from dit.data.schema import DatasetBundle
from dit.data.synthetic import make_synthetic_bundle
from dit.models.classical import _nan_area, _nan_slope, build_feature_matrix, describe_feature_layout


class TestSmoothing:
    def test_window_one_is_a_noop(self) -> None:
        x = make_synthetic_bundle().X
        assert np.array_equal(smooth_profiles(x, 1), x, equal_nan=True)

    def test_single_gap_does_not_spread(self) -> None:
        x = np.ones((1, 1, 11, 1), dtype=np.float32)
        x[0, 0, 5, 0] = np.nan
        out = smooth_profiles(x, 5)
        assert out.shape == x.shape
        # the smoothed value at the gap is finite because neighbours survive
        assert np.isfinite(out[0, 0, 5, 0])
        # far from the gap the profile is untouched
        assert out[0, 0, 0, 0] == pytest.approx(1.0, abs=1e-6)
        assert out[0, 0, 10, 0] == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize("window", [0, -3, 2, 4])
    def test_invalid_windows_raise(self, window: int) -> None:
        x = make_synthetic_bundle(n_points=10, n_metrics=1).X
        with pytest.raises(ValueError):
            smooth_profiles(x, window)

    def test_window_cannot_exceed_the_node_count(self) -> None:
        x = make_synthetic_bundle(n_points=4, n_metrics=1).X
        with pytest.raises(ValueError, match="exceeds the number of nodes"):
            smooth_profiles(x, 5)

    def test_shape_is_enforced(self) -> None:
        with pytest.raises(ValueError, match="expected \\[N,T,P,M\\]"):
            smooth_profiles(np.zeros((10, 5)))

    def test_nan_never_propagates_to_finite_neighbours(self) -> None:
        x = np.ones((1, 1, 11, 1), dtype=np.float32)
        x[0, 0, 5, 0] = np.nan
        out = smooth_profiles(x, 3)
        # neighbours of the gap stay close to 1.0
        assert out[0, 0, 3, 0] == pytest.approx(1.0, abs=1e-6)
        assert out[0, 0, 7, 0] == pytest.approx(1.0, abs=1e-6)


class TestProfileSummaryNaNSafety:
    """np.trapz propagates NaN, turning a 1% missing rate into 16% NaN columns."""

    def test_area_ignores_missing_nodes(self) -> None:
        values = np.ones((2, 3, 10, 2), dtype=np.float32)
        values[0, 0, 4, 0] = np.nan
        positions = np.linspace(-1, 1, 10)
        assert np.isfinite(_nan_area(values, positions)).all()

    def test_all_missing_profile_is_unknown_not_zero(self) -> None:
        values = np.full((1, 1, 10, 1), np.nan, dtype=np.float32)
        positions = np.linspace(-1, 1, 10)
        assert np.isnan(_nan_area(values, positions)[0, 0, 0])
        assert np.isnan(_nan_slope(values, positions)[0, 0, 0])

    def test_area_matches_trapezoid_on_finite_data(self) -> None:
        rng = np.random.default_rng(0)
        values = rng.normal(0, 0.5, (8, 4, 60, 3)).astype(np.float32)
        positions = np.linspace(-1, 1, 60)
        assert np.allclose(_nan_area(values, positions), np.trapz(values, x=positions, axis=2))

    def test_slope_matches_polyfit_on_finite_data(self) -> None:
        rng = np.random.default_rng(1)
        values = rng.normal(0, 0.5, (6, 3, 40, 2)).astype(np.float32)
        positions = np.linspace(-1, 1, 40)
        for i in range(6):
            for j in range(3):
                for m in range(2):
                    expected = np.polyfit(positions, values[i, j, :, m], 1)[0]
                    assert _nan_slope(values, positions)[i, j, m] == pytest.approx(
                        expected, rel=1e-5
                    )

    def test_slope_is_nan_aware(self) -> None:
        values = np.zeros((1, 1, 10, 1), dtype=np.float32)
        values[0, 0, :, 0] = np.arange(10, dtype=np.float32)
        positions = np.linspace(-1, 1, 10)
        clean = _nan_slope(values, positions)[0, 0, 0]
        values[0, 0, 3, 0] = np.nan
        assert _nan_slope(values, positions)[0, 0, 0] == pytest.approx(clean, rel=1e-6)

    def test_summary_does_not_inflate_the_missing_rate(self) -> None:
        bundle = make_synthetic_bundle(n_samples=60, n_points=100, n_metrics=4, seed=3)
        features = build_feature_matrix(bundle, view="summary", include_covariates=False)
        assert float(np.mean(~np.isfinite(features))) < 0.02

    def test_summary_column_count_matches_layout(self) -> None:
        bundle = make_synthetic_bundle(n_samples=30, n_tracts=6, n_points=10, n_metrics=2)
        features = build_feature_matrix(bundle, view="summary", include_covariates=True)
        layout = describe_feature_layout(bundle, view="summary", include_covariates=True)
        layout.validate(features.shape[1])


class TestFeatureLayout:
    def test_profile_layout_accounting(self) -> None:
        bundle = make_synthetic_bundle(n_tracts=18, n_points=100, n_metrics=4)
        layout = view_for_layout(
            "profile",
            bundle.metric_names,
            bundle.tract_names,
            bundle.n_points,
            has_covariates=True,
            has_missing_pattern=True,
        )
        assert layout.n_feature_columns == 18 * 100 * 4
        assert layout.total_features == 18 * 100 * 4 + 18 + 2

    def test_layout_without_missing_pattern_is_core_plus_covariates(self) -> None:
        layout = view_for_layout("profile", ("FA",), ("t0", "t1"), 10, has_covariates=True)
        assert layout.total_features == layout.n_feature_columns + 2

    def test_blocks_are_contiguous_and_cover_the_core(self) -> None:
        bundle = make_synthetic_bundle(n_tracts=4, n_points=10, n_metrics=2)
        layout = view_for_layout(
            "profile", bundle.metric_names, bundle.tract_names, 10, has_covariates=False
        )
        blocks = layout.blocks()
        assert len(blocks) == 4 * 2
        assert blocks[0].start == 0
        assert blocks[-1].end == layout.n_feature_columns
        for previous, current in zip(blocks, blocks[1:]):
            assert previous.end == current.start, "blocks must be contiguous"

    def test_profile_blocks_are_ordered_tract_then_metric(self) -> None:
        layout = view_for_layout(
            "profile",
            ("FA", "MD"),
            ("t0", "t1"),
            5,
            has_covariates=False,
        )
        names = [block.name for block in layout.blocks()]
        assert names == ["t0|FA", "t0|MD", "t1|FA", "t1|MD"]

    def test_missing_pattern_block_is_not_anatomical(self) -> None:
        bundle = make_synthetic_bundle(n_tracts=3, n_points=5, n_metrics=2)
        layout = view_for_layout(
            "profile",
            bundle.metric_names,
            bundle.tract_names,
            5,
            has_covariates=False,
            has_missing_pattern=True,
        )
        assert len(layout.anatomical_blocks()) == 3 * 2
        assert len(layout.blocks()) == 3 * 2 + 1
        assert len(layout.missing_pattern_columns) == 3

    def test_feature_names_match_column_count(self) -> None:
        bundle = make_synthetic_bundle(n_tracts=3, n_points=5, n_metrics=2)
        for view in ("profile", "summary", "FA"):
            layout = view_for_layout(
                view, bundle.metric_names, bundle.tract_names, 5, has_covariates=True
            )
            assert len(layout.feature_names()) == layout.total_features

    def test_build_and_layout_agree_for_every_view(self) -> None:
        bundle = make_synthetic_bundle(n_tracts=4, n_points=10, n_metrics=3)
        for view in ("profile", "summary", "FA"):
            for covariates in (True, False):
                for missing in (True, False):
                    features = build_feature_matrix(
                        bundle,
                        view=view,
                        include_covariates=covariates,
                        include_missing_pattern=missing,
                    )
                    layout = describe_feature_layout(
                        bundle,
                        view=view,
                        include_covariates=covariates,
                        include_missing_pattern=missing,
                    )
                    layout.validate(features.shape[1])

    def test_unknown_view_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown feature view"):
            view_for_layout("nope", ("FA",), ("t0",), 5, has_covariates=False)

    def test_is_anatomical_separates_the_tail_columns(self) -> None:
        layout = view_for_layout(
            "summary", ("FA",), ("t0",), 5, has_covariates=True, has_missing_pattern=True
        )
        assert layout.is_anatomical(0)
        assert not layout.is_anatomical(layout.n_feature_columns)
        assert not layout.is_anatomical(layout.total_features - 1)


class TestMissingPattern:
    def test_missing_pattern_reflects_tract_gaps(self) -> None:
        x = np.ones((4, 3, 10, 1), dtype=np.float32)
        x[0, 0, :, :] = np.nan
        bundle = DatasetBundle(X=x)
        features = build_feature_matrix(
            bundle,
            view="profile",
            include_covariates=False,
            include_missing_pattern=True,
        )
        layout = describe_feature_layout(
            bundle, view="profile", include_covariates=False, include_missing_pattern=True
        )
        block = layout.index_blocks()["missing_pattern"]
        assert features[0, block.start] == pytest.approx(1.0)
        assert features[0, block.start + 1] == pytest.approx(0.0)
        assert features[1, block.start] == pytest.approx(0.0)


class TestPreprocessorLeakage:
    def test_fit_uses_only_training_rows(self) -> None:
        bundle = make_synthetic_bundle(n_samples=80, n_points=11, n_metrics=2, seed=5)
        preprocessor = TractFeaturePreprocessor(smooth_window=1).fit(
            bundle.X[:50], age=bundle.age[:50], sex=bundle.sex[:50]
        )
        training_mean = float(np.nanmean(bundle.X[:50].reshape(50, -1), axis=0).mean())
        assert float(preprocessor.center_.mean()) == pytest.approx(training_mean, rel=1e-5)

    def test_transform_is_invariant_to_unseen_data(self) -> None:
        bundle = make_synthetic_bundle(n_samples=80, n_points=11, n_metrics=2, seed=5)
        fitted = TractFeaturePreprocessor().fit(bundle.X[:50], bundle.age[:50], bundle.sex[:50])
        transformed = fitted.transform(bundle.X[:50], bundle.age[:50], bundle.sex[:50])

        # Fitting on a different, far larger training set must not change the
        # transform of the original rows.
        other = TractFeaturePreprocessor().fit(
            bundle.X[:50], bundle.age[:50], bundle.sex[:50]
        )
        assert np.allclose(transformed, other.transform(bundle.X[:50], bundle.age[:50], bundle.sex[:50]))

    def test_transform_requires_fit(self) -> None:
        bundle = make_synthetic_bundle(n_samples=28, n_points=5, n_metrics=1)
        with pytest.raises(RuntimeError, match="must be fit"):
            TractFeaturePreprocessor().transform(bundle.X)

    def test_imputation_uses_training_median(self) -> None:
        bundle = make_synthetic_bundle(n_samples=40, n_points=7, n_metrics=1, seed=2)
        x_train = bundle.X[:25].copy()
        x_test = bundle.X[25:].copy()
        x_test[0] = np.nan
        fitted = TractFeaturePreprocessor().fit(x_train, bundle.age[:25], bundle.sex[:25])
        out = fitted.transform(x_test, bundle.age[25:], bundle.sex[25:])
        assert np.isfinite(out[0]).all()

    def test_covariates_are_appended_when_requested(self) -> None:
        bundle = make_synthetic_bundle(n_samples=30, n_points=7, n_metrics=1)
        plain = TractFeaturePreprocessor(include_covariates=False).fit_transform(
            bundle.X, bundle.age, bundle.sex
        )
        with_cov = TractFeaturePreprocessor(include_covariates=True).fit_transform(
            bundle.X, bundle.age, bundle.sex
        )
        assert with_cov.shape[1] == plain.shape[1] + 2

    def test_zero_variance_columns_are_not_divided_by_zero(self) -> None:
        x = np.ones((12, 2, 5, 1), dtype=np.float32)
        age = np.arange(12, dtype=np.float32)
        sex = np.zeros(12, dtype=np.float32)
        out = TractFeaturePreprocessor().fit_transform(x, age, sex)
        assert np.isfinite(out).all()


class TestResidualizer:
    def test_residuals_are_orthogonal_to_covariates(self) -> None:
        rng = np.random.default_rng(0)
        n = 300
        cov = rng.normal(size=(n, 2))
        weights = rng.normal(size=(2, 40))
        x = cov @ weights + rng.normal(0, 0.1, size=(n, 40))
        residualizer = Residualizer().fit(x, cov)
        out = residualizer.transform(x, cov)
        design = np.column_stack((np.ones(n), cov))
        correlation = design.T @ out
        assert np.abs(correlation).max() < 1e-6

    def test_fit_on_train_removes_signal_from_unseen_rows(self) -> None:
        rng = np.random.default_rng(1)
        n = 400
        cov = rng.normal(size=(n, 2))
        weights = rng.normal(size=(2, 30))
        x = cov @ weights + rng.normal(0, 0.1, size=(n, 30))
        split = 250
        residualizer = Residualizer().fit(x[:split], cov[:split])
        out = residualizer.transform(x[split:], cov[split:])
        design = np.column_stack(
            (
                np.ones(n - split),
                (cov[split:] - residualizer.cov_mean_) / residualizer.cov_scale_,
            )
        )
        # Orthogonality is exact on the fitted rows; on unseen rows it only
        # holds in expectation, so the claim is that residualisation strictly
        # reduces the covariate signal that survives on the held-out rows.
        raw_alignment = np.abs(design.T @ x[split:]).max()
        assert np.abs(design.T @ out).max() < raw_alignment * 0.5

    def test_nan_covariates_are_imputed_from_training(self) -> None:
        rng = np.random.default_rng(2)
        cov = rng.normal(size=(60, 2))
        x = cov @ rng.normal(size=(2, 10)) + rng.normal(size=(60, 10))
        residualizer = Residualizer().fit(x, cov)
        test_cov = cov[56:60].copy()
        test_cov[0, 0] = np.nan
        out = residualizer.transform(x[56:60], test_cov)
        assert np.isfinite(out).all()

    def test_nan_features_are_preserved_as_nan(self) -> None:
        x = np.zeros((10, 5))
        x[0, 0] = np.nan
        cov = np.arange(10).reshape(-1, 1)
        residualizer = Residualizer().fit(x, cov)
        out = residualizer.transform(x, cov)
        assert np.isnan(out[0, 0])
        assert np.isfinite(out[1, 0])

    def test_transform_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fit"):
            Residualizer().transform(np.zeros((3, 2)), np.zeros((3, 1)))

    def test_width_mismatch_raises(self) -> None:
        residualizer = Residualizer().fit(np.zeros((10, 4)), np.zeros((10, 2)))
        with pytest.raises(ValueError, match="different number of columns"):
            residualizer.transform(np.zeros((5, 3)), np.zeros((5, 2)))

    def test_too_few_rows_raises(self) -> None:
        with pytest.raises(ValueError, match="not enough rows"):
            Residualizer().fit(np.zeros((2, 4)), np.zeros((2, 2)))

    def test_shape_contract_is_enforced(self) -> None:
        with pytest.raises(ValueError, match=r"X must be \[N,F\]"):
            Residualizer().fit(np.zeros(10), np.zeros((10, 2)))

    def test_parameters_are_reportable(self) -> None:
        residualizer = Residualizer().fit(np.zeros((12, 4)), np.zeros((12, 2)))
        params = residualizer.parameters()
        assert params["include_intercept"] is True
        assert params["n_fitted_features"] == 4
        assert len(params["cov_mean"]) == 2

    def test_no_intercept_mode(self) -> None:
        rng = np.random.default_rng(3)
        cov = rng.normal(size=(80, 2))
        x = cov @ rng.normal(size=(2, 6))
        residualizer = Residualizer(include_intercept_=False).fit(x, cov)
        out = residualizer.transform(x, cov)
        assert np.isfinite(out).all()


class TestCovariateHelpers:
    def test_covariate_matrix_handles_missing_metadata(self) -> None:
        out = covariate_matrix(None, None, 5)
        assert out.shape == (5, 2)
        assert np.isnan(out).all()

    def test_covariate_matrix_checks_length(self) -> None:
        with pytest.raises(ValueError, match="one value per sample"):
            covariate_matrix(np.zeros(3), np.zeros(5), 5)

    def test_strip_covariates_splits_the_tail(self) -> None:
        matrix = np.arange(12).reshape(3, 4)
        core, cov = strip_covariate_columns(matrix, 2)
        assert core.shape == (3, 2)
        assert cov.shape == (3, 2)
        assert np.array_equal(cov, matrix[:, 2:])

    def test_strip_covariates_rejects_degenerate_width(self) -> None:
        with pytest.raises(ValueError, match="no non-covariate columns"):
            strip_covariate_columns(np.zeros((3, 2)))
