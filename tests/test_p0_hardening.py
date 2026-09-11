"""Regression tests for the P0 credibility fixes in the optimization plan.

Covers: residualization fitted inside the inner folds with an explicit
missing-value scheme (P0-1), explicit missing-value masks through the deep
path (P0-2), per-class early-stop/calibration halves and train-only class
weights (P0-3), the configured epoch budget (P0-4), and the scorer / class
space / calibration contracts (P0-5).
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from dit.data.covariates import Residualizer, ResidualizeCovariates
from dit.data.synthetic import make_synthetic_bundle
from dit.models.classical import align_probabilities, make_search_estimator


def _sklearn():
    import sklearn

    return sklearn


def _domain_train():
    """Import the deep training loop, skipping the test when torch is absent."""

    pytest.importorskip("torch", reason="deep-path regressions require torch")
    import dit.models.domain_train as module

    return module


def _covariate_matrix(
    features: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    n = features.shape[0]
    age = 60 + 8 * rng.random(n)
    sex = rng.integers(0, 2, n).astype(float)
    return np.column_stack((age, sex))


class TestResidualizerMissingValues:
    def test_column_with_gaps_is_still_residualized(self) -> None:
        rng = np.random.default_rng(7)
        n = 60
        cov = _covariate_matrix(np.zeros((n, 1)), rng)
        base = rng.normal(size=(n, 4))
        age_effect = np.outer(cov[:, 0] - cov[:, 0].mean(), np.array([2.0, 0.0, 0.0, 0.0]))
        values = base + age_effect
        values[3, 2] = np.nan
        values[10, 2] = np.nan

        residualizer = Residualizer().fit(values, cov)
        out = residualizer.transform(values, cov)

        # The previously-skipped column now receives coefficients, so the
        # demographic component is removed from it instead of passing through.
        assert np.any(residualizer.design_[:, 2] != 0)
        finite_rows = np.isfinite(out[:, 2])
        corr = np.corrcoef(out[finite_rows, 2], cov[finite_rows, 0])[0, 1]
        assert abs(corr) < 0.2

    def test_parameters_report_real_counts(self) -> None:
        rng = np.random.default_rng(3)
        values = rng.normal(size=(40, 3))
        values[0, 1] = np.nan
        params = Residualizer().fit(values, _covariate_matrix(values, rng)).parameters()
        assert params["n_fitted_features"] == 3
        assert params["n_train_rows"] == 40
        assert params["n_columns_with_imputation"] == 1

    def test_all_nan_column_stays_nan_and_gets_zero_coefficient(self) -> None:
        rng = np.random.default_rng(5)
        values = rng.normal(size=(40, 3))
        values[:, 2] = np.nan
        residualizer = Residualizer().fit(values, _covariate_matrix(values, rng))
        assert np.all(residualizer.design_[:, 2] == 0)
        out = residualizer.transform(values, _covariate_matrix(values, rng))
        assert np.all(np.isnan(out[:, 2]))

    def test_no_finite_values_at_all_is_refused(self) -> None:
        values = np.full((30, 3), np.nan)
        cov = _covariate_matrix(values, np.random.default_rng(1))
        with pytest.raises(ValueError, match="no finite feature values"):
            Residualizer().fit(values, cov)


class TestPipelineResidualization:
    def _matrix_with_covariates(self, n: int = 80, seed: int = 11) -> np.ndarray:
        rng = np.random.default_rng(seed)
        features = rng.normal(size=(n, 6))
        cov = _covariate_matrix(features, rng)
        features = features + np.outer(cov[:, 0] - 65.0, np.full(6, 0.05))
        return np.column_stack((features, cov))

    def test_step_strips_covariates_and_returns_residuals(self) -> None:
        matrix = self._matrix_with_covariates()
        step = ResidualizeCovariates().fit(matrix)
        out = step.transform(matrix)
        assert out.shape == (matrix.shape[0], matrix.shape[1] - 2)
        # Residuals are orthogonal to the covariates they were regressed on.
        for column in range(out.shape[1]):
            corr = np.corrcoef(out[:, column], matrix[:, -2])[0, 1]
            assert abs(corr) < 0.2

    def test_step_inside_the_search_is_refit_per_inner_fold(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import KFold

        matrix = self._matrix_with_covariates(n=90)
        labels = np.repeat([0, 1], 45)

        seen_rows: list[int] = []
        original_fit = ResidualizeCovariates.fit

        def spy(self: ResidualizeCovariates, X, y=None):
            seen_rows.append(int(np.asarray(X).shape[0]))
            return original_fit(self, X, y)

        with mock.patch.object(ResidualizeCovariates, "fit", spy):
            search = make_search_estimator(
                "logistic", cv=KFold(n_splits=3, shuffle=True, random_state=0), residualize=True
            )
            search.fit(matrix, labels)

        full = matrix.shape[0]
        # Candidate fits see inner-training subsets; only the final refit sees
        # every row. Seeing subsets proves the regression is fold-local rather
        # than fitted once on the whole outer training fold.
        assert any(count < full for count in seen_rows)
        assert seen_rows[-1] == full

    def test_experiment_residualize_reports_pipeline_fitted_parameters(self) -> None:
        from dit.evaluation.experiment import ExperimentConfig, run_experiment

        bundle = make_synthetic_bundle(n_samples=60, n_tracts=3, n_points=10, n_metrics=2, seed=4)
        config = ExperimentConfig(
            model="logistic",
            covariate_strategy="residualize",
            n_splits=2,
            inner_splits=2,
            seed=1,
        )
        result = run_experiment(bundle, config)
        for fold in result.folds:
            params = fold["residualizer"]
            # 3 tracts x 2 metrics x 4 summary statistics: the model sees the
            # residualized anatomical columns only, covariates are dropped.
            assert params["n_fitted_features"] == 24
            assert params["n_train_rows"] == fold["n_train"]


class TestDeepMaskPropagation:
    def _profiles(self, n: int = 12, seed: int = 2) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        profiles = rng.normal(size=(n, 3, 8, 2)).astype(np.float32)
        labels = np.repeat([0, 1], n // 2)
        profiles[1, 0, 2, 1] = np.nan
        profiles[2, 2, :, :] = np.nan
        profiles[5, :, 7, 0] = np.nan
        return profiles, labels

    def test_transformer_output_is_invariant_to_fill_under_fixed_mask(self) -> None:
        _domain_train()  # importorskip: the transformer sits behind torch
        import torch

        from dit.models.tract_transformer import TractTransformer

        torch.manual_seed(0)
        model = TractTransformer(
            n_tracts=3, n_points=8, n_metrics=2, n_classes=2, d_model=16, n_heads=4,
            n_layers=1, feedforward_dim=32, include_covariates=False,
        )
        model.eval()
        rng = np.random.default_rng(6)
        profiles = rng.normal(size=(4, 3, 8, 2)).astype(np.float32)
        profiles[1, 0, 2, 1] = np.nan
        profiles[2, 2, :, :] = np.nan
        mask = torch.isfinite(torch.as_tensor(profiles))
        filled_a = torch.as_tensor(np.nan_to_num(profiles, nan=1.37))
        filled_b = torch.as_tensor(np.nan_to_num(profiles, nan=-9.1))
        with torch.no_grad():
            reference = model(torch.as_tensor(profiles))
            variant_a = model(filled_a, valid_mask=mask)
            variant_b = model(filled_b, valid_mask=mask)
        assert torch.allclose(reference, variant_a, atol=1e-6)
        assert torch.allclose(variant_a, variant_b, atol=1e-6)

    def test_classifier_forwards_explicit_masks_on_every_call(self) -> None:
        domain_train = _domain_train()
        DomainAlignedClassifier = domain_train.DomainAlignedClassifier
        DomainTrainConfig = domain_train.DomainTrainConfig

        from dit.models.tract_transformer import TractTransformer

        profiles, labels = self._profiles(n=16)
        captured: list[object] = []
        original = TractTransformer.forward

        def spy(self, x, *, valid_mask=None, **kwargs):
            captured.append(valid_mask)
            return original(self, x, valid_mask=valid_mask, **kwargs)

        with mock.patch.object(TractTransformer, "forward", spy):
            classifier = DomainAlignedClassifier(
                DomainTrainConfig(
                    epochs=2, batch_size=4, patience=2, seed=1,
                    alignment="coral", calibration="temperature",
                    validation_fraction=0.25,
                )
            )
            classifier.fit(profiles, labels)
            classifier.predict_proba(profiles)

        assert captured
        assert all(mask is not None for mask in captured)


class TestHoldoutRoles:
    def test_calibration_halves_cover_every_class(self) -> None:
        domain_train = _domain_train()
        DomainAlignedClassifier = domain_train.DomainAlignedClassifier
        DomainTrainConfig = domain_train.DomainTrainConfig

        labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
        classifier = DomainAlignedClassifier(
            DomainTrainConfig(validation_fraction=0.3, calibration="temperature")
        )
        train, validation, calibration = classifier._split_indices(labels)
        assert not set(train) & set(validation)
        assert not set(train) & set(calibration)
        for label in (0, 1, 2):
            assert np.any(labels[validation] == label)
            assert np.any(labels[calibration] == label)

    def test_single_class_holdout_row_cannot_serve_both_roles(self) -> None:
        domain_train = _domain_train()
        DomainAlignedClassifier = domain_train.DomainAlignedClassifier
        DomainTrainConfig = domain_train.DomainTrainConfig

        labels = np.array([0] * 20 + [1] * 3)
        classifier = DomainAlignedClassifier(
            DomainTrainConfig(validation_fraction=0.2, calibration="temperature")
        )
        with pytest.raises(ValueError, match="both early stopping and calibration"):
            classifier._split_indices(labels)

    def test_class_weights_come_from_training_rows_only(self) -> None:
        domain_train = _domain_train()
        DomainAlignedClassifier = domain_train.DomainAlignedClassifier
        DomainTrainConfig = domain_train.DomainTrainConfig
        _class_weights = domain_train._class_weights

        profiles = np.random.default_rng(9).normal(size=(40, 2, 6, 2)).astype(np.float32)
        # Heavily skewed training set; the holdout must not soften it.
        labels_full = np.array([0] * 32 + [1] * 8)
        config = DomainTrainConfig(
            epochs=1, batch_size=8, patience=1, seed=5, class_weights=True
        )
        classifier = DomainAlignedClassifier(config).fit(profiles, labels_full)

        train, _, _ = classifier._split_indices(labels_full)
        expected = _class_weights(
            np.asarray([labels_full[row] for row in train.tolist()]), 2, classifier.device
        ).cpu().numpy().tolist()
        all_rows = _class_weights(labels_full, 2, classifier.device).cpu().numpy().tolist()
        assert classifier.class_weights_ == pytest.approx(expected)
        # The skew is strong enough that holdout labels *would* have mattered.
        assert all_rows != pytest.approx(expected)


class TestEpochBudget:
    def test_search_respects_the_configured_epoch_budget(self) -> None:
        domain_train = _domain_train()
        DomainTrainConfig = domain_train.DomainTrainConfig
        search_domain_classifier = domain_train.search_domain_classifier

        bundle = make_synthetic_bundle(n_samples=40, n_tracts=2, n_points=8, n_metrics=2, seed=3)
        labels = (bundle.y == 2).astype(int)
        winner, report = search_domain_classifier(
            bundle.X,
            labels,
            config=DomainTrainConfig(epochs=2, batch_size=8, patience=1, seed=1),
        )
        assert "epochs" not in report["best_params"]
        assert "learning_rate" in report["best_params"]
        assert winner.parameters()["epochs_requested"] == 2
        assert winner.parameters()["epochs_run"] <= 2


class TestScorerAndClassSpaceContracts:
    def test_align_probabilities_rejects_incomplete_class_sets(self) -> None:
        probabilities = np.array([[0.7, 0.3]])
        with pytest.raises(ValueError, match="never saw class"):
            align_probabilities(probabilities[:, :1], [0], 2)
        with pytest.raises(ValueError, match="outside the task range"):
            align_probabilities(np.array([[0.7, 0.3]]), [0, 2], 2)
        with pytest.raises(ValueError, match="duplicated class"):
            align_probabilities(np.array([[0.7, 0.3]]), [1, 1], 2)
        aligned = align_probabilities(np.array([[0.3, 0.7]]), [1, 0], 2)
        assert aligned.tolist() == [[0.7, 0.3]]

    def test_inner_validation_slices_must_cover_every_class(self) -> None:
        from sklearn.model_selection import GroupKFold

        from dit.evaluation.experiment import _require_inner_class_support

        y = np.array([0, 0, 1, 1])
        groups = np.array([0, 0, 1, 1])
        cv = GroupKFold(n_splits=2)
        with pytest.raises(ValueError, match="partial class set"):
            _require_inner_class_support(cv, y, groups)

        y_ok = np.array([0, 1, 0, 1])
        _require_inner_class_support(cv, y_ok, groups)

    def test_aggregate_declares_selection_metric_and_prediction_rule(self) -> None:
        from dit.evaluation.experiment import ExperimentConfig, run_experiment

        bundle = make_synthetic_bundle(n_samples=60, n_tracts=3, n_points=10, n_metrics=2, seed=2)
        config = ExperimentConfig(
            model="logistic", n_splits=2, inner_splits=2, seed=1
        )
        result = run_experiment(bundle, config)
        assert result.aggregate["selection_metric"] == "balanced_accuracy"
        assert result.aggregate["prediction_rule"] == "predict_proba_argmax"

    def test_calibrated_svm_predict_agrees_with_proba_argmax(self) -> None:
        matrix = np.random.default_rng(0).normal(size=(60, 5))
        labels = np.repeat([0, 1], 30)
        search = make_search_estimator("linear_svm", cv=3)
        search.fit(matrix, labels)
        probabilities = search.predict_proba(matrix)
        assert np.array_equal(search.predict(matrix), probabilities.argmax(axis=1))

    def test_calibration_wrapper_contains_the_full_pipeline(self) -> None:
        if tuple(int(part) for part in _sklearn().__version__.split(".")[:2]) < (1, 9):
            pytest.skip("calibration-wraps-pipeline layout requires sklearn >= 1.9")
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        matrix = np.random.default_rng(1).normal(size=(60, 5))
        labels = np.repeat([0, 1], 30)
        search = make_search_estimator("linear_svm", cv=2)
        search.fit(matrix, labels)
        wrapper = search.best_estimator_.named_steps["model"]
        inner = wrapper.estimator
        assert isinstance(inner, Pipeline)
        assert "imputer" in inner.named_steps
        assert isinstance(inner.named_steps["scaler"], StandardScaler)
        # Grid keys reach the inner estimator through the nested pipeline.
        assert any(key.startswith("model__estimator__model__") for key in search.param_grid)
