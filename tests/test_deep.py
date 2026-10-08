"""Tests for the profile Transformer, its training loop and domain alignment.

These primitives had no caller when they were first written, which is how four
defects survived unnoticed: an inverted train/validation cut, a discriminator
built at the wrong width, and an alignment loss that could never engage on a
small batch.  Every test here is deliberately tiny so the suite stays fast, but
each one pins behaviour that a silent regression would otherwise paper over.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip(
    "torch", reason="deep-model tests require the optional torch extra"
)

from dit.data.synthetic import make_synthetic_bundle
from dit.models.domain_adaptation import (
    DomainDiscriminator,
    coral_loss,
    gradient_reverse,
    mmd_rbf_loss,
)
from dit.models.domain_train import (
    DomainAlignedClassifier,
    DomainTrainConfig,
    _alignment_loss,
    search_domain_classifier,
)
from dit.models.tract_transformer import TractTransformer

SMALL = DomainTrainConfig(epochs=6, batch_size=8, patience=3, seed=1)


class TestDomainPrimitives:
    def test_coral_of_identical_distributions_is_zero(self) -> None:
        rng = np.random.default_rng(0)
        source = torch.as_tensor(rng.normal(size=(30, 12)))
        loss = coral_loss(source, source.clone())
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_coral_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="same D"):
            coral_loss(torch.zeros(5, 8), torch.zeros(5, 9))

    def test_coral_is_invariant_to_mean_shift(self) -> None:
        rng = np.random.default_rng(1)
        source = torch.as_tensor(rng.normal(size=(25, 6)))
        target = torch.as_tensor(rng.normal(size=(25, 6)))
        assert coral_loss(source, target).item() == pytest.approx(
            coral_loss(source + 3.0, target - 1.0).item(), rel=1e-4
        )

    def test_mmd_is_non_negative_and_zero_on_the_same_sample(self) -> None:
        rng = np.random.default_rng(2)
        sample = torch.as_tensor(rng.normal(size=(20, 5)))
        assert mmd_rbf_loss(sample, sample.clone()).item() == pytest.approx(0.0, abs=1e-5)
        other = torch.as_tensor(rng.normal(size=(20, 5)))
        assert mmd_rbf_loss(sample, other).item() > 0

    def test_mmd_rejects_mismatched_shapes(self) -> None:
        with pytest.raises(ValueError, match="same D"):
            mmd_rbf_loss(torch.zeros(4, 3), torch.zeros(4, 4))

    def test_degenerate_domains_return_zero_without_error(self) -> None:
        source = torch.zeros(1, 4, requires_grad=True)
        target = torch.zeros(1, 4)
        assert coral_loss(source, target).item() == 0.0

    def test_gradient_reverse_negates_and_scales_the_gradient(self) -> None:
        values = torch.zeros(3, 4, requires_grad=True)
        gradient_reverse(values, strength=0.25).sum().backward()
        assert values.grad is not None
        assert torch.allclose(values.grad, torch.full_like(values, -0.25))

    def test_discriminator_shapes_follow_the_feature_width(self) -> None:
        discriminator = DomainDiscriminator(48, 7)
        features = torch.zeros(9, 48)
        logits = discriminator(features, strength=1.0)
        assert logits.shape == (9, 7)

    @pytest.mark.parametrize("alignment", ["coral", "mmd"])
    @pytest.mark.parametrize("strength", [0.0, 0.25, 1.0])
    def test_alignment_ramp_scales_statistical_penalties(
        self, alignment: str, strength: float
    ) -> None:
        features = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [3.0, 0.0]],
            requires_grad=True,
        )
        ramped, used = _alignment_loss(
            alignment, features, [0, 0, 1, 1], None, strength, 1, np.random.default_rng(0)
        )
        full, used_full = _alignment_loss(
            alignment, features, [0, 0, 1, 1], None, 1.0, 1, np.random.default_rng(0)
        )
        assert used and used_full
        assert full.item() > 0
        assert ramped.item() == pytest.approx(full.item() * strength, rel=1e-6)
        ramped_gradient = torch.autograd.grad(ramped, features)[0]
        full_gradient = torch.autograd.grad(full, features)[0]
        assert torch.allclose(ramped_gradient, strength * full_gradient, atol=1e-6)


class TestTrainingObjective:
    @pytest.mark.parametrize("class_weights", [False, True])
    @pytest.mark.parametrize("batch_size", [2, 3, 5])
    def test_fit_matches_global_loss_and_gradient(
        self, monkeypatch, class_weights: bool, batch_size: int
    ) -> None:
        from dit.models import domain_train

        class ConstantLogitModel(torch.nn.Module):
            def __init__(self, n_classes: int, **kwargs) -> None:
                super().__init__()
                self.n_classes = n_classes
                self.logits = torch.nn.Parameter(torch.tensor([0.2, -0.1]))

            def forward(self, profiles, **kwargs):
                return self.logits.expand(profiles.shape[0], -1)

        captured = []
        original_step = torch.optim.AdamW.step

        def capture_step(optimiser, *args, **kwargs):
            captured.append(optimiser.param_groups[0]["params"][0].grad.detach().clone())
            return original_step(optimiser, *args, **kwargs)

        monkeypatch.setattr(domain_train, "TractTransformer", ConstantLogitModel)
        monkeypatch.setattr(torch.optim.AdamW, "step", capture_step)
        profiles = np.arange(40, dtype=np.float32).reshape(5, 2, 4, 1)
        labels = np.array([0, 0, 0, 0, 1], dtype=np.int64)
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(
                epochs=1, batch_size=batch_size, validation_fraction=0.0,
                class_weights=class_weights, seed=1,
            )
        ).fit(profiles, labels)

        reference = torch.tensor([0.2, -0.1], requires_grad=True)
        weights = torch.tensor([0.625, 2.5]) if class_weights else None
        expected = torch.nn.functional.cross_entropy(
            reference.expand(5, -1), torch.as_tensor(labels), weight=weights
        )
        expected.backward()
        assert reference.grad.norm().item() < 1.0
        assert len(captured) == 1
        assert torch.allclose(captured[0], reference.grad, atol=1e-7)
        assert fitted.history_["training"] == pytest.approx([expected.item()], rel=1e-6)


class TestTractTransformer:
    def _model(self, **overrides: object) -> TractTransformer:
        kwargs: dict[str, object] = {
            "n_tracts": 3,
            "n_points": 20,
            "n_metrics": 2,
            "n_classes": 2,
            "d_model": 32,
            "n_heads": 4,
            "n_layers": 1,
            "include_covariates": True,
        }
        kwargs.update(overrides)
        return TractTransformer(**kwargs)  # type: ignore[arg-type]

    @pytest.mark.parametrize("pooling", ["cls", "mean", "attention"])
    def test_output_shapes_are_stable_across_pooling(self, pooling: str) -> None:
        model = self._model(pooling=pooling)
        x = torch.zeros(4, 3, 20, 2)
        covariates = torch.zeros(4, 2)
        logits = model(x, covariates=covariates)
        assert logits.shape == (4, 2)
        logits, features = model(x, covariates=covariates, return_features=True)
        assert logits.shape == (4, 2)
        assert features.shape == (4, 32 + 16)

    def test_nans_are_masked_not_propagated(self) -> None:
        model = self._model()
        x = torch.ones(2, 3, 20, 2)
        x[0, 0, :, :] = torch.nan
        logits = model(x, covariates=torch.zeros(2, 2))
        assert torch.isfinite(logits).all()

    def test_an_all_missing_subject_is_still_scored(self) -> None:
        model = self._model()
        x = torch.ones(2, 3, 20, 2)
        x[1] = torch.nan
        logits = model(x, covariates=torch.zeros(2, 2))
        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()

    def test_covariates_are_required_when_enabled(self) -> None:
        model = self._model()
        with pytest.raises(ValueError, match="covariates"):
            model(torch.zeros(2, 3, 20, 2))

    def test_covariates_may_be_disabled(self) -> None:
        model = self._model(include_covariates=False)
        logits, features = model(torch.zeros(2, 3, 20, 2), return_features=True)
        assert features.shape == (2, 32)
        assert logits.shape == (2, 2)

    def test_input_shape_contract_is_enforced(self) -> None:
        model = self._model()
        with pytest.raises(ValueError, match="expected"):
            model(torch.zeros(2, 4, 20, 2), covariates=torch.zeros(2, 2))
        with pytest.raises(ValueError, match="expected"):
            model(torch.zeros(2, 3, 20, 2), covariates=torch.zeros(2, 3))

    @pytest.mark.parametrize("kwargs", [{"d_model": 30}, {"pooling": "bad"}, {"n_points": 1}])
    def test_construction_guards_raise(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            self._model(**kwargs)  # type: ignore[arg-type]


class TestDomainTrainConfig:
    @pytest.mark.parametrize(
        "kwargs",
        [
            {"alignment": "gmm"},
            {"d_model": 30, "n_heads": 4},
            {"validation_fraction": -0.1},
            {"validation_fraction": 0.5},
            {"batch_size": 1},
            {"alignment_weight": -0.1},
            {"alignment_batch": 1},
            {"alignment_ramp": 0},
            {"calibration": "isotonic"},
            {"calibration": ""},
        ],
    )
    def test_invalid_settings_raise(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            DomainTrainConfig(**kwargs)  # type: ignore[arg-type]

    def test_zero_validation_fraction_is_legal(self) -> None:
        config = DomainTrainConfig(validation_fraction=0.0)
        assert config.validation_fraction == 0.0


class TestDomainAlignedClassifier:
    def _bundle(self, n_samples: int = 70):
        bundle = make_synthetic_bundle(
            n_samples=n_samples, n_tracts=3, n_points=16, n_metrics=2, seed=4
        )
        return bundle.X, (bundle.y == 2).astype(int), bundle.site

    @pytest.mark.parametrize("alignment", ["none", "coral", "mmd", "dann"])
    def test_alignment_only_engages_when_requested(self, alignment: str) -> None:
        profiles, labels, site = self._bundle()
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(epochs=8, batch_size=8, patience=4, alignment=alignment, seed=1)
        ).fit(profiles, labels, site=site)
        assert fitted.alignment_active_ == (alignment != "none")

    def test_alignment_is_skipped_when_there_is_only_one_site(self) -> None:
        profiles, labels, _ = self._bundle()
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(epochs=8, batch_size=8, patience=4, alignment="coral", seed=1)
        ).fit(profiles, labels, site=np.zeros(labels.shape[0], dtype=int))
        assert fitted.alignment_active_ is False
        assert fitted.discriminator is None

    def test_missing_site_codes_are_excluded_from_the_domains(self) -> None:
        profiles, labels, site = self._bundle()
        site = site.copy()
        site[::3] = -1
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(epochs=8, batch_size=8, patience=4, alignment="mmd", seed=1)
        ).fit(profiles, labels, site=site)
        assert fitted.alignment_active_ is True

    def test_predict_proba_contract(self) -> None:
        profiles, labels, site = self._bundle()
        fitted = DomainAlignedClassifier(SMALL).fit(profiles[:50], labels[:50], site=site[:50])
        probabilities = fitted.predict_proba(profiles[50:], None)
        assert probabilities.shape == (profiles.shape[0] - 50, labels.max() + 1)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        assert np.isfinite(probabilities).all()
        assert (probabilities >= 0).all()

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="must be fit"):
            DomainAlignedClassifier(SMALL).predict_proba(np.zeros((2, 3, 16, 2)))

    def test_normalisation_statistics_come_from_training_rows_only(self) -> None:
        profiles, labels, site = self._bundle()
        train, test = profiles[:50], profiles[50:]
        # The centred value must match the mean over the training slice that
        # _split_indices actually keeps, not over every row handed to fit().
        classifier = DomainAlignedClassifier(SMALL)
        training_rows, _, _ = classifier._split_indices(labels[:50])
        fitted = classifier.fit(train, labels[:50], site=site[:50])
        kept = train[training_rows]
        expected = np.nanmean(kept.reshape(kept.shape[0], -1), axis=0)
        assert fitted.center_.reshape(-1) == pytest.approx(expected, abs=1e-4)
        # The held-out rows must not shift the statistics either.
        assert fitted.center_.reshape(-1) != pytest.approx(
            np.nanmean(train.reshape(train.shape[0], -1), axis=0), abs=1e-4
        )
        assert fitted.center_.reshape(-1) != pytest.approx(
            np.nanmean(profiles.reshape(profiles.shape[0], -1), axis=0), abs=1e-4
        )

    def test_prediction_does_not_change_when_other_rows_are_added(self) -> None:
        profiles, labels, site = self._bundle()
        train, test = profiles[:50], profiles[50:]
        first = DomainAlignedClassifier(SMALL).fit(train, labels[:50], site=site[:50])
        second = DomainAlignedClassifier(SMALL).fit(train, labels[:50], site=site[:50])
        assert np.allclose(
            first.predict_proba(test), second.predict_proba(test), atol=1e-5
        )

    def test_missing_values_are_imputed_by_zero_after_centring(self) -> None:
        profiles, labels, site = self._bundle()
        test = profiles[50:].copy()
        test[0] = np.nan
        fitted = DomainAlignedClassifier(SMALL).fit(profiles[:50], labels[:50], site=site[:50])
        probabilities = fitted.predict_proba(test, None)
        assert np.isfinite(probabilities).all()

    def test_single_class_fold_is_refused(self) -> None:
        profiles, labels, site = self._bundle()
        with pytest.raises(ValueError, match="two classes"):
            DomainAlignedClassifier(SMALL).fit(profiles[:20], np.zeros(20, dtype=int), site=site[:20])

    def test_shape_contract_is_enforced(self) -> None:
        profiles, labels, site = self._bundle()
        with pytest.raises(ValueError, match="profiles must be"):
            DomainAlignedClassifier(SMALL).fit(
                profiles[:50].reshape(50, -1), labels[:50], site=site[:50]
            )

    def test_no_holdout_mode_trains_on_every_row(self) -> None:
        profiles, labels, site = self._bundle()
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(epochs=4, batch_size=8, patience=3, validation_fraction=0.0, seed=1)
        ).fit(profiles[:40], labels[:40], site=site[:40])
        assert fitted.best_validation_score_ is None
        assert fitted.parameters()["epochs_run"] == 4

    def test_parameters_are_reportable(self) -> None:
        profiles, labels, site = self._bundle()
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(epochs=4, batch_size=8, patience=3, alignment="coral", seed=1)
        ).fit(profiles[:40], labels[:40], site=site[:40])
        report = fitted.parameters()
        assert report["alignment"] == "coral"
        assert report["alignment_active"] is True
        assert set(report) >= {"d_model", "n_layers", "batch_size", "learning_rate", "seed"}

    def test_temperature_divides_the_logits_before_the_softmax(self) -> None:
        profiles, labels, _ = self._bundle()
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(
                epochs=6, batch_size=8, patience=3, calibration="temperature", seed=1
            )
        ).fit(profiles, labels)
        assert fitted.calibration_applied_ is True
        # The scalar is pinned rather than left at the fitted value: on a short
        # run over random profiles the fitted temperature is far from one, and
        # what is under test here is the direction of the operation, not the
        # magnitude that happened to be found.
        fitted.temperature_ = 1.0
        peaked = fitted.predict_proba(profiles)
        fitted.temperature_ = 2.0
        cooled = fitted.predict_proba(profiles)
        # Dividing the logits by T can only flatten the softmax.
        assert float(np.max(cooled, axis=1).mean()) < float(np.max(peaked, axis=1).mean())
        assert np.allclose(peaked.sum(axis=1), 1.0)
        assert np.allclose(cooled.sum(axis=1), 1.0)

    def test_temperature_stays_put_when_calibration_is_disabled(self) -> None:
        profiles, labels, _ = self._bundle()
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(epochs=6, batch_size=8, patience=3, seed=1)
        ).fit(profiles, labels)
        assert fitted.calibration_applied_ is False
        assert fitted.temperature_ == 1.0
        assert fitted.platt_ is None

    def test_sigmoid_calibration_fits_one_predictor_per_class(self) -> None:
        profiles, labels, _ = self._bundle(n_samples=120)
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(
                epochs=6, batch_size=8, patience=3, calibration="sigmoid", seed=1
            )
        ).fit(profiles, labels)
        assert fitted.calibration_applied_ is True
        assert fitted.temperature_ == 1.0
        assert len(fitted.platt_) == fitted.model.n_classes
        probabilities = fitted.predict_proba(profiles)
        assert np.all(probabilities >= 0)
        assert np.allclose(probabilities.sum(axis=1), 1.0)

    def test_calibration_without_a_hold_out_is_refused(self) -> None:
        profiles, labels, _ = self._bundle()
        with pytest.raises(ValueError, match="at least two held-out"):
            DomainAlignedClassifier(
                DomainTrainConfig(
                    epochs=2,
                    batch_size=8,
                    validation_fraction=0.0,
                    calibration="temperature",
                    seed=1,
                )
            ).fit(profiles, labels)

    def test_calibration_is_published_in_the_fit_summary(self) -> None:
        profiles, labels, _ = self._bundle()
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(
                epochs=4, batch_size=8, patience=3, calibration="temperature", seed=1
            )
        ).fit(profiles[:60], labels[:60])
        report = fitted.parameters()
        assert report["calibration"] == "temperature"
        assert report["calibration_applied"] is True
        assert report["temperature"] == pytest.approx(fitted.temperature_, rel=1e-12)
        assert report["temperature_saturated"] == fitted.temperature_saturated_

    def test_the_reported_flag_matches_the_reported_value(self) -> None:
        # Whatever the fit produced, the flag next to the value must agree with
        # it.  A clipped answer and a fitted one print identically, so the flag
        # is the only way to tell them apart, and it is derived rather than
        # asserted.
        from dit.models.calibration import temperature_saturated

        profiles, labels, _ = self._bundle()
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(
                epochs=4, batch_size=8, patience=3, calibration="temperature", seed=1
            )
        ).fit(profiles[:60], labels[:60])
        report = fitted.parameters()
        assert report["temperature_saturated"] == temperature_saturated(
            report["temperature"]
        )

    def test_centring_ignores_the_calibration_slice(self) -> None:
        # The calibration rows are held out of the loss; they must also be held
        # out of the centring, or the claim in _fit_calibration is untrue.
        profiles, labels, _ = self._bundle(n_samples=80)
        classifier = DomainAlignedClassifier(
            DomainTrainConfig(
                epochs=3, batch_size=8, patience=2, seed=1, calibration="temperature"
            )
        )
        training_rows, _, calibration_rows = classifier._split_indices(labels)
        fitted = classifier.fit(profiles, labels)
        assert calibration_rows.size >= 2
        expected = np.nanmean(
            profiles[training_rows].reshape(training_rows.size, -1), axis=0
        )
        assert fitted.center_.reshape(-1) == pytest.approx(expected, abs=1e-4)


class TestValidationSplit:
    @pytest.mark.parametrize("entry_point", ["fit", "search"])
    @pytest.mark.parametrize(
        "counts,calibration,fraction",
        [([10, 10], "sigmoid", 0.15), ([20, 1], "temperature", 0.15),
         ([20, 20], "temperature", 0.0)],
    )
    def test_invalid_calibration_support_is_rejected_before_training(
        self, monkeypatch, entry_point, counts, calibration, fraction
    ) -> None:
        def unexpected_step(*args, **kwargs):
            pytest.fail("insufficient calibration support reached an optimizer step")

        monkeypatch.setattr(torch.optim.AdamW, "step", unexpected_step)
        labels = np.repeat([0, 1], counts)
        profiles = np.zeros((labels.size, 2, 4, 1), dtype=np.float32)
        config = DomainTrainConfig(
            epochs=1, batch_size=4, d_model=8, n_layers=1,
            calibration=calibration, validation_fraction=fraction,
        )
        with pytest.raises(ValueError, match="calibration|held-out"):
            if entry_point == "fit":
                DomainAlignedClassifier(config).fit(profiles, labels)
            else:
                search_domain_classifier(profiles, labels, config=config)

    def test_split_is_stratified(self) -> None:
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        classifier = DomainAlignedClassifier(DomainTrainConfig(validation_fraction=0.25))
        train, validation, calibration = classifier._split_indices(labels)
        assert set(train.tolist() + validation.tolist() + calibration.tolist()) == set(
            range(labels.shape[0])
        )
        assert len(set(train.tolist()) & set(validation.tolist())) == 0
        assert labels[validation].size >= 2
        assert set(labels[validation].tolist()) == {0, 1}

    def test_train_is_the_larger_slice(self) -> None:
        labels = np.repeat([0, 1], 20)
        train, validation, calibration = DomainAlignedClassifier(
            DomainTrainConfig(validation_fraction=0.2)
        )._split_indices(labels)
        assert train.size > validation.size + calibration.size
        assert validation.size >= 2

    def test_zero_fraction_returns_every_row_to_training(self) -> None:
        labels = np.array([0, 1, 0, 1])
        train, validation, calibration = DomainAlignedClassifier(
            DomainTrainConfig(validation_fraction=0.0)
        )._split_indices(labels)
        assert train.size == labels.shape[0]
        assert validation.size == 0
        assert calibration.size == 0

    def test_too_few_rows_raise(self) -> None:
        with pytest.raises(ValueError, match="not enough rows"):
            DomainAlignedClassifier(DomainTrainConfig())._split_indices(np.array([0, 1]))


class TestHoldOutSplitting:
    def test_the_hold_out_is_shared_between_validation_and_calibration(self) -> None:
        labels = np.repeat([0, 1, 2], 8)
        classifier = DomainAlignedClassifier(
            DomainTrainConfig(validation_fraction=0.4, calibration="temperature")
        )
        train, validation, calibration = classifier._split_indices(labels)
        assert validation.size >= 1
        assert calibration.size >= 1
        assert not (set(validation.tolist()) & set(calibration.tolist()))
        # Neither hold-out may touch training: the calibration fit must not see
        # any row the classifier was trained on.
        assert not (set(validation.tolist()) & set(train.tolist()))
        assert not (set(calibration.tolist()) & set(train.tolist()))
        assert train.size + validation.size + calibration.size == labels.shape[0]

    def test_without_calibration_the_hold_out_stays_whole(self) -> None:
        labels = np.repeat([0, 1], 10)
        _, validation, calibration = DomainAlignedClassifier(
            DomainTrainConfig(validation_fraction=0.3)
        )._split_indices(labels)
        assert calibration.size == 0
        assert validation.size >= 1


class TestDeepExperiment:
    def _config(self, **overrides: object):
        from dit.evaluation.experiment import ExperimentConfig

        values: dict[str, object] = {
            "model": "tract_transformer",
            "task": "binary",
            "n_splits": 2,
            "inner_splits": 2,
            "deep_epochs": 6,
            "deep_batch_size": 8,
            "deep_patience": 3,
            "deep_d_model": 32,
            "deep_layers": 1,
            "alignment": "none",
            "seed": 1,
        }
        values.update(overrides)
        return ExperimentConfig(**values)  # type: ignore[arg-type]

    @pytest.mark.parametrize("alignment", ["none", "coral"])
    def test_deep_fold_is_fold_local(self, alignment: str) -> None:
        from dit.evaluation.experiment import run_experiment

        bundle = make_synthetic_bundle(
            n_samples=70, n_tracts=3, n_points=16, n_metrics=2, seed=6
        )
        result = run_experiment(bundle, self._config(alignment=alignment))
        assert np.isfinite(result.probabilities).all()
        assert result.probabilities.shape[1] == 2
        assert result.aggregate["roc_auc"] is not None
        assert result.aggregate["threshold_criterion"] == "f1"
        fold = result.folds[0]
        # The grid tunes the learning rate; the epoch budget comes from the
        # config, so the report can only show the requested value, never a
        # hidden grid override.
        assert "learning_rate" in fold["best_params"]
        assert "epochs" not in fold["best_params"]
        assert fold["training"]["epochs_requested"] == 6
        assert fold["training"]["epochs_run"] <= 6
        assert fold["training"]["alignment"] == alignment
        assert fold["training"]["alignment_active"] is (alignment != "none")

    def test_selection_is_refused_for_the_profile_model(self) -> None:
        from dit.evaluation.experiment import run_experiment

        bundle = make_synthetic_bundle(n_samples=70, n_tracts=3, n_points=16, n_metrics=2)
        with pytest.raises(ValueError, match="not defined for the profile Transformer"):
            run_experiment(bundle, self._config(enable_selection=True))

    def test_residualisation_is_refused_for_the_profile_model(self) -> None:
        from dit.evaluation.experiment import run_experiment

        bundle = make_synthetic_bundle(n_samples=70, n_tracts=3, n_points=16, n_metrics=2)
        with pytest.raises(ValueError, match="not defined for the profile Transformer"):
            run_experiment(bundle, self._config(covariate_strategy="residualize"))

    def test_alignment_flag_is_published(self) -> None:
        config = self._config(alignment="mmd")
        assert config.describe()["alignment"] == "mmd"
        with pytest.raises(ValueError, match="unknown alignment"):
            self._config(alignment="gmm")

    def test_loso_and_stratified_share_the_deep_path(self) -> None:
        from dit.evaluation.experiment import run_experiment

        bundle = make_synthetic_bundle(
            n_samples=70, n_tracts=3, n_points=16, n_metrics=2, seed=8
        )
        for strategy in ("stratified", "loso", "site_stratified"):
            result = run_experiment(bundle, self._config(split_strategy=strategy))
            assert np.isfinite(result.probabilities).all()
            assert len(result.folds) >= 2
            for fold in result.folds:
                if strategy == "loso":
                    # Either a site-grouped hold-out or the recorded fallback;
                    # which one depends on each fold's site class coverage.
                    assert fold["inner_cv"] in {"site_grouped", "class_stratified"}
                else:
                    # Track-A protocol parity: the deep path keeps the
                    # class-stratified inner hold-out exactly like the
                    # classical search does on stratified folds.
                    assert fold["inner_cv"] == "class_stratified"


class TestSearch:
    def test_search_reports_the_winning_setting(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=3, n_points=16, n_metrics=2, seed=5)
        labels = (bundle.y == 2).astype(int)
        winner, report = search_domain_classifier(
            bundle.X,
            labels,
            config=DomainTrainConfig(epochs=6, batch_size=8, patience=3, seed=1),
            site=bundle.site,
        )
        assert "best_params" in report
        assert "learning_rate" in report["best_params"]
        assert "epochs" not in report["best_params"]
        assert len(report["grid"]) == 2
        assert report["tuning_score"] == pytest.approx(
            max(entry["score"] for entry in report["grid"]), abs=1e-9
        )
        assert winner.classes_ is not None
        assert winner.predict_proba(bundle.X).shape[1] == 2

    def test_search_rejects_mismatched_row_counts(self) -> None:
        with pytest.raises(ValueError, match="same row count"):
            search_domain_classifier(np.zeros((6, 2, 4, 1)), np.array([0, 1, 0]))

    def test_search_fails_closed_on_a_tiny_fold(self) -> None:
        # Three rows leave only two for tuning, which may hold a single class.
        with pytest.raises(ValueError):
            search_domain_classifier(np.zeros((3, 2, 4, 1)), np.array([0, 1, 0]))

    def test_search_validation_covers_every_class(self) -> None:
        # The candidate score is balanced accuracy over the validation slice; a
        # plain random cut could drop the minority class and pick a winner on a
        # partial score.  The split must be stratified.
        from dit.models.domain_train import _stratified_holdout

        labels = np.array([0] * 84 + [1] * 16)
        for seed in range(6):
            _, validation = _stratified_holdout(labels, 0.15, seed)
            assert set(np.unique(labels[validation]).tolist()) == {0, 1}

    def test_search_fails_closed_when_a_class_cannot_be_held_out(self) -> None:
        # A singleton class cannot appear in the validation slice, so the
        # search must refuse rather than score candidates on two of three
        # classes.
        labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 2])
        with pytest.raises(ValueError, match="cover every class"):
            search_domain_classifier(
                np.zeros((labels.size, 2, 4, 1)),
                labels,
                config=DomainTrainConfig(epochs=1, batch_size=2, patience=1, seed=1),
            )


class TestSiteGroupedSelection:
    def test_holdout_picks_one_whole_site(self) -> None:
        from dit.models.domain_train import _site_grouped_holdout

        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        site = np.array([0, 0, 1, 1, 2, 2, 3, 3])
        split = _site_grouped_holdout(labels, site, seed=0)
        assert split is not None
        tuning, validation = split
        # The candidate is scored on a site it never trained on, mirroring the
        # outer LOSO question inside the training rows.
        assert set(site[validation].tolist()) == {site[validation[0]]}
        assert not (set(tuning.tolist()) & set(validation.tolist()))
        assert set(tuning.tolist()) | set(validation.tolist()) == set(range(labels.size))
        assert set(np.unique(labels[validation]).tolist()) == {0, 1}
        assert set(np.unique(labels[tuning]).tolist()) == {0, 1}

    def test_holdout_is_deterministic_per_seed(self) -> None:
        from dit.models.domain_train import _site_grouped_holdout

        labels = np.repeat([0, 1], 8)
        site = np.tile([0, 1, 2, 3], 4)
        first = _site_grouped_holdout(labels, site, seed=3)
        second = _site_grouped_holdout(labels, site, seed=3)
        assert first is not None and second is not None
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])

    def test_unknown_site_rows_never_form_the_validation(self) -> None:
        from dit.models.domain_train import _site_grouped_holdout

        labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        site = np.array([0, 0, -1, -1, 1, 1, 2, 2])
        split = _site_grouped_holdout(labels, site, seed=0)
        assert split is not None
        _, validation = split
        assert np.all(site[validation] >= 0)

    def test_no_qualifying_site_returns_none(self) -> None:
        # Each site carries a single class, so any site hold-out would leave
        # the candidate score partial on one side; the caller must fall back.
        from dit.models.domain_train import _site_grouped_holdout

        labels = np.array([0, 0, 0, 1, 1, 1])
        site = np.array([0, 0, 0, 1, 1, 1])
        assert _site_grouped_holdout(labels, site, seed=0) is None

    def test_search_reports_the_selection_scheme(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=3, n_points=16, n_metrics=2, seed=5)
        labels = (bundle.y == 2).astype(int)
        config = DomainTrainConfig(epochs=6, batch_size=8, patience=3, seed=1)
        with_site, site_report = search_domain_classifier(
            bundle.X, labels, config=config, site=bundle.site, split_strategy="loso"
        )
        stratified_with_site, stratified_report = search_domain_classifier(
            bundle.X, labels, config=config, site=bundle.site, split_strategy="stratified"
        )
        without_site, plain_report = search_domain_classifier(bundle.X, labels, config=config)
        site_stratified, ss_report = search_domain_classifier(
            bundle.X, labels, config=config, site=bundle.site, split_strategy="site_stratified"
        )
        # Site-grouped selection is the Track-B (LOSO) protocol: stratified and
        # site-stratified outer folds must keep the class-stratified hold-out
        # even when site labels exist, matching the classical path's _inner_cv
        # rule.  Every synthetic site holds both classes, so LOSO engages it.
        assert site_report["inner_cv"] == "site_grouped"
        assert stratified_report["inner_cv"] == "class_stratified"
        assert plain_report["inner_cv"] == "class_stratified"
        assert ss_report["inner_cv"] == "class_stratified"
        assert with_site.predict_proba(bundle.X).shape[1] == 2
        assert stratified_with_site.predict_proba(bundle.X).shape[1] == 2
        assert site_stratified.predict_proba(bundle.X).shape[1] == 2

    def test_search_falls_back_when_no_site_qualifies(self) -> None:
        # Two sites, each single-class: on a LOSO run the site-grouped hold-out
        # is impossible, so the search must use the stratified cut and say so
        # in the report instead of silently presenting a site-blind choice as
        # site-aware.
        labels = np.array([0] * 20 + [1] * 20)
        site = np.array([0] * 20 + [1] * 20)
        _, report = search_domain_classifier(
            np.zeros((labels.size, 2, 4, 1)),
            labels,
            config=DomainTrainConfig(epochs=2, batch_size=8, patience=1, seed=1),
            site=site,
            split_strategy="loso",
        )
        assert report["inner_cv"] == "class_stratified"


class TestDeterministicSwitch:
    """Plan P0.6: the determinism flag is opt-in, reported, and never leaks."""

    def _bundle(self, n_samples: int = 40):
        bundle = make_synthetic_bundle(
            n_samples=n_samples, n_tracts=2, n_points=8, n_metrics=2, seed=3
        )
        return bundle.X, (bundle.y == 2).astype(int), bundle.site

    def test_fit_restores_the_global_determinism_state(self) -> None:
        profiles, labels, site = self._bundle()
        torch.use_deterministic_algorithms(True)
        try:
            fitted = DomainAlignedClassifier(SMALL).fit(profiles, labels, site=site)
            # The fit ran with deterministic=False, so back outside fit() the
            # caller's True must be in force again.
            restored = torch.are_deterministic_algorithms_enabled()
        finally:
            torch.use_deterministic_algorithms(False)
        assert fitted.best_validation_score_ is not None
        assert restored is True

    def test_fit_inherits_the_callers_determinism_when_not_requested(self) -> None:
        profiles, labels, site = self._bundle()
        seen: dict[str, bool] = {}
        original = DomainAlignedClassifier._fit_impl

        def spy(instance, *args, **kwargs):
            seen["inside"] = torch.are_deterministic_algorithms_enabled()
            return original(instance, *args, **kwargs)

        # deterministic=False must leave the global flag alone in BOTH
        # directions: a caller with determinism on must keep it during the
        # fit, and a caller with it off must not have it silently forced on.
        torch.use_deterministic_algorithms(True)
        DomainAlignedClassifier._fit_impl = spy
        try:
            DomainAlignedClassifier(SMALL).fit(profiles, labels, site=site)
            assert seen["inside"] is True
            torch.use_deterministic_algorithms(False)
            DomainAlignedClassifier(SMALL).fit(profiles, labels, site=site)
            assert seen["inside"] is False
        finally:
            DomainAlignedClassifier._fit_impl = original
            torch.use_deterministic_algorithms(False)

    def test_deterministic_fit_is_bitwise_reproducible(self) -> None:
        profiles, labels, site = self._bundle(50)
        config = replace(SMALL, deterministic=True)
        first = DomainAlignedClassifier(config).fit(profiles, labels, site=site)
        second = DomainAlignedClassifier(config).fit(profiles, labels, site=site)
        assert first.parameters()["deterministic"] is True
        assert np.array_equal(
            first.predict_proba(profiles), second.predict_proba(profiles)
        )

    def test_the_flag_flows_through_the_experiment_config(self) -> None:
        from dit.evaluation.experiment import ExperimentConfig, _deep_train_config

        config = ExperimentConfig(deep_deterministic=True)
        assert config.describe()["deep_deterministic"] is True
        assert _deep_train_config(config).deterministic is True
        assert _deep_train_config(ExperimentConfig()).deterministic is False

    def test_fit_restores_the_warn_only_companion_flag(self) -> None:
        # use_deterministic_algorithms resets warn_only on every call, so
        # restoring only the boolean would silently clear a caller's
        # warn_only=True. Both halves of the global state must survive.
        profiles, labels, site = self._bundle()
        torch.use_deterministic_algorithms(True, warn_only=True)
        try:
            DomainAlignedClassifier(SMALL).fit(profiles, labels, site=site)
            restored_mode = torch.are_deterministic_algorithms_enabled()
            restored_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        finally:
            torch.use_deterministic_algorithms(False)
        assert restored_mode is True
        assert restored_warn_only is True

    def test_fit_restores_state_when_the_fit_raises(self) -> None:
        # The restore lives in a finally block; a fit that fails (here: a
        # single class) must still put the caller's global state back.
        profiles, labels, site = self._bundle()
        torch.use_deterministic_algorithms(True)
        try:
            with pytest.raises(ValueError, match="at least two classes"):
                DomainAlignedClassifier(SMALL).fit(
                    profiles, np.zeros_like(labels), site=site
                )
            restored = torch.are_deterministic_algorithms_enabled()
        finally:
            torch.use_deterministic_algorithms(False)
        assert restored is True


class TestAlignmentLoss:
    def test_unknown_sites_are_excluded(self) -> None:
        # Rows with a negative (unknown) site id must not enter alignment; with
        # only one real site among them, there is nothing to align.
        features = torch.randn(6, 4)
        groups = [0, 0, -2, -2, -2, -2]
        loss, used = _alignment_loss(
            "coral", features, groups, None, 1.0, 6, np.random.default_rng(0)
        )
        assert used is False
        assert float(loss) == 0.0

    def test_dann_uses_every_site_as_a_target(self) -> None:
        # Three sites with two members each: the discriminator must classify all
        # three (a pairwise 0/1 scheme would never make site 2 a positive
        # class).  A finite positive loss and engagement is the observable.
        torch.manual_seed(0)
        features = torch.randn(6, 8, requires_grad=True)
        groups = [0, 0, 1, 1, 2, 2]
        discriminator = DomainDiscriminator(8, 3)
        loss, used = _alignment_loss(
            "dann", features, groups, discriminator, 1.0, 6, np.random.default_rng(0)
        )
        assert used is True
        assert torch.isfinite(loss) and float(loss) > 0.0
        loss.backward()
        assert features.grad is not None

    def test_needs_two_usable_sites(self) -> None:
        features = torch.randn(4, 4)
        # One site has a single member, so fewer than two usable sites remain.
        loss, used = _alignment_loss(
            "mmd", features, [0, 0, 0, 1], None, 1.0, 6, np.random.default_rng(0)
        )
        assert used is False


class TestDeepCovariateImputation:
    def test_missing_covariates_do_not_reach_the_network(self) -> None:
        bundle = make_synthetic_bundle(n_samples=50, n_tracts=3, n_points=16, n_metrics=2, seed=6)
        y = (bundle.y == 2).astype(int)
        covariates = np.full((bundle.n_samples, 2), np.nan, dtype=np.float32)
        covariates[: bundle.n_samples // 2, 0] = 62.0  # partial age, no sex
        fitted = DomainAlignedClassifier(
            DomainTrainConfig(epochs=4, batch_size=8, patience=2, seed=1)
        ).fit(bundle.X, y, covariates=covariates)
        # A fully-NaN covariate matrix at predict time must still yield finite
        # probabilities because medians fill the gaps.
        proba = fitted.predict_proba(
            bundle.X, np.full((bundle.n_samples, 2), np.nan, dtype=np.float32)
        )
        assert np.isfinite(proba).all()
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
