"""Tests for cross-model soft voting.

The important property is not that the ensemble scores higher, but that every
averaged probability stays out of fold for the subject it describes and that
the weights come from inside the same outer fold.  Both are checked here by
construction rather than by eyeballing a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from dit.data.synthetic import make_synthetic_bundle
from dit.evaluation.experiment import (
    DEFAULT_ENSEMBLE_MODELS,
    ExperimentConfig,
    _base_config_for_ensemble,
    _fold_weights,
    run_configured_experiment,
    run_ensemble,
)

BASELINE = ExperimentConfig(
    model="ensemble",
    task="binary",
    n_splits=2,
    inner_splits=2,
    ensemble_models=("linear_svm", "logistic"),
    seed=1,
)


class TestFoldWeights:
    def test_equal_weighting_is_uniform(self) -> None:
        weights = _fold_weights([0.9, 0.1, 0.6], "equal")
        assert np.allclose(weights, 1.0 / 3)

    def test_inner_score_weighting_is_normalised(self) -> None:
        weights = _fold_weights([0.8, 0.4, 0.2], "inner_score")
        assert weights.sum() == pytest.approx(1.0)
        assert weights[0] > weights[1] > weights[2]

    def test_uninformative_models_get_no_weight(self) -> None:
        weights = _fold_weights([0.5, 0.0, -0.2], "inner_score")
        assert weights[1] == 0.0
        assert weights[2] == 0.0
        assert weights[0] == pytest.approx(1.0)

    def test_all_uninformative_falls_back_to_equal(self) -> None:
        weights = _fold_weights([0.0, -0.1, np.nan], "inner_score")
        assert np.allclose(weights, 1.0 / 3)

    def test_nan_scores_do_not_contaminate(self) -> None:
        weights = _fold_weights([0.6, np.nan], "inner_score")
        assert np.isfinite(weights).all()
        assert weights.sum() == pytest.approx(1.0)

    def test_single_model_gets_full_weight(self) -> None:
        assert _fold_weights([0.7], "inner_score").tolist() == [1.0]


class TestEnsembleRun:
    def test_class_names_follow_the_task_view_not_the_global_table(self) -> None:
        # A binary view renumbers the labels (0=NC, 1=AD); naming them from the
        # global canonical table reported the disease class as MCI.
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=13)
        result = run_ensemble(bundle, BASELINE)
        assert result.aggregate["class_names"] == ["NC", "AD"]

    def test_every_subject_gets_a_probability(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=9)
        result = run_ensemble(bundle, BASELINE)
        # The binary view drops MCI, so the row count is the view's, not the
        # bundle's.
        n_binary = bundle.task_view("binary").n_samples
        assert result.probabilities.shape == (n_binary, 2)
        assert np.isfinite(result.probabilities).all()
        assert np.allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-5)
        assert (result.probabilities >= 0).all()

    def test_weights_are_recomputed_per_fold(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=10)
        result = run_ensemble(bundle, BASELINE)
        for fold in result.folds:
            assert set(fold["weights"]) == set(BASELINE.ensemble_models)
            assert sum(fold["weights"].values()) == pytest.approx(1.0, abs=1e-6)
            for entry in fold["base_models"]:
                assert entry["model"] in BASELINE.ensemble_models

    def test_base_models_are_reported(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=11)
        result = run_ensemble(bundle, BASELINE)
        assert set(result.aggregate["ensemble_models"]) == set(BASELINE.ensemble_models)
        assert set(result.aggregate["base_model_scores"]) == set(BASELINE.ensemble_models)
        for score in result.aggregate["base_model_scores"].values():
            assert "accuracy" in score
            assert "roc_auc" in score

    def test_equal_weighting_gives_a_valid_result(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=12)
        result = run_ensemble(bundle, replace_config(BASELINE, ensemble_weighting="equal"))
        assert np.allclose(result.probabilities.sum(axis=1), 1.0, atol=1e-5)
        assert result.aggregate["threshold_criterion"] == "f1"

    def test_weighting_schemes_agree_on_structure(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=13)
        inner = run_ensemble(bundle, BASELINE)
        equal = run_ensemble(bundle, replace_config(BASELINE, ensemble_weighting="equal"))
        assert inner.probabilities.shape == equal.probabilities.shape
        assert len(inner.folds) == len(equal.folds)
        assert inner.threshold == equal.threshold
        # Different weights must change the average unless the base models
        # disagree about nothing, so the two schemes may not coincide.
        assert not np.allclose(inner.probabilities, equal.probabilities)
        assert inner.aggregate["ensemble_weighting"] == "inner_score"
        assert equal.aggregate["ensemble_weighting"] == "equal"

    def test_adaptive_threshold_is_not_used_for_cv_performance(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=14)
        result = run_ensemble(bundle, replace_config(BASELINE, threshold_criterion="balanced"))
        argmax = np.argmax(result.probabilities, axis=1)
        assert result.aggregate["threshold_criterion"] == "balanced"
        assert result.aggregate["threshold_evaluation"] == "outer_oof_argmax"
        assert np.array_equal(result.predictions, argmax)
        for metric in ("accuracy", "balanced_accuracy", "macro_f1"):
            assert result.aggregate[metric] == pytest.approx(result.oof_argmax[metric])
        assert result.thresholded_selection_metrics["scope"] == "all_outer_oof_used_for_threshold_fit"
        assert result.thresholded_selection_metrics["not_for_performance_comparison"] is True

    def test_fixed_threshold_keeps_argmax_predictions(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=141)
        result = run_ensemble(bundle, replace_config(BASELINE, threshold_criterion="fixed"))
        assert np.array_equal(result.predictions, np.argmax(result.probabilities, axis=1))
        assert result.threshold.thresholds == (0.5, 0.5)

    def test_single_base_model_matches_a_solo_run(self) -> None:
        from dit.evaluation.experiment import run_experiment

        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=15)
        solo_config = replace_config(BASELINE, model="linear_svm", ensemble_models=("linear_svm",))
        ensemble = run_ensemble(bundle, solo_config)
        solo = run_experiment(bundle, replace_config(solo_config, model="linear_svm"))
        assert ensemble.probabilities.shape == solo.probabilities.shape
        # One base model gets weight one, so the average is that model itself.
        assert np.allclose(ensemble.probabilities, solo.probabilities, atol=1e-6)
        assert np.array_equal(ensemble.predictions, solo.predictions)

    def test_loso_and_stratified_share_the_ensemble_path(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=16)
        for strategy in ("stratified", "loso"):
            result = run_ensemble(bundle, replace_config(BASELINE, split_strategy=strategy))
            assert np.isfinite(result.probabilities).all()
            assert len(result.folds) >= 2

    def test_selection_is_still_applied_within_each_base_run(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=17)
        result = run_ensemble(bundle, replace_config(BASELINE, enable_selection=True))
        assert np.isfinite(result.probabilities).all()

    def test_key_names_the_ensemble(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=18)
        result = run_ensemble(bundle, BASELINE)
        assert result.key.startswith("ensemble|")
        assert result.model_name == "ensemble"

    def test_dispatch_selects_the_ensemble(self) -> None:
        bundle = make_synthetic_bundle(n_samples=70, n_tracts=4, n_points=16, n_metrics=2, seed=19)
        result = run_configured_experiment(bundle, BASELINE)
        assert result.model_name == "ensemble"
        solo = run_configured_experiment(bundle, replace_config(BASELINE, model="logistic"))
        assert solo.model_name == "logistic"


class TestEnsembleConfig:
    def test_defaults_are_the_documented_lineup(self) -> None:
        config = ExperimentConfig()
        assert config.ensemble_models == DEFAULT_ENSEMBLE_MODELS
        assert config.ensemble_weighting == "inner_score"

    def test_defaults_exclude_the_profile_model(self) -> None:
        # Kept out of the default lineup for cost, not for correctness: naming
        # it explicitly is supported, and the ensemble calibrates it for you.
        assert "tract_transformer" not in DEFAULT_ENSEMBLE_MODELS

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"ensemble_weighting": "greedy"},
            {"ensemble_models": ()},
            {"ensemble_models": ("linear_svm", "linear_svm")},
            {"deep_calibration": "isotonic"},
        ],
    )
    def test_invalid_settings_raise(self, kwargs: dict[str, object]) -> None:
        with pytest.raises(ValueError):
            ExperimentConfig(**kwargs)  # type: ignore[arg-type]


def replace_config(config: ExperimentConfig, **overrides: object):
    from dataclasses import replace

    return replace(config, **overrides)  # type: ignore[arg-type]


class TestDeepBaseCalibration:
    """Soft voting averages probabilities, so a deep base must be calibrated."""

    @pytest.mark.parametrize("name", ["tract_transformer", "domain_transformer", "transformer"])
    def test_a_deep_base_gets_temperature_calibration(self, name: str) -> None:
        base = _base_config_for_ensemble(ExperimentConfig(ensemble_models=(name,)), name)
        assert base.model == name
        assert base.deep_calibration == "temperature"

    def test_classical_bases_are_left_alone(self) -> None:
        # Classical models keep the unset sentinel: they never read the
        # Transformer's flags, so there is nothing to rewrite.
        base = _base_config_for_ensemble(
            ExperimentConfig(ensemble_models=("linear_svm",)), "linear_svm"
        )
        assert base.deep_calibration is None

    @pytest.mark.parametrize("choice", ["none", "temperature", "sigmoid"])
    def test_an_explicit_choice_is_never_overwritten(self, choice: str) -> None:
        base = _base_config_for_ensemble(
            ExperimentConfig(ensemble_models=("tract_transformer",), deep_calibration=choice),
            "tract_transformer",
        )
        assert base.deep_calibration == choice

    def test_the_effective_calibration_reaches_the_fold_report(self) -> None:
        pytest.importorskip("torch", reason="deep ensemble integration requires torch")
        # Auto-applied calibration is decided inside run_ensemble, so it has to
        # be published there: a reader of evaluation.json otherwise cannot tell
        # whether the averaged probabilities were calibrated at all.
        bundle = make_synthetic_bundle(n_samples=80, n_tracts=4, n_points=16, n_metrics=2, seed=21)
        config = replace_config(
            BASELINE,
            ensemble_models=("logistic", "tract_transformer"),
            deep_epochs=2,
            deep_patience=1,
        )
        result = run_ensemble(bundle, config)
        assert result.folds
        for fold in result.folds:
            entries = {entry["model"]: entry for entry in fold["base_models"]}
            assert set(entries) == {"logistic", "tract_transformer"}
            assert entries["tract_transformer"]["deep_calibration"] == "temperature"
            assert entries["logistic"]["deep_calibration"] is None

