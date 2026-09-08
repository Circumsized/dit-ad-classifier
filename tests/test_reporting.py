"""Fold serialization: provenance must survive the trip to JSON.

``run_ensemble`` decides whether a deep member was calibrated in a local
variable, and ``DomainAlignedClassifier`` reports its calibration scalar in its
own summary.  Both are invisible in ``evaluation.json`` unless the fold table
carries them through, which is what makes an averaged probability auditable.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from dit.evaluation.experiment import ExperimentConfig, ExperimentResult, run_ensemble
from dit.evaluation.threshold import ThresholdPolicy
from types import SimpleNamespace

from dit.evaluation.reporting import (
    _fold_table,
    assemble_report,
    build_covariate_ablation_comparisons,
    render_markdown,
    write_json,
)


class TestFoldTable:
    def test_provenance_keys_are_kept(self) -> None:
        class FakeResult:
            folds = [
                {
                    "fold": "0",
                    "n_train": 40,
                    "n_test": 15,
                    "accuracy": 0.8,
                    "weights": {"logistic": 0.5, "tract_transformer": 0.5},
                    "base_models": [
                        {
                            "model": "logistic",
                            "deep_calibration": None,
                            "inner_best_score": 0.8,
                            "fold_balanced_accuracy": 0.8,
                        }
                    ],
                    "training": {"calibration": "temperature", "temperature": 1.7, "epochs_run": 21},
                    "best_params": {"model__C": 2.0},
                    "tuning_score": 0.75,
                    "alignment_grid": [{"params": {"epochs": 40}, "score": 0.71}],
                    "confusion_matrix": [[9, 1], [2, 3]],
                    "n_features_used": 192,
                    "task": "binary",
                    "split_strategy": "loso",
                    "test_index_digest": "digest-0",
                    "test_class_counts": {"0": 8, "1": 7},
                    "held_out_site": "site-0",
                    "a_dropped_key": "should not appear",
                }
            ]

        rows = _fold_table(FakeResult())
        row = rows[0]
        for key in (
            "weights",
            "base_models",
            "training",
            "best_params",
            "tuning_score",
            "alignment_grid",
            "confusion_matrix",
            "n_features_used",
            "task",
            "split_strategy",
            "test_index_digest",
            "test_class_counts",
            "held_out_site",
        ):
            assert key in row, key
        assert "a_dropped_key" not in row
        # The payload must round-trip, or the report is not a report.
        assert json.loads(json.dumps(row))["base_models"][0]["model"] == "logistic"

    def test_missing_provenance_is_omitted_not_nulled(self) -> None:
        # A classical fold has no deep provenance; it must not show up as null,
        # which would read as "calibration was off" rather than "not applicable".
        class FakeResult:
            folds = [{"fold": "0", "accuracy": 0.7}]

        assert "training" not in _fold_table(FakeResult())[0]
        assert "base_models" not in _fold_table(FakeResult())[0]


def test_ensemble_fold_report_carries_the_deep_bases_calibration(tmp_path) -> None:
    pytest.importorskip("torch", reason="deep ensemble integration requires torch")
    from dit.data.synthetic import make_synthetic_bundle

    bundle = make_synthetic_bundle(n_samples=80, n_tracts=4, n_points=16, n_metrics=2, seed=21)
    config = ExperimentConfig(
        model="ensemble",
        task="binary",
        n_splits=2,
        inner_splits=2,
        ensemble_models=("logistic", "tract_transformer"),
        deep_epochs=2,
        deep_patience=1,
        seed=1,
    )
    ensemble_result = run_ensemble(bundle, config)
    rows = _fold_table(ensemble_result)
    assert rows
    for row in rows:
        entries = {entry["model"]: entry for entry in row["base_models"]}
        assert entries["tract_transformer"]["deep_calibration"] == "temperature"
        assert entries["logistic"]["deep_calibration"] is None
    payload = json.dumps(rows, indent=2)
    assert "deep_calibration" in payload

    evaluation = ensemble_result.as_evaluation_result()
    serialized = evaluation.to_dict()
    assert "threshold" in serialized
    assert serialized["threshold"]["criterion"] == config.threshold_criterion
    assert serialized["thresholded_selection_metrics"]["not_for_performance_comparison"] is True


def test_report_uses_explicit_result_policy_without_composite_key() -> None:
    from dit.evaluation.experiment import ExperimentConfig, ExperimentResult
    from dit.evaluation.threshold import ThresholdPolicy

    policy = ThresholdPolicy((0.25, 0.75), (0, 1), "balanced", 0.8)
    result = ExperimentResult(
        config=ExperimentConfig(model="linear_svm", covariate_strategy="none"),
        model_name="linear_svm",
        aggregate={"accuracy": 0.8, "balanced_accuracy": 0.8, "macro_f1": 0.8},
        folds=(),
        predictions=np.array([0, 1]),
        probabilities=np.array([[0.8, 0.2], [0.2, 0.8]]),
        threshold=policy,
    )
    payload = assemble_report(
        [result.as_evaluation_result()],
        dataset_summary={},
        configuration={},
    )

    entry = payload["results"][0]
    assert entry["threshold"] == policy.parameters()
    assert "paired_comparisons" not in payload
    assert "Decision policies" in render_markdown(payload)
    assert json.dumps(payload, allow_nan=False)


def _ablation_fold(identifier: str, score: float) -> dict[str, object]:
    return {
        "fold": identifier,
        "task": "binary",
        "split_strategy": "stratified",
        "test_index_digest": f"digest-{identifier}",
        "test_class_counts": {"0": 4, "1": 4},
        "n_test": 8,
        "balanced_accuracy": score,
    }


def _ablation_result(covariate_strategy: str, scores: list[float], **overrides):
    values = {
        "model": "linear_svm",
        "task": "binary",
        "split_strategy": "stratified",
        "feature_view": "summary",
        "covariate_strategy": covariate_strategy,
        "n_splits": 5,
        "inner_splits": 2,
        "seed": 7,
    }
    values.update(overrides)
    config = ExperimentConfig(**values)
    return SimpleNamespace(
        config=config,
        key="linear_svm|binary|stratified|summary|" + covariate_strategy + "|5",
        folds=tuple(_ablation_fold(str(index), value) for index, value in enumerate(scores)),
    )


def test_covariate_ablation_comparisons_are_predeclared_and_matched() -> None:
    feature = _ablation_result("feature", [0.5] * 5)
    none = _ablation_result("none", [0.6] * 5)
    residualize = _ablation_result("residualize", [0.4] * 5)
    # This run has a different feature view, so it has no same-setting feature
    # reference and must not join the group merely because its fold IDs match.
    unrelated = _ablation_result("none", [0.9] * 5, feature_view="profile")

    comparisons = build_covariate_ablation_comparisons(
        [feature, none, residualize, unrelated]
    )

    assert len(comparisons) == 2
    assert {item["candidate_covariate_strategy"] for item in comparisons} == {
        "none",
        "residualize",
    }
    assert all(item["metric"] == "balanced_accuracy" for item in comparisons)
    assert all(item["scope"] == "predeclared_exploratory_covariate_ablation" for item in comparisons)
    assert all(item["p_value_holm"] is not None for item in comparisons)


def test_paired_comparison_section_renders_and_serializes_strictly(tmp_path) -> None:
    feature = _ablation_result("feature", [0.5] * 5)
    candidate = _ablation_result("none", [0.6] * 5)
    comparison = build_covariate_ablation_comparisons([feature, candidate])
    policy = ThresholdPolicy((0.5, 0.5), (0, 1), "fixed", 0.5)
    result = ExperimentResult(
        config=ExperimentConfig(model="linear_svm"),
        model_name="linear_svm",
        aggregate={"accuracy": 0.8, "balanced_accuracy": 0.8, "macro_f1": 0.8},
        folds=(),
        predictions=np.array([0, 1]),
        probabilities=np.array([[0.8, 0.2], [0.2, 0.8]]),
        threshold=policy,
    )
    payload = assemble_report(
        [result.as_evaluation_result()],
        dataset_summary={},
        configuration={},
        paired_comparisons=comparison,
    )

    assert payload["paired_comparisons"] == comparison
    assert "Paired ablation comparisons" in render_markdown(payload)
    output = write_json(tmp_path / "report.json", payload)
    assert json.loads(output.read_text(encoding="utf-8"))["paired_comparisons"]


def test_write_json_refuses_nonstandard_nan(tmp_path) -> None:
    with pytest.raises(ValueError, match="Out of range"):
        write_json(tmp_path / "bad.json", {"value": float("nan")})
