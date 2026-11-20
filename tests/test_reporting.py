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
from dit.evaluation.site_balance import paired_fold_comparison
from types import SimpleNamespace

from dit.evaluation.reporting import (
    _fold_table,
    assemble_report,
    build_covariate_ablation_comparisons,
    render_markdown,
    write_json,
)


class TestFoldTable:
    def test_every_recorded_fold_key_reaches_the_report(self) -> None:
        # Contract reversal (2026-11-20): the table used to filter fold keys
        # through an explicit allowlist, which silently dropped real
        # diagnostics whenever a runner gained a key (inner_best_score,
        # inner_budget, residualizer, epochs_requested, weighting_fallback
        # all vanished).  The project rule is that provenance is never
        # silently dropped, so unknown keys now pass through.
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
                    "data_digest": "snapshot-0",
                    "test_class_counts": {"0": 8, "1": 7},
                    "held_out_site": "site-0",
                    "inner_cv": "site_grouped",
                    "a_new_diagnostic": "now appears",
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
            "inner_cv",
            "alignment_grid",
            "confusion_matrix",
            "n_features_used",
            "task",
            "split_strategy",
            "test_index_digest",
            "data_digest",
            "test_class_counts",
            "held_out_site",
            "a_new_diagnostic",
        ):
            assert key in row, key
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


def test_report_round_trip_preserves_data_snapshot_identity() -> None:
    def fold(data_digest: str, score: float) -> dict[str, object]:
        return {
            "fold": "0",
            "task": "binary",
            "split_strategy": "stratified",
            "test_index_digest": "indices-0",
            "data_digest": data_digest,
            "test_class_counts": {"0": 4, "1": 4},
            "n_test": 8,
            "balanced_accuracy": score,
            "accuracy": score,
            "macro_f1": score,
        }

    class FakeResult:
        def __init__(self, rows):
            self.model = "linear_svm"
            self.task = "binary"
            self.split_strategy = "stratified"
            self.feature_view = "summary/none"
            self.aggregate = {"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5}
            self.folds = (rows,)

    reference = FakeResult(fold("snapshot-a", 0.5))
    candidate = FakeResult(fold("snapshot-b", 0.6))
    payload = assemble_report(
        [reference, candidate], dataset_summary={}, configuration={}
    )
    restored = json.loads(json.dumps(payload, allow_nan=False))
    saved_reference = restored["results"][0]["folds"]
    saved_candidate = restored["results"][1]["folds"]
    assert saved_reference[0]["data_digest"] == "snapshot-a"
    assert saved_candidate[0]["data_digest"] == "snapshot-b"
    comparison = paired_fold_comparison(saved_reference, saved_candidate)
    assert comparison["status"] == "not_comparable"
    assert "data snapshot" in comparison["reason"]


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


def test_write_json_maps_non_finite_to_null(tmp_path) -> None:
    # write_json runs _json_safe before the strict dump, so a non-finite
    # float lands as null instead of either a bare NaN token or a crash --
    # the same contract assemble_report has always had at its boundary.
    write_json(tmp_path / "bad.json", {"value": float("nan")})
    restored = json.loads((tmp_path / "bad.json").read_text(encoding="utf-8"))
    assert restored["value"] is None


def test_markdown_tables_escape_data_controlled_cells() -> None:
    """A hostile feature view or model name must not break the table row.

    ``feature_view`` can be a .mat metric name and result keys embed model
    names, so pipe/backtick/bracket metacharacters must be neutralised.
    """
    hostile = SimpleNamespace(
        model="lin|ear`<b>[x](y)",
        task="bin|ary",
        split_strategy="stratif|ied",
        feature_view="MD`</td>`",
        aggregate={"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5, "roc_auc": None},
        folds=(),
    )
    payload = assemble_report([hostile], dataset_summary={}, configuration={})
    row = next(line for line in render_markdown(payload).splitlines() if line.startswith("| lin"))
    # 11 columns -> exactly 12 structural pipes (escaped \| do not split).
    assert row.replace("\|", "").count("|") == 12
    assert r"lin\|ear" in row
    assert "&lt;b&gt;" in row
    assert r"\[x\]" in row


def test_paired_comparison_table_escapes_reference_and_candidate() -> None:
    comparison = {
        "reference": "mod|el`x`",
        "candidate": "cand<y>",
        "metric": "balanced_accuracy",
        "n_pairs": 5,
        "mean_delta": 0.01,
        "p_value": 0.5,
        "p_value_holm": 0.5,
        "status": "comparable",
    }
    payload = assemble_report(
        [], dataset_summary={}, configuration={}, paired_comparisons=[comparison]
    )
    row = next(
        line
        for line in render_markdown(payload).splitlines()
        if line.startswith("| mod")
    )
    # 8 columns -> exactly 9 structural pipes (escaped \| do not split).
    assert row.replace("\|", "").count("|") == 9
    assert r"mod\|el" in row and "cand&lt;y&gt;" in row


def test_evaluation_result_to_dict_round_trips_without_torch() -> None:
    # to_dict had its only coverage inside a torch-gated test; the core CI
    # job never exercised it.
    from dit.evaluation.runner import EvaluationResult
    from dit.evaluation.threshold import ThresholdPolicy

    result = EvaluationResult(
        model="linear_svm",
        task="binary",
        split_strategy="stratified",
        feature_view="summary/feature",
        aggregate={"accuracy": 0.5},
        folds=({"fold": "0", "accuracy": 0.5},),
        predictions=np.asarray([0, 1]),
        probabilities=np.asarray([[0.6, 0.4], [0.3, 0.7]]),
        threshold=ThresholdPolicy((0.5, 0.5), (0, 1), "f1", 0.5),
        thresholded_selection_metrics={"scope": "x"},
    )
    payload = result.to_dict()
    assert payload["model"] == "linear_svm"
    assert payload["predictions"] == [0, 1]
    assert payload["threshold"]["criterion"] == "f1"
    assert payload["thresholded_selection_metrics"] == {"scope": "x"}
    json.dumps(payload, allow_nan=False)


def test_paired_comparison_caveats_cannot_forge_markdown_structure() -> None:
    # Caveats and reasons reach the report through this public API; an
    # unescaped newline would forge headings and tables in the Markdown.
    comparison = {
        "reference": "a",
        "candidate": "b",
        "metric": "balanced_accuracy",
        "n_pairs": 5,
        "mean_delta": 0.01,
        "p_value": 0.5,
        "p_value_holm": 0.5,
        "status": "tested",
        "caveats": ["ok\r\n\r\n# Injected heading\n\n| fake | table |\n|---|---|\n| a | b |"],
    }
    payload = assemble_report(
        [], dataset_summary={}, configuration={}, paired_comparisons=[comparison]
    )
    lines = render_markdown(payload).splitlines()
    # The caveat stays inside its blockquote line; no forged heading or table
    # row starts a new structural line.
    assert not any(line.startswith("# Injected") for line in lines)
    assert not any(line.startswith("| fake") for line in lines)
    caveat_line = next(line for line in lines if line.startswith("> ") and "Injected" in line)
    assert "\n" not in caveat_line


def test_fold_table_passes_through_every_recorded_diagnostic() -> None:
    # The old explicit allowlist silently dropped real fold diagnostics
    # (inner_best_score, inner_budget, residualizer, epochs_requested,
    # weighting_fallback); the table now passes unknown keys through.
    from dit.evaluation.runner import EvaluationResult

    fold = {
        "fold": "0",
        "n_train": 10,
        "n_test": 5,
        "accuracy": 0.8,
        "balanced_accuracy": 0.8,
        "macro_f1": 0.8,
        "sensitivity": 0.9,
        "specificity": 0.7,
        "inner_best_score": 0.77,
        "inner_budget": {"n_candidates": 9, "n_fits": 19},
        "weighting_fallback": True,
        "residualizer": {"include_intercept": True},
        "test_index_digest": "d" * 64,
        "task": "binary",
        "split_strategy": "stratified",
        "test_class_counts": {"0": 3, "1": 2},
        "fold_comparable": True,
    }
    result = EvaluationResult(
        model="ensemble",
        task="binary",
        split_strategy="stratified",
        feature_view="summary/feature",
        aggregate={},
        folds=(fold,),
        predictions=np.zeros(2, dtype=int),
        probabilities=np.full((2, 2), 0.5),
    )
    row = _fold_table(result)[0]
    for key in (
        "inner_best_score",
        "inner_budget",
        "weighting_fallback",
        "residualizer",
        "test_index_digest",
    ):
        assert key in row, key
    json.dumps(row, allow_nan=False)


def test_markdown_results_table_renders_null_aggregates_as_dash() -> None:
    # A non-finite aggregate metric becomes null in JSON; the Markdown table
    # must render "-" instead of crashing on float(None) after the JSON was
    # already written.
    hostile = SimpleNamespace(
        model="linear_svm",
        task="binary",
        split_strategy="stratified",
        feature_view="summary/feature",
        aggregate={
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_f1": None,
            "roc_auc": None,
            "sensitivity": None,
            "specificity": None,
        },
        folds=(),
    )
    payload = assemble_report([hostile], dataset_summary={}, configuration={})
    rendered = render_markdown(payload)
    row = next(line for line in rendered.splitlines() if line.startswith("| linear_svm"))
    assert row.count("|") == 12
    # Pin the cells, not mere presence: rendering "None"/"nan" here would
    # pass an existence check but mislead every reader.
    cells = [cell.strip() for cell in row.split("|")]
    assert cells[5:11] == ["-"] * 6
    # And the same payload's JSON keeps the keys with null values.
    serialized = json.loads(json.dumps(payload, allow_nan=False))
    aggregate = serialized["results"][0]["aggregate"]
    assert all(
        aggregate[key] is None
        for key in ("accuracy", "macro_f1", "balanced_accuracy", "roc_auc",
                    "sensitivity", "specificity")
    )


def test_fold_spread_treats_explicit_none_as_unrecorded() -> None:
    # float(None) used to crash the report assembly when a fold recorded an
    # explicit null metric.
    fold = {
        "fold": "0",
        "n_train": 10,
        "n_test": 5,
        "accuracy": None,
        "balanced_accuracy": 0.8,
        "macro_f1": 0.7,
    }
    hostile = SimpleNamespace(
        model="linear_svm",
        task="binary",
        split_strategy="stratified",
        feature_view="summary/feature",
        aggregate={"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5, "roc_auc": None},
        folds=(fold,),
    )
    payload = assemble_report([hostile], dataset_summary={}, configuration={})
    spread = payload["results"][0]["fold_metric_spread"]
    assert spread["accuracy"]["n_folds"] == 0  # None filtered as non-finite
    json.dumps(payload, allow_nan=False)


def test_results_table_finite_values_render_three_decimals() -> None:
    # Pins the template-equivalence side of the null guard: a preformatted
    # string must render exactly like the old `{value:.3f}` template did.
    hostile = SimpleNamespace(
        model="linear_svm",
        task="binary",
        split_strategy="stratified",
        feature_view="summary/feature",
        aggregate={
            "accuracy": 0.8333333,
            "balanced_accuracy": 0.8125,
            "macro_f1": 0.790123,
            "roc_auc": 0.9,
            "sensitivity": 0.88,
            "specificity": 0.75,
        },
        folds=(),
    )
    payload = assemble_report([hostile], dataset_summary={}, configuration={})
    row = next(line for line in render_markdown(payload).splitlines() if line.startswith("| linear_svm"))
    assert row == (
        "| linear_svm | binary | stratified | summary/feature | "
        "0.833 | 0.900 | 0.790 | 0.812 | 0.880 | 0.750 | 0 |"
    )
    assert row.count("|") == 12


def test_markdown_json_blocks_sanitise_non_finite_values() -> None:
    # _json_block shares the write_json boundary: an unsanitised payload with
    # a NaN section must render as null, not crash strict serialisation.
    hostile = SimpleNamespace(
        model="linear_svm",
        task="binary",
        split_strategy="stratified",
        feature_view="summary/feature",
        aggregate={"accuracy": 0.5, "balanced_accuracy": 0.5, "macro_f1": 0.5, "roc_auc": None},
        folds=(),
    )
    payload = assemble_report(
        [hostile],
        dataset_summary={"missing_fraction": float("nan"), "n_samples": 30},
        configuration={},
    )
    rendered = render_markdown(payload)
    assert "NaN" not in rendered
    assert '"missing_fraction": null' in rendered
