"""Site-balance reporting: per-site numbers must match the view they describe."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dit.data.synthetic import make_synthetic_bundle
from dit.evaluation.experiment import ExperimentConfig, run_experiment
from dit.evaluation.provenance import fold_provenance
from dit.evaluation.site_balance import (
    fold_macro_summary,
    holm_adjust,
    paired_fold_comparison,
    per_site_metrics,
    site_composition,
    stable_predictors,
)


def test_binary_view_class_names_use_the_supplied_label_map() -> None:
    # A binary view renumbers labels to 0=NC, 1=AD; the global canonical table
    # would name class 1 MCI.
    site = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([0, 1, 0, 1, 1, 0])
    rows = per_site_metrics(site, y, y, label_map={0: "NC", 1: "AD"})
    assert rows[0]["classes"] == {"NC": 2, "AD": 1}
    assert rows[1]["classes"] == {"NC": 1, "AD": 2}


def test_shape_mismatch_between_view_and_predictions_still_raises() -> None:
    site = np.arange(6)
    y = np.array([0, 1, 0, 1, 0, 1])
    with pytest.raises(ValueError, match="must match in shape"):
        per_site_metrics(site, y, y[:4])


def test_binary_run_reports_per_site_metrics_through_the_cli_bundle() -> None:
    # The bundle used to pass full-dataset site/labels against binary-view
    # predictions, and the ValueError fallback silently replaced per-site
    # metrics with an "incomplete metadata" note.
    from dit.cli.main import _bundle

    dataset = make_synthetic_bundle(n_samples=60, n_sites=3, n_classes=3, seed=42)
    config = ExperimentConfig(
        model="linear_svm",
        task="binary",
        split_strategy="stratified",
        feature_view="summary",
        n_splits=2,
        inner_splits=2,
        seed=42,
    )
    result = run_experiment(dataset, config)
    for fold in result.folds:
        assert fold["task"] == "binary"
        assert fold["split_strategy"] == "stratified"
        assert len(fold["test_index_digest"]) == 64
        assert sum(fold["test_class_counts"].values()) == fold["n_test"]
        assert "held_out_site" not in fold
    payload = _bundle([result], dataset, {"task": "binary"})
    assert "note" not in (payload.get("site_composition") or {})
    assert all(
        set(row["classes"]) == {"NC", "AD"}
        for row in payload["site_composition"]["sites"]
    )
    per_site = payload["per_site"][result.key]
    assert len(per_site) == 3
    assert all(row["classes"].keys() == {"NC", "AD"} for row in per_site)


def test_site_composition_flags_degenerate_and_thin_sites() -> None:
    site = np.array([0, 0, 0, 0, 0, 1, 1])
    y = np.array([0, 0, 0, 0, 1, 1, 1])
    report = site_composition(site, y)
    # Site 0 holds one MCI-style member (thin but usable); site 1 holds a
    # single class (its accuracy would be meaningless).
    assert report["degenerate_sites"] == ["1"]
    assert report["thin_sites"] == ["0"]


def test_site_composition_reports_missing_expected_classes() -> None:
    site = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y = np.array([0, 1, 2, 0, 2, 0, 1, 1, 2])
    report = site_composition(site, y, min_per_class=1)
    site_one = next(row for row in report["sites"] if row["site"] == "1")
    assert site_one["classes"] == {"NC": 2, "MCI": 0, "AD": 1}
    assert site_one["missing_classes"] == ["MCI"]
    assert site_one["thin"] is True


def test_site_composition_uses_the_supplied_label_map() -> None:
    # A two-class source has canonical 1=AD; the global table would name it MCI.
    site = np.array([0, 0, 0, 1, 1, 1])
    y = np.array([0, 1, 0, 1, 1, 0])
    report = site_composition(site, y, label_map={0: "NC", 1: "AD"})
    assert set(report["sites"][0]["classes"]) == {"NC", "AD"}


def test_bundle_names_binary_source_sites_without_mci() -> None:
    # End-to-end: a genuine two-class bundle must not have its AD class
    # reported as MCI in the site-composition block of the report.
    from dit.cli.main import _bundle

    dataset = make_synthetic_bundle(n_samples=60, n_sites=3, n_classes=2, seed=7)
    config = ExperimentConfig(
        model="linear_svm",
        task="multiclass",
        split_strategy="stratified",
        feature_view="summary",
        n_splits=2,
        inner_splits=2,
        seed=7,
    )
    result = run_experiment(dataset, config)
    payload = _bundle([result], dataset, {"task": "multiclass"})
    for row in payload["site_composition"]["sites"]:
        assert set(row["classes"]) <= {"NC", "AD"}


def test_bundle_rejects_mixed_task_views() -> None:
    # Binary drops MCI while multiclass keeps it, so one site table cannot
    # honestly describe both result cohorts.
    from dit.cli.main import _bundle

    dataset = make_synthetic_bundle(n_samples=60, n_sites=3, n_classes=3, seed=8)
    binary = run_experiment(
        dataset,
        ExperimentConfig(model="linear_svm", task="binary", n_splits=2, inner_splits=2, seed=8),
    )
    multiclass = run_experiment(
        dataset,
        ExperimentConfig(model="linear_svm", task="multiclass", n_splits=2, inner_splits=2, seed=8),
    )
    with pytest.raises(ValueError, match="one task"):
        _bundle([binary, multiclass], dataset, {})


def _fold(
    identifier: str,
    metric: float,
    *,
    digest: str | None = None,
    task: str = "binary",
    strategy: str = "loso",
    counts: dict[str, int] | None = None,
) -> dict[str, object]:
    return {
        "fold": identifier,
        "task": task,
        "split_strategy": strategy,
        "test_index_digest": digest or f"digest-{identifier}",
        "test_class_counts": counts or {"0": 4, "1": 4},
        "held_out_site": identifier if strategy == "loso" else None,
        "n_test": 8,
        "balanced_accuracy": metric,
    }


def test_fold_provenance_is_non_identifying_and_stable() -> None:
    labels = np.array([0, 1, 0, 1])
    first = fold_provenance(
        np.array([3, 1]), labels, fold_id="site-a", task="binary", split_strategy="loso",
        site=np.array(["a", "b", "a", "b"]),
    )
    second = fold_provenance(
        np.array([1, 3]), labels, fold_id="site-a", task="binary", split_strategy="loso",
        site=np.array(["a", "b", "a", "b"]),
    )
    assert first["test_index_digest"] == second["test_index_digest"]
    assert first["held_out_site"] == "b"
    assert first["test_class_counts"] == {"0": 0, "1": 2}
    assert "subject" not in json.dumps(first).lower()


def test_paired_comparison_joins_by_id_digest_not_input_order() -> None:
    reference = [_fold("0", 0.50), _fold("1", 0.60), _fold("2", 0.70)]
    candidate = [_fold("2", 0.90), _fold("0", 0.60), _fold("1", 0.50)]
    result = paired_fold_comparison(reference, candidate)
    assert result["status"] == "tested"
    assert [pair["fold"] for pair in result["pairs"]] == ["0", "1", "2"]
    assert result["mean_delta"] == pytest.approx(1 / 15)
    assert result["wins"] == 2
    assert result["losses"] == 1


def test_paired_comparison_rejects_different_manifest_or_protocol() -> None:
    reference = [_fold("0", 0.5), _fold("1", 0.6)]
    wrong_digest = [_fold("0", 0.6, digest="different"), _fold("1", 0.7)]
    result = paired_fold_comparison(reference, wrong_digest)
    assert result["status"] == "not_comparable"
    assert "digest" in str(result["reason"])

    different_protocol = [_fold("0", 0.6, strategy="stratified"), _fold("1", 0.7, strategy="stratified")]
    result = paired_fold_comparison(reference, different_protocol)
    assert result["status"] == "not_comparable"
    assert "task or split strategy" in str(result["reason"])


def test_paired_comparison_excludes_missing_class_and_stays_json_safe() -> None:
    reference = [_fold("0", 0.5), _fold("1", 0.6, counts={"0": 8, "1": 0})]
    candidate = [_fold("0", 0.6), _fold("1", 0.7, counts={"0": 8, "1": 0})]
    result = paired_fold_comparison(reference, candidate)
    assert result["status"] == "not_tested"
    assert result["excluded_units"][0]["fold"] == "1"
    assert json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    ("n_folds", "expected_minimum"), [(5, 0.0625), (7, 0.015625)]
)
def test_paired_comparison_reports_exact_small_sample_resolution(
    n_folds: int, expected_minimum: float
) -> None:
    reference = [_fold(str(index), 0.5) for index in range(n_folds)]
    candidate = [_fold(str(index), 0.6) for index in range(n_folds)]
    result = paired_fold_comparison(reference, candidate)
    assert result["status"] == "tested"
    assert result["min_attainable_p"] == pytest.approx(expected_minimum)
    assert result["p_value"] == pytest.approx(expected_minimum)


def test_paired_comparison_all_zero_is_not_tested_and_json_safe() -> None:
    reference = [_fold(str(index), 0.5) for index in range(5)]
    result = paired_fold_comparison(reference, list(reversed(reference)))
    assert result["status"] == "not_tested"
    assert result["p_value"] == 1.0
    assert result["n_nonzero"] == 0
    assert json.dumps(result, allow_nan=False)


def test_holm_adjustment_is_monotonic_and_never_below_raw_p() -> None:
    comparisons = [
        {"status": "tested", "p_value": 0.04},
        {"status": "tested", "p_value": 0.01},
        {"status": "not_tested", "p_value": None},
    ]
    adjusted = holm_adjust(comparisons)
    valid = sorted(
        (item["p_value"], item["p_value_holm"])
        for item in adjusted
        if item["p_value"] is not None
    )
    assert valid[0][1] >= valid[0][0]
    assert valid[1][1] >= valid[1][0]
    assert valid[0][1] <= valid[1][1]
    assert adjusted[2]["p_value_holm"] is None


def test_fold_macro_summary_excludes_incomparable_folds() -> None:
    folds = [
        {"balanced_accuracy": 0.8, "fold_comparable": True, "held_out_site": "0"},
        {"balanced_accuracy": 0.6, "fold_comparable": True, "held_out_site": "1"},
        {"balanced_accuracy": 0.2, "fold_comparable": False, "held_out_site": "2"},
    ]
    summary = fold_macro_summary(folds, metrics=("balanced_accuracy",))
    assert summary["n_folds"] == 3
    assert summary["n_comparable_folds"] == 2
    assert summary["incomparable_folds"] == ["2"]
    # The 0.2 incomparable fold must not drag the unweighted mean down.
    assert summary["per_metric"]["balanced_accuracy"]["mean"] == pytest.approx(0.7)


def test_stable_predictors_counts_selection_frequency_across_folds() -> None:
    folds = [
        {"selection": {"retained_blocks": ["A", "B"], "retained_nodes": {"A": [1, 2]}}},
        {"selection": {"retained_blocks": ["A"], "retained_nodes": {"A": [1]}}},
        {"accuracy": 0.9},  # a fold without selection is ignored
    ]
    result = stable_predictors(folds)
    assert result["n_selection_folds"] == 2
    top_block = result["blocks"][0]
    assert top_block["block"] == "A" and top_block["folds"] == 2
    assert top_block["frequency"] == pytest.approx(1.0)
    node_a1 = next(n for n in result["nodes"] if n["block"] == "A" and n["node"] == 1)
    assert node_a1["folds"] == 2


def test_selection_enforces_hard_feature_cap() -> None:
    dataset = make_synthetic_bundle(n_samples=60, n_sites=3, n_classes=3, seed=5)
    config = ExperimentConfig(
        model="logistic",
        task="binary",
        split_strategy="stratified",
        feature_view="profile",
        enable_selection=True,
        max_features=80,
        n_splits=2,
        inner_splits=2,
        seed=5,
    )
    result = run_experiment(dataset, config)
    for fold in result.folds:
        # 80 anatomical columns plus the two trailing covariate columns.
        assert fold["n_features_used"] <= 80 + 2
