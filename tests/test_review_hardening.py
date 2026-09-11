"""Regression tests for review hardening fixes.

Covers two findings from the 2026-09 code review:
- ``_json_safe`` must neutralize numpy integer/boolean scalars, not just
  floats, or a later report field crashes strict JSON serialization.
- ``evaluate_metric_ensemble`` must fail as loudly as ``evaluate_classical``
  when an outer fold leaves a subject unpredicted.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import dit.evaluation.runner as runner
from dit.data.synthetic import make_synthetic_bundle
from dit.evaluation.reporting import assemble_report, write_json


def test_report_normalizes_numpy_integer_and_boolean(tmp_path):
    payload = assemble_report(
        [],
        dataset_summary={"n_samples": np.int64(3)},
        configuration={"enabled": np.bool_(True)},
    )

    target = write_json(tmp_path / "report.json", payload)
    restored = json.loads(target.read_text(encoding="utf-8"))

    assert restored["dataset"]["n_samples"] == 3
    assert restored["configuration"]["enabled"] is True


def test_metric_ensemble_rejects_incomplete_outer_predictions(monkeypatch):
    dataset = make_synthetic_bundle(
        n_samples=18,
        n_tracts=2,
        n_points=5,
        n_metrics=2,
        n_sites=3,
        n_classes=2,
    )
    view = dataset.task_view("binary")
    train_indices = np.arange(view.n_samples - 1)
    test_indices = np.asarray([view.n_samples - 1])

    class FakeSearch:
        classes_ = np.asarray([0, 1])
        best_params_ = {}

        def fit(self, features, labels, **kwargs):
            return self

        def predict_proba(self, features):
            return np.tile(np.asarray([[0.6, 0.4]]), (len(features), 1))

    monkeypatch.setattr(
        runner,
        "split_indices",
        lambda *args, **kwargs: [(train_indices, test_indices, "0")],
    )
    monkeypatch.setattr(
        runner,
        "_inner_cv",
        lambda *args, **kwargs: (object(), None),
    )
    monkeypatch.setattr(
        runner,
        "make_search_estimator",
        lambda *args, **kwargs: FakeSearch(),
    )

    with pytest.raises(
        RuntimeError,
        match="cross-validation did not predict every subject",
    ):
        runner.evaluate_metric_ensemble(
            dataset,
            task="binary",
            split_strategy="stratified",
            inner_splits=2,
        )
