"""Regression tests for review hardening fixes.

Covers a finding from the 2026-09 code review:
- ``_json_safe`` must neutralize numpy integer/boolean scalars, not just
  floats, or a later report field crashes strict JSON serialization.

(The companion finding about ``evaluate_metric_ensemble`` failing loudly on
incomplete outer predictions died with the compatibility runner itself: the
CLI path is ``run_experiment``, whose OOF-completeness guard is pinned by
tests/test_experiment_oof_argmax.py.)
"""

from __future__ import annotations

import json

import numpy as np

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
