"""Deletion contract: removed symbols must stay removed.

Three commits in this remediation claimed to delete ``experiment_payload_of_base``
while leaving the definition in place, and no test noticed -- F821/F841 can
see undefined names and unused locals, but a defined-but-call-free function is
invisible to both. These assertions pin the deletions themselves: one line
would have caught the phantom.
"""

from __future__ import annotations


def test_removed_evaluation_helpers_are_gone() -> None:
    import dit.evaluation.experiment as experiment
    import dit.evaluation.runner as runner
    import dit.evaluation.threshold as threshold

    for name in ("evaluate_classical", "evaluate_metric_ensemble", "_inner_cv"):
        assert not hasattr(runner, name), f"{runner.__name__}.{name} must stay deleted"
    for name in ("_apply_residualizer", "to_fold_dicts", "experiment_payload_of_base"):
        assert not hasattr(experiment, name), f"{experiment.__name__}.{name} must stay deleted"
    for name in ("apply_policy", "threshold_sweep"):
        assert not hasattr(threshold, name), f"{threshold.__name__}.{name} must stay deleted"


def test_removed_data_helpers_are_gone() -> None:
    import dit.data as data
    import dit.data.layout as layout
    import dit.data.preprocessing as preprocessing
    import dit.data.selection as selection
    import dit.data.sklearn_compat as compat

    assert not hasattr(data, "TractFeaturePreprocessor")
    assert not hasattr(preprocessing, "TractFeaturePreprocessor")
    assert not hasattr(selection.SparseBlockSelector, "make_pipeline")
    assert not hasattr(compat, "has_new_logistic_api")
    assert not hasattr(layout, "expected_feature_count")
    assert not hasattr(layout, "VALID_VIEWS")


def test_removed_cli_wrapper_is_gone() -> None:
    import dit.cli.main as main

    assert not hasattr(main, "experiment_payload_of_base"), (
        f"{main.__name__}.experiment_payload_of_base must stay deleted "
        "(deletion ledger: docs/REMEDIATION_2026-11-19.md, C11/R4b)"
    )
