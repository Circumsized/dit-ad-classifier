"""Fail-closed branches of ``paired_fold_comparison``.

Every rejection path in ``dit/evaluation/site_balance.py`` must return
``not_comparable`` with a reason instead of pairing folds that were not
evaluated on the same rows, task, protocol or class manifest.  One legal
fixture feeds one mutation per branch.  The positive control at the top proves
the fixture itself is comparable: without it, every rejection below would also
pass on a fixture that can never be paired.
"""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from dit.evaluation.site_balance import paired_fold_comparison


def _fold(
    identifier: str,
    digest: str,
    metric: float,
    *,
    task: str = "binary",
    strategy: str = "loso",
    counts: dict[str, int] | None = None,
    n_test: int = 8,
) -> dict[str, object]:
    """One fold report in the shape ``fold_provenance`` emits."""

    return {
        "fold": identifier,
        "task": task,
        "split_strategy": strategy,
        "test_index_digest": digest,
        "test_class_counts": {"0": 4, "1": 4} if counts is None else counts,
        "held_out_site": identifier if strategy == "loso" else None,
        "n_test": n_test,
        "balanced_accuracy": metric,
    }


def _pair(n: int = 3) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """A legal reference/candidate pair over ``n`` matched folds."""

    reference = [_fold(str(index), f"digest-{index}", 0.50 + 0.01 * index) for index in range(n)]
    candidate = [_fold(str(index), f"digest-{index}", 0.55 + 0.01 * index) for index in range(n)]
    return reference, candidate


def _drop(key: str, *, side: str = "reference", index: int = 0) -> Callable:
    def mutate(reference: list[dict[str, object]], candidate: list[dict[str, object]]) -> None:
        (reference if side == "reference" else candidate)[index].pop(key)

    return mutate


def _set(key: str, value: object, *, side: str = "reference", index: int = 0) -> Callable:
    def mutate(reference: list[dict[str, object]], candidate: list[dict[str, object]]) -> None:
        (reference if side == "reference" else candidate)[index][key] = value

    return mutate


def _set_all(key: str, value: object, *, side: str = "candidate") -> Callable:
    def mutate(reference: list[dict[str, object]], candidate: list[dict[str, object]]) -> None:
        for fold in reference if side == "reference" else candidate:
            fold[key] = value

    return mutate


def test_the_shared_fixture_is_comparable_so_rejections_are_not_vacuous() -> None:
    # Positive control: if this ever stops being "tested", every rejection
    # below would pass for the wrong reason (an unpairable fixture).
    reference, candidate = _pair()
    result = paired_fold_comparison(reference, candidate)
    assert result["status"] == "tested"
    assert result["reason"] is None
    assert result["n_pairs"] == 3
    assert result["n_input_folds"] == 3
    assert json.dumps(result, allow_nan=False)


# (id, mutation applied to the legal pair, substring the reason must carry).
# Branches (a)-(d) share one guard, so they share one reason sentence; the
# per-case value is proving that each *field's* absence is caught.
REJECTIONS: list[tuple[str, Callable, str]] = [
    # (a) missing fold ID
    ("fold-id-missing", _drop("fold"), "every fold needs ID, task, strategy and test-index digest"),
    # (b) test-index digest missing / wrong type / empty
    ("digest-missing", _drop("test_index_digest"), "test-index digest"),
    ("digest-not-a-string", _set("test_index_digest", 12345), "test-index digest"),
    ("digest-empty", _set("test_index_digest", ""), "test-index digest"),
    # (c) task missing / empty / wrong type
    ("task-missing", _drop("task"), "task, strategy"),
    ("task-empty", _set("task", ""), "task, strategy"),
    ("task-not-a-string", _set("task", 2), "task, strategy"),
    # (d) split strategy missing / empty
    ("strategy-missing", _drop("split_strategy"), "strategy and test-index digest"),
    ("strategy-empty", _set("split_strategy", ""), "strategy and test-index digest"),
    # (e) duplicate fold ID, on either side
    ("duplicate-fold-id-reference", _set("fold", "0", index=1), "duplicate fold ID"),
    ("duplicate-fold-id-candidate", _set("fold", "0", side="candidate", index=1), "duplicate fold ID"),
    # (f) one side mixes two tasks or two strategies
    ("mixed-task-reference", _set("task", "multiclass", index=1), "one task and split strategy"),
    ("mixed-strategy-candidate", _set("split_strategy", "stratified", side="candidate", index=1), "one task and split strategy"),
    # (g) empty fold list
    ("empty-reference-list", lambda r, c: r.clear(), "at least one fold"),
    ("empty-candidate-list", lambda r, c: c.clear(), "at least one fold"),
    # (h) fold-ID / digest sets do not match
    ("fold-id-set-mismatch", _set("fold", "9", index=2), "do not match"),
    ("digest-set-mismatch", _set("test_index_digest", "other", side="candidate", index=2), "do not match"),
    # (i) n_test differs, including a key present on only one side
    ("n-test-differs", _set("n_test", 7, side="candidate", index=1), "different n_test"),
    ("n-test-missing-on-one-side", _drop("n_test", side="candidate", index=1), "different n_test"),
    # (j) held-out site differs, including a key present on only one side
    ("held-out-site-differs", _set("held_out_site", "site-z", side="candidate", index=1), "different held-out site"),
    ("held-out-site-missing-on-one-side", _drop("held_out_site", side="candidate", index=1), "different held-out site"),
    # (k) class-count manifest corrupt or mismatched
    ("counts-missing", _drop("test_class_counts", index=1), "class-count manifest"),
    ("counts-empty-mapping", _set("test_class_counts", {}, index=1), "class-count manifest"),
    ("counts-bool-value", _set("test_class_counts", {"0": True, "1": 4}, index=1), "class-count manifest"),
    ("counts-negative-value", _set("test_class_counts", {"0": -1, "1": 4}, index=1), "class-count manifest"),
    ("counts-fractional-value", _set("test_class_counts", {"0": 4.5, "1": 4}, index=1), "class-count manifest"),
    ("counts-string-value", _set("test_class_counts", {"0": "4", "1": 4}, index=1), "class-count manifest"),
    ("counts-not-a-mapping", _set("test_class_counts", [4, 4], index=1), "class-count manifest"),
    ("counts-values-differ", _set("test_class_counts", {"0": 5, "1": 3}, side="candidate", index=1), "different test class counts"),
    # (l) metric missing or not numeric
    ("metric-missing", _drop("balanced_accuracy", side="candidate", index=1), "lacks metric"),
    ("metric-not-numeric", _set("balanced_accuracy", "high", index=1), "lacks metric"),
    # (m) non-finite metric
    ("metric-nan", _set("balanced_accuracy", float("nan"), side="candidate", index=1), "non-finite"),
    ("metric-inf", _set("balanced_accuracy", float("inf"), index=1), "non-finite"),
    # cross-side protocol mismatch: both sides internally consistent
    ("cross-side-task", _set_all("task", "multiclass"), "task or split strategy does not match"),
    ("cross-side-strategy", _set_all("split_strategy", "stratified"), "task or split strategy does not match"),
]


@pytest.mark.parametrize(
    ("case", "mutate", "expected"),
    REJECTIONS,
    ids=[case for case, _, _ in REJECTIONS],
)
def test_rejections_are_fail_closed(case: str, mutate: Callable, expected: str) -> None:
    reference, candidate = _pair()
    mutate(reference, candidate)
    result = paired_fold_comparison(reference, candidate)
    assert result["status"] == "not_comparable", f"{case} produced {result['status']}: {result['reason']}"
    assert expected in str(result["reason"])
    # A refused comparison never reports pairs or a p-value.
    assert result["pairs"] == []
    assert result["n_pairs"] == 0
    assert result["p_value"] is None
    assert json.dumps(result, allow_nan=False)


def test_single_matched_fold_is_not_tested_rather_than_comparable() -> None:
    # (n) one pair cannot support a two-sided signed-rank test.
    reference, candidate = _pair(n=1)
    result = paired_fold_comparison(reference, candidate)
    assert result["status"] == "not_tested"
    assert "fewer than two matched folds" in str(result["reason"])
    assert result["n_input_folds"] == 1
    assert result["n_pairs"] == 1
    assert result["p_value"] is None
    assert json.dumps(result, allow_nan=False)


def test_one_eligible_fold_after_exclusion_is_not_tested() -> None:
    # (n) the same refusal when an eligibility exclusion, not the input size,
    # is what leaves a single pair.
    reference = [_fold("0", "digest-0", 0.5), _fold("1", "digest-1", 0.6, counts={"0": 8, "1": 0})]
    candidate = [_fold("0", "digest-0", 0.6), _fold("1", "digest-1", 0.7, counts={"0": 8, "1": 0})]
    result = paired_fold_comparison(reference, candidate)
    assert result["status"] == "not_tested"
    assert "fewer than two matched folds" in str(result["reason"])
    assert [unit["fold"] for unit in result["excluded_units"]] == ["1"]
    assert result["n_pairs"] == 1
    assert json.dumps(result, allow_nan=False)


def test_strategy_aliases_are_not_silently_normalised() -> None:
    # ``loso`` and ``leave_one_site_out`` are aliases for ``split_indices``
    # but not for the comparison context, which only lowercases and swaps
    # dashes.  A differently spelled strategy is refused, not assumed equal.
    reference = [_fold("0", "digest-0", 0.5)]
    reference[0]["split_strategy"] = "LOSO"
    candidate = [_fold("0", "digest-0", 0.6)]
    candidate[0]["split_strategy"] = "leave-one-site-out"
    result = paired_fold_comparison(reference, candidate)
    assert result["status"] == "not_comparable"
    assert "task or split strategy does not match" in str(result["reason"])
