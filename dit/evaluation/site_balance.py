"""Site composition and matched-fold evaluation summaries.

Leave-one-site-out only means something when the held-out site contains the
classes required by the task. This module makes site imbalance explicit and
provides descriptive fold spread plus an ID-aware, exploratory paired test for
predeclared ablation comparisons.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence

import numpy as np

from dit.data.schema import CANONICAL_NAMES

MIN_PER_CLASS = 3


def _class_namer(label_map: Mapping[int, str] | None):
    table = label_map if label_map is not None else CANONICAL_NAMES
    return lambda cls: table.get(int(cls), str(cls))


def site_composition(
    site: np.ndarray,
    y: np.ndarray,
    *,
    min_per_class: int = MIN_PER_CLASS,
    label_map: Mapping[int, str] | None = None,
) -> dict[str, object]:
    """Return per-site class counts and degeneracy flags.

    Every site reports every class present in the task view, including zero
    counts. A site is ``thin`` when any expected class has fewer than
    ``min_per_class`` members, and ``degenerate`` when only one class is
    observed, so incomparable LOSO folds cannot hide behind omitted keys.
    """

    groups = np.asarray(site).reshape(-1)
    labels = np.asarray(y).reshape(-1)
    if groups.shape != labels.shape:
        raise ValueError("site and labels must have the same shape")
    if np.any(groups == -1):
        raise ValueError("site contains missing (-1) values")

    name_of = _class_namer(label_map)
    expected_labels = sorted(int(label) for label in np.unique(labels).tolist())
    rows: list[dict[str, object]] = []
    for identifier in sorted(set(groups.tolist()), key=str):
        mask = groups == identifier
        members = labels[mask]
        observed_labels = set(int(label) for label in np.unique(members).tolist())
        counts = {
            name_of(label): int(np.sum(members == label))
            for label in expected_labels
        }
        missing_classes = [
            name_of(label) for label in expected_labels if label not in observed_labels
        ]
        min_count = min(counts.values()) if counts else 0
        rows.append(
            {
                "site": str(identifier),
                "n": int(members.size),
                "classes": counts,
                "n_classes": len(observed_labels),
                "n_expected_classes": len(expected_labels),
                "missing_classes": missing_classes,
                "thin": min_count < min_per_class,
                "degenerate": len(observed_labels) == 1,
            }
        )

    n_folds = len(rows)
    return {
        "n_sites": n_folds,
        "min_per_class": int(min_per_class),
        "n_subjects": int(labels.size),
        "degenerate_sites": [row["site"] for row in rows if row["degenerate"]],
        # A degenerate site is already listed separately; keep ``thin_sites``
        # as the existing non-degenerate, low-support category while each row's
        # own ``thin`` flag still records the missing-class condition.
        "thin_sites": [
            row["site"] for row in rows if row["thin"] and not row["degenerate"]
        ],
        "balanced_fraction": float(
            np.mean([not row["thin"] for row in rows]) if rows else 0.0
        ),
        "max_fold_size": int(max((row["n"] for row in rows), default=0)),
        "min_fold_size": int(min((row["n"] for row in rows), default=0)),
        "sites": rows,
    }


def per_site_metrics(
    site: np.ndarray,
    y: np.ndarray,
    predictions: np.ndarray,
    *,
    label_map: Mapping[int, str] | None = None,
) -> list[dict[str, object]]:
    """Accuracy and class mix within each site, for stratified reporting."""

    groups = np.asarray(site).reshape(-1)
    labels = np.asarray(y, dtype=int).reshape(-1)
    pred = np.asarray(predictions, dtype=int).reshape(-1)
    if not (groups.shape == labels.shape == pred.shape):
        raise ValueError("site, labels and predictions must match in shape")

    name_of = _class_namer(label_map)
    rows: list[dict[str, object]] = []
    for identifier in sorted(set(groups.tolist()), key=str):
        mask = groups == identifier
        members = labels[mask]
        counts = {
            name_of(cls): int(n)
            for cls, n in zip(*np.unique(members, return_counts=True))
        }
        rows.append(
            {
                "site": str(identifier),
                "n": int(members.size),
                "classes": counts,
                "accuracy": float(np.mean(pred[mask] == members)),
                "balanced_accuracy": float(
                    np.nanmean(
                        [
                            float(
                                np.mean((pred[mask] == cls) & (members == cls))
                                / np.mean(members == cls)
                            )
                            if np.any(members == cls)
                            else np.nan
                            for cls in sorted(set(members.tolist()))
                        ]
                    )
                ),
            }
        )
    return rows


def fold_macro_summary(
    fold_reports: Sequence[Mapping[str, object]],
    metrics: Sequence[str] = ("accuracy", "balanced_accuracy", "macro_f1"),
) -> dict[str, object]:
    """Unweighted per-fold summary that complements the pooled OOF aggregate.

    The pooled aggregate weights every subject equally, so larger LOSO sites
    dominate.  This reports the unweighted mean/median/range across folds and
    excludes folds flagged ``fold_comparable=False`` (a held-out site missing a
    class), whose metric is on a different label set and must not be pooled.
    """

    comparable = [
        fold for fold in fold_reports if fold.get("fold_comparable", True)
    ]
    incomparable = [
        str(fold.get("held_out_site", fold.get("fold")))
        for fold in fold_reports
        if not fold.get("fold_comparable", True)
    ]
    summary: dict[str, object] = {
        "n_folds": len(fold_reports),
        "n_comparable_folds": len(comparable),
        "incomparable_folds": incomparable,
    }
    per_metric: dict[str, object] = {}
    for metric in metrics:
        values = [
            float(fold[metric])
            for fold in comparable
            if metric in fold and np.isfinite(float(fold[metric]))
        ]
        if not values:
            per_metric[metric] = {"mean": None, "median": None, "min": None, "max": None}
            continue
        per_metric[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    summary["per_metric"] = per_metric
    return summary


def stable_predictors(
    fold_reports: Sequence[Mapping[str, object]],
    *,
    top_k: int = 25,
) -> dict[str, object]:
    """Cross-fold selection frequency of anatomical blocks and nodes.

    The competition asks for the consistency of the estimated predictors.  Each
    outer fold selects its features independently, so a predictor that survives
    in many folds is more trustworthy than one selected once.  This counts, over
    the folds that actually ran selection, how often each block and each
    (block, node) was retained, and reports the frequency as count over the
    number of selection folds.  It is descriptive: it does not claim a p-value.
    """

    selections = [
        fold["selection"]
        for fold in fold_reports
        if isinstance(fold.get("selection"), Mapping)
    ]
    n_folds = len(selections)
    if n_folds == 0:
        return {"n_selection_folds": 0, "blocks": [], "nodes": []}

    block_counts: dict[str, int] = {}
    node_counts: dict[tuple[str, int], int] = {}
    for selection in selections:
        for block in selection.get("retained_blocks", []) or []:
            block_counts[str(block)] = block_counts.get(str(block), 0) + 1
        retained_nodes = selection.get("retained_nodes", {}) or {}
        for block, nodes in retained_nodes.items():
            for node in nodes or []:
                key = (str(block), int(node))
                node_counts[key] = node_counts.get(key, 0) + 1

    blocks = [
        {"block": name, "folds": count, "frequency": count / n_folds}
        for name, count in block_counts.items()
    ]
    blocks.sort(key=lambda item: (-item["folds"], item["block"]))
    nodes = [
        {"block": block, "node": node, "folds": count, "frequency": count / n_folds}
        for (block, node), count in node_counts.items()
    ]
    nodes.sort(key=lambda item: (-item["folds"], item["block"], item["node"]))
    return {
        "n_selection_folds": n_folds,
        "blocks": blocks[:top_k],
        "nodes": nodes[:top_k],
    }


def pairwise_fold_delta(fold_metrics: list[float]) -> dict[str, object]:
    """Summarize the dispersion of one result's finite fold metrics.

    This is descriptive only. It does not compare two models and does not
    compute significance; use :func:`paired_fold_comparison` for a validated,
    matched-fold comparison.
    """

    values = [float(value) for value in fold_metrics if np.isfinite(value)]
    if len(values) < 2:
        return {
            "n_folds": len(values),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "spread": None,
        }
    return {
        "n_folds": len(values),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "spread": float(np.max(values) - np.min(values)),
    }


def _comparison_payload(
    *,
    status: str,
    reason: str | None,
    metric: str | None,
    n_input_folds: int = 0,
    n_pairs: int = 0,
    n_nonzero: int = 0,
    mean_delta: float | None = None,
    median_delta: float | None = None,
    wins: int = 0,
    losses: int = 0,
    ties: int = 0,
    statistic: float | None = None,
    p_value: float | None = None,
    min_attainable_p: float | None = None,
    pairs: list[dict[str, object]] | None = None,
    excluded_units: list[dict[str, object]] | None = None,
    caveats: list[str] | None = None,
) -> dict[str, object]:
    """Build a stable, JSON-safe paired-comparison record."""

    return {
        "status": status,
        "reason": reason,
        "metric": metric,
        "direction": "candidate_minus_reference",
        "n_input_folds": n_input_folds,
        "n_pairs": n_pairs,
        "n_nonzero": n_nonzero,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "statistic": statistic,
        "p_value": p_value,
        "p_value_holm": None,
        "min_attainable_p": min_attainable_p,
        "zero_method": "zsplit",
        "alternative": "two-sided",
        "method": "auto",
        "pairs": pairs or [],
        "excluded_units": excluded_units or [],
        "caveats": caveats or [],
    }


def _fold_map(
    folds: Sequence[Mapping[str, object]],
) -> tuple[
    dict[tuple[str, str], Mapping[str, object]] | None,
    tuple[str, str] | None,
    str | None,
]:
    """Map fold reports by ID/digest and validate their shared provenance."""

    mapped: dict[tuple[str, str], Mapping[str, object]] = {}
    fold_ids: set[str] = set()
    context: tuple[str, str] | None = None
    for fold in folds:
        raw_id = fold.get("fold")
        digest = fold.get("test_index_digest")
        task = fold.get("task")
        strategy = fold.get("split_strategy")
        if (
            raw_id is None
            or not isinstance(digest, str)
            or not digest
            or not isinstance(task, str)
            or not task
            or not isinstance(strategy, str)
            or not strategy
        ):
            return None, None, "every fold needs ID, task, strategy and test-index digest provenance"
        fold_id = str(raw_id)
        if fold_id in fold_ids:
            return None, None, f"duplicate fold ID {fold_id!r}"
        fold_ids.add(fold_id)
        current_context = (task, strategy.lower().replace("-", "_"))
        if context is not None and current_context != context:
            return None, None, "folds do not share one task and split strategy"
        context = current_context
        mapped[(fold_id, digest)] = fold
    if not mapped:
        return None, None, "at least one fold is required"
    return mapped, context, None


def _class_counts(fold: Mapping[str, object]) -> dict[str, int] | None:
    """Validate the class-count manifest without silently coercing corruption."""

    raw = fold.get("test_class_counts")
    if not isinstance(raw, Mapping) or not raw:
        return None
    counts: dict[str, int] = {}
    for label, value in raw.items():
        if isinstance(value, bool):
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        if parsed < 0 or parsed != value:
            return None
        counts[str(label)] = parsed
    return counts


def _minimum_two_sided_p(n_nonzero: int) -> float | None:
    """Smallest exact two-sided sign/rank p-value for this many nonzero pairs."""

    return None if n_nonzero < 1 else float(2.0 / (2**n_nonzero))


def paired_fold_comparison(
    reference_folds: Sequence[Mapping[str, object]],
    candidate_folds: Sequence[Mapping[str, object]],
    *,
    metric: str = "balanced_accuracy",
    rounding: int = 12,
) -> dict[str, object]:
    """Compare matched outer folds with a two-sided Wilcoxon signed-rank test.

    Pairing requires identical fold IDs and non-identifying test-index digests,
    plus a shared task, split strategy, test size and class-count manifest.
    Differences are candidate minus reference. Folds missing an expected class
    are listed as exclusions, never silently included in a partial metric.

    This remains exploratory cross-validation evidence, not an independent
    clinical-sample test: outer folds share training rows and small ``n`` limits
    p-value resolution.
    """

    reference, reference_context, reason = _fold_map(reference_folds)
    if reason is not None or reference is None:
        return _comparison_payload(
            status="not_comparable",
            reason=f"reference folds: {reason}",
            metric=metric,
        )
    candidate, candidate_context, reason = _fold_map(candidate_folds)
    if reason is not None or candidate is None:
        return _comparison_payload(
            status="not_comparable",
            reason=f"candidate folds: {reason}",
            metric=metric,
        )
    if reference_context != candidate_context:
        return _comparison_payload(
            status="not_comparable",
            reason="task or split strategy does not match",
            metric=metric,
        )
    if set(reference) != set(candidate):
        return _comparison_payload(
            status="not_comparable",
            reason="fold IDs or test-index digests do not match",
            metric=metric,
        )

    pairs: list[dict[str, object]] = []
    excluded_units: list[dict[str, object]] = []
    differences: list[float] = []
    for key in sorted(reference):
        reference_fold, candidate_fold = reference[key], candidate[key]
        if reference_fold.get("n_test") != candidate_fold.get("n_test"):
            return _comparison_payload(
                status="not_comparable",
                reason=f"fold {key[0]!r} has different n_test",
                metric=metric,
            )
        if reference_fold.get("held_out_site") != candidate_fold.get("held_out_site"):
            return _comparison_payload(
                status="not_comparable",
                reason=f"fold {key[0]!r} has a different held-out site",
                metric=metric,
            )
        reference_counts = _class_counts(reference_fold)
        candidate_counts = _class_counts(candidate_fold)
        if reference_counts is None or candidate_counts is None:
            return _comparison_payload(
                status="not_comparable",
                reason=f"fold {key[0]!r} lacks a valid test class-count manifest",
                metric=metric,
            )
        if reference_counts != candidate_counts:
            return _comparison_payload(
                status="not_comparable",
                reason=f"fold {key[0]!r} has different test class counts",
                metric=metric,
            )
        if any(count == 0 for count in reference_counts.values()):
            excluded_units.append(
                {
                    "fold": key[0],
                    "held_out_site": candidate_fold.get("held_out_site"),
                    "reason": "test fold is missing an expected class",
                    "test_class_counts": reference_counts,
                }
            )
            continue
        try:
            reference_value = float(reference_fold[metric])
            candidate_value = float(candidate_fold[metric])
        except (KeyError, TypeError, ValueError):
            return _comparison_payload(
                status="not_comparable",
                reason=f"fold {key[0]!r} lacks metric {metric!r}",
                metric=metric,
            )
        if not (np.isfinite(reference_value) and np.isfinite(candidate_value)):
            return _comparison_payload(
                status="not_comparable",
                reason=f"fold {key[0]!r} has a non-finite {metric!r}",
                metric=metric,
            )
        delta = round(candidate_value - reference_value, rounding)
        differences.append(delta)
        pairs.append(
            {
                "fold": key[0],
                "test_index_digest": key[1],
                "held_out_site": candidate_fold.get("held_out_site"),
                "reference": reference_value,
                "candidate": candidate_value,
                "delta": delta,
            }
        )

    delta_values = np.asarray(differences, dtype=float)
    caveats = [
        "Exploratory cross-validation comparison: paired folds share training data."
    ]
    if excluded_units:
        caveats.append(
            f"Excluded {len(excluded_units)} fold(s) missing an expected test class."
        )
    if delta_values.size < 2:
        return _comparison_payload(
            status="not_tested",
            reason="fewer than two matched folds remain after eligibility checks",
            metric=metric,
            n_input_folds=len(reference),
            n_pairs=int(delta_values.size),
            pairs=pairs,
            excluded_units=excluded_units,
            caveats=caveats,
        )

    n_nonzero = int(np.count_nonzero(delta_values))
    minimum = _minimum_two_sided_p(n_nonzero)
    if minimum is not None and minimum > 0.05:
        caveats.append(
            f"With {n_nonzero} nonzero paired folds, a two-sided exact test cannot reach p<0.05."
        )
    result = _comparison_payload(
        status="not_tested" if n_nonzero == 0 else "tested",
        reason="all matched fold differences are zero" if n_nonzero == 0 else None,
        metric=metric,
        n_input_folds=len(reference),
        n_pairs=int(delta_values.size),
        n_nonzero=n_nonzero,
        mean_delta=float(np.mean(delta_values)),
        median_delta=float(np.median(delta_values)),
        wins=int(np.sum(delta_values > 0)),
        losses=int(np.sum(delta_values < 0)),
        ties=int(np.sum(delta_values == 0)),
        p_value=1.0 if n_nonzero == 0 else None,
        min_attainable_p=minimum,
        pairs=pairs,
        excluded_units=excluded_units,
        caveats=caveats,
    )
    if n_nonzero == 0:
        return result

    try:
        from scipy.stats import wilcoxon

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            statistic, p_value = wilcoxon(
                delta_values,
                zero_method="zsplit",
                alternative="two-sided",
                method="auto",
            )
    except (ImportError, ValueError, Warning) as exc:
        result["status"] = "not_tested"
        result["reason"] = f"Wilcoxon could not be computed: {exc}"
        return result

    if not (np.isfinite(statistic) and np.isfinite(p_value)):
        result["status"] = "not_tested"
        result["reason"] = "Wilcoxon returned a non-finite result"
        return result
    result["statistic"] = float(statistic)
    result["p_value"] = float(p_value)
    return result


def holm_adjust(comparisons: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Add Holm-adjusted p-values to valid comparisons without changing order."""

    adjusted = [dict(comparison) for comparison in comparisons]
    eligible = [
        (index, float(item["p_value"]))
        for index, item in enumerate(adjusted)
        if item.get("status") == "tested" and item.get("p_value") is not None
    ]
    for item in adjusted:
        item["p_value_holm"] = None
    previous = 0.0
    total = len(eligible)
    for rank, (index, p_value) in enumerate(sorted(eligible, key=lambda item: item[1])):
        value = min(1.0, max(previous, (total - rank) * p_value))
        adjusted[index]["p_value_holm"] = float(value)
        previous = value
    return adjusted
