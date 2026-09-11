"""Assemble and persist evaluation reports.

Reports record the split strategy, feature view, covariate policy, threshold
criterion, per-fold detail, and the runtime environment, so results stay
reproducible.
"""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from dit.evaluation.site_balance import (
    holm_adjust,
    paired_fold_comparison,
    pairwise_fold_delta,
    stable_predictors,
)
from dit.evaluation.threshold import ThresholdPolicy


def environment_fingerprint(seed: int | None = None) -> dict[str, object]:
    """Record the runtime that produced a report."""

    versions: dict[str, str] = {"python": platform.python_version()}
    for name in ("numpy", "scipy", "sklearn", "torch", "pandas", "matplotlib"):
        try:
            module = __import__(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except ImportError:
            versions[name] = "not installed"
    fingerprint: dict[str, object] = {
        "platform": platform.platform(),
        "versions": versions,
        "seed": seed,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if commit.returncode == 0 and commit.stdout.strip():
            fingerprint["git_commit"] = commit.stdout.strip()
        # A dirty worktree means the short commit alone does not pin the code
        # that ran, so record that and how many files differ.
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if status.returncode == 0:
            changed = [line for line in status.stdout.splitlines() if line.strip()]
            fingerprint["git_dirty"] = bool(changed)
            fingerprint["git_changed_files"] = len(changed)
    except (OSError, subprocess.SubprocessError):
        pass
    return fingerprint


def _fold_table(result) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for fold in result.folds:
        row: dict[str, object] = {
            "fold": fold.get("fold"),
            "n_train": fold.get("n_train"),
            "n_test": fold.get("n_test"),
            "accuracy": fold.get("accuracy"),
            "balanced_accuracy": fold.get("balanced_accuracy"),
            "macro_f1": fold.get("macro_f1"),
            "sensitivity": fold.get("sensitivity"),
            "specificity": fold.get("specificity"),
            "roc_auc": fold.get("roc_auc"),
            "ece": fold.get("ece"),
            "brier_score": fold.get("brier_score"),
        }
        # Fit provenance for auditing ensembles: ``weights`` and
        # ``base_models`` carry the per-fold ensemble composition, including
        # whether a deep member was calibrated.
        for key in (
            "best_params",
            "tuning_score",
            "training",
            "alignment_grid",
            "weights",
            "base_models",
            "confusion_matrix",
            "n_features_used",
            "task",
            "split_strategy",
            "test_index_digest",
            "test_class_counts",
            "held_out_site",
            "fold_comparable",
            "selection",
        ):
            value = fold.get(key)
            if value is not None:
                row[key] = value
        rows.append({key: value for key, value in row.items() if value is not None})
    return rows


def _covariate_comparison_key(result) -> tuple[tuple[str, object], ...]:
    """Fingerprint settings that must match before covariate ablation pairing.

    The covariate strategy itself is deliberately excluded.  Everything else
    that changes data, folds, model behavior or tuning must match, otherwise a
    difference cannot be attributed to the covariate choice.
    """

    config = getattr(result, "config", None)
    if config is None:
        return tuple()
    values = dict(config.describe())
    values.pop("covariate_strategy", None)
    return tuple(sorted((str(key), _freeze(value)) for key, value in values.items()))


def _freeze(value: object) -> object:
    """Make config values sortable/hashable for a provenance fingerprint."""

    if isinstance(value, dict):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def build_covariate_ablation_comparisons(results: list) -> list[dict[str, object]]:
    """Return only predeclared matched-fold covariate ablation comparisons.

    ``feature`` is the reference. Arbitrary model/view combinations, testing a
    selected winner, and mixing split strategies are not compared.
    """

    groups: dict[tuple[tuple[str, object], ...], dict[str, object]] = {}
    for result in results:
        config = getattr(result, "config", None)
        if config is None:
            continue
        strategy = getattr(config, "covariate_strategy", None)
        if strategy not in {"none", "feature", "residualize"}:
            continue
        groups.setdefault(_covariate_comparison_key(result), {})[strategy] = result

    comparisons: list[dict[str, object]] = []
    for _, variants in sorted(groups.items(), key=lambda item: repr(item[0])):
        reference = variants.get("feature")
        if reference is None:
            continue
        for candidate_name in ("none", "residualize"):
            candidate = variants.get(candidate_name)
            if candidate is None:
                continue
            comparison = paired_fold_comparison(
                reference.folds,
                candidate.folds,
                metric="balanced_accuracy",
            )
            comparison.update(
                {
                    "scope": "predeclared_exploratory_covariate_ablation",
                    "reference": reference.key,
                    "candidate": candidate.key,
                    "reference_covariate_strategy": "feature",
                    "candidate_covariate_strategy": candidate_name,
                }
            )
            comparisons.append(comparison)
    return holm_adjust(comparisons)


def assemble_report(
    results: list,
    *,
    dataset_summary: dict[str, object],
    configuration: dict[str, object],
    threshold_policies: dict[str, ThresholdPolicy] | None = None,
    site_composition: dict[str, object] | None = None,
    per_site_metrics: dict[str, object] | None = None,
    paired_comparisons: list[dict[str, object]] | None = None,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build the JSON payload for one experimental session."""

    payload: dict[str, object] = {
        "schema_version": 1,
        "configuration": configuration,
        "dataset": dataset_summary,
        "environment": environment_fingerprint(configuration.get("seed")),
        "results": [],
    }
    for result in results:
        entry: dict[str, object] = {
            "model": result.model,
            "task": result.task,
            "split_strategy": result.split_strategy,
            "feature_view": result.feature_view,
            "aggregate": result.aggregate,
            "fold_metric_spread": {
                metric: pairwise_fold_delta([float(fold.get(metric, np.nan)) for fold in result.folds])
                for metric in ("accuracy", "balanced_accuracy", "macro_f1")
                if all(metric in fold for fold in result.folds)
            },
            "folds": _fold_table(result),
        }
        # Prefer the explicit policy carried by the result.  The mapping
        # fallback keeps compatibility with callers that still provide the
        # pre-existing threshold_policies argument, while avoiding an implicit
        # key convention that can omit policies from the report.
        policy = getattr(result, "threshold", None)
        if policy is None:
            policy = (threshold_policies or {}).get(result_key(result))
        if policy is not None:
            entry["threshold"] = policy.parameters()
        selection_metrics = getattr(result, "thresholded_selection_metrics", None)
        if selection_metrics is not None:
            entry["thresholded_selection_metrics"] = selection_metrics
        # When feature selection ran, report how consistently each anatomical
        # block/node was chosen across folds.
        if any(isinstance(fold.get("selection"), dict) for fold in result.folds):
            entry["stable_predictors"] = stable_predictors(result.folds)
        payload["results"].append(entry)

    if site_composition is not None:
        payload["site_composition"] = site_composition
    if per_site_metrics is not None:
        payload["per_site"] = per_site_metrics
    if paired_comparisons:
        payload["paired_comparisons"] = paired_comparisons
    if extra:
        payload["extra"] = extra
    return _json_safe(payload)


def result_key(result) -> str:
    return f"{result.model}|{result.task}|{result.split_strategy}|{result.feature_view}"


def _json_safe(value: object) -> object:
    """Replace non-finite floats before strict JSON serialization.

    Unavailable metrics become JSON ``null`` instead of the non-standard
    ``NaN`` token. Scalar and container types are preserved recursively.
    """

    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if np.isfinite(numeric) else None
    if isinstance(value, dict):
        # Integer keys stay in the in-memory contract; ``json.dumps`` turns
        # them into JSON object names at the file boundary.
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def write_json(path: str | Path, payload: dict[str, object]) -> Path:
    """Write the report as JSON, creating parent directories."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=False, allow_nan=False), encoding="utf-8"
    )
    return target


def render_markdown(payload: dict[str, object]) -> str:
    """Render a report as Markdown for the paper and README."""

    config = payload["configuration"]
    lines: list[str] = [
        f"# AI4AD AFQ evaluation report",
        "",
        f"_Generated {payload['environment']['generated_utc']}_",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## Dataset",
        "",
        _json_block(payload["dataset"]),
        "",
        "## Results",
        "",
        _results_table(payload["results"]),
    ]
    decision_policies = [
        {
            "model": entry["model"],
            "threshold": entry.get("threshold"),
            "thresholded_selection_metrics": entry.get("thresholded_selection_metrics"),
        }
        for entry in payload["results"]
        if "threshold" in entry
    ]
    if decision_policies:
        lines += [
            "",
            "## Decision policies",
            "",
            _json_block(decision_policies),
        ]
    if payload.get("paired_comparisons"):
        lines += [
            "",
            "## Paired ablation comparisons",
            "",
            "Exploratory matched-fold Wilcoxon comparisons. Fold-level effects are "
            "reported with raw and Holm-adjusted p-values; shared training data and "
            "small fold counts limit inferential strength.",
            "",
            _paired_comparisons_table(payload["paired_comparisons"]),
        ]
    lines += [
        "",
        "## Environment",
        "",
        _json_block(payload["environment"]),
    ]
    if payload.get("site_composition"):
        lines += ["", "## Site composition", "", _json_block(payload["site_composition"])]
    if payload.get("per_site"):
        lines += ["", "## Per-site performance", "", _json_block(payload["per_site"])]
    return "\n".join(lines) + "\n"


def write_markdown(path: str | Path, payload: dict[str, object]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render_markdown(payload), encoding="utf-8")
    return target


def _json_block(value: object) -> str:
    return "```json\n" + json.dumps(value, indent=2, allow_nan=False) + "\n```"


def _paired_comparisons_table(comparisons: list[dict[str, object]]) -> str:
    """Render paired-comparison effects without hiding unavailable p-values."""

    lines = [
        "| Reference | Candidate | Metric | Pairs | Mean Δ | raw p | Holm p | Status |",
        "|---|---|---|---:|---:|---:|---:|---|",
    ]
    for comparison in comparisons:
        raw_p = comparison.get("p_value")
        holm_p = comparison.get("p_value_holm")
        mean = comparison.get("mean_delta")
        lines.append(
            "| {reference} | {candidate} | {metric} | {pairs} | {mean} | {raw} | {holm} | {status} |".format(
                reference=comparison.get("reference", "-"),
                candidate=comparison.get("candidate", "-"),
                metric=comparison.get("metric", "-"),
                pairs=comparison.get("n_pairs", 0),
                mean="-" if mean is None else format(float(mean), "+.4f"),
                raw="-" if raw_p is None else format(float(raw_p), ".4f"),
                holm="-" if holm_p is None else format(float(holm_p), ".4f"),
                status=comparison.get("status", "unknown"),
            )
        )
        for caveat in comparison.get("caveats", []):
            lines.append(f"> {caveat}")
        if comparison.get("reason"):
            lines.append(f"> {comparison['reason']}")
    return "\n".join(lines)


def _results_table(results: list[dict[str, object]]) -> str:
    header = (
        "| Model | Task | Split | Features | ACC | AUC | F1 | bal.ACC | "
        "sens | spec | folds |"
    )
    rule = "|" + "---|" * 11
    lines = [header, rule]
    for entry in results:
        aggregate = entry["aggregate"]
        folds = entry.get("folds", [])
        lines.append(
            "| {model} | {task} | {split} | {view} | {acc:.3f} | {auc} | {f1:.3f} | "
            "{bal:.3f} | {sens} | {spec} | {nfolds} |".format(
                model=entry["model"],
                task=entry["task"],
                split=entry["split_strategy"],
                view=entry["feature_view"],
                acc=float(aggregate["accuracy"]),
                auc=(f"{float(aggregate['roc_auc']):.3f}" if aggregate.get("roc_auc") is not None else "-"),
                f1=float(aggregate["macro_f1"]),
                bal=float(aggregate["balanced_accuracy"]),
                sens=(
                    f"{float(aggregate['sensitivity']):.3f}"
                    if aggregate.get("sensitivity") is not None
                    else "-"
                ),
                spec=(
                    f"{float(aggregate['specificity']):.3f}"
                    if aggregate.get("specificity") is not None
                    else "-"
                ),
                nfolds=len(folds),
            )
        )
    return "\n".join(lines)
