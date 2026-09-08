"""Map model importance back to anatomical units."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from dit.data.layout import FeatureLayout


def block_permutation_importance(
    estimator: Callable,
    X: np.ndarray,
    y: np.ndarray,
    layout: FeatureLayout,
    *,
    n_repeats: int = 10,
    seed: int = 42,
    holdout_fraction: float = 0.3,
) -> dict[str, object]:
    """Shuffle each (tract, metric) block at a time and measure the score drop.

    Only the held-out rows are permuted, so the drop reflects out-of-sample
    dependence rather than how well the model memorised the training block.  A
    negative score means shuffling helped, which usually means the block is
    noise the model was fitting.
    """

    if not callable(getattr(estimator, "score", None)):
        raise ValueError("estimator must expose .score(X, y)")
    values = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y).reshape(-1)
    layout.validate(values.shape[1])
    if not (0.0 < holdout_fraction < 1.0):
        raise ValueError("holdout_fraction must be in (0, 1)")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    rng = np.random.default_rng(seed)
    order = rng.permutation(values.shape[0])
    cut = int(values.shape[0] * (1.0 - holdout_fraction))
    eval_pos = order[cut:]
    if eval_pos.size < 10:
        raise ValueError("holdout is too small for permutation importance")
    eval_x = values[eval_pos]
    eval_y = labels[eval_pos]

    baseline = float(estimator.score(eval_x, eval_y))
    rows: list[dict[str, object]] = []
    for block in layout.anatomical_blocks():
        columns = list(block.columns)
        drops: list[float] = []
        for _ in range(n_repeats):
            permuted = eval_x.copy()
            permutation = rng.permutation(eval_pos.size)
            permuted[:, columns] = eval_x[permutation][:, columns]
            drops.append(baseline - float(estimator.score(permuted, eval_y)))
        rows.append(
            {
                "block": block.name,
                "tract": block.tract_index,
                "metric": block.metric_index,
                "n_nodes": block.size,
                "importance": float(np.mean(drops)),
                "std": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
            }
        )
    rows.sort(key=lambda row: row["importance"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    return {
        "baseline_score": baseline,
        "n_repeats": n_repeats,
        "n_holdout": int(eval_pos.size),
        "blocks": rows,
        "top_blocks": [row["block"] for row in rows[:10]],
    }


def node_importance_from_coefficients(
    estimator,
    layout: FeatureLayout,
    *,
    model_step: str = "model",
) -> list[dict[str, object]]:
    """Per-node coefficient magnitudes from a fitted pipeline.

    Returns every anatomical column ranked by absolute coefficient, so the
    output is directly the "estimated predictors" list the competition asks for.
    Covariate and missing-pattern columns are excluded.
    """

    pipeline = _resolve_steps(estimator)
    model = pipeline[model_step] if model_step in pipeline else _last_model(pipeline)
    coef = _find_coefficients(model)
    if coef is None:
        raise ValueError(f"estimator step {model_step!r} exposes no coef_")
    magnitudes = np.abs(np.asarray(coef, dtype=float))
    if magnitudes.ndim == 2:
        magnitudes = magnitudes.max(axis=0)
    if magnitudes.size > layout.n_feature_columns:
        magnitudes = magnitudes[: layout.n_feature_columns]

    rows: list[dict[str, object]] = []
    for block in layout.anatomical_blocks():
        for offset, column in enumerate(block.columns):
            rows.append(
                {
                    "block": block.name,
                    "tract_index": block.tract_index,
                    "tract": layout.tract_names[block.tract_index],
                    "metric_index": block.metric_index,
                    "metric": layout.metric_names[block.metric_index],
                    "stat": block.stat,
                    "node": offset if block.size > 1 else None,
                    "column": int(column),
                    "magnitude": float(magnitudes[column]),
                }
            )
    rows.sort(key=lambda row: row["magnitude"], reverse=True)
    return rows


def _find_coefficients(model) -> np.ndarray | None:
    """Locate ``coef_``, descending through calibration and search wrappers."""

    current = model
    for _ in range(6):
        coef = getattr(current, "coef_", None)
        if coef is not None:
            return coef
        calibrated = getattr(current, "calibrated_classifiers_", None)
        if calibrated:
            current = calibrated[0].estimator
            continue
        best = getattr(current, "best_estimator_", None)
        if best is not None and best is not current:
            current = best
            continue
        break
    return None


def _resolve_steps(estimator) -> dict[str, object]:
    if hasattr(estimator, "steps"):
        return dict(estimator.steps)
    if hasattr(estimator, "best_estimator_"):
        inner = estimator.best_estimator_
        return dict(inner.steps) if hasattr(inner, "steps") else {"model": inner}
    return {"model": estimator}


def _last_model(pipeline: dict[str, object]) -> object:
    return list(pipeline.values())[-1]
