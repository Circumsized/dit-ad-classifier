"""Decision-threshold selection on out-of-fold probabilities.

A raw ``argmax`` assumes the priors are equal and the likelihood ratio is
balanced.  In this dataset NC and AD differ in age and site composition, so the
cost of missing AD (sensitivity loss) is not the same as mislabeling a control.
The competition reports sensitivity and specificity separately, which means the
operating point matters, not just the ranking.

Thresholds may be fit on out-of-fold probabilities for a final deployment
policy: each probability was produced by a model that did not see its subject.
That does not make the policy's own selection score an unbiased evaluation
metric, because the threshold search still observes all OOF labels.  Evaluation
code must keep the deployment policy separate from the outer-fold performance
estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRIDS: dict[str, tuple[float, ...]] = {
    "f1": tuple(np.linspace(0.05, 0.95, 181).tolist()),
    "balanced": tuple(np.linspace(0.05, 0.95, 181).tolist()),
    "fixed": (0.5,),
}


@dataclass(frozen=True)
class ThresholdPolicy:
    """A fitted decision rule over class probabilities."""

    thresholds: tuple[float, ...]
    class_ids: tuple[int, ...]
    criterion: str
    score: float

    @property
    def n_classes(self) -> int:
        return len(self.thresholds)

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply the frozen thresholds to a probability matrix."""

        prob = np.asarray(probabilities, dtype=float)
        if prob.ndim != 2:
            raise ValueError("probabilities must be [N, C]")
        if prob.shape[1] != self.n_classes:
            raise ValueError(
                f"probabilities have {prob.shape[1]} columns, policy has "
                f"{self.n_classes}"
            )
        scores = prob - np.asarray(self.thresholds, dtype=float).reshape(1, -1)
        return np.asarray(self.class_ids, dtype=int)[np.argmax(scores, axis=1)]

    def parameters(self) -> dict[str, object]:
        return {
            "thresholds": dict(zip(self.class_ids, self.thresholds)),
            "criterion": self.criterion,
            "score": self.score,
        }


def _confusion(y: np.ndarray, pred: np.ndarray, label: int) -> tuple[int, int, int, int]:
    tp = int(np.sum((pred == label) & (y == label)))
    fp = int(np.sum((pred == label) & (y != label)))
    fn = int(np.sum((pred != label) & (y == label)))
    return tp, fp, fn, tp + fp + fn


def _score_confusion(y: np.ndarray, pred: np.ndarray, classes: list[int], criterion: str) -> float:
    if criterion == "balanced":
        # Mean of disease sensitivity and reference specificity: the two
        # quantities the competition reports alongside accuracy.
        disease = max(classes)
        tp, fp, fn, _ = _confusion(y, pred, disease)
        recall = tp / (tp + fn) if tp + fn else 0.0
        specificity = _tn(y, pred, disease) / (_tn(y, pred, disease) + fp) if (
            _tn(y, pred, disease) + fp
        ) else 0.0
        return float((recall + specificity) / 2.0)

    f1_scores: list[float] = []
    for label in classes:
        tp, fp, fn, _ = _confusion(y, pred, label)
        support = tp + fn
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / support if support else 0.0
        f1_scores.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return float(np.mean(f1_scores))


def _tn(y: np.ndarray, pred: np.ndarray, label: int) -> int:
    return int(np.sum((pred != label) & (y != label)))


def fit_threshold(
    y: np.ndarray,
    probabilities: np.ndarray,
    *,
    criterion: str = "f1",
    classes: list[int] | None = None,
) -> ThresholdPolicy:
    """Search per-class thresholds against out-of-fold probabilities.

    ``f1`` maximises the mean per-class F1, which is the metric the competition
    reports.  ``balanced`` maximises the mean of disease sensitivity and
    reference specificity.  ``fixed`` skips the search and keeps 0.5.

    The rule this search optimises is ``argmax(prob - t)`` with one offset
    ``t_c`` per class.  The offsets only matter *relative to each other*: a
    single shared constant shifts every column equally and provably cannot
    change a single prediction, which is why the search is per-class coordinate
    descent rather than one scalar swept over the grid.  Each pass tries every
    grid value for one class while holding the others fixed, and stops when a
    full pass improves nothing.  The starting point (all offsets 0.5) is the
    plain argmax rule, so the search can only move the score up, never down.
    """

    if criterion not in GRIDS:
        raise ValueError(f"unknown criterion {criterion!r}; use {sorted(GRIDS)}")
    labels = np.asarray(y, dtype=int).reshape(-1)
    prob = np.asarray(probabilities, dtype=float)
    if prob.ndim != 2:
        raise ValueError("probabilities must be [N, C]")
    if prob.shape[0] != labels.shape[0]:
        raise ValueError("y and probabilities must have the same row count")

    ids = tuple(sorted(set(labels.tolist())) if classes is None else classes)
    if len(ids) < 2:
        raise ValueError("threshold fitting needs at least two present classes")
    columns = [int(label) for label in ids]
    if any(label >= prob.shape[1] for label in columns):
        raise ValueError("probability matrix does not cover every label")

    class_ids = np.asarray(ids, dtype=int)

    def _predictions(offsets: list[float]) -> np.ndarray:
        scores = prob[:, columns] - np.asarray(offsets, dtype=float).reshape(1, -1)
        return class_ids[np.argmax(scores, axis=1)]

    def _score(offsets: list[float]) -> float:
        return _score_confusion(labels, _predictions(offsets), list(ids), criterion)

    if criterion == "fixed":
        offsets = [0.5] * len(ids)
        score = _score(offsets)
        if not np.isfinite(score):
            raise ValueError("threshold search produced no finite score")
        return ThresholdPolicy(tuple(offsets), tuple(ids), criterion, score)

    grid = GRIDS[criterion]
    offsets = [0.5] * len(ids)
    best_score = _score(offsets)
    if not np.isfinite(best_score):
        raise ValueError("threshold search produced no finite score")

    for _pass in range(2):
        improved = False
        for index in range(len(ids)):
            current = offsets[index]
            best_offset, best_for_class = current, best_score
            for candidate in grid:
                if candidate == current:
                    continue
                offsets[index] = candidate
                value = _score(offsets)
                if np.isfinite(value) and value > best_for_class + 1e-12:
                    best_offset, best_for_class = candidate, value
            offsets[index] = best_offset
            if best_for_class > best_score + 1e-12:
                best_score = best_for_class
                improved = True
        if not improved:
            break

    return ThresholdPolicy(tuple(offsets), tuple(ids), criterion, best_score)


def apply_policy(policy: ThresholdPolicy, probabilities: np.ndarray) -> np.ndarray:
    """Apply an already-fitted policy without selecting a new threshold."""

    return policy.predict(probabilities)


def threshold_sweep(
    y: np.ndarray,
    probabilities: np.ndarray,
    criterion: str = "f1",
    n_points: int = 33,
) -> list[dict[str, float]]:
    """Return sensitivity/specificity/F1 at a range of thresholds for plotting.

    The sweep moves the disease-class operating point: a row is predicted
    diseased when ``p_disease >= t``, otherwise it falls to the best remaining
    class.  Sweeping one shared constant off every column would leave the
    argmax -- and therefore every row of the table -- unchanged.
    """

    labels = np.asarray(y, dtype=int).reshape(-1)
    prob = np.asarray(probabilities, dtype=float)
    if prob.ndim != 2 or prob.shape[0] != labels.shape[0]:
        raise ValueError("y and probabilities must describe the same non-empty rows")
    if labels.size == 0:
        raise ValueError("y cannot be empty")
    classes = sorted(set(labels.tolist()))
    if len(classes) < 2:
        raise ValueError("threshold sweep needs at least two present classes")
    if any(label < 0 or label >= prob.shape[1] for label in classes):
        raise ValueError("probability matrix does not cover every label")
    disease = max(classes)
    other_columns = [column for column in classes if column != disease]
    other_ids = np.asarray(other_columns, dtype=int)
    grid = np.linspace(0.0, 1.0, n_points)
    rows: list[dict[str, float]] = []
    for candidate in grid:
        fallback = other_ids[np.argmax(prob[:, other_columns], axis=1)]
        predictions = np.where(prob[:, disease] >= candidate, disease, fallback)
        tp, fp, fn, _ = _confusion(labels, predictions, disease)
        support = tp + fn
        recall = tp / support if support else float("nan")
        tn = _tn(labels, predictions, disease)
        specificity = tn / (tn + fp) if tn + fp else float("nan")
        precision = tp / (tp + fp) if tp + fp else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else float("nan")
        )
        rows.append(
            {
                "threshold": float(candidate),
                "sensitivity": float(recall),
                "specificity": float(specificity),
                "f1_disease": float(f1),
                "accuracy": float(np.mean(predictions == labels)),
            }
        )
    return rows
