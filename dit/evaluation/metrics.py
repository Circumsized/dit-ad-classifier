"""Dependency-light classification metrics for reproducible reports."""

from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, labels=None) -> np.ndarray:
    true = np.asarray(y_true, dtype=int).reshape(-1)
    pred = np.asarray(y_pred, dtype=int).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    classes = np.asarray(sorted(set(true) | set(pred)) if labels is None else labels, dtype=int)
    lookup = {int(label): index for index, label in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for actual, guessed in zip(true, pred):
        if int(actual) in lookup and int(guessed) in lookup:
            matrix[lookup[int(actual)], lookup[int(guessed)]] += 1
    return matrix


def expected_calibration_error(y_true, probabilities, n_bins: int = 10) -> float:
    """Compute confidence-based expected calibration error."""

    true = np.asarray(y_true, dtype=int).reshape(-1)
    prob = _as_probability_matrix(probabilities)
    if prob.shape[0] != true.shape[0]:
        raise ValueError("probabilities and labels must have the same row count")
    confidence = np.max(prob, axis=1)
    correct = (np.argmax(prob, axis=1) == true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        if index == n_bins - 1:
            members = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            members = (confidence >= edges[index]) & (confidence < edges[index + 1])
        if np.any(members):
            ece += np.mean(members) * abs(np.mean(correct[members]) - np.mean(confidence[members]))
    return float(ece)


def classification_metrics(y_true, y_pred, probabilities=None) -> dict[str, object]:
    """Return competition and reliability metrics in JSON-friendly form."""

    true = np.asarray(y_true, dtype=int).reshape(-1)
    pred = np.asarray(y_pred, dtype=int).reshape(-1)
    if true.shape != pred.shape or true.size == 0:
        raise ValueError("non-empty y_true and y_pred must have the same shape")
    classes = np.asarray(sorted(set(true) | set(pred)), dtype=int)
    matrix = confusion_matrix(true, pred, labels=classes)
    recalls = np.divide(
        np.diag(matrix),
        matrix.sum(axis=1),
        out=np.full(len(classes), np.nan, dtype=float),
        where=matrix.sum(axis=1) > 0,
    )
    precisions = np.divide(
        np.diag(matrix),
        matrix.sum(axis=0),
        out=np.full(len(classes), np.nan, dtype=float),
        where=matrix.sum(axis=0) > 0,
    )
    f1 = np.divide(
        2 * precisions * recalls,
        precisions + recalls,
        out=np.zeros(len(classes), dtype=float),
        where=(precisions + recalls) > 0,
    )
    weights = matrix.sum(axis=1) / float(true.size)
    result: dict[str, object] = {
        "n_samples": int(true.size),
        "accuracy": float(np.mean(true == pred)),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "macro_f1": float(np.nanmean(f1)),
        # Weighted F1 sums per-class F1 weighted by class support; dividing by
        # the weights would cancel them and yield the macro mean instead.
        "weighted_f1": float(np.sum(f1 * weights)),
        "precision": {str(label): float(value) for label, value in zip(classes, precisions)},
        "class_recall": {str(label): float(value) for label, value in zip(classes, recalls)},
        "class_f1": {str(label): float(value) for label, value in zip(classes, f1)},
        "confusion_matrix": matrix.tolist(),
    }
    if len(classes) == 2:
        # Binary convention: class 0 is the reference, class 1 the disease.
        result["specificity"] = float(recalls[0])
        result["sensitivity"] = float(recalls[1])
        result["f1_disease"] = float(f1[1])
        result["precision_disease"] = float(precisions[1])

    if probabilities is not None:
        prob = _as_probability_matrix(probabilities)
        if prob.shape[0] != true.shape[0]:
            raise ValueError("probabilities and labels must have the same row count")
        if prob.shape[1] <= int(np.max(true)):
            raise ValueError("probability columns do not cover all labels")
        aucs = []
        for label in np.unique(true):
            target = (true == label).astype(int)
            if len(np.unique(target)) == 2:
                aucs.append(_binary_auc(target, prob[:, int(label)]))
        result["roc_auc"] = float(np.mean(aucs)) if aucs else None
        if len(classes) == 2:
            result["auc_disease"] = _binary_auc((true == classes[1]).astype(int), prob[:, int(classes[1])])
        one_hot = np.eye(prob.shape[1], dtype=float)[true]
        result["brier_score"] = float(np.mean(np.sum((prob - one_hot) ** 2, axis=1)))
        result["ece"] = expected_calibration_error(true, prob)
    return result


def competition_metrics(metrics: dict[str, object]) -> dict[str, float | None]:
    """Extract the three scores the competition reports.

    ACC, AUC and F-Score on both the binary and three-class tasks. Everything
    else in ``classification_metrics`` is diagnostic.
    """

    aucs = metrics.get("roc_auc")
    return {
        "accuracy": float(metrics["accuracy"]),
        "auc": None if aucs is None else float(aucs),
        "f1": float(metrics["macro_f1"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
    }


def _as_probability_matrix(probabilities) -> np.ndarray:
    prob = np.asarray(probabilities, dtype=float)
    if prob.ndim == 1:
        prob = np.column_stack((1.0 - prob, prob))
    if prob.ndim != 2 or prob.shape[1] < 2:
        raise ValueError("probabilities must have shape [N,C] or [N]")
    if np.any(~np.isfinite(prob)) or np.any(prob < -1e-7) or np.any(prob > 1 + 1e-7):
        raise ValueError("probabilities must be finite and in [0,1]")
    row_sum = prob.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("probability rows must have positive mass")
    return prob / row_sum


def _binary_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    positive = np.asarray(y_true, dtype=int) == 1
    n_pos = int(np.sum(positive))
    n_neg = int(positive.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = _average_ranks(np.asarray(score, dtype=float))
    rank_sum = float(np.sum(ranks[positive]))
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks
