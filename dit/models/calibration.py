"""Post-hoc probability calibration for the profile Transformer.

Class weighting changes the posterior targeted by cross-entropy, so raw
softmax values need calibration before they enter a probability ensemble.
Accuracy and AUC alone do not establish probability calibration.

Both helpers are fit on rows outside training and early stopping. Temperature
scaling preserves argmax; the one-vs-rest logistic maps can change predictions.
"""

from __future__ import annotations

from typing import Any

import numpy as np


CALIBRATIONS: tuple[str, ...] = ("none", "temperature", "sigmoid")

# Search range for the temperature in log space, i.e. T in
# [e**-5, e**5] ~ [0.007, 148]. Beyond that the loss is flat, so a boundary
# answer means the data asked for something implausible. Public so a caller
# can tell a clipped answer from a fitted one; see :func:`temperature_saturated`.
TEMPERATURE_LOG_BOUNDS: tuple[float, float] = (-5.0, 5.0)
_GOLDEN_RATIO: float = (np.sqrt(5.0) + 1.0) / 2.0


def temperature_saturated(value: float) -> bool:
    """Whether a fitted temperature landed on the edge of the search range.

    Reported alongside the value: on degenerate input the objective is flat and
    the search walks to the wall, so ``148.4`` can be either a genuine fit or a
    clipped one, and the number alone does not distinguish them.
    """

    low = np.exp(TEMPERATURE_LOG_BOUNDS[0] + 1e-6)
    high = np.exp(TEMPERATURE_LOG_BOUNDS[1] - 1e-6)
    return value <= low or value >= high


def nll_at_temperature(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean negative log-likelihood of ``logits / temperature`` on the labels."""

    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must be aligned [N,C] and [N]")
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
    # ``log_probs`` is already in log space; applying log again would make the
    # objective a function of the log of a log, with its own minimum, and send
    # the search to the edge of the range.
    gathered = log_probs[np.arange(labels.shape[0]), labels]
    return float(-np.mean(gathered))


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one temperature that minimises NLL on ``logits``.

    Golden-section search in log space, so the scan is uniform in the quantity
    being fitted. Returns ``1.0`` for degenerate input.
    """

    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must be aligned [N,C] and [N]")
    if labels.shape[0] < 2:
        raise ValueError("temperature calibration needs at least two rows")
    if np.any(labels < 0) or int(labels.max()) >= logits.shape[1]:
        raise ValueError("labels must address a class in the logits")
    if not np.isfinite(logits).all():
        # Non-finite logits make all bracketing comparisons false, leaving the
        # interval at its initial bounds and returning a boundary value.
        raise ValueError(
            "logits must be finite; a non-finite value makes the objective "
            "undefined and the search returns the range boundary"
        )

    low, high = TEMPERATURE_LOG_BOUNDS
    first = high - (high - low) / _GOLDEN_RATIO
    second = low + (high - low) / _GOLDEN_RATIO
    first_nll = nll_at_temperature(logits, labels, np.exp(first))
    second_nll = nll_at_temperature(logits, labels, np.exp(second))

    # 200 is a safety cap; the convergence break below fires near iteration 50,
    # so the literal is not a tuning knob.
    for _ in range(200):
        if first_nll < second_nll:
            high, second, second_nll = second, first, first_nll
            first = high - (high - low) / _GOLDEN_RATIO
            first_nll = nll_at_temperature(logits, labels, np.exp(first))
        else:
            low = first
            first, first_nll = second, second_nll
            second = low + (high - low) / _GOLDEN_RATIO
            second_nll = nll_at_temperature(logits, labels, np.exp(second))
        if abs(high - low) < 1e-9:
            break

    return float(np.exp((low + high) / 2.0))


def fit_platt_scalars(probabilities: np.ndarray, labels: np.ndarray) -> list[Any]:
    """Fit one binary logistic map on the probability vector per target class.

    This is a multivariate one-vs-rest map, not a univariate Platt sigmoid.
    Each class needs two positive and two negative calibration rows. A lower
    true-class probability can correct overconfidence, so retaining a fixed
    fraction of the raw probability is not a validity condition. Small-sample
    generalization still requires validation outside this fitting slice.
    """

    from sklearn.linear_model import LogisticRegression

    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must be aligned [N,C] and [N]")
    if labels.size < 2:
        raise ValueError("sigmoid calibration needs at least two rows")
    if probabilities.shape[1] < 2:
        raise ValueError("probabilities must cover at least two classes")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if np.any((probabilities < 0) | (probabilities > 1)) or not np.allclose(
        probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6
    ):
        raise ValueError("probabilities must be in [0,1] and each row must sum to one")
    if np.any(labels < 0) or int(labels.max()) >= probabilities.shape[1]:
        # Without this, a label vector naming a class the matrix lacks is
        # truncated: only ``range(probabilities.shape[1])`` classes get fitted
        # and the rest are dropped from the output.
        raise ValueError("labels must address a class in the probabilities")

    scalars: list[Any] = []
    for class_index in range(probabilities.shape[1]):
        target = (labels == class_index).astype(int)
        positives = int(target.sum())
        if positives < 2 or target.size - positives < 2:
            raise ValueError(
                f"class {class_index} has {positives} of {target.size} "
                "calibration rows; each class needs at least two of each outcome"
            )
        scalars.append(
            LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(
                probabilities, target
            )
        )

    return scalars


def apply_platt_scalars(probabilities: np.ndarray, scalars: list[Any]) -> np.ndarray:
    """Apply fitted Platt scalars and renormalise to a valid row distribution."""

    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] < 2:
        raise ValueError("probabilities must be a two-dimensional matrix with at least two columns")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("probabilities rows must sum to one")
    if len(scalars) != probabilities.shape[1]:
        raise ValueError("one scalar per probability column is required")
    corrected = np.zeros_like(probabilities)
    for class_index, scalar in enumerate(scalars):
        corrected[:, class_index] = np.asarray(scalar.predict_proba(probabilities), dtype=float)[:, 1]
    if not np.isfinite(corrected).all() or np.any(corrected < 0):
        raise ValueError("calibration map must return finite, non-negative probabilities")
    total = corrected.sum(axis=1, keepdims=True)
    if np.any(~np.isfinite(total)) or np.any(total <= 0):
        raise ValueError("calibrated probabilities must have finite, positive row mass")
    return corrected / total
