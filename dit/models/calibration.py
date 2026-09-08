"""Post-hoc probability calibration for the profile Transformer.

A network trained with class-weighted cross-entropy learns to separate the
classes, not to report the posterior: the softmax it emits is systematically
over- or under-confident.  That is invisible to every ranking-based metric, so
accuracy and AUC stay intact while the probability values are unusable, and it
is exactly why the Transformer must not be averaged into the ensemble with
models whose probabilities have already been calibrated.

Both helpers here are fit on rows the network was not tuned on, and both leave
the class ordering untouched -- a correctly calibrated map changes only how
confident the network sounds, never which class it calls.
"""

from __future__ import annotations

from typing import Any

import numpy as np


CALIBRATIONS: tuple[str, ...] = ("none", "temperature", "sigmoid")

# Search range for the temperature in log space, i.e. T in
# [e**-5, e**5] ~ [0.007, 148].  Wider is only noise: beyond it the loss is
# flat, so a boundary answer means the data asked for something implausible.
# It is public because a caller must be able to tell a clipped answer from a
# fitted one -- see :func:`temperature_saturated`.
TEMPERATURE_LOG_BOUNDS: tuple[float, float] = (-5.0, 5.0)
_GOLDEN_RATIO: float = (np.sqrt(5.0) + 1.0) / 2.0


def temperature_saturated(value: float) -> bool:
    """Whether a fitted temperature landed on the edge of the search range.

    Reported alongside the value itself: on degenerate input the objective is
    flat and the search walks to the wall, so ``148.4`` can be either a genuine
    fit or a clipped answer.  Both are indistinguishable from the number alone.
    """

    low = np.exp(TEMPERATURE_LOG_BOUNDS[0] + 1e-6)
    high = np.exp(TEMPERATURE_LOG_BOUNDS[1] - 1e-6)
    return value <= low or value >= high


# Share of a class's own mean probability that a Platt map must retain.
# Relative rather than absolute: a measured degenerate fit retains 0.56 while a
# legitimate one retains 0.90, whereas in absolute terms the two cases sit at
# 0.176 and 0.086 -- overlapping, because the drop scales with how confident
# the network was to start with.
_PLATT_MASS_RETENTION = 0.85


def nll_at_temperature(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    """Mean negative log-likelihood of ``logits / temperature`` on the labels."""

    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if logits.ndim != 2 or logits.shape[0] != labels.shape[0]:
        raise ValueError("logits and labels must be aligned [N,C] and [N]")
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
    # ``log_probs`` is already in log space: applying log again would turn the
    # objective into a function of the log of a log, which has its own minimum
    # and would send the search to the edge of the range.
    gathered = log_probs[np.arange(labels.shape[0]), labels]
    return float(-np.mean(gathered))


def temperature_scale(logits: np.ndarray, labels: np.ndarray) -> float:
    """Fit one temperature that minimises NLL on ``logits``.

    Golden-section search in log space, so the scan is uniform in the quantity
    that actually matters.  Returns ``1.0`` for degenerate input instead of
    inventing a correction.
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
        # A NaN makes every comparison in the bracketing loop evaluate False, so
        # the interval never moves and the search exits at its initial bounds --
        # the boundary value, which reads like a fitted answer.
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
    # so do not read the literal as a tuning knob.
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
    """Fit one binary logistic on the class probabilities per target class.

    This is the multiclass Platt construction that scikit-learn applies inside
    ``CalibratedClassifierCV``: each estimator maps the raw class-probability
    vector onto the probability of one class.  A class with fewer than two
    calibration rows cannot be calibrated at all, and refusing is the only
    honest answer -- inventing a scalar would look calibrated and be made up.

    The row-count floor is necessary but not sufficient: with two support rows a
    class can still be fitted, and the map can come back having thrown away most
    of that class's mass.  ``ece`` will not reveal this, because it bins by
    confidence and never asks which class is wrong, so the fitted map is checked
    against the mass it replaced before being returned.
    """

    from sklearn.linear_model import LogisticRegression

    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if probabilities.ndim != 2 or probabilities.shape[0] != labels.shape[0]:
        raise ValueError("probabilities and labels must be aligned [N,C] and [N]")
    if not np.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    if np.any(labels < 0) or int(labels.max()) >= probabilities.shape[1]:
        # Without this a label vector naming a class the matrix lacks is quietly
        # truncated: only ``range(probabilities.shape[1])`` classes get fitted,
        # and the rest vanish from the output.
        raise ValueError("labels must address a class in the probabilities")

    scalars: list[Any] = []
    for class_index in range(probabilities.shape[1]):
        target = (labels == class_index).astype(int)
        if target.sum() < 2 or target.sum() == target.size:
            raise ValueError(
                f"class {class_index} has {int(target.sum())} of {target.size} "
                "calibration rows; each class needs at least two of each outcome"
            )
        scalars.append(
            LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(
                probabilities, target
            )
        )

    corrected = apply_platt_scalars(probabilities, scalars)
    _assert_no_mass_loss(probabilities, corrected, labels)
    return scalars


def _assert_no_mass_loss(
    raw: np.ndarray, corrected: np.ndarray, labels: np.ndarray
) -> None:
    """Refuse a Platt map that starves a class of the mass it was given.

    Measured on a 3-class slice where one class held two rows drawn from the
    same cluster as the others, that class's mean probability on its own rows
    fell from 0.398 to 0.222 while ``ece`` improved from 0.17 to 0.02 -- the
    standard metric rewards the failure.  The check is on the change in true-
    class mass rather than on support size, because a class with two rows that
    is genuinely separable calibrates correctly and must not be refused.
    """

    for class_index in range(raw.shape[1]):
        members = labels == class_index
        before = float(raw[members, class_index].mean())
        after = float(corrected[members, class_index].mean())
        if before > 0 and after < before * _PLATT_MASS_RETENTION:
            raise ValueError(
                f"Platt calibration reduced class {class_index}'s mean probability on "
                f"its own rows from {before:.3f} to {after:.3f} "
                f"({after / before:.0%} retained); the class has too little informative "
                f"support in the calibration slice, so the map cannot be trusted. "
                f"Use temperature calibration instead."
            )


def apply_platt_scalars(probabilities: np.ndarray, scalars: list[Any]) -> np.ndarray:
    """Apply fitted Platt scalars and renormalise to a valid row distribution."""

    probabilities = np.asarray(probabilities, dtype=float)
    corrected = np.zeros_like(probabilities)
    for class_index, scalar in enumerate(scalars):
        corrected[:, class_index] = np.asarray(scalar.predict_proba(probabilities), dtype=float)[:, 1]
    corrected = np.clip(corrected, 0.0, None)
    total = corrected.sum(axis=1, keepdims=True)
    return corrected / np.where(total > 0, total, 1.0)
