"""Calibration primitive tests, checked against known-correct posteriors.

The temperature fit is validated against a construction with an exact answer:
logits produced by inflating a true log-posterior by a factor of ``k`` are
recalibrated by ``T = k``, so any other answer is wrong rather than merely
imprecise.
"""

from __future__ import annotations

import numpy as np
import pytest

from dit.evaluation.metrics import expected_calibration_error
from dit.models.calibration import (
    TEMPERATURE_LOG_BOUNDS,
    apply_platt_scalars,
    fit_platt_scalars,
    nll_at_temperature,
    temperature_saturated,
    temperature_scale,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


def _true_logits(n_samples: int, n_classes: int, *, seed: int = 0, concentration: float = 20.0) -> tuple[np.ndarray, np.ndarray]:
    """Logits whose softmax equals the distribution the labels were drawn from."""

    rng = np.random.default_rng(seed)
    posteriors = rng.dirichlet([concentration] * n_classes, size=n_samples)
    # Labels are sampled from the posterior rather than taken as its argmax.
    # A network that is always right is never punished for overconfidence, so
    # the fitted temperature just runs to the edge of the search range.
    labels = np.array([rng.choice(n_classes, p=row) for row in posteriors])
    return np.log(posteriors), labels


def test_temperature_recovers_a_known_inflation_factor() -> None:
    logits, labels = _true_logits(8000, 3)
    inflated = logits * 3.0
    assert temperature_scale(inflated, labels) == pytest.approx(3.0, abs=0.05)


def test_temperature_moves_in_the_right_direction() -> None:
    logits, labels = _true_logits(4000, 2, seed=1)
    assert temperature_scale(logits * 4.0, labels) > 1.0
    assert temperature_scale(logits / 4.0, labels) < 1.0


def test_fitted_temperature_beats_the_unscaled_loss() -> None:
    logits, labels = _true_logits(4000, 3, seed=2)
    inflated = logits * 2.5
    temperature = temperature_scale(inflated, labels)
    assert nll_at_temperature(inflated, labels, temperature) < nll_at_temperature(inflated, labels, 1.0)


def test_temperature_respects_the_search_range() -> None:
    logits, labels = _true_logits(2000, 2, seed=3)
    fitted = temperature_scale(logits * 10.0, labels)
    assert np.exp(-5.0) <= fitted <= np.exp(5.0)


def test_temperature_rejects_misaligned_inputs() -> None:
    logits, labels = _true_logits(200, 3)
    with pytest.raises(ValueError, match="aligned"):
        temperature_scale(logits, labels[:-1])
    with pytest.raises(ValueError, match="address a class"):
        temperature_scale(logits, labels + 1)
    with pytest.raises(ValueError, match="at least two"):
        temperature_scale(logits[:1], labels[:1])


def test_platt_scalars_shrink_the_calibration_error() -> None:
    logits, labels = _true_logits(3000, 3, seed=4)
    probabilities = _softmax(logits)
    inflated = _softmax(logits * 2.0)

    scalars = fit_platt_scalars(inflated, labels)
    corrected = apply_platt_scalars(inflated, scalars)
    assert expected_calibration_error(labels, corrected) < expected_calibration_error(labels, inflated)


def test_platt_output_is_a_valid_distribution() -> None:
    logits, labels = _true_logits(1200, 3, seed=5)
    probabilities = _softmax(logits)
    corrected = apply_platt_scalars(probabilities, fit_platt_scalars(probabilities, labels))
    assert np.all(corrected >= 0)
    assert np.allclose(corrected.sum(axis=1), 1.0)


def test_apply_matches_the_documented_construction() -> None:
    # Each class gets its own binary predictor, and the outputs are rescaled to
    # a distribution.  Pinned against a hand-built reference rather than a
    # tolerance on behaviour, which is what the wrapper must produce exactly.
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(11)
    probabilities = rng.dirichlet([1.0] * 3, size=60)
    labels = np.argmax(probabilities, axis=1)
    scalars = [
        LogisticRegression(C=np.inf, solver="lbfgs", max_iter=1000).fit(
            probabilities, (labels == class_index).astype(int)
        )
        for class_index in range(3)
    ]
    corrected = apply_platt_scalars(probabilities, scalars)
    expected = np.column_stack(
        [scalar.predict_proba(probabilities)[:, 1] for scalar in scalars]
    )
    expected = expected / expected.sum(axis=1, keepdims=True)
    assert np.allclose(corrected, expected)


def test_apply_is_deterministic() -> None:
    logits, labels = _true_logits(400, 3, seed=12)
    probabilities = _softmax(logits)
    scalars = fit_platt_scalars(probabilities, labels)
    assert np.allclose(
        apply_platt_scalars(probabilities, scalars), apply_platt_scalars(probabilities, scalars)
    )


def _uninformative_minority(n_minority: int, *, seed: int = 3, informative: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """A three-class slice whose last class is drawn from the same cluster."""

    rng = np.random.default_rng(seed)
    rows = rng.normal(size=(18 + n_minority, 3)) * 0.1 + 0.4
    if informative:
        rows[-n_minority:] = rng.normal(size=(n_minority, 3)) * 0.1 + [0.9, 0.05, 0.05]
    rows = rows / rows.sum(axis=1, keepdims=True)
    labels = np.array([0, 1] * (rows.shape[0] // 2), dtype=int)
    if labels.size < rows.shape[0]:
        labels = np.concatenate([labels, np.zeros(rows.shape[0] - labels.size, dtype=int)])
    labels[-n_minority:] = 2
    return rows, labels


class TestPlattMassLoss:
    """The map must not starve a class of the mass the network gave it."""

    @pytest.mark.parametrize("n_minority", [2, 3, 4])
    def test_refuses_when_a_rare_class_is_uninformative(self, n_minority: int) -> None:
        # Measured: class 2's mean probability on its own rows fell 0.398 ->
        # 0.222 while ece improved 0.17 -> 0.02, so the standard metric rewards
        # the failure and cannot be used as the check.
        probabilities, labels = _uninformative_minority(n_minority)
        with pytest.raises(ValueError, match="reduced class"):
            fit_platt_scalars(probabilities, labels)

    @pytest.mark.parametrize("n_minority", [2, 3, 4, 5])
    def test_accepts_a_rare_class_that_is_separable(self, n_minority: int) -> None:
        # The guard is on the change in mass, not on support size: two rows
        # that are genuinely distinguishable calibrate fine and must pass.
        probabilities, labels = _uninformative_minority(n_minority, seed=7, informative=True)
        scalars = fit_platt_scalars(probabilities, labels)
        assert len(scalars) == 3

    def test_a_well_behaved_fit_is_not_flagged(self) -> None:
        logits, labels = _true_logits(3000, 3, seed=4)
        probabilities = _softmax(logits * 2.0)
        assert len(fit_platt_scalars(probabilities, labels)) == 3


class TestInputValidation:
    def test_platt_refuses_labels_beyond_the_matrix(self) -> None:
        # Two columns, labels naming class 2: previously only two classes were
        # fitted and the rest vanished from the output with no warning.
        probabilities = np.full((10, 2), 0.5)
        probabilities[0, 0] = 0.9
        with pytest.raises(ValueError, match="must address a class"):
            fit_platt_scalars(probabilities, np.array([0, 1, 2] * 3 + [0]))

    def test_platt_refuses_non_finite_probabilities(self) -> None:
        probabilities = np.full((12, 3), 1 / 3)
        probabilities[4] = np.nan
        with pytest.raises(ValueError, match="finite"):
            fit_platt_scalars(probabilities, np.array([0, 1, 2] * 4))

    def test_temperature_refuses_non_finite_logits(self) -> None:
        logits, labels = _true_logits(200, 3, seed=13)
        logits[9] = np.nan
        with pytest.raises(ValueError, match="finite"):
            temperature_scale(logits, labels)

    def test_nll_refuses_misaligned_inputs(self) -> None:
        logits, labels = _true_logits(200, 3, seed=14)
        with pytest.raises(ValueError, match="aligned"):
            nll_at_temperature(logits, labels[:10], 1.0)

    def test_temperature_saturated_flags_the_search_boundaries(self) -> None:
        low = np.exp(TEMPERATURE_LOG_BOUNDS[0])
        high = np.exp(TEMPERATURE_LOG_BOUNDS[1])
        assert temperature_saturated(low)
        assert temperature_saturated(high)
        assert not temperature_saturated(1.0)
        assert not temperature_saturated(7.0)

    def test_temperature_reports_saturation_on_degenerate_input(self) -> None:
        # A flat objective has no minimum, so the search walks to the wall and
        # the answer must be labelled as clipped rather than fitted.
        logits = np.zeros((400, 2))
        labels = np.array([0, 1] * 200)
        fitted = temperature_scale(logits, labels)
        assert temperature_saturated(fitted)
        assert fitted == pytest.approx(np.exp(TEMPERATURE_LOG_BOUNDS[1]), rel=1e-6)


def test_platt_refuses_a_class_without_two_outcomes() -> None:
    probabilities = np.array([[0.9, 0.05, 0.05], [0.9, 0.05, 0.05]] * 5)
    labels = np.array([0, 1] * 5)
    with pytest.raises(ValueError, match="calibration rows"):
        fit_platt_scalars(probabilities, labels)


def test_platt_refuses_misaligned_inputs() -> None:
    probabilities = np.full((10, 3), 1 / 3)
    with pytest.raises(ValueError, match="aligned"):
        fit_platt_scalars(probabilities, np.arange(9))
