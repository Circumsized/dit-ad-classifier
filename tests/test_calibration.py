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


@pytest.mark.parametrize("positive_probability", [0.0, np.nan, np.inf])
def test_apply_refuses_invalid_calibrated_mass(positive_probability: float) -> None:
    class InvalidMap:
        def predict_proba(self, probabilities):
            return np.tile([1.0, positive_probability], (probabilities.shape[0], 1))

    expected_message = "calibration map" if not np.isfinite(positive_probability) else "calibrated probabilities"
    with pytest.raises(ValueError, match=expected_message):
        apply_platt_scalars(np.full((4, 2), 0.5), [InvalidMap(), InvalidMap()])


class _FixedMap:
    def __init__(self, positive: float):
        self.positive = positive

    def predict_proba(self, probabilities):
        return np.column_stack(
            [np.full(probabilities.shape[0], 1.0 - self.positive),
             np.full(probabilities.shape[0], self.positive)]
        )


def test_apply_requires_one_map_per_class() -> None:
    with pytest.raises(ValueError, match="one scalar per probability column"):
        apply_platt_scalars(np.full((4, 2), 0.5), [_FixedMap(0.5)])


def test_apply_rejects_invalid_input_probabilities() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        apply_platt_scalars(np.array([[0.8, 0.8], [0.5, 0.5]]), [_FixedMap(0.5), _FixedMap(0.5)])


def test_apply_rejects_negative_or_nonfinite_map_output() -> None:
    for positive in (-0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="calibration map"):
            apply_platt_scalars(
                np.full((4, 2), 0.5), [_FixedMap(positive), _FixedMap(positive)]
            )


def test_apply_is_deterministic() -> None:
    logits, labels = _true_logits(400, 3, seed=12)
    probabilities = _softmax(logits)
    scalars = fit_platt_scalars(probabilities, labels)
    assert np.allclose(
        apply_platt_scalars(probabilities, scalars), apply_platt_scalars(probabilities, scalars)
    )


def _separable_minority(n_minority: int, *, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Valid probabilities with a distinct last-class cluster."""

    rng = np.random.default_rng(seed)
    rows = rng.dirichlet([10.0] * 3, size=18 + n_minority)
    rows[-n_minority:] = rng.dirichlet([1.0, 1.0, 30.0], size=n_minority)
    labels = np.array([0, 1] * 9 + [2] * n_minority, dtype=int)
    return rows, labels


class TestCalibrationPosteriors:
    @pytest.mark.parametrize("n_minority", [2, 3, 4])
    def test_constant_scores_recover_class_priors(self, n_minority: int) -> None:
        labels = np.array([0] * 9 + [1] * 9 + [2] * n_minority)
        raw = np.tile([0.3, 0.3, 0.4], (labels.size, 1))
        corrected = apply_platt_scalars(raw, fit_platt_scalars(raw, labels))
        expected = np.bincount(labels, minlength=3) / labels.size
        assert corrected == pytest.approx(np.tile(expected, (labels.size, 1)), abs=1e-3)

    def test_accepts_legitimate_overconfidence_correction(self) -> None:
        raw = np.tile([0.9, 0.1], (200, 1))
        labels = np.tile([0, 1], 100)
        scalars = fit_platt_scalars(raw, labels)
        held_out = np.tile([0.9, 0.1], (40, 1))
        held_out_labels = np.tile([0, 1], 20)
        corrected = apply_platt_scalars(held_out, scalars)
        assert corrected == pytest.approx(np.full((40, 2), 0.5), abs=1e-3)
        rows = np.arange(held_out_labels.size)
        raw_nll = -np.log(held_out[rows, held_out_labels]).mean()
        corrected_nll = -np.log(corrected[rows, held_out_labels]).mean()
        assert corrected_nll == pytest.approx(np.log(2), abs=1e-3)
        assert corrected_nll < raw_nll
        assert expected_calibration_error(held_out_labels, corrected) < 1e-3

    @pytest.mark.parametrize("n_minority", [2, 3, 4, 5])
    def test_accepts_a_rare_class_that_is_separable(self, n_minority: int) -> None:
        probabilities, labels = _separable_minority(n_minority, seed=7)
        corrected = apply_platt_scalars(probabilities, fit_platt_scalars(probabilities, labels))
        assert np.isfinite(corrected).all()
        assert np.allclose(corrected.sum(axis=1), 1.0)

    def test_a_well_behaved_fit_is_not_flagged(self) -> None:
        logits, labels = _true_logits(3000, 3, seed=4)
        probabilities = _softmax(logits * 2.0)
        assert len(fit_platt_scalars(probabilities, labels)) == 3


class TestInputValidation:
    @pytest.mark.parametrize("labels", [np.array([0, 1, 1, 1]), np.array([0, 0, 0, 1])])
    def test_platt_requires_two_positive_and_two_negative_rows(self, labels) -> None:
        probabilities = np.full((labels.size, 2), 0.5)
        with pytest.raises(ValueError, match="calibration rows"):
            fit_platt_scalars(probabilities, labels)

    @pytest.mark.parametrize("bad_value", [-0.1, 1.1])
    def test_platt_refuses_probabilities_outside_unit_interval(self, bad_value: float) -> None:
        probabilities = np.full((12, 2), 0.5)
        probabilities[0] = [bad_value, 1.0 - bad_value]
        with pytest.raises(ValueError, match="probabilities"):
            fit_platt_scalars(probabilities, np.array([0, 1] * 6))

    def test_platt_refuses_non_normalized_probabilities(self) -> None:
        with pytest.raises(ValueError, match="probabilities"):
            fit_platt_scalars(np.full((12, 2), 0.2), np.array([0, 1] * 6))

    def test_platt_refuses_empty_calibration_rows(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            fit_platt_scalars(np.empty((0, 2)), np.empty(0, dtype=int))

    def test_platt_refuses_labels_beyond_the_matrix(self) -> None:
        # Two columns, labels naming class 2: previously only two classes were
        # fitted and the rest vanished from the output with no warning.
        probabilities = np.full((10, 2), 0.5)
        probabilities[0] = [0.9, 0.1]
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
