import numpy as np
import pytest

from pamssw.bias import GaussianBiasTerm, QuadraticBiasTerm
from pamssw.softening import LocalSofteningModel, PairSofteningTerm
from pamssw.state import State
from pamssw.walker import ProposalPotential


def test_gaussian_bias_lowers_directional_curvature_at_center():
    center = np.zeros(6)
    direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    term = GaussianBiasTerm(center=center, direction=direction, sigma=0.5, weight=0.4)

    corrected = term.directional_curvature_shift()

    assert np.isclose(corrected, -1.6)


def test_gaussian_bias_hvp_matches_center_curvature_shift():
    center = np.zeros(6)
    direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    term = GaussianBiasTerm(center=center, direction=direction, sigma=0.5, weight=0.4)

    hvp = term.hvp_contribution(direction, center)

    np.testing.assert_allclose(hvp, -1.6 * direction)
    assert float(direction @ hvp) == np.float64(term.directional_curvature_shift())


def test_gaussian_bias_hvp_matches_finite_difference_gradient():
    center = np.zeros(6)
    positions = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    bias_direction = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    probe_direction = np.array([0.5, -0.25, 0.0, 0.0, 0.0, 0.0])
    probe_direction = probe_direction / np.linalg.norm(probe_direction)
    term = GaussianBiasTerm(center=center, direction=bias_direction, sigma=0.7, weight=0.9)
    epsilon = 1e-6

    _, grad_plus = term.evaluate(positions + epsilon * probe_direction)
    _, grad_minus = term.evaluate(positions - epsilon * probe_direction)
    finite_difference = (grad_plus - grad_minus) / (2.0 * epsilon)

    np.testing.assert_allclose(
        term.hvp_contribution(probe_direction, positions),
        finite_difference,
        rtol=1e-6,
        atol=1e-8,
    )


def test_quadratic_bias_matches_gaussian_through_center_curvature():
    center = np.zeros(6)
    direction = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    gaussian = GaussianBiasTerm(
        center=center,
        direction=direction,
        sigma=0.7,
        weight=0.9,
    )
    quadratic = QuadraticBiasTerm(
        center=center,
        direction=direction,
        sigma=0.7,
        weight=0.9,
    )

    gaussian_energy, gaussian_gradient = gaussian.evaluate(center)
    quadratic_energy, quadratic_gradient = quadratic.evaluate(center)

    assert quadratic_energy == pytest.approx(gaussian_energy)
    np.testing.assert_allclose(quadratic_gradient, gaussian_gradient)
    assert quadratic.directional_curvature_shift() == pytest.approx(
        gaussian.directional_curvature_shift()
    )
    np.testing.assert_allclose(
        quadratic.hvp_contribution(quadratic.direction, center),
        gaussian.hvp_contribution(gaussian.direction, center),
    )


def test_quadratic_bias_hvp_matches_finite_difference_gradient():
    center = np.zeros(6)
    positions = np.array([0.2, 0.1, 0.0, 0.0, 0.0, 0.0])
    bias_direction = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    probe_direction = np.array([0.5, -0.25, 0.0, 0.0, 0.0, 0.0])
    probe_direction = probe_direction / np.linalg.norm(probe_direction)
    term = QuadraticBiasTerm(
        center=center,
        direction=bias_direction,
        sigma=0.7,
        weight=0.9,
    )
    epsilon = 1e-6

    _, grad_plus = term.evaluate(positions + epsilon * probe_direction)
    _, grad_minus = term.evaluate(positions - epsilon * probe_direction)
    finite_difference = (grad_plus - grad_minus) / (2.0 * epsilon)

    np.testing.assert_allclose(
        term.hvp_contribution(probe_direction, positions),
        finite_difference,
        rtol=1e-6,
        atol=1e-8,
    )


def test_proposal_parts_sum_true_multiple_bias_and_softening_with_one_raw_call():
    class CountingCalculator:
        def __init__(self):
            self.calls = 0

        def evaluate_flat(self, flat_positions, template):
            self.calls += 1
            return 1.25, np.array([0.4, -0.2, 0.1, -0.3, 0.5, -0.4])

    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.2, 0.0, 0.0], [1.2, 0.2, 0.0]]),
    )
    flat_positions = state.flatten_positions()
    calculator = CountingCalculator()
    potential = ProposalPotential(
        calculator,
        biases=[
            GaussianBiasTerm(center=np.zeros(6), direction=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), sigma=0.7, weight=0.4),
            GaussianBiasTerm(center=np.zeros(6), direction=np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]), sigma=0.9, weight=0.8),
        ],
        softening=LocalSofteningModel(
            [PairSofteningTerm(atom_i=0, atom_j=1, reference_distance=0.8, width=0.3, strength=0.5)]
        ),
    )

    assert hasattr(potential, "evaluate_parts")
    parts = potential.evaluate_parts(flat_positions, state)

    assert calculator.calls == 1
    assert parts.total_energy == pytest.approx(parts.true_energy + parts.bias_energy + parts.softening_energy)
    np.testing.assert_allclose(
        parts.total_gradient,
        parts.true_gradient + parts.bias_gradient + parts.softening_gradient,
    )


def test_proposal_parts_rejects_true_gradient_shape_mismatch():
    class MatrixGradientCalculator:
        def evaluate_flat(self, flat_positions, template):
            return 0.0, np.zeros((1, 3))

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    potential = ProposalPotential(MatrixGradientCalculator())

    with pytest.raises(ValueError, match="true_gradient.*flat_positions"):
        potential.evaluate_parts(state.flatten_positions(), state)


def test_proposal_batch_preserves_analytic_bias_and_softening_parts():
    class BatchCalculator:
        supports_batch_evaluation = True

        def evaluate_flat_many(self, flat_positions, templates):
            return tuple(
                (1.25, np.array([0.4, -0.2, 0.1, -0.3, 0.5, -0.4]))
                for _ in flat_positions
            )

    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.2, 0.0, 0.0], [1.2, 0.2, 0.0]]),
    )
    bias = GaussianBiasTerm(
        center=np.zeros(6),
        direction=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        sigma=0.7,
        weight=0.4,
    )
    softening = LocalSofteningModel(
        [
            PairSofteningTerm(
                atom_i=0,
                atom_j=1,
                reference_distance=0.8,
                width=0.3,
                strength=0.5,
            )
        ]
    )
    potential = ProposalPotential(
        BatchCalculator(),
        biases=[bias],
        softening=softening,
    )
    first = state.flatten_positions()
    second = first + np.array([0.01, 0.0, 0.0, 0.0, -0.02, 0.0])

    parts = potential.evaluate_parts_many((first, second), (state, state))

    assert len(parts) == 2
    for positions, value in zip((first, second), parts):
        bias_energy, bias_gradient = bias.evaluate(positions)
        softening_energy, softening_gradient = softening.evaluate(positions)
        assert value.true_energy == pytest.approx(1.25)
        assert value.bias_energy == pytest.approx(bias_energy)
        assert value.softening_energy == pytest.approx(softening_energy)
        np.testing.assert_allclose(value.bias_gradient, bias_gradient)
        np.testing.assert_allclose(value.softening_gradient, softening_gradient)
        np.testing.assert_allclose(
            value.total_gradient,
            value.true_gradient + bias_gradient + softening_gradient,
        )


def test_gaussian_bias_reports_mic_image_signature():
    term = GaussianBiasTerm(
        center=np.array([0.0, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        sigma=0.7,
        weight=0.4,
    )
    cell = np.diag([10.0, 10.0, 10.0])

    assert term.mic_image_signature(np.array([4.9, 0.0, 0.0]), cell, (True, True, True)) == (
        0,
        0,
        0,
    )
    assert term.mic_image_signature(np.array([5.1, 0.0, 0.0]), cell, (True, True, True)) == (
        1,
        0,
        0,
    )
    assert term.mic_image_signature(np.array([5.1, 0.0, 0.0]), cell, (False, False, False)) == ()
