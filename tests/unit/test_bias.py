import numpy as np

from pamssw.bias import GaussianBiasTerm


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
