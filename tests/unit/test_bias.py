import numpy as np

from pamssw.bias import GaussianBiasTerm


def test_gaussian_bias_lowers_directional_curvature_at_center():
    center = np.zeros(6)
    direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    term = GaussianBiasTerm(center=center, direction=direction, sigma=0.5, weight=0.4)

    corrected = term.directional_curvature_shift()

    assert np.isclose(corrected, -1.6)
