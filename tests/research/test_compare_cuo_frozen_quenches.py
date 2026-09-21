import numpy as np

from research.ga_ssw.compare_cuo_frozen_quenches import _frozen_bias


def test_frozen_gaussian_changes_energy_and_gradient():
    q = np.array([1.0, 0.0])
    terms = [{"center": [0.0, 0.0], "direction": [1.0, 0.0],
              "weight": 2.0, "width": 1.0}]

    energy, gradient = _frozen_bias(3.0, np.zeros(2), q, terms)

    expected = 3.0 + 2.0 * np.exp(-.5)
    assert np.isclose(energy, expected)
    assert np.isclose(gradient[0], -2.0 * np.exp(-.5))
    assert np.isclose(gradient[1], 0.0)
    assert energy != 3.0
    assert not np.array_equal(gradient, np.zeros(2))
