import numpy as np

from pamssw.pbc import mic_displacement, mic_distance_matrix, wrap_positions


def test_mic_displacement_wraps_periodic_axis_only():
    cell = np.diag([5.0, 5.0, 10.0])
    lhs = np.array([[0.2, 0.2, 9.5]])
    rhs = np.array([[4.8, 4.9, 0.5]])

    delta = mic_displacement(lhs, rhs, cell, (True, True, False))

    np.testing.assert_allclose(delta, np.array([[0.4, 0.3, 9.0]]))


def test_wrap_positions_wraps_periodic_axes_only():
    cell = np.diag([5.0, 5.0, 10.0])
    positions = np.array([[5.2, -0.2, 11.0]])

    wrapped = wrap_positions(positions, cell, (True, True, False))

    np.testing.assert_allclose(wrapped, np.array([[0.2, 4.8, 11.0]]))


def test_mic_distance_matrix_uses_short_periodic_distance():
    cell = np.diag([5.0, 5.0, 5.0])
    positions = np.array([[0.2, 0.0, 0.0], [4.8, 0.0, 0.0]])

    distances = mic_distance_matrix(positions, cell, (True, True, True))

    assert distances[0, 1] == distances[1, 0]
    np.testing.assert_allclose(distances[0, 1], 0.4)
