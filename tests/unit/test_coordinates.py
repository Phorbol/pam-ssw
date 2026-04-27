import numpy as np

from pamssw.coordinates import CartesianCoordinates, TangentVector
from pamssw.state import State


def test_cartesian_coordinates_round_trip_preserves_state_metadata():
    state = State(
        numbers=np.array([29, 8]),
        positions=np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 1.0]]),
        cell=np.diag([5.0, 5.0, 15.0]),
        pbc=(True, True, False),
        fixed_mask=np.array([True, False]),
    )

    coordinates = CartesianCoordinates.from_state(state)
    restored = coordinates.to_state(coordinates.values)

    assert restored.pbc == state.pbc
    np.testing.assert_allclose(restored.cell, state.cell)
    np.testing.assert_array_equal(restored.fixed_mask, state.fixed_mask)
    np.testing.assert_allclose(restored.positions, state.positions)


def test_cartesian_tangent_displaces_only_movable_atoms():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        fixed_mask=np.array([True, False]),
    )
    coordinates = CartesianCoordinates.from_state(state)
    tangent = TangentVector(np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0]))

    displaced = coordinates.displace(tangent, 0.5)

    np.testing.assert_allclose(displaced.positions[0], state.positions[0])
    np.testing.assert_allclose(displaced.positions[1], np.array([1.5, 1.0, 1.5]))
