import numpy as np

from pamssw.rigid import project_out_rigid_body_modes, rigid_body_overlap
from pamssw.state import State


def _tetrahedron() -> State:
    return State(
        numbers=np.full(4, 18),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    )


def test_project_out_rigid_body_modes_removes_translation_and_rotation():
    state = _tetrahedron()
    translation = np.tile(np.array([1.0, -0.5, 0.25]), state.n_atoms)
    centered = state.positions - state.positions.mean(axis=0, keepdims=True)
    rotation = np.cross(np.array([0.0, 0.0, 1.0]), centered).reshape(-1)
    internal = np.array([1.0, -1.0, 0.5, -0.5, 0.25, 0.75, -0.25, 0.5, -0.75, 0.1, -0.2, 0.3])
    direction = translation + rotation + internal

    projected = project_out_rigid_body_modes(state, direction)

    assert rigid_body_overlap(state, projected) < 1e-10
    assert np.linalg.norm(projected) > 1e-8


def test_rigid_projection_is_disabled_for_periodic_states():
    state = State(
        numbers=np.full(4, 18),
        positions=_tetrahedron().positions,
        cell=np.eye(3) * 10.0,
        pbc=(True, True, True),
    )
    direction = np.arange(state.n_atoms * 3, dtype=float) + 1.0

    projected = project_out_rigid_body_modes(state, direction)

    np.testing.assert_allclose(projected, direction)
