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


def test_periodic_state_projects_only_periodic_translations():
    state = State(
        numbers=np.full(4, 18),
        positions=_tetrahedron().positions,
        cell=np.diag([5.0, 5.0, 10.0]),
        pbc=(True, True, False),
    )
    direction = np.tile(np.array([1.0, 1.0, 1.0]), state.n_atoms)

    projected = project_out_rigid_body_modes(state, direction)

    projected = projected.reshape(state.n_atoms, 3)
    np.testing.assert_allclose(projected[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(projected[:, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(projected[:, 2], 1.0)
