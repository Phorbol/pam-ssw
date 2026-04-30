from __future__ import annotations

import numpy as np
import pytest

from pamssw.generalized_coordinates import (
    CellDOFMask,
    GeneralizedCoordinates,
    generalized_rigid_body_basis,
    project_out_generalized_rigid_modes,
)
from pamssw.state import State


def periodic_state() -> State:
    return State(
        numbers=np.array([6, 6, 8]),
        positions=np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5]]),
        cell=np.diag([5.0, 6.0, 7.0]),
        pbc=(True, True, True),
        fixed_mask=np.array([False, False, True]),
    )


def test_shape_6_pack_unpack_roundtrip() -> None:
    mask = CellDOFMask("shape_6")
    matrix = np.array([[0.1, 0.2, 0.3], [0.2, -0.1, 0.4], [0.3, 0.4, 0.05]])
    np.testing.assert_allclose(mask.unpack(mask.pack(matrix)), matrix)


def test_fixed_cell_roundtrip_preserves_state() -> None:
    state = periodic_state()
    gcoord = GeneralizedCoordinates.from_state(state, "fixed_cell")
    restored = gcoord.to_state(gcoord.to_q(state))
    np.testing.assert_allclose(restored.positions, state.positions)
    np.testing.assert_allclose(restored.cell, state.cell)


def test_volume_only_changes_volume() -> None:
    state = periodic_state()
    gcoord = GeneralizedCoordinates.from_state(state, "volume_only")
    q = gcoord.to_q(state)
    q[gcoord.atomic_size] = 0.1
    expanded = gcoord.to_state(q)
    assert np.linalg.det(expanded.cell) > np.linalg.det(state.cell)


def test_slab_xy_preserves_z_cell_vector() -> None:
    state = periodic_state()
    gcoord = GeneralizedCoordinates.from_state(state, "slab_xy")
    q = gcoord.to_q(state)
    q[gcoord.atomic_size : gcoord.atomic_size + 3] = [0.05, -0.02, 0.01]
    strained = gcoord.to_state(q)
    np.testing.assert_allclose(strained.cell[2], state.cell[2], atol=1e-12)


def test_fractional_wrap_only_periodic_axes() -> None:
    state = periodic_state()
    gcoord = GeneralizedCoordinates.from_state(state, "shape_6")
    q = gcoord.to_q(state)
    q[: gcoord.atomic_size] += 2.25
    wrapped = gcoord.fractional_wrap(q)
    assert np.all((wrapped[: gcoord.atomic_size] >= 0.0) & (wrapped[: gcoord.atomic_size] < 1.0))


def test_fixed_cartesian_semantics_keep_fixed_positions() -> None:
    state = periodic_state()
    gcoord = GeneralizedCoordinates.from_state(state, "volume_only", fixed_atom_cell_semantics="fixed_cartesian")
    q = gcoord.to_q(state)
    q[gcoord.atomic_size] = 0.2
    strained = gcoord.to_state(q)
    np.testing.assert_allclose(strained.positions[state.fixed_mask], state.positions[state.fixed_mask])


def test_generalized_rigid_projection_removes_fractional_translation() -> None:
    state = periodic_state()
    gcoord = GeneralizedCoordinates.from_state(state, "shape_6")
    basis = generalized_rigid_body_basis(gcoord)
    assert basis.shape == (gcoord.size, 3)
    direction = basis[:, 0] + 0.1 * np.ones(gcoord.size)
    projected = project_out_generalized_rigid_modes(gcoord, direction)
    assert abs(gcoord.metric.dot(projected, basis[:, 0])) < 1e-10


def test_no_active_dof_raises() -> None:
    state = periodic_state()
    state = State(state.numbers, state.positions, state.cell, state.pbc, np.ones(state.n_atoms, dtype=bool))
    with pytest.raises(ValueError):
        GeneralizedCoordinates.from_state(state, "fixed_cell")
