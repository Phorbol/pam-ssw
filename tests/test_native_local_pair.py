import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from pamssw.standalone.native_local_pair import native_local_pair


def test_local_pair_matches_finite_geometry_and_records_draws():
    atoms = Atoms("C4", positions=[[0, 0, 0], [4, 0, 0], [0, 1.2, 0], [4, 1.2, 0]])
    rec = native_local_pair(atoms, (0, 1), rng=iter([0.1, 0.9, 0.1, 0.9]))
    assert rec.raw_direction.shape == (4, 3)
    assert rec.draw_count == 2
    assert rec.accepted_count <= 8
    assert rec.selections
    assert np.allclose(atoms.positions, [[0, 0, 0], [4, 0, 0], [0, 1.2, 0], [4, 1.2, 0]])


def test_pair_gate_rejects_too_close_pair_before_draws():
    atoms = Atoms("C3", positions=[[0, 0, 0], [.5, 0, 0], [1.2, 2, 0]])
    rec = native_local_pair(atoms, (0, 1), rng=iter(()))
    assert np.all(np.isfinite(rec.raw_direction))
    assert rec.accepted_count == 0


def test_bool_mask_is_applied_and_constraints_are_rejected():
    atoms = Atoms("C2", positions=[[0, 0, 0], [2, 0, 0]])
    mask = np.array([True, False, True, True, True, True], dtype=bool)
    rec = native_local_pair(atoms, (0, 1), rng=iter([0.5, 0.5]), freedom_mask=mask)
    assert rec.raw_direction[0, 1] == 0
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match="constraints"):
        native_local_pair(atoms, (0, 1), rng=iter([0.5, 0.5]))


def test_neighbor_lists_and_marker_follow_archived_signs():
    atoms = Atoms("C4", positions=[[0, 0, 0], [4, 0, 0], [0, 1.2, 0], [4, 1.2, 0]])
    plus = native_local_pair(atoms, (0, 1), rng=iter([0.5, 0.5]), marker=1)
    minus = native_local_pair(atoms, (0, 1), rng=iter([0.5, 0.5]), marker=-1)
    assert plus.neighbor_lists == ((2,), (3,))
    assert plus.draw_count == 2
    assert np.allclose(plus.raw_direction, [[1, 0, 0], [-1, 0, 0],
                                             [0.766261028176921, -0.229878308453076, 0],
                                             [-0.766261028176921, -0.229878308453076, 0]])
    assert np.allclose(minus.raw_direction, -plus.raw_direction)
    assert plus.selections[0]["status"] == "accepted"


def test_each_endpoint_has_independent_four_accept_cap():
    positions = [[0, 0, 0], [10, 0, 0]]
    positions += [[0, 1.2, 0], [0, -1.2, 0], [0, 0, 1.2], [0, 0, -1.2]]
    positions += [[10, 1.2, 0], [10, -1.2, 0], [10, 0, 1.2], [10, 0, -1.2]]
    atoms = Atoms("C10", positions=positions)
    draws = [0.01, 0.26, 0.51, 0.76] * 2
    rec = native_local_pair(atoms, (0, 1), rng=iter(draws))
    assert rec.draw_count == 8
    assert rec.accepted_count == 8


def test_pair_gate_exact_c_radius_is_accepted():
    # C radius=.76 A from the recovered species_radius table: .6*(.76+.76)=.912.
    atoms = Atoms("C2", positions=[[0, 0, 0], [.912, 0, 0]])
    rec = native_local_pair(atoms, (0, 1), rng=iter([.5, .5]))
    assert np.allclose(rec.raw_direction[:2], [[1, 0, 0], [-1, 0, 0]])
