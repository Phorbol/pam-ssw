import numpy as np
from ase import Atoms
from pamssw.standalone.native_pair_group import native_local_pair_group


def test_pair_group_overlapping_masks_and_endpoint_override():
    atoms = Atoms('H4', positions=[[0, 0, 0], [2, 1, 0], [5, 5, 0], [9, 9, 9]])
    out = native_local_pair_group(atoms, (0, 1), [1, 1, 0, 0], [0, 1, 1, 0])
    u = np.array([2., 1., 0.]) / np.sqrt(5.)
    expected = np.zeros((4, 3)); expected[0] = 1.2*u; expected[1] = -1.2*u
    expected[2] = -u
    np.testing.assert_allclose(out, expected)


def test_pair_group_nonaxial_geometry_and_nonmember_initial():
    atoms = Atoms('H4', positions=[[1, 2, 3], [2, 4, 8], [20, 30, 40], [7, 8, 9]])
    initial = np.arange(12, dtype=float).reshape(4, 3)
    got = native_local_pair_group(atoms, (0, 1), [1, 0, 1, 0], [0, 1, 0, 0], initial)
    assert np.array_equal(got[3], initial[3])
    assert np.linalg.norm(got[2]) > 0

import pytest

@pytest.mark.parametrize('bad_pair', [(0.0, 1), (True, 1), (0, 4)])
def test_pair_group_rejects_non_integer_or_invalid_pair(bad_pair):
    atoms = Atoms('H4', positions=np.zeros((4, 3)))
    atoms.positions[1, 0] = 1
    with pytest.raises(ValueError):
        native_local_pair_group(atoms, bad_pair, [1]*4, [0]*4)


def test_pair_group_rejects_ase_constraints():
    atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
    from ase.constraints import FixAtoms
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match='constraints'):
        native_local_pair_group(atoms, (0, 1), [1, 1], [0, 0])
