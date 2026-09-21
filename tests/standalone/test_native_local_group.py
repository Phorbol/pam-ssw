import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from pamssw.standalone.native_local_group import native_local_group


def test_native_local_group_cross_product_and_masks():
    atoms = Atoms("H4", positions=[[0, 0, 0], [3, 4, 0], [9, 0, 0], [3, 1, 2]])
    groups = np.array([1, 1, 1, 1], dtype=np.int32)
    mask = np.ones(12, dtype=bool)
    mask[8] = False
    got = native_local_group(atoms, (0, 1), groups, mask)
    expected = np.zeros((4, 3))
    # Explicit component form keeps this check independent of np.cross.
    for k in (0, 1, 2, 3):
        a = atoms.positions[k] - atoms.positions[0]
        b = atoms.positions[k] - atoms.positions[1]
        expected[k] = [a[1] * b[2] - a[2] * b[1],
                       a[2] * b[0] - a[0] * b[2],
                       a[0] * b[1] - a[1] * b[0]]
    expected *= mask.reshape(4, 3)
    assert np.allclose(got, expected)
    assert np.allclose(got[3], [8, -6, -9])


def test_native_local_group_preserves_pair_distances_to_first_order():
    atoms = Atoms("H4", positions=[[0, 0, 0], [3, 4, 0], [9, 0, 0], [3, 1, 2]])
    got = native_local_group(atoms, (0, 1), np.ones(4, dtype=np.int32))
    for k in (2, 3):
        assert abs(np.dot(atoms.positions[k] - atoms.positions[0], got[k])) < 1e-12
        assert abs(np.dot(atoms.positions[k] - atoms.positions[1], got[k])) < 1e-12


def test_native_local_group_is_translation_and_rotation_covariant():
    atoms = Atoms("H4", positions=[[0, 0, 0], [3, 4, 0], [9, 0, 0], [3, 1, 2]])
    groups = np.array([0, 0, 1, 1], dtype=np.int32)
    base = native_local_group(atoms, (0, 1), groups)
    theta = 0.37
    rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                         [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    moved = Atoms("H4", positions=atoms.positions @ rotation.T + [11, -2, 5])
    transformed = native_local_group(moved, (0, 1), groups)
    assert np.allclose(transformed, base @ rotation.T)


def test_native_local_group_requires_explicit_pair_and_group_mask():
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError):
        native_local_group(atoms, (0, 1), np.ones(3, dtype=np.int32))
    with pytest.raises(ValueError):
        native_local_group(atoms, (0, 0), np.ones(2, dtype=np.int32))


def test_native_local_group_rejects_periodic_and_constrained_inputs():
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]], cell=np.eye(3), pbc=True)
    with pytest.raises(ValueError):
        native_local_group(atoms, (0, 1), np.ones(2, dtype=np.int32))
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError):
        native_local_group(atoms, (0, 1), np.ones(2, dtype=np.int32))
