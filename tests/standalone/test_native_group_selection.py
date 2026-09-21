"""Geometry contracts; frozen pair/mask outputs from the uploaded ELF.

These tests establish neither PES-search efficacy nor the complete native
direction controller. The original-instruction comparison is archived separately.
"""
import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.native_local_group import select_native_local_group


def fixture():
    atoms = Atoms('H5', positions=[[0, 0, 0], [1, 0, 0], [0, 2, 0],
                                 [2, 2, 1], [-2, 1, 1]])
    reference = atoms.positions + np.arange(5)[:, None] * [.1, .02, .03]
    return atoms, reference


def test_matches_native_pair_and_group_for_noncollinear_input():
    atoms, reference = fixture()
    got = select_native_local_group(reference, atoms, iter([.5]))
    assert got.pair == (3, 4)  # Native one-based [4, 5].
    assert got.group_mask.tolist() == [0, 0, 1, 0, 1]
    assert got.draw_count == 1


def test_native_eligibility_uses_first_mask_component():
    atoms, reference = fixture()
    mask = np.array([[1, 0, 0], [0, 1, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]], bool)
    got = select_native_local_group(reference, atoms, iter([.5]), mask)
    assert got.pair == (4, 2)  # Native one-based [5, 3].
    assert got.group_mask.tolist() == [0, 0, 1, 0, 0]


def test_empty_candidate_set_does_not_draw_or_invent_an_axis():
    atoms = Atoms('H2', positions=[[0, 0, 0], [.5, 0, 0]])
    got = select_native_local_group(atoms.positions, atoms, iter([]))
    assert got.pair == (1, None)
    assert not got.group_mask.any()
    assert got.draw_count == 0


def test_selection_is_rigid_motion_invariant_away_from_ties():
    atoms, reference = fixture()
    base = select_native_local_group(reference, atoms, iter([.5]))
    rotation = np.array([[0, -1., 0], [1., 0, 0], [0, 0, 1.]])
    atoms.positions = atoms.positions @ rotation + [17., -23., 4.]
    changed = select_native_local_group(reference @ rotation + [17., -23., 4.],
                                       atoms, iter([.5]))
    assert changed.pair == base.pair
    assert np.array_equal(changed.group_mask, base.group_mask)


def test_unsupported_geometry_and_empty_active_set_are_explicit():
    atoms, reference = fixture()
    with pytest.raises(ValueError, match='active'):
        select_native_local_group(reference, atoms, iter([]), np.zeros((5, 3), bool))
    atoms.pbc = True
    with pytest.raises(ValueError, match='nonperiodic'):
        select_native_local_group(reference, atoms, iter([]))
