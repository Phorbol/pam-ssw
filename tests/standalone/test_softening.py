"""Numerical/contract checks only, not LS-SSW scientific validation."""
import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.softening import FrozenBondSoftening, LSResponseState

ENERGIES = {(1, 1): 4.0}
LENGTHS = {(1, 1): 1.2}


def make(atoms):
    return FrozenBondSoftening.from_atoms(
        atoms, bond_energies=ENERGIES, bond_lengths=LENGTHS
    )


def test_pair_potential_force_and_frozen_reference():
    initial = Atoms('HH', positions=[[0, 0, 0], [1, 0, 0]])
    soft = make(initial)
    moved = initial.copy()
    moved.positions[1, 0] = 1.3  # Beyond selection threshold: retained this step.
    energy, forces = soft.evaluate(moved)
    expected = 0.03 * 4.0 * np.exp(-0.3 / 0.2)
    assert energy == pytest.approx(expected)
    np.testing.assert_allclose(forces, [[-expected / 0.2, 0, 0], [expected / 0.2, 0, 0]])
    h = 1e-6
    for atom in range(2):
        for axis in range(3):
            plus, minus = moved.copy(), moved.copy()
            plus.positions[atom, axis] += h
            minus.positions[atom, axis] -= h
            numerical = -(soft.evaluate(plus)[0] - soft.evaluate(minus)[0]) / (2*h)
            assert forces[atom, axis] == pytest.approx(numerical, abs=1e-9)
    initial.positions[:] = 0  # Caller mutation cannot alter frozen references.
    assert soft.evaluate(moved)[0] == pytest.approx(expected)


def test_periodic_mic_translation_and_changed_cell_rejection():
    atoms = Atoms('HH', positions=[[0.2, 0, 0], [3.2, 0, 0]], cell=[4, 5, 6], pbc=True)
    soft = make(atoms)
    energy, forces = soft.evaluate(atoms)
    assert energy == pytest.approx(0.12)
    shifted = atoms.copy()
    shifted.positions[1] += shifted.cell[0]
    np.testing.assert_allclose(soft.evaluate(shifted)[1], forces, atol=1e-12)
    changed = atoms.copy()
    changed.cell[0, 0] += 0.01
    with pytest.raises(ValueError, match='cell'):
        soft.evaluate(changed)
    changed = atoms.copy()
    changed.pbc = False
    with pytest.raises(ValueError, match='periodic'):
        soft.evaluate(changed)


def test_response_uses_true_energy_and_renormalizes_next_neighbors():
    atoms = Atoms('HHH', positions=[[0, 0, 0], [1, 0, 0], [3, 0, 0]])
    current = make(atoms)  # One pair; total A=0.12 eV.
    following = atoms.copy()
    following.positions[2, 0] = 2  # Two bonds in next step.
    controller = LSResponseState(target_per_atom=0.02)
    updated = controller.update(current, following, energy_before=-5, energy_after=-4.97,
                                bond_energies=ENERGIES, bond_lengths=LENGTHS)
    # P=0.01 eV/atom; total A_next=0.12 - 3*1.8*(.01-.02)=.174.
    assert controller.last_response == pytest.approx(0.01)
    assert controller.steps == 1
    assert updated.strengths == pytest.approx((0.087, 0.087))
    assert updated.evaluate(following)[0] == pytest.approx(0.174)


def test_reject_negative_update_without_mutating_response_state():
    atoms = Atoms('HH', positions=[[0, 0, 0], [1, 0, 0]])
    controller = LSResponseState(target_per_atom=0.01)
    with pytest.raises(ValueError, match='strength'):
        controller.update(make(atoms), atoms, energy_before=0, energy_after=1,
                          bond_energies=ENERGIES, bond_lengths=LENGTHS)
    assert controller.steps == 0
    assert controller.last_response is None


def test_empty_missing_tables_and_changed_composition_rejected():
    with pytest.raises(ValueError, match='pairs'):
        make(Atoms('HH', positions=[[0, 0, 0], [3, 0, 0]]))
    with pytest.raises(ValueError, match='table'):
        FrozenBondSoftening.from_atoms(Atoms('HO', positions=[[0,0,0], [1,0,0]]),
                                       bond_energies=ENERGIES, bond_lengths=LENGTHS)
    atoms = Atoms('HH', positions=[[0,0,0], [1,0,0]])
    soft = make(atoms)
    changed = atoms.copy()
    changed.numbers[1] = 8
    with pytest.raises(ValueError, match='identity'):
        soft.evaluate(changed)
    changed = atoms.copy()
    changed.positions[1] = 0
    with pytest.raises(ValueError, match='distance'):
        soft.evaluate(changed)


def test_public_constructor_owns_mutable_inputs():
    atoms = Atoms('HH', positions=[[0,0,0], [1,0,0]])
    numbers = atoms.numbers.copy()
    cell = atoms.cell.array.copy()
    pbc = atoms.pbc.copy()
    pairs = [[0, 1]]
    references = [1.0]
    strengths = [0.12]
    soft = FrozenBondSoftening(numbers, cell, pbc, pairs, references, strengths)
    numbers[0] = 8;cell[0,0] = 10;pbc[0] = True
    pairs[0][0] = 1;references[0] = 2;strengths[0] = 9
    assert soft.evaluate(atoms)[0] == pytest.approx(.12)
    assert soft.pairs == ((0,1),)


@pytest.mark.parametrize('pairs', [[(-1,1)], [(0,2)], [(0,0)], [(True,0)], [(0,1),(1,0)]])
def test_public_constructor_rejects_invalid_or_duplicate_pairs(pairs):
    with pytest.raises(ValueError, match='pairs'):
        FrozenBondSoftening((1,1), ((0.,0.,0.),)*3, (False,)*3,
                            pairs, [1.]*len(pairs), [.12]*len(pairs))
