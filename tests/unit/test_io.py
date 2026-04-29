import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import write

from pamssw import read_state, state_from_atoms, state_to_atoms


def test_state_from_atoms_preserves_cell_pbc_and_fixatoms():
    atoms = Atoms(
        numbers=[46, 8, 8],
        positions=[[0.0, 0.0, 0.0], [1.9, 0.0, 0.0], [0.0, 1.9, 0.0]],
        cell=np.diag([5.0, 5.0, 12.0]),
        pbc=[True, True, False],
    )
    atoms.set_constraint(FixAtoms(indices=[0]))

    state = state_from_atoms(atoms)

    assert state.numbers.tolist() == [46, 8, 8]
    assert np.allclose(state.positions, atoms.positions)
    assert np.allclose(state.cell, atoms.cell.array)
    assert state.pbc == (True, True, False)
    assert state.fixed_mask.tolist() == [True, False, False]


def test_read_state_reads_any_ase_supported_file(tmp_path):
    path = tmp_path / "structure.xyz"
    atoms = Atoms(numbers=[18, 18], positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
    write(path, atoms)

    state = read_state(path)

    assert state.numbers.tolist() == [18, 18]
    assert np.allclose(state.positions, atoms.positions)
    assert state.cell is None
    assert state.pbc == (False, False, False)


def test_state_to_atoms_round_trips_state_metadata():
    atoms = Atoms(
        numbers=[1, 1],
        positions=[[0.1, 0.0, 0.0], [9.8, 0.0, 0.0]],
        cell=np.diag([10.0, 10.0, 20.0]),
        pbc=[True, True, False],
    )
    state = state_from_atoms(atoms)

    round_trip = state_to_atoms(state)

    assert round_trip.numbers.tolist() == [1, 1]
    assert np.allclose(round_trip.positions, atoms.positions)
    assert np.allclose(round_trip.cell.array, atoms.cell.array)
    assert round_trip.pbc.tolist() == [True, True, False]
