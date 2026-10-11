"""Contract checks for the experimental fixed-radius shape pullback."""
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from pamssw.standalone.shape_softening import FrozenShapeBondSoftening
from pamssw.standalone.softening import FrozenBondSoftening, LSResponseState


ENERGIES = {(1, 1): 4.0}
LENGTHS = {(1, 1): 1.3}


def make(atoms):
    return FrozenShapeBondSoftening.from_atoms(
        atoms, bond_energies=ENERGIES, bond_lengths=LENGTHS
    )


def test_shape_pullback_force_matches_energy_finite_difference():
    atoms = Atoms('HHH', positions=[[0.0, 0.0, 0.0], [1.0, 0.1, 0.0],
                                    [0.2, 1.1, 0.3]])
    bias = make(atoms)
    evaluation = atoms.copy()
    evaluation.positions[2] += [0.35, -0.2, 0.25]
    energy, forces = bias.evaluate(evaluation)
    assert energy > 0
    step = 1e-6
    for atom in range(len(atoms)):
        for axis in range(3):
            plus, minus = evaluation.copy(), evaluation.copy()
            plus.positions[atom, axis] += step
            minus.positions[atom, axis] -= step
            numerical_force = -(bias.evaluate(plus)[0] - bias.evaluate(minus)[0]) / (2*step)
            assert forces[atom, axis] == pytest.approx(numerical_force, abs=2e-8)
    np.testing.assert_allclose(forces.sum(axis=0), 0.0, atol=1e-12)
    relative = evaluation.positions - evaluation.positions.mean(axis=0)
    assert np.sum(forces * relative) == pytest.approx(0.0, abs=1e-12)


def test_shape_pullback_force_rotates_with_geometry():
    atoms = Atoms('HHH', positions=[[0, 0, 0], [1, 0.1, 0], [0.2, 1.1, 0.3]])
    bias = make(atoms)
    evaluation = atoms.copy()
    evaluation.positions[2] += [0.3, -0.15, 0.2]
    _, forces = bias.evaluate(evaluation)
    angle = 0.73
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                         [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    rotated = evaluation.copy()
    rotated.positions[:] = evaluation.positions @ rotation.T + [1.2, -0.8, 0.5]
    rotated_energy, rotated_forces = bias.evaluate(rotated)
    energy, _ = bias.evaluate(evaluation)
    assert rotated_energy == pytest.approx(energy, abs=1e-12)
    np.testing.assert_allclose(rotated_forces, forces @ rotation.T, atol=1e-12)


def test_shape_energy_is_translation_and_uniform_scale_invariant():
    atoms = Atoms('HHH', positions=[[0, 0, 0], [1, 0.1, 0], [0.2, 1.1, 0.3]])
    bias = make(atoms)
    expected = bias.evaluate(atoms)[0]
    transformed = atoms.copy()
    transformed.positions[:] = 2.4 * atoms.positions + [3.1, -0.7, 1.8]
    assert bias.evaluate(transformed)[0] == pytest.approx(expected, abs=1e-12)


def test_response_rebuild_freezes_new_centered_radius():
    atoms = Atoms('HHH', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    current = make(atoms)
    following = atoms.copy()
    following.positions[:] = [[0, 0, 0], [2, 0, 0], [0, 1, 0]]
    controller = LSResponseState(target_per_atom=0.01)
    rebuilt = controller.update(current, following, energy_before=-1, energy_after=-1,
                                bond_energies=ENERGIES, bond_lengths=LENGTHS)
    expected_radius = np.linalg.norm(following.positions - following.positions.mean(axis=0))
    assert rebuilt.radius0 == pytest.approx(expected_radius)
    assert rebuilt.radius0 != pytest.approx(current.radius0)
    assert rebuilt.evaluate(following)[0] == pytest.approx(sum(rebuilt.strengths))


@pytest.mark.parametrize('change', ['periodic', 'constrained', 'collapsed'])
def test_shape_geometry_rejects_unsupported_or_singular_domain(change):
    atoms = Atoms('HH', positions=[[0, 0, 0], [1, 0, 0]])
    if change == 'periodic':
        atoms.pbc = True
    elif change == 'constrained':
        atoms.set_constraint(FixAtoms(indices=[0]))
    elif change == 'collapsed':
        atoms.positions[1] = atoms.positions[0]
    with pytest.raises(ValueError):
        make(atoms)


def test_direct_shape_constructor_rejects_periodic_frozen_state():
    atoms = Atoms('HH', positions=[[0, 0, 0], [1, 0, 0]], cell=[4, 4, 4], pbc=True)
    periodic = FrozenBondSoftening.from_atoms(
        atoms, bond_energies=ENERGIES, bond_lengths=LENGTHS
    )
    with pytest.raises(ValueError, match='periodic'):
        FrozenShapeBondSoftening(
            periodic.numbers, periodic.cell, periodic.pbc, periodic.pairs,
            periodic.reference_distances, periodic.strengths, periodic.xi,
            periodic.energy_filter, radius0=1.0,
        )
