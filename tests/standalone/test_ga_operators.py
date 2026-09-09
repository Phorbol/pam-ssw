"""Geometry checks on an existing water archive; no calculator or search runs."""
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.ga_operators import (
    cut_monomers, gametes_match, mutate_single_monomer,
    rotate_coordinates, SamplingExhausted,
)


@pytest.fixture
def water():
    # Real uploaded-run archive, not a generated all-atom model.
    path = Path(__file__).resolve().parents[2] / 'research/ga_ssw/evidence/staged-water/final-arc/0.arc'
    rows = [line.split() for line in path.read_text().splitlines() if 'CORE' in line]
    atoms = Atoms([row[0] for row in rows], positions=[[float(x) for x in row[1:4]] for row in rows])
    return atoms, tuple(tuple(range(i, i + 3)) for i in range(0, len(atoms), 3))


def test_original_rotation_is_rigid_and_reproducible(water):
    atoms, _ = water
    xyz = atoms.positions.copy()
    a = rotate_coordinates(xyz, np.random.default_rng(123))
    b = rotate_coordinates(xyz, np.random.default_rng(123))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_allclose(a @ a.T, xyz @ xyz.T, atol=1e-10)
    np.testing.assert_array_equal(atoms.positions, xyz)


def test_single_monomer_mutation_selects_best_and_rotates_about_origin(water):
    atoms, groups = water
    original = atoms.positions.copy()
    result = mutate_single_monomer([atoms, atoms.copy()], [2., -1.], groups, np.random.default_rng(15), max_attempts=1000)
    assert result.parent_index == 1
    selected = list(groups[result.group_index])
    rest = sorted(set(range(len(atoms))) - set(selected))
    np.testing.assert_array_equal(result.atoms.positions[rest], original[rest])
    before, after = original[selected], result.atoms.positions[selected]
    np.testing.assert_allclose(after @ after.T, before @ before.T, atol=1e-10)
    # The source rotates absolute coordinates, not about the monomer centroid.
    assert np.linalg.norm(after.mean(axis=0) - before.mean(axis=0)) > 1.
    np.testing.assert_array_equal(atoms.positions, original)
    assert result.atoms.calc is None


def test_cut_keeps_whole_molecules_and_recovered_restore_frame(water):
    atoms, groups = water
    original = atoms.positions.copy()
    result = cut_monomers(atoms, groups, np.random.default_rng(17), max_attempts=1000)
    assert abs(len(result.son.group_ids) - len(result.daughter.group_ids)) == 1
    assert gametes_match(result.son, result.daughter, len(groups))
    assert not gametes_match(result.son, result.son, len(groups))
    centered = original - original.mean(axis=0)
    angle = np.arctan(-1. / result.plane_slope)
    c, s = np.cos(angle), np.sin(angle)
    # Original CutMC restores full groups with this y rotation alone.
    y = np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])
    for gamete in (result.son, result.daughter):
        for gid, fragment in zip(gamete.group_ids, gamete.fragments):
            assert fragment.get_chemical_formula() == 'H2O'
            np.testing.assert_allclose(fragment.positions, centered[list(groups[gid])] @ y)
    np.testing.assert_array_equal(atoms.positions, original)


def test_degenerate_cut_reports_budget_exhaustion():
    atoms = Atoms('HH', positions=np.zeros((2, 3)))
    with pytest.raises(SamplingExhausted):
        cut_monomers(atoms, ((0,), (1,)), np.random.default_rng(4), max_attempts=2)


def test_no_nontrivial_monomer_and_overlapping_groups_fail():
    atoms = Atoms('HH', positions=[[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match='non-singleton'):
        mutate_single_monomer([atoms], [0.], ((0,), (1,)), np.random.default_rng(1), max_attempts=1000)
    with pytest.raises(ValueError, match='partition'):
        cut_monomers(atoms, ((0, 1), (1,)), np.random.default_rng(1), max_attempts=10)


def test_fit_binding_matches_original_jar_water_oracle():
    import json
    from pamssw.standalone.ga_operators import fit_binding
    fixture = json.loads((Path(__file__).parent / 'fixtures/type3_docking.json').read_text())
    atoms = Atoms(fixture['numbers'], positions=fixture['positions'])
    result = fit_binding(atoms[:3], atoms[3:], min_distance=1.5, accuracy=5)
    np.testing.assert_allclose(result.atoms.positions, fixture['expected_positions'], rtol=0, atol=2e-12)
    np.testing.assert_array_equal(result.atoms.numbers, atoms.numbers)


def test_complete_crossmc_butt_preserves_group_identity_and_source_cell(water):
    from pamssw.standalone.ga_operators import dock_gametes
    atoms, groups = water
    cut = cut_monomers(atoms, groups, np.random.default_rng(17), max_attempts=1000)
    child = dock_gametes(cut.son, cut.daughter, len(groups))
    assert len(child.atoms) == len(atoms)
    assert child.groups == groups
    for group in groups:
        np.testing.assert_allclose(child.atoms[list(group)].get_all_distances(), atoms[list(group)].get_all_distances(), atol=2e-12)
    np.testing.assert_allclose(child.atoms.cell.lengths(), np.ptp(child.atoms.positions, axis=0) + .5)
    assert not child.atoms.pbc.any()


def test_three_water_mutation_modes_are_actual_geometries(water):
    from pamssw.standalone.ga_operators import mutate_type3
    atoms, groups = water
    # Three fragments from the supplied archive suffice to test recombination.
    sample = atoms[:9]
    subset = groups[:3]
    result = mutate_type3([sample], [0.], subset, (0, 0, 0), (1, 1, 1), np.random.default_rng(4), max_selection_attempts=1000)
    assert [c.operation for c in result] == ['rotation_recombination', 'monomer_reconstruction', 'single_monomer_rotation']
    for candidate in result:
        assert len(candidate.atoms) == 9
        assert candidate.groups == subset
        assert candidate.group_parent_indices == (0, 0, 0)
        for group in subset:
            np.testing.assert_allclose(candidate.atoms[list(group)].get_all_distances(), sample[list(group)].get_all_distances(), atol=2e-12)
    assert result[-1].details['rotated_group'] in range(3)


def test_type3_rejects_known_bad_mutable_monomer_branch(water):
    from pamssw.standalone.ga_operators import mutate_type3
    atoms, groups = water
    with pytest.raises(NotImplementedError, match='changeType=1'):
        mutate_type3([atoms], [0.], groups, (1,) + (0,) * 14, (0, 1, 0), np.random.default_rng(1), max_selection_attempts=1000)


def test_propose_type3_returns_budget_status_with_real_children(water):
    from pamssw.standalone.ga_operators import propose_type3
    atoms, groups = water
    result = propose_type3([atoms, atoms.copy(), atoms.copy()], [0., 1., 2.], groups,
                          (0,) * 15, np.random.default_rng(5), min_ga=4,
                          bond_limits={}, max_batches=1, max_cut_attempts=1000, max_pair_attempts=10000)
    # Java floor quotas give only one crossover per G=4 batch; never pad.
    assert result.status == 'budget_exhausted'
    assert len(result.candidates) == 1
    child = result.candidates[0]
    assert child.operation == 'crossover'
    assert len(child.group_parent_indices) == 15
    assert set(child.group_parent_indices) <= {0, 1, 2}
    assert result.batches == 1
    np.testing.assert_array_equal(child.atoms.numbers, atoms.numbers)


def test_fixed_monomer_reconstruction_matches_original_jar_replayed_draws():
    import json
    from pamssw.standalone.ga_operators import mutate_type3
    fixture = json.loads((Path(__file__).parent / 'fixtures/type3_reconstruction.json').read_text())
    class DrawReplay:
        def __init__(self):
            self.draws = iter(fixture['draws'])
        def random(self):
            return next(self.draws)
    atoms = Atoms(fixture['numbers'], positions=fixture['positions'])
    result = mutate_type3([atoms], [0.], ((0, 1, 2), (3, 4, 5)), (0, 0), (0, 1, 0), DrawReplay(), max_selection_attempts=1000)
    np.testing.assert_allclose(result[0].atoms.positions, fixture['expected_positions'], rtol=0, atol=2e-12)


def test_type3_proposal_completes_full_batches_without_truncation(water):
    from collections import Counter
    from pamssw.standalone.ga_operators import propose_type3
    atoms, groups = water
    sample, subset = atoms[:9], groups[:3]
    result = propose_type3([sample, sample.copy(), sample.copy()], [0., 1., 2.], subset,
                          (0,) * 3, np.random.default_rng(7), min_ga=8, bond_limits={},
                          max_batches=2, max_cut_attempts=1000, max_pair_attempts=10000)
    assert result.status == 'target_reached'
    assert len(result.candidates) == 10  # G=8 gives five per batch, no truncation.
    assert Counter(c.operation for c in result.candidates) == dict(crossover=4, rotation_recombination=2, monomer_reconstruction=2, single_monomer_rotation=2)


@pytest.mark.parametrize('invalid', ['periodic', 'constrained', 'nonfinite', 'empty'])
def test_direct_gametes_reject_unsupported_fragments_before_flatten(water, invalid):
    from ase.constraints import FixAtoms
    from pamssw.standalone.ga_operators import MolecularGamete, dock_gametes
    atoms, _ = water
    fragment = atoms[:3]
    if invalid == 'periodic':
        fragment.pbc = True
    elif invalid == 'constrained':
        fragment.set_constraint(FixAtoms(indices=[0]))
    elif invalid == 'nonfinite':
        fragment.positions[0, 0] = np.nan
    else:
        fragment = Atoms()
    son = MolecularGamete((0,), (fragment,))
    daughter = MolecularGamete((1,), (atoms[3:6],))
    with pytest.raises(ValueError):
        dock_gametes(son, daughter, 2)


def test_singleton_rejection_sampling_has_explicit_budget():
    atoms = Atoms('HHH', positions=[[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    class AlwaysFirst:
        def random(self):
            return 0.
    with pytest.raises(SamplingExhausted, match='non-singleton'):
        mutate_single_monomer([atoms], [0.], ((0,), (1, 2)), AlwaysFirst(), max_attempts=2)
