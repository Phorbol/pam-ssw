"""Representation and checkpoint contracts for the opt-in molecular pool."""
import copy

import ase
import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from research.ga_ssw.pool_starter_adapter import PoolStarterAdapter
from pamssw.standalone.starter_selection import StarterObservation, StarterPoolSnapshot
from test_pool_starter_adapter import assert_payload_equal, Choose


def pool(identity='ase_permute_v1'):
    return PoolStarterAdapter(mode='uniform', energy_tol=.001, rmsd_tol=.1,
                              identity_matcher=identity)


def obs(index, atoms):
    return StarterObservation(index, atoms, 0., 0.)


def test_default_contract_stays_legacy_v1():
    old = PoolStarterAdapter(mode='uniform', energy_tol=.001, rmsd_tol=.1)
    explicit = pool('ordered_v1')
    assert old.checkpoint_contract() == explicit.checkpoint_contract()
    assert old.export_state()['version'] == 1
    assert 'identity_matcher' not in old.checkpoint_contract()
    explicit.restore_state(old.export_state())
    assert_payload_equal(old.export_state(), explicit.export_state())


def test_permuted_c60_is_one_entry_and_keeps_input_order():
    initial = molecule('C60')
    permuted = initial[np.random.default_rng(3).permutation(60)]
    permuted.rotate(35, 'z')
    permuted.translate([2., -1., .3])
    positions = permuted.positions.copy()
    a = pool()
    assert a(StarterPoolSnapshot((obs(0, initial), obs(1, permuted)),
                                0, 1, 0, 2), Choose(0)) is None
    assert a.mapping == [0, 0]
    assert len(a.archive.entries) == 1
    assert a.archive.entries[0].duplicate_hits == 1
    assert a.archive.entries[0].node_successes == 0
    assert a.outcomes[-1].is_duplicate
    assert not a.outcomes[-1].is_new_minimum
    np.testing.assert_array_equal(permuted.positions, positions)
    np.testing.assert_array_equal(a.archive.entries[0].state.positions, initial.positions)


def test_restore_retains_matcher_and_duplicate_accounting():
    initial = molecule('H2O')
    first = (obs(0, initial),)
    a = pool()
    a(StarterPoolSnapshot(first, 0, None, 0, 1), Choose(0))
    payload = a.export_state()
    assert payload['version'] == 2
    assert payload['contract']['identity_matcher']['ase_version'] == ase.__version__
    b = pool().restore_state(payload)
    second = first + (obs(1, initial[[0, 2, 1]]),)
    snap = StarterPoolSnapshot(second, 0, 1, 1, 2)
    assert a(snap, Choose(0)) == b(snap, Choose(0))
    assert a.mapping == [0, 0]
    assert_payload_equal(a.export_state(), b.export_state())


@pytest.mark.parametrize('source,target', [('ordered_v1', 'ase_permute_v1'),
                                          ('ase_permute_v1', 'ordered_v1')])
def test_cross_mode_restore_fails_before_mutation(source, target):
    destination = pool(target)
    before = destination.export_state()
    with pytest.raises(ValueError, match='version|contract'):
        destination.restore_state(pool(source).export_state())
    assert_payload_equal(before, destination.export_state())


def test_ase_version_mismatch_is_rejected():
    a = pool()
    payload = copy.deepcopy(a.export_state())
    payload['contract']['identity_matcher']['ase_version'] = 'different'
    with pytest.raises(ValueError, match='contract'):
        a.restore_state(payload)


def test_periodic_input_rejected_before_insertion():
    a = pool()
    atoms = molecule('H2O')
    atoms.set_cell([10, 10, 10])
    atoms.pbc = True
    before = a.export_state()
    with pytest.raises(ValueError, match='nonperiodic'):
        a(StarterPoolSnapshot((obs(0, atoms),), 0, None, 0, 1), Choose(0))
    assert_payload_equal(a.export_state(), before)


def test_mismatched_composition_is_distinct_and_single_atom_is_supported():
    a = pool()
    items = (obs(0, Atoms('H', positions=[[0, 0, 0]])),
             obs(1, Atoms('H', positions=[[9, 0, 0]])),
             obs(2, Atoms('He', positions=[[0, 0, 0]])))
    a(StarterPoolSnapshot(items, 0, 2, 0, 3), Choose(0))
    assert a.mapping == [0, 0, 1]


def test_unknown_matcher_is_rejected():
    with pytest.raises(ValueError, match='identity_matcher'):
        pool('invalid')
