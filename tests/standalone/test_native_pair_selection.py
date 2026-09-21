"""Native outcomes on canonical in-cell coordinates; not search efficacy.

The earlier centered C6H6 (8,3) native True fixture was outside the native
coordinate chart. Its contrary result is retained in
native-pair-coordinate-chart-20260917.json, not used as a physical constraint.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from pamssw.standalone.native_pair_selection import native_pair_allowed, refresh_native_pair


@pytest.mark.parametrize('name,pair,allowed', [
    ('C2H6', (0, 5), False), ('C2H6', (5, 2), False),
    ('CH3OH', (4, 1), False), ('C6H6', (8, 3), False),
    ('C6H6', (8, 7), True),
    ('C6H6', (8, 0), False),
])
def test_frozen_native_checks(name, pair, allowed):
    atoms = molecule(name)
    atoms.positions += 15
    assert native_pair_allowed(atoms, pair) == allowed


def test_isolated_formula_is_translation_invariant():
    atoms = molecule('C6H6')
    base = native_pair_allowed(atoms, (8, 3))
    atoms.positions += 15
    assert native_pair_allowed(atoms, (8, 3)) == base


def test_pair_check_has_no_calculator_dependency():
    atoms = Atoms('H2', positions=[[0, 0, 0], [3, 0, 0]])
    assert native_pair_allowed(atoms, (0, 1))
    assert atoms.calc is None


def test_nonperiodic_domain_is_explicit():
    atoms = molecule('C2H6')
    atoms.pbc = True
    with pytest.raises(ValueError, match='nonperiodic'):
        native_pair_allowed(atoms, (0, 1))


def native_rng(prefix):
    yield from prefix
    rng = np.random.default_rng(817)
    while True:
        yield rng.random()


@pytest.mark.parametrize('name,prefix,pair,draws,allowed,rejections', [
    ('C2H6', [0., .2, 0.], (0, 5), 331, False, 150),
    ('C6H6', [.9, .7, 0.], (8, 7), 7, True, 0),
    ('CH3OH', [0., .2, 0.], (0, 1), 153, False, 150),
])
def test_refresh_preserves_native_exit_distinction(name, prefix, pair, draws, allowed, rejections):
    atoms = molecule(name)
    atoms.positions += 15
    got = refresh_native_pair(atoms, (0, 1), native_rng(prefix))
    assert got.pair == pair
    assert got.draw_count == draws
    assert got.geometry_accepted == allowed
    assert got.distance_or_fixatom_rejections == rejections
    assert got.stop_reason == ('geometry_accepted' if allowed else 'rejection_limit')


def test_refresh_keeps_absent_second_if_no_distance_candidate():
    atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
    got = refresh_native_pair(atoms, (0, None), iter([0.] * 153))
    assert got.pair == (0, None)
    assert not got.geometry_accepted
