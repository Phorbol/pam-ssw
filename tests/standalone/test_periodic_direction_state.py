"""Periodic geometry identity and controller routing for recovered directions."""
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.recovered_direction import (
    RecoveredDirectionController, RecoveredDirectionSettings,
)


def settings(**kwargs):
    values = dict(
        ratio_local=5, local_probability=1., group_threshold=.5,
        pre_rotmax=2, rotmax=4, pre_ftol=1e-3, ftol=2e-3,
        metric='euclidean', max_force_calls=8, geometry='periodic_local')
    values.update(kwargs)
    return RecoveredDirectionSettings(**values)


def crystal():
    return Atoms('C6', positions=[
        [.1, .1, .1], [1.4, .1, .2], [.1, 2.2, .2],
        [2.8, 2.1, 1.], [4.5, 1.2, .9], [8., 0., 0.],
    ], cell=[[7., 0., 0.], [1.1, 7.5, 0.], [.2, .4, 9.]], pbc=True)


def test_geometry_option_defaults_and_validates():
    legacy = RecoveredDirectionSettings(
        ratio_local=5, local_probability=1., group_threshold=.5,
        pre_rotmax=2, rotmax=4, pre_ftol=1e-3, ftol=2e-3,
        metric='euclidean', max_force_calls=8)
    assert legacy.geometry == 'nonperiodic'
    with pytest.raises(ValueError, match='geometry'):
        replace(legacy, geometry='periodic_global')


def test_periodic_checkpoint_captures_immutable_cell_and_pbc_and_resume_checks_them():
    atoms = crystal()
    controller = RecoveredDirectionController(settings())
    controller.initialize(atoms, atoms.copy(), iter([.2] * 100))
    state = controller.checkpoint_state()
    assert state.geometry_cell == tuple(tuple(float(x) for x in row) for row in atoms.cell.array)
    assert state.geometry_pbc == (True, True, True)
    with pytest.raises((AttributeError, TypeError)):
        state.geometry_cell[0][0] = 3.
    controller.validate_geometry(atoms.copy())
    state.validate_geometry(atoms.copy())
    changed_cell = atoms.copy()
    changed_cell.cell[0, 0] += .01
    with pytest.raises(ValueError, match='cell'):
        controller.validate_geometry(changed_cell)
    with pytest.raises(ValueError, match='cell/PBC identity'):
        state.validate_geometry(changed_cell)
    changed_pbc = atoms.copy()
    changed_pbc.pbc = [True, True, False]
    with pytest.raises(ValueError, match='PBC|pbc'):
        controller.validate_geometry(changed_pbc)
    with pytest.raises(ValueError, match='cell/PBC identity'):
        state.validate_geometry(changed_pbc)
    RecoveredDirectionController(settings()).restore_checkpoint_state(state)


def test_periodic_checkpoint_rejects_missing_identity_and_geometry_mode_switch():
    from dataclasses import replace

    atoms = crystal()
    controller = RecoveredDirectionController(settings())
    controller.initialize(atoms, atoms.copy(), iter([.2] * 100))
    state = controller.checkpoint_state()
    with pytest.raises(ValueError, match='requires cell and PBC'):
        replace(state, geometry_cell=None)
    nonperiodic = RecoveredDirectionSettings(
        ratio_local=5, local_probability=1., group_threshold=.5,
        pre_rotmax=2, rotmax=4, pre_ftol=1e-3, ftol=2e-3,
        metric='euclidean', max_force_calls=8)
    with pytest.raises(ValueError, match='settings'):
        RecoveredDirectionController(nonperiodic).restore_checkpoint_state(state)


def test_randomized_periodic_selection_maps_pair_and_group_back_to_original_order(monkeypatch):
    from pamssw.standalone.native_local_group import LocalGroupSelection
    from pamssw.standalone import periodic_direction

    atoms = crystal()
    order = np.array([2, 0, 5, 1, 4, 3])
    active = np.array([True, False, True, True, False, True])
    calls = {}

    class RNG:
        def permutation(self, count):
            assert count == len(atoms)
            return order

    def select(reference, ordered_atoms, rng, active_mask=None):
        calls['active_mask'] = active_mask.copy()
        calls['reference'] = np.asarray(reference).copy()
        calls['pbc'] = ordered_atoms.pbc.copy()
        return LocalGroupSelection((0, 1), np.array([1, 0, 1, 0, 0, 0]), 1)

    monkeypatch.setattr(periodic_direction, 'select_periodic_direction_group', select)
    controller = RecoveredDirectionController(
        settings(startup_order='randomized'), active_mask=active)
    controller.initialize(atoms, atoms.copy(), RNG())
    np.testing.assert_array_equal(calls['active_mask'], active[order])
    np.testing.assert_array_equal(calls['reference'], atoms.positions[order])
    np.testing.assert_array_equal(calls['pbc'], atoms.pbc)
    checkpoint = controller.checkpoint_state()
    assert checkpoint.pair == (int(order[0]), int(order[1]))
    np.testing.assert_array_equal(checkpoint.group, [0, 0, 1, 0, 0, 1])
    assert checkpoint.refresh is None


def test_periodic_initialization_rejects_invalid_geometry_before_rng_use():
    atoms = crystal()
    rng = iter([.2] * 100)
    bad = atoms.copy()
    bad.cell[2] = bad.cell[1]
    with pytest.raises(ValueError, match='full.rank|cell'):
        RecoveredDirectionController(settings()).initialize(bad, bad.copy(), rng)
    assert next(rng) == .2


def test_periodic_selection_bypasses_native_pair_refresh_and_tracks_landing_cell():
    atoms = crystal()
    controller = RecoveredDirectionController(settings())
    controller.initialize(atoms, atoms.copy(), iter([.2] * 100))
    assert controller.diagnostics['refresh'] is None
    work = atoms.copy()
    controller.begin_escape(atoms, work, iter([.2] * 100))
    controller.save_gaussian_center(work.positions.copy())
    landing = atoms.copy()
    landing.positions[0, 0] += .2
    controller.observe_landing(landing, iter([.2] * 100))
    assert controller.diagnostics['refresh'] is None
    assert controller.checkpoint_state().geometry_cell == tuple(
        tuple(float(x) for x in row) for row in atoms.cell.array)
    controller.validate_geometry(landing)
