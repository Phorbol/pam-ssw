"""Activity-restricted geometry contract for recovered directions."""
import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.native_direction_control import (
    LocalDirectionState, generate_local_direction,
)
from pamssw.standalone.recovered_direction import (
    RecoveredDirectionCheckpointState, RecoveredDirectionController,
    RecoveredDirectionSettings,
)


def settings():
    return RecoveredDirectionSettings(
        ratio_local=5, local_probability=1., group_threshold=.5,
        pre_rotmax=2, rotmax=4, pre_ftol=1e-3, ftol=2e-3,
        metric='euclidean', max_force_calls=8)


def geometry():
    # The last atom is a fixed geometric reference outside the active subset.
    return Atoms('C6', positions=[
        [0., 0., 0.], [1.4, 0., .1], [0., 2.2, .1],
        [2.8, 2.1, 1.], [-2.5, 1.2, .9], [8., 0., 0.],
    ])


def test_active_generator_projects_every_component_and_stage_seed():
    atoms = geometry()
    active = np.array([True, True, True, True, True, False])
    coefficients = np.zeros(10)
    coefficients[1] = 1.
    coefficients[4] = .6
    got = generate_local_direction(
        atoms, np.ones((6, 3)), coefficients, (0, 5), np.ones(6, int),
        iter([.17]), group_marker=0, active_mask=active)
    assert np.isclose(np.linalg.norm(got.direction), 1.)
    np.testing.assert_array_equal(got.direction[~active], 0.)


def test_active_stage_history_has_zero_fixed_support():
    atoms = geometry()
    active = np.array([True, True, True, True, True, False])
    coefficients = np.zeros(10)
    coefficients[4] = .6
    state = LocalDirectionState((0, 5), np.zeros(6, int), coefficients,
                                group_marker=0, active_mask=active)
    center = atoms.positions.copy()
    state.save_gaussian_center(center)
    atoms.positions[~active] += [0., .1, .2]
    got = state.update(atoms, lambda: .17)
    np.testing.assert_array_equal(got.direction[~active], 0.)


def test_active_controller_selection_keeps_fixed_axis_reference_but_active_support():
    atoms = geometry()
    active = np.array([True, True, True, True, True, False])
    controller = RecoveredDirectionController(settings(), active_mask=active)
    controller.initialize(atoms, atoms.copy(), iter([.2] * 100))
    state = controller.checkpoint_state()
    assert state.pair[0] in np.flatnonzero(active)
    assert state.pair[1] is None or state.pair[1] != state.pair[0]
    assert not np.any(state.group[~active])


def test_all_active_selection_preserves_existing_score_and_single_draw():
    from pamssw.standalone.native_local_group import select_native_local_group
    from pamssw.standalone.recovered_direction import _select_active_direction_group
    atoms = geometry()
    atoms.positions[-1] = [4., 0., 0.]
    active = np.ones(len(atoms), dtype=bool)
    expected = select_native_local_group(atoms.positions, atoms, iter([.37]))
    actual = _select_active_direction_group(atoms.positions, atoms, iter([.37]), active)
    assert actual.pair == expected.pair
    assert actual.draw_count == expected.draw_count
    np.testing.assert_array_equal(actual.group_mask, expected.group_mask)


def test_active_selection_rejects_empty_existing_score_band_without_redraw():
    atoms = Atoms('C2', positions=[[0., 0., 0.], [1., 0., 0.]])
    draws = iter([.37])
    controller = RecoveredDirectionController(settings(), active_mask=np.array([True, False]))
    with pytest.raises(ValueError, match='score band has no second'):
        controller.initialize(atoms, atoms.copy(), draws)
    assert next(draws) == .37


def test_active_checkpoint_carries_immutable_mask_and_rejects_mismatch():
    atoms = geometry()
    active = np.array([True, True, True, True, True, False])
    controller = RecoveredDirectionController(settings(), active_mask=active)
    controller.initialize(atoms, atoms.copy(), iter([.2] * 100))
    checkpoint = controller.checkpoint_state()
    np.testing.assert_array_equal(checkpoint.active_mask, active)
    assert not checkpoint.active_mask.flags.writeable
    mismatched = RecoveredDirectionController(settings(), active_mask=np.ones(6, bool))
    with pytest.raises(ValueError, match='active_mask'):
        mismatched.restore_checkpoint_state(checkpoint)


@pytest.mark.parametrize('corruption, message', [
    ('inactive_group', 'group.*active'),
    ('fixed_first', 'pair.*active'),
    ('native_refresh', 'must not contain.*refresh'),
    ('selection_group', 'selection.*group'),
    ('selection_pair', 'selection.*pair'),
])
def test_active_checkpoint_restore_revalidates_corrupt_state(corruption, message):
    from dataclasses import replace
    from pamssw.standalone.native_local_group import LocalGroupSelection

    atoms = geometry()
    active = np.array([True, True, True, True, True, False])
    controller = RecoveredDirectionController(settings(), active_mask=active)
    controller.initialize(atoms, atoms.copy(), iter([.2] * 100))
    state = controller.checkpoint_state()
    forged = object.__new__(RecoveredDirectionCheckpointState)
    for name in ('settings', 'pair', 'group', 'group_marker', 'selection',
                 'refresh', 'active_mask'):
        object.__setattr__(forged, name, getattr(state, name))
    selection = state.selection
    if corruption == 'inactive_group':
        group = state.group.copy()
        group[np.flatnonzero(~active)[0]] = 1
        object.__setattr__(forged, 'group', group)
    elif corruption == 'fixed_first':
        object.__setattr__(forged, 'pair', (int(np.flatnonzero(~active)[0]), state.pair[1]))
    elif corruption == 'native_refresh':
        from pamssw.standalone.native_pair_selection import PairRefreshResult
        object.__setattr__(forged, 'refresh', PairRefreshResult(
            state.pair, True, 'geometry_accepted', 1, 0, 0, 0))
    elif corruption == 'selection_group':
        group = selection.group_mask.copy()
        group[0] = 1 - group[0]
        object.__setattr__(forged, 'selection', replace(selection, group_mask=group))
    else:
        other = (state.pair[1] + 1) % len(atoms)
        object.__setattr__(forged, 'selection', replace(
            selection, pair=(state.pair[0], other)))
    with pytest.raises(ValueError, match=message):
        RecoveredDirectionController(settings(), active_mask=active).restore_checkpoint_state(
            forged)


def test_all_fixed_active_controller_rejects_before_selection_draw():
    atoms = geometry()
    draws = iter([.2] * 100)
    with pytest.raises(ValueError, match='active'):
        RecoveredDirectionController(settings(), active_mask=np.zeros(6, bool)).initialize(
            atoms, atoms.copy(), draws)
    assert next(draws) == .2


def test_explicit_none_preserves_legacy_generator_draw_and_output():
    atoms = geometry()
    coefficients = np.zeros(10)
    coefficients[1] = 1.
    args = (atoms, np.zeros((6, 3)), coefficients, (0, 2),
            np.zeros(6, int))
    without = generate_local_direction(*args, iter([.17]), group_marker=0)
    explicit_none = generate_local_direction(*args, iter([.17]), group_marker=0,
                                             active_mask=None)
    np.testing.assert_array_equal(explicit_none.direction, without.direction)
