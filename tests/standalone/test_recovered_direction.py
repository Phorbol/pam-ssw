import numpy as np
import pytest
import json
from ase import Atoms

from pamssw.standalone.recovered_direction import (
    RecoveredDirectionController, RecoveredDirectionSettings,
)


def settings(**changes):
    values = dict(ratio_local=5, local_probability=1., group_threshold=.5,
                  pre_rotmax=2, rotmax=4, pre_ftol=1e-3, ftol=2e-3,
                  metric='euclidean', max_force_calls=8)
    values.update(changes)
    return RecoveredDirectionSettings(**values)


def rng(seed=19):
    generator = np.random.default_rng(seed)
    while True:
        yield float(generator.random())


def geometry(offset=0.):
    return Atoms('H5', positions=np.array([
        [0., 0., 0.], [1., 0., .2], [0., 2.4, 0.],
        [3.4, 2.2, 1.], [-3.2, 1.1, 1.2],
    ]) + offset)


def test_explicit_settings_reject_invalid_budget_and_metric():
    with pytest.raises(ValueError, match='>=2'):
        settings(max_force_calls=1)
    with pytest.raises(ValueError, match='metric'):
        settings(metric='automatic')


def test_c1_radius_policy_is_explicit_and_restricted_by_default():
    assert settings().c1_radius_policy == 'restricted'
    assert settings(c1_radius_policy='per_atom').c1_radius_policy == 'per_atom'
    with pytest.raises(ValueError, match='c1_radius_policy'):
        settings(c1_radius_policy='anything_else')


def test_controller_uses_pre_ls_reference_and_saved_gaussian_center():
    initial = geometry()
    quenched = geometry(); quenched.positions += np.arange(5)[:, None] * [.03, .01, -.02]
    controller = RecoveredDirectionController(settings())
    stream = rng()
    controller.initialize(initial, quenched, stream)
    before = controller.diagnostics
    assert before['pair'][1] is not None
    assert len(before['selection']['group_mask']) == 5

    current = quenched.copy()
    work = quenched.copy(); work.positions += [.02, -.01, .03]
    first = controller.begin_escape(current, work, stream)
    assert np.isclose(np.linalg.norm(first.direction), 1.)
    diag = controller.diagnostics
    np.testing.assert_allclose(diag['outer_reference'], current.positions)
    assert diag['coefficients'][4] > 0

    center = work.copy()
    controller.save_gaussian_center(center)
    center.positions[:] = 100.
    work.positions += np.linspace(0., .04, 15).reshape(5, 3)
    updated = controller.update_direction(work, stream)
    assert np.isclose(np.linalg.norm(updated.direction), 1.)
    np.testing.assert_allclose(controller.diagnostics['outer_reference'], current.positions)


def test_landing_refresh_is_independent_of_mc_result_and_diagnostics_are_copies():
    initial = geometry()
    controller = RecoveredDirectionController(settings())
    stream = rng(71)
    controller.initialize(initial, initial.copy(), stream)
    work = initial.copy()
    controller.begin_escape(initial, work, stream)
    landing = work.copy()
    landing.positions += np.arange(5)[:, None] * [.08, -.03, .02]
    controller.observe_landing(landing, stream)
    diag = controller.diagnostics
    assert diag['pair'][1] is not None
    assert not diag['escape_active']
    assert diag['selection']['draw_count'] in (0, 1)
    assert diag['refresh']['draw_count'] >= 4
    json.dumps(diag)
    diag['group'][:] = [99] * len(diag['group'])
    diag['selection']['group_mask'][:] = [99] * len(diag['selection']['group_mask'])
    diag['outer_reference'][0][0] = 99
    fresh = controller.diagnostics
    stored_group = np.asarray(fresh['group'])
    assert np.all((stored_group == 0) | (stored_group == 1))
    group = np.asarray(fresh['selection']['group_mask'])
    assert np.all((group == 0) | (group == 1))
    np.testing.assert_allclose(fresh['outer_reference'], initial.positions)


def test_missing_pair_fails_without_fallback():
    atoms = Atoms('H2', positions=[[0., 0., 0.], [1., 0., 0.]])
    controller = RecoveredDirectionController(settings())
    with pytest.raises(ValueError, match='complete atom pair'):
        controller.initialize(atoms, atoms.copy(), iter([0.] * 200))
