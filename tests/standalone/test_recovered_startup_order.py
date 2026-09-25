"""Startup relabeling contract; no claim about same-seed search efficiency."""
from dataclasses import replace
import pickle

import numpy as np
import pytest
from ase.build import molecule

from pamssw.standalone.recovered_direction import (
    RecoveredDirectionController, RecoveredDirectionSettings,
)


def settings(**kwargs):
    return RecoveredDirectionSettings(ratio_local=50, local_probability=.5,
        group_threshold=.5, pre_rotmax=1, rotmax=1, pre_ftol=.2, ftol=.02,
        metric='euclidean', max_force_calls=8, **kwargs)


def test_startup_setting_default_validation_and_missing_pickle_field():
    old = settings()
    assert old.startup_order == 'legacy'
    old.__dict__.pop('startup_order')
    restored = pickle.loads(pickle.dumps(old))
    assert restored.startup_order == 'legacy'
    assert restored == settings()
    with pytest.raises(ValueError, match='startup_order'):
        settings(startup_order='unknown')


def test_randomized_startup_maps_active_and_diagnostic_state_and_rng():
    atoms = molecule('C60')
    reference = atoms.copy()
    reference.positions += np.arange(len(atoms))[:, None] * [.003, -.001, .002]
    before = atoms.positions.copy()
    rng = np.random.default_rng(73)
    control_rng = np.random.default_rng(73)
    order = control_rng.permutation(len(atoms))
    expected = RecoveredDirectionController(settings())
    expected.initialize(reference[order], atoms[order], control_rng)
    actual = RecoveredDirectionController(settings(startup_order='randomized'))
    actual.initialize(reference, atoms, rng)
    a, b = actual.checkpoint_state(), expected.checkpoint_state()
    assert a.pair == tuple(int(order[i]) for i in b.pair)
    np.testing.assert_array_equal(a.group[order], b.group)
    assert a.selection.pair == tuple(None if i is None else int(order[i]) for i in b.selection.pair)
    np.testing.assert_array_equal(a.selection.group_mask[order], b.selection.group_mask)
    assert a.refresh.pair == tuple(None if i is None else int(order[i]) for i in b.refresh.pair)
    assert replace(a.refresh, pair=b.refresh.pair) == b.refresh
    assert a.selection.draw_count == b.selection.draw_count
    assert rng.bit_generator.state == control_rng.bit_generator.state
    np.testing.assert_array_equal(atoms.positions, before)


def test_restore_does_not_reinitialize_and_landing_refresh_stays_legacy():
    atoms = molecule('C60')
    actual = RecoveredDirectionController(settings(startup_order='randomized'))
    rng = np.random.default_rng(73)
    actual.initialize(atoms, atoms, rng)
    restored = RecoveredDirectionController(actual.settings)
    restored.restore_checkpoint_state(pickle.loads(pickle.dumps(actual.checkpoint_state())))
    legacy = RecoveredDirectionController(settings())
    legacy.restore_checkpoint_state(replace(actual.checkpoint_state(), settings=settings()))
    other_rng = np.random.default_rng()
    other_rng.bit_generator.state = rng.bit_generator.state
    actual.begin_escape(atoms, atoms, rng)
    legacy.begin_escape(atoms, atoms, other_rng)
    actual.observe_landing(atoms, rng)
    legacy.observe_landing(atoms, other_rng)
    assert actual.diagnostics == legacy.diagnostics
    assert rng.bit_generator.state == other_rng.bit_generator.state
    assert restored.diagnostics['pair'] is not None


def test_pool_restart_uses_same_randomized_initialization():
    from pamssw.standalone.paper_reference import _prepare_pool_restart
    atoms = molecule('C60')
    cfg = settings(startup_order='randomized')
    a, b = np.random.default_rng(81), np.random.default_rng(81)
    _, _, restarted = _prepare_pool_restart(atoms, ls=None, recovered_direction=cfg, rng=a)
    expected = RecoveredDirectionController(cfg)
    expected.initialize(atoms, atoms, b)
    assert restarted.diagnostics == expected.diagnostics
    assert a.bit_generator.state == b.bit_generator.state


@pytest.mark.parametrize('pool', [False, True])
def test_old_pickle_and_startup_mismatch_before_rng_or_pes(monkeypatch, pool):
    import pamssw.standalone.paper_reference as driver
    from pamssw.standalone.surface import QuenchResult
    from test_recovered_direction_checkpoint import config, FlatSurface
    from test_pool_direction_checkpoint import Selector
    atoms = molecule('C60')
    monkeypatch.setattr(driver, 'quench', lambda atoms, surface, **kw:
        QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'stub'))
    kwargs = dict(starter_selector=Selector(), selector_rng=np.random.default_rng(3)) if pool else {}
    kwargs['progress_callback'] = lambda event: False
    result = driver.run_ssw(atoms, FlatSurface(), steps=0, config=config(),
        rng=np.random.default_rng(73), recovered_direction=settings(), **kwargs)
    checkpoint = result.checkpoint
    assert checkpoint.schema_version == (5 if pool else 4)
    checkpoint.recovered_direction_state.settings.__dict__.pop('startup_order')
    old = pickle.loads(pickle.dumps(checkpoint))
    resumed = driver.run_ssw(atoms, FlatSurface(), steps=0, config=config(),
        rng=np.random.default_rng(99), checkpoint=old, **kwargs)
    assert resumed.checkpoint.rng_state == checkpoint.rng_state
    rng = np.random.default_rng(99)
    before = rng.bit_generator.state
    surface = FlatSurface()
    with pytest.raises(ValueError, match='settings'):
        driver.run_ssw(atoms, surface, steps=0, config=config(), rng=rng,
            recovered_direction=settings(startup_order='randomized'), checkpoint=old, **kwargs)
    assert surface.requests == 0
    assert rng.bit_generator.state == before
