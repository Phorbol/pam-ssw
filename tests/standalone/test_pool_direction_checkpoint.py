import numpy as np
import pytest
from ase.cluster import Icosahedron

from pamssw.standalone.native_mc import NativeMCSettings
from pamssw.standalone.paper_reference import (
    SSWConfig, load_ssw_checkpoint, run_ssw,
)
from pamssw.standalone.surface import QuenchResult
from pamssw.standalone.recovered_direction import RecoveredDirectionSettings


class FlatSurface:
    def __init__(self):
        self.requests = 0

    def evaluate(self, atoms):
        self.requests += 1
        return 0., np.zeros_like(atoms.positions)


class Selector:
    def __init__(self):
        self.calls = 0

    def checkpoint_contract(self):
        return {'identity': 'pool-direction-fixture', 'version': 1,
                'config': {'choice': 'none'}}

    def export_state(self):
        return {'calls': self.calls}

    def restore_state(self, payload):
        self.calls = int(payload['calls'])

    def __call__(self, snapshot, selector_rng):
        self.calls += 1
        selector_rng.random()
        return None


def _direction_settings():
    return RecoveredDirectionSettings(
        ratio_local=50, local_probability=.5, group_threshold=.5,
        pre_rotmax=1, rotmax=1, pre_ftol=.01, ftol=.01,
        metric='euclidean', max_force_calls=8)


def _config():
    return SSWConfig(width=.1, rotation_bias=1., max_gaussians=2,
        temperature_K=150., fmax=.03, relax_steps=2, fd_step=.001,
        rotation_hvp=2, rotation_tol=.02, direction_sampling='global',
        cluster_frame='direction_only')


def _stub_quench(monkeypatch):
    import pamssw.standalone.paper_reference as driver
    monkeypatch.setattr(driver, 'quench', lambda atoms, surface, **kwargs:
        QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'stub'))


def _run(path, selector, selector_rng, *, steps, checkpoint=None,
         monkeypatch=None):
    if monkeypatch is not None:
        _stub_quench(monkeypatch)
    return run_ssw(
        Icosahedron('Cu', 2), FlatSurface(), steps=steps, config=_config(),
        rng=np.random.default_rng(19), recovered_direction=_direction_settings(),
        mc=NativeMCSettings(.1, 99999), starter_selector=selector,
        selector_rng=selector_rng, checkpoint_path=path, checkpoint=checkpoint)


def test_schema5_direction_native_mc_split_replays_success_and_terminal_failure(monkeypatch, tmp_path):
    full_selector = Selector()
    full = _run(tmp_path / 'full.pkl', full_selector, np.random.default_rng(23),
                steps=2, monkeypatch=monkeypatch)
    split_path = tmp_path / 'split.pkl'
    _run(split_path, Selector(), np.random.default_rng(23), steps=1,
         monkeypatch=monkeypatch)
    checkpoint = load_ssw_checkpoint(split_path)
    assert checkpoint.schema_version == 5
    resumed_selector = Selector()
    resumed = _run(split_path, resumed_selector, np.random.default_rng(999),
                    steps=1, checkpoint=checkpoint, monkeypatch=monkeypatch)

    # This degenerate flat fixture reaches the existing native acos guard on
    # its second outer step. Replay must preserve the failure, not hide it.
    assert full.status == resumed.status == 'evaluation_failed'
    assert full.records[0].error is None
    assert full.records[1].error == resumed.records[1].error
    assert 'native acos outside numerical domain' in full.records[1].error
    assert full_selector.calls > 0
    assert resumed.evaluation_requests == full.evaluation_requests
    assert resumed.checkpoint.rng_state == full.checkpoint.rng_state
    assert [r.starter_selection for r in resumed.records] == [r.starter_selection for r in full.records]
    np.testing.assert_array_equal(resumed.current.positions, full.current.positions)
    assert resumed.checkpoint.pool_state['state'] == full.checkpoint.pool_state['state']
    assert resumed.checkpoint.pool_state['selector_rng_state'] == full.checkpoint.pool_state['selector_rng_state']
    assert resumed.checkpoint.recovered_direction_state.pair == full.checkpoint.recovered_direction_state.pair
    np.testing.assert_array_equal(resumed.checkpoint.recovered_direction_state.group,
                                  full.checkpoint.recovered_direction_state.group)
    assert resumed.checkpoint.recovered_direction_state.group_marker == full.checkpoint.recovered_direction_state.group_marker
    assert resumed.checkpoint.native_mc_state == full.checkpoint.native_mc_state


def test_schema5_terminal_checkpoint_is_not_resumable(monkeypatch, tmp_path):
    import pamssw.standalone.paper_reference as driver
    _stub_quench(monkeypatch)
    monkeypatch.setattr(driver, 'native_metropolis',
                        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError('fixture MC failure')))
    path = tmp_path / 'terminal.pkl'
    result = _run(path, Selector(), np.random.default_rng(23), steps=1,
                   monkeypatch=monkeypatch)
    checkpoint = load_ssw_checkpoint(path)
    assert result.status == 'mc_failed' and checkpoint.schema_version == 5
    class UntouchedSelector(Selector):
        def export_state(self):
            raise AssertionError('terminal resume must not call selector export')

    surface = FlatSurface()
    with pytest.raises(ValueError, match='terminal checkpoint'):
        run_ssw(Icosahedron('Cu', 2), surface, steps=1, config=_config(),
                rng=np.random.default_rng(19), recovered_direction=_direction_settings(),
                mc=NativeMCSettings(.1, 99999), starter_selector=UntouchedSelector(),
                selector_rng=np.random.default_rng(23), checkpoint=checkpoint)
    assert surface.requests == 0


def test_plain_selector_is_refused_before_pes_when_checkpointing(tmp_path):
    surface = FlatSurface()
    with pytest.raises(ValueError, match='checkpoint_contract'):
        run_ssw(Icosahedron('Cu', 2), surface, steps=1, config=_config(),
                rng=np.random.default_rng(19), starter_selector=lambda *_: None,
                selector_rng=np.random.default_rng(23), checkpoint_path=tmp_path / 'plain.pkl')
    assert surface.requests == 0
