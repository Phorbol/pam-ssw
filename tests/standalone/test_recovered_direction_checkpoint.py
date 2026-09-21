import numpy as np
import pytest
from ase.cluster import Icosahedron

from pamssw.standalone.paper_reference import (
    SSWConfig, load_ssw_checkpoint, run_ssw,
)
from pamssw.standalone.surface import QuenchResult
from pamssw.standalone.recovered_direction import (
    RecoveredDirectionCheckpointState, RecoveredDirectionSettings,
)


class FlatSurface:
    requests = 0
    def evaluate(self, atoms):
        self.requests += 1
        return 0.0, np.zeros_like(atoms.positions)


def settings():
    return RecoveredDirectionSettings(
        ratio_local=50, local_probability=.5, group_threshold=.5,
        pre_rotmax=1, rotmax=1, pre_ftol=.01, ftol=.01,
        metric='euclidean', max_force_calls=8)


def config():
    return SSWConfig(width=.1, rotation_bias=1., max_gaussians=2,
        temperature_K=0., fmax=.03, relax_steps=2, fd_step=.001,
        rotation_hvp=2, rotation_tol=.02, direction_sampling='global',
        cluster_frame='direction_only')


def test_recovered_direction_checkpoint_replays_second_outer_step(monkeypatch, tmp_path):
    import pamssw.standalone.paper_reference as driver
    monkeypatch.setattr(driver, 'quench', lambda atoms, surface, **kwargs:
        QuenchResult(atoms.copy(), 0., 0., True, 0, 0, 'stub'))
    atoms = Icosahedron('Cu', 2)
    uninterrupted = run_ssw(atoms.copy(), FlatSurface(), steps=2,
        config=config(), rng=np.random.default_rng(19), recovered_direction=settings())
    checkpoint_path = tmp_path / 'state.pkl'
    run_ssw(atoms.copy(), FlatSurface(), steps=1, config=config(),
        rng=np.random.default_rng(19), recovered_direction=settings(),
        checkpoint_path=checkpoint_path)
    checkpoint = load_ssw_checkpoint(checkpoint_path)
    assert checkpoint.schema_version == 4
    resumed = run_ssw(atoms.copy(), FlatSurface(), steps=1, config=config(),
        rng=np.random.default_rng(19), recovered_direction=settings(),
        checkpoint=checkpoint)
    assert len(resumed.records) == len(uninterrupted.records) == 2
    assert resumed.records[1].climb[0]['recovered_direction']['pair'] == uninterrupted.records[1].climb[0]['recovered_direction']['pair']
    np.testing.assert_allclose(
        resumed.records[1].climb[0]['recovered_direction']['proposal'],
        uninterrupted.records[1].climb[0]['recovered_direction']['proposal'])


def test_legacy_checkpoint_without_direction_state_is_rejected_before_pes(tmp_path, monkeypatch):
    import pamssw.standalone.paper_reference as driver
    atoms = Icosahedron('Cu', 2)
    monkeypatch.setattr(driver, 'quench', lambda atoms, surface, **kwargs: QuenchResult(
        atoms.copy(), 0., 0., True, 0, 0, 'stub'))
    path = tmp_path / 'ordinary.pkl'
    run_ssw(atoms.copy(), FlatSurface(), steps=1, config=config(),
            rng=np.random.default_rng(19), checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    surface = FlatSurface()
    with pytest.raises(ValueError, match='direction state'):
        run_ssw(atoms.copy(), surface, steps=1, config=config(),
                rng=np.random.default_rng(19), recovered_direction=settings(),
                checkpoint=checkpoint)
    assert surface.requests == 0


def _checkpoint_with_direction(tmp_path, monkeypatch):
    import pamssw.standalone.paper_reference as driver
    monkeypatch.setattr(driver, 'quench', lambda atoms, surface, **kwargs: QuenchResult(
        atoms.copy(), 0., 0., True, 0, 0, 'stub'))
    atoms = Icosahedron('Cu', 2); path = tmp_path / 'direction.pkl'
    run_ssw(atoms.copy(), FlatSurface(), steps=1, config=config(),
            rng=np.random.default_rng(19), recovered_direction=settings(),
            checkpoint_path=path)
    return atoms, load_ssw_checkpoint(path)


def test_direction_checkpoint_settings_mismatch_is_before_pes(tmp_path, monkeypatch):
    atoms, checkpoint = _checkpoint_with_direction(tmp_path, monkeypatch)
    surface = FlatSurface()
    changed = RecoveredDirectionSettings(**{**settings().__dict__, 'ratio_local': 51})
    with pytest.raises(ValueError, match='settings'):
        run_ssw(atoms, surface, steps=1, config=config(),
                rng=np.random.default_rng(19), recovered_direction=changed,
                checkpoint=checkpoint)
    assert surface.requests == 0


@pytest.mark.parametrize('field,value', [
    ('pair', (True, 1)), ('group', np.zeros(12, dtype=np.int32)),
    ('group_marker', 1.5),
])
def test_direction_checkpoint_state_is_revalidated_before_pes(tmp_path, monkeypatch, field, value):
    atoms, checkpoint = _checkpoint_with_direction(tmp_path, monkeypatch)
    state = object.__new__(RecoveredDirectionCheckpointState)
    for name in ('settings', 'pair', 'group', 'group_marker'):
        object.__setattr__(state, name, getattr(checkpoint.recovered_direction_state, name))
    object.__setattr__(state, field, value)
    checkpoint.recovered_direction_state = state
    surface = FlatSurface()
    with pytest.raises((TypeError, ValueError), match='checkpoint|direction'):
        run_ssw(atoms, surface, steps=0, config=config(),
                rng=np.random.default_rng(19), recovered_direction=settings(),
                checkpoint=checkpoint)
    assert surface.requests == 0


def test_schema4_rejects_invalid_mc_or_rotation_payload(tmp_path, monkeypatch):
    from dataclasses import replace
    import pamssw.standalone.paper_reference as driver
    atoms, checkpoint = _checkpoint_with_direction(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='native MC'):
        driver._validate_ssw_checkpoint(replace(checkpoint, mc_settings=object(), native_mc_state=object()))
    with pytest.raises(ValueError, match='rotation'):
        driver._validate_ssw_checkpoint(replace(checkpoint, recovered_rotation=object()))


def test_direction_checkpoint_preserves_selection_refresh_diagnostics(tmp_path, monkeypatch):
    import pamssw.standalone.paper_reference as driver
    monkeypatch.setattr(driver, 'quench', lambda atoms, surface, **kwargs: QuenchResult(
        atoms.copy(), 0., 0., True, 0, 0, 'stub'))
    atoms = Icosahedron('Cu', 2)
    uninterrupted = run_ssw(atoms.copy(), FlatSurface(), steps=2,
        config=config(), rng=np.random.default_rng(19), recovered_direction=settings())
    path = tmp_path / 'diag.pkl'
    run_ssw(atoms.copy(), FlatSurface(), steps=1, config=config(),
            rng=np.random.default_rng(19), recovered_direction=settings(), checkpoint_path=path)
    resumed = run_ssw(atoms.copy(), FlatSurface(), steps=1, config=config(),
            rng=np.random.default_rng(19), recovered_direction=settings(),
            checkpoint=load_ssw_checkpoint(path))
    a = uninterrupted.records[1].climb[0]['recovered_direction']
    b = resumed.records[1].climb[0]['recovered_direction']
    assert b['selection'] == a['selection']
    assert b['refresh'] == a['refresh']


def test_direction_checkpoint_resumes_native_mc(monkeypatch, tmp_path):
    from dataclasses import replace
    from pamssw.standalone import NativeMCSettings
    atoms, checkpoint = _checkpoint_with_direction(tmp_path, monkeypatch)
    cfg = replace(config(), temperature_K=150.)
    path = tmp_path / 'native-mc.pkl'
    run_ssw(atoms, FlatSurface(), steps=1, config=cfg,
            rng=np.random.default_rng(19), recovered_direction=settings(),
            mc=NativeMCSettings(energy_tol=.1, maxtrap=99999), checkpoint_path=path)
    result = run_ssw(atoms, FlatSurface(), steps=0, config=cfg,
            rng=np.random.default_rng(98765),
            mc=NativeMCSettings(energy_tol=.1, maxtrap=99999),
            checkpoint=load_ssw_checkpoint(path))
    assert len(result.records) == 1
    assert result.status == 'completed', [(r.status, r.error) for r in result.records]
