"""Completed-attempt pause and resume contracts on Cu/EMT."""

import numpy as np
import pytest
from ase.calculators.emt import EMT

from pamssw.standalone.paper_reference import (
    load_ssw_checkpoint, run_ls_ssw, run_ssw, save_ssw_checkpoint,
)
from pamssw.standalone.surface import ASESurface
from test_ssw_checkpoint import _case


def test_paper_ls_callback_pauses_after_complete_step_and_resumes(tmp_path):
    atoms, config, ls = _case()
    full_rng = np.random.default_rng(19)
    full = run_ssw(atoms, ASESurface(EMT()), steps=2, config=config,
                   rng=full_rng, ls=ls, checkpoint_callback=lambda cp: False)
    path = tmp_path / 'pause.pkl'
    seen = []
    first_rng = np.random.default_rng(19)
    first_surface = ASESurface(EMT())

    def pause(checkpoint):
        seen.append((checkpoint.next_index, checkpoint.status,
                     checkpoint.response.steps, checkpoint.evaluation_requests))
        save_ssw_checkpoint(path, checkpoint)
        checkpoint.current.positions[:] = 1000.0
        checkpoint.response.steps = 1000
        checkpoint.records[0].last_atoms.positions[:] = 1000.0
        return True

    first = run_ls_ssw(atoms, first_surface, steps=2, config=config,
                       rng=first_rng, ls=ls, checkpoint_callback=pause)
    stored = load_ssw_checkpoint(path)
    assert first.status == 'paused'
    assert first.checkpoint.status == stored.status == 'completed'
    assert seen == [(1, 'completed', 1, first.evaluation_requests)]
    assert len(first.records) == 1
    assert first.evaluation_requests == first_surface.requests
    assert first.checkpoint.response.steps == 1
    np.testing.assert_array_equal(first.current.positions, stored.current.positions)
    np.testing.assert_array_equal(first.records[0].last_atoms.positions,
                                  stored.records[0].last_atoms.positions)
    resumed_rng = np.random.default_rng(999)
    resumed_surface = ASESurface(EMT())
    resumed = run_ssw(atoms, resumed_surface, steps=1, config=config,
                      rng=resumed_rng, ls=ls, checkpoint=stored)
    assert resumed.status == full.status == 'completed'
    assert resumed.evaluation_requests == full.evaluation_requests
    assert first.evaluation_requests + resumed_surface.requests == full.evaluation_requests
    assert [record.index for record in resumed.records] == [0, 1]
    assert [record.ls_update for record in resumed.records] == [record.ls_update for record in full.records]
    assert resumed.checkpoint.response.steps == full.checkpoint.response.steps
    np.testing.assert_array_equal(resumed.current.positions, full.current.positions)
    assert resumed_rng.bit_generator.state == full_rng.bit_generator.state


def test_callback_none_keeps_snapshot_disabled(monkeypatch):
    import pamssw.standalone.paper_reference as ref
    atoms, config, ls = _case()
    monkeypatch.setattr(ref, '_checkpoint_copy', lambda value: (_ for _ in ()).throw(
        AssertionError('unexpected snapshot')))
    result = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                     rng=np.random.default_rng(19), ls=ls, checkpoint_callback=None)
    assert result.status == 'completed' and result.checkpoint is None


def test_callback_only_runs_at_resumable_boundaries():
    atoms, config, ls = _case()
    seen = []
    result = run_ssw(atoms, ASESurface(EMT()), steps=0, config=config,
                     rng=np.random.default_rng(19), ls=ls,
                     checkpoint_callback=lambda cp: seen.append(cp) or True)
    assert result.status == 'completed' and seen == []
    assert result.checkpoint is not None and result.checkpoint.next_index == 0


def test_noncallable_callback_rejected_without_pes():
    atoms, config, ls = _case()
    surface = ASESurface(EMT())
    with pytest.raises(TypeError, match='checkpoint_callback'):
        run_ssw(atoms, surface, steps=1, config=config,
                rng=np.random.default_rng(19), ls=ls, checkpoint_callback=True)
    assert surface.requests == 0


def test_callback_only_pool_checkpoint_preserves_selector_and_landing_order(tmp_path):
    from research.ga_ssw.pool_starter_adapter import PoolStarterAdapter

    atoms, config, ls = _case()
    def policy():
        return PoolStarterAdapter(mode='uniform', energy_tol=1e-5, rmsd_tol=1e-3)
    full_selector = policy()
    full_rng, full_selector_rng = np.random.default_rng(19), np.random.default_rng(23)
    full = run_ssw(atoms, ASESurface(EMT()), steps=3, config=config, ls=ls,
                   rng=full_rng, starter_selector=full_selector,
                   selector_rng=full_selector_rng)
    path = tmp_path / 'pool-callback.pkl'
    def pause(cp):
        save_ssw_checkpoint(path, cp)
        return cp.next_index == 1
    first = run_ssw(atoms, ASESurface(EMT()), steps=3, config=config, ls=ls,
                    rng=np.random.default_rng(19), starter_selector=policy(),
                    selector_rng=np.random.default_rng(23), checkpoint_callback=pause)
    assert first.status == 'paused'
    assert first.checkpoint.schema_version == 5
    resumed_selector = policy()
    resumed_rng, resumed_selector_rng = np.random.default_rng(999), np.random.default_rng(999)
    resumed = run_ssw(atoms, ASESurface(EMT()), steps=2, config=config, ls=ls,
                      rng=resumed_rng, starter_selector=resumed_selector,
                      selector_rng=resumed_selector_rng,
                      checkpoint=load_ssw_checkpoint(path))
    assert resumed.status == full.status == 'completed'
    assert resumed.evaluation_requests == full.evaluation_requests
    assert [r.starter_selection for r in resumed.records] == [r.starter_selection for r in full.records]
    assert [r.ls_update for r in resumed.records] == [r.ls_update for r in full.records]
    assert [r.landing.energy if r.landing else None for r in resumed.records] == [
        r.landing.energy if r.landing else None for r in full.records]
    assert resumed_rng.bit_generator.state == full_rng.bit_generator.state
    assert resumed_selector_rng.bit_generator.state == full_selector_rng.bit_generator.state
    np.testing.assert_array_equal(resumed.current.positions, full.current.positions)


def test_native_ls_wrapper_accepts_pause_callback():
    from pamssw.standalone.ls_native_reference import NativeLSSettings, run_native_ls_ssw
    atoms, config, _ = _case()
    ls = NativeLSSettings({(29, 29): 1.}, {(29, 29): 3.}, target_mev_per_atom=1.)
    seen = []
    result = run_native_ls_ssw(atoms, ASESurface(EMT()), steps=2, config=config,
        rng=np.random.default_rng(19), ls=ls,
        checkpoint_callback=lambda cp: seen.append(cp.response.steps) or True)
    assert result.status == 'paused'
    assert seen == [1]
    assert result.checkpoint.response.steps == 1


def test_terminal_ls_update_never_calls_callback(monkeypatch):
    from pamssw.standalone.softening import LSResponseState
    atoms, config, ls = _case()
    def fail_update(*args, **kwargs):
        raise ValueError('intentional response failure')
    monkeypatch.setattr(LSResponseState, 'update', fail_update)
    seen = []
    result = run_ssw(atoms, ASESurface(EMT()), steps=2, config=config,
                     rng=np.random.default_rng(19), ls=ls,
                     checkpoint_callback=lambda cp: seen.append(cp) or True)
    assert result.status == result.checkpoint.status == 'ls_update_failed'
    assert seen == []
