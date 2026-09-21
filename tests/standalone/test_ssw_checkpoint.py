import numpy as np
import pytest
from copy import deepcopy
from ase import Atoms
from ase.calculators.emt import EMT

from pamssw.standalone.paper_reference import (
    LSSettings, SSWConfig, load_ssw_checkpoint, run_ssw, save_ssw_checkpoint,
)
from pamssw.standalone.surface import ASESurface, quench


def _case():
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
    seed = quench(atoms, ASESurface(EMT()), fmax=1e-6, steps=100)
    config = SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e-5, relax_steps=100, fd_step=1e-4,
        rotation_hvp=8, rotation_tol=1e-3, direction_sampling='global')
    ls = LSSettings(bond_energies={(29, 29): 1.}, bond_lengths={(29, 29): 3.},
                    target_per_atom=.001)
    return seed.atoms, config, ls


def test_fixed_cell_ls_checkpoint_resume_matches_continuous(tmp_path):
    atoms, config, ls = _case()
    continuous = run_ssw(atoms, ASESurface(EMT()), steps=2, config=config,
                         rng=np.random.default_rng(19), ls=ls)
    path = tmp_path / 'ssw.pkl'
    first = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                    rng=np.random.default_rng(19), ls=ls, checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    assert checkpoint.next_index == 1
    resumed = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                       rng=np.random.default_rng(999), ls=ls, checkpoint=checkpoint)
    assert resumed.status == continuous.status == 'completed'
    assert [r.index for r in resumed.records] == [0, 1]
    assert resumed.evaluation_requests == continuous.evaluation_requests
    np.testing.assert_allclose(resumed.current.positions, continuous.current.positions)
    for got, expected in zip(resumed.records, continuous.records):
        assert got.index == expected.index
        assert got.status == expected.status and got.accepted == expected.accepted
        np.testing.assert_allclose(got.last_atoms.positions, expected.last_atoms.positions)
    assert first.evaluation_requests < resumed.evaluation_requests


def test_checkpoint_rejects_mismatched_parameters_before_pes_request(tmp_path):
    atoms, config, ls = _case()
    path = tmp_path / 'ssw.pkl'
    run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
            rng=np.random.default_rng(19), ls=ls, checkpoint_path=path)
    changed = SSWConfig(width=.2, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e-5, relax_steps=100, fd_step=1e-4,
        rotation_hvp=8, rotation_tol=1e-3, direction_sampling='global')
    surface = ASESurface(EMT())
    with pytest.raises(ValueError, match='SSWConfig'):
        run_ssw(atoms, surface, steps=1, config=changed,
                rng=np.random.default_rng(19), ls=ls,
                checkpoint=load_ssw_checkpoint(path))
    assert surface.requests == 0


def test_terminal_failure_checkpoint_is_not_resumable(tmp_path):
    atoms, config, ls = _case()
    path = tmp_path / 'ssw.pkl'
    run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
            rng=np.random.default_rng(19), ls=ls, checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    checkpoint.status = 'evaluation_failed'
    save_ssw_checkpoint(path, checkpoint)
    with pytest.raises(ValueError, match='terminal checkpoint'):
        run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                rng=np.random.default_rng(19), ls=ls,
                checkpoint=load_ssw_checkpoint(path))


def test_default_path_does_not_snapshot(monkeypatch):
    import pamssw.standalone.paper_reference as ref
    atoms, config, ls = _case()
    monkeypatch.setattr(ref, '_checkpoint_copy', lambda value: (_ for _ in ()).throw(AssertionError('unexpected snapshot')))
    result = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                     rng=np.random.default_rng(19), ls=ls)
    assert result.checkpoint is None


def test_steps_zero_disk_boundary_and_existing_path_are_explicit(tmp_path):
    atoms, config, ls = _case(); path = tmp_path / 'boundary.pkl'
    first_surface = ASESurface(EMT())
    first = run_ssw(atoms, first_surface, steps=0, config=config,
                    rng=np.random.default_rng(19), ls=ls, checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    assert checkpoint.next_index == 0 and len(checkpoint.records) == 0
    resumed_surface = ASESurface(EMT())
    resumed = run_ssw(atoms, resumed_surface, steps=0, config=config,
                       rng=np.random.default_rng(999), ls=ls, checkpoint=checkpoint)
    assert resumed_surface.requests == 0
    assert resumed.evaluation_requests == first.evaluation_requests
    fresh = run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                    rng=np.random.default_rng(19), ls=ls, checkpoint_path=path)
    assert fresh.records[0].index == 0


def test_ls_prequench_surface_failure_saves_paid_terminal_checkpoint(tmp_path):
    atoms, config, ls = _case()
    baseline = ASESurface(EMT()); initial = quench(atoms, baseline, fmax=config.fmax,
                                                    steps=config.relax_steps)
    class FailAfter:
        def __init__(self, allowed):
            self.surface = ASESurface(EMT()); self.allowed = allowed
        @property
        def requests(self): return self.surface.requests
        def evaluate(self, value):
            if self.requests >= self.allowed:
                self.surface.evaluate(value)
                raise RuntimeError('intentional next surface request failure')
            return self.surface.evaluate(value)
    surface = FailAfter(baseline.requests); path = tmp_path / 'failed.pkl'
    result = run_ssw(initial.atoms, surface, steps=1, config=config,
                     rng=np.random.default_rng(19), ls=ls, checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    assert result.status == checkpoint.status == 'ls_prequench_failed'
    assert checkpoint.evaluation_requests == result.evaluation_requests == surface.requests
    assert checkpoint.records[0].evaluation_requests >= 1
    with pytest.raises(ValueError, match='terminal checkpoint'):
        run_ssw(initial.atoms, ASESurface(EMT()), steps=0, config=config,
                rng=np.random.default_rng(19), ls=ls, checkpoint=checkpoint)


def test_native_ls_state_survives_split_resume(tmp_path):
    from pamssw.standalone.ls_native_reference import NativeLSSettings, run_native_ls_ssw
    atoms, config, _ = _case()
    settings = NativeLSSettings({(29, 29): 1.}, {(29, 29): 3.}, target_mev_per_atom=1.)
    continuous = run_native_ls_ssw(atoms, ASESurface(EMT()), steps=2, config=config,
                                   rng=np.random.default_rng(19), ls=settings,
                                   checkpoint_path=tmp_path / 'continuous.pkl')
    path = tmp_path / 'native.pkl'
    first = run_native_ls_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                              rng=np.random.default_rng(19), ls=settings, checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    resumed = run_native_ls_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
                                rng=np.random.default_rng(999), ls=settings, checkpoint=checkpoint)
    assert checkpoint.response.steps == 1 and resumed.checkpoint is not None
    assert resumed.records[1].index == 1
    assert resumed.evaluation_requests == continuous.evaluation_requests
    assert resumed.records[1].ls_update == continuous.records[1].ls_update
    assert resumed.checkpoint.response.steps == 2
    assert resumed.checkpoint.response.state.table
    assert resumed.checkpoint.response.state.table != checkpoint.response.state.table
    assert resumed.checkpoint.response.state.table == continuous.checkpoint.response.state.table
    assert resumed.checkpoint.rng_state == continuous.checkpoint.rng_state


@pytest.mark.parametrize('kind', ['schema', 'index', 'cost', 'rng', 'type'])
def test_invalid_checkpoint_rejected_before_any_pes_request(tmp_path, kind):
    atoms, config, ls = _case(); path = tmp_path / 'valid.pkl'
    run_ssw(atoms, ASESurface(EMT()), steps=1, config=config,
            rng=np.random.default_rng(19), ls=ls, checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    if kind == 'schema': checkpoint.schema_version = 99
    elif kind == 'index': checkpoint.next_index += 1
    elif kind == 'cost': checkpoint.evaluation_requests += 1
    elif kind == 'rng': checkpoint.rng_state = dict(checkpoint.rng_state, bit_generator='PCG64DXSM')
    else: checkpoint = object()
    surface = ASESurface(EMT())
    with pytest.raises((TypeError, ValueError)):
        run_ssw(atoms, surface, steps=1, config=config, rng=np.random.default_rng(19),
                ls=ls, checkpoint=checkpoint)
    assert surface.requests == 0
