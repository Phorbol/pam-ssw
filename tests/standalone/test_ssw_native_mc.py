import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.calculators.calculator import Calculator, all_changes

from pamssw.standalone.native_mc import NativeMCSettings
from pamssw.standalone.paper_reference import (
    SSWConfig, load_ssw_checkpoint, run_ssw,
)
from pamssw.standalone.surface import ASESurface


def _config():
    return SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e-5, relax_steps=100, fd_step=1e-4,
        rotation_hvp=8, rotation_tol=1e-3, direction_sampling='global')


def _atoms():
    return Atoms('Cu2', positions=[[0., 0., 0.], [2.5, 0., 0.]])


class _Harmonic(Calculator):
    implemented_properties = ['energy', 'forces']
    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {'energy': float((atoms.positions ** 2).sum() / 2),
                        'forces': -atoms.positions.copy()}


def test_default_checkpoint_remains_schema_one_and_has_no_mc_telemetry(tmp_path):
    path = tmp_path / 'paper.pkl'
    result = run_ssw(_atoms(), ASESurface(EMT()), steps=1, config=_config(),
                     rng=np.random.default_rng(19), checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    assert checkpoint.schema_version == 1
    assert getattr(checkpoint, 'mc_settings', None) is None
    assert getattr(checkpoint, 'native_mc_state', None) is None
    assert result.records[0].mc_telemetry is None


def test_native_mc_split_resume_is_exact_and_persists_counter(tmp_path):
    settings = NativeMCSettings(energy_tol=1e-3, maxtrap=2)
    continuous = run_ssw(_atoms(), ASESurface(EMT()), steps=2, config=_config(),
                         rng=np.random.default_rng(19), mc=settings,
                         checkpoint_path=tmp_path / 'continuous.pkl')
    path = tmp_path / 'native.pkl'
    run_ssw(_atoms(), ASESurface(EMT()), steps=1, config=_config(),
            rng=np.random.default_rng(19), mc=settings, checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    assert checkpoint.schema_version == 2
    assert checkpoint.native_mc_state is not None
    assert all(record.mc_telemetry is not None for record in continuous.records)
    assert [record.mc_telemetry.nsame_for_acceptance for record in continuous.records] == [1, 2]
    resumed = run_ssw(_atoms(), ASESurface(EMT()), steps=1, config=_config(),
                      rng=np.random.default_rng(999), mc=settings,
                      checkpoint=checkpoint)
    assert resumed.checkpoint.native_mc_state == continuous.checkpoint.native_mc_state
    assert resumed.checkpoint.rng_state == continuous.checkpoint.rng_state
    assert [r.mc_telemetry for r in resumed.records] == [r.mc_telemetry for r in continuous.records]
    assert [r.accepted for r in resumed.records] == [r.accepted for r in continuous.records]
    np.testing.assert_allclose(resumed.current.positions, continuous.current.positions)
    assert resumed.evaluation_requests == continuous.evaluation_requests


def test_native_mc_rejects_schema_one_checkpoint_before_pes(tmp_path):
    path = tmp_path / 'paper.pkl'
    run_ssw(_atoms(), ASESurface(EMT()), steps=1, config=_config(),
            rng=np.random.default_rng(19), checkpoint_path=path)
    checkpoint = load_ssw_checkpoint(path)
    surface = ASESurface(EMT())
    with pytest.raises(ValueError, match='native MC state'):
        run_ssw(_atoms(), surface, steps=1, config=_config(),
                rng=np.random.default_rng(19), mc=NativeMCSettings(1e-3, 2),
                checkpoint=checkpoint)
    assert surface.requests == 0


@pytest.mark.parametrize("landing_converged,landing_energy", [(False, -1.), (True, -1.), (True, 1000.)])
def test_native_mc_draws_only_for_qualified_true_landing(monkeypatch, tmp_path, landing_converged, landing_energy):
    from types import SimpleNamespace
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.surface import QuenchResult

    quench_calls = []
    def controlled_quench(atoms, surface, *, terms=(), **kwargs):
        quench_calls.append(bool(terms))
        is_landing = len(quench_calls) == 3
        return QuenchResult(atoms.copy(), landing_energy if is_landing else 0., 0.,
                            landing_converged if is_landing else True,
                            0, 0, 'modified' if terms else 'true', None)
    class CountingRNG:
        def __init__(self):
            self.inner = np.random.default_rng(7)
            self.draws = []
        def random(self):
            value = self.inner.random()
            self.draws.append(value)
            return value
        def __getattr__(self, name):
            return getattr(self.inner, name)

    monkeypatch.setattr(paper, 'quench', controlled_quench)
    monkeypatch.setattr(paper, 'paper_biased_direction',
        lambda *a, **kw: SimpleNamespace(direction=np.array([[1., 0., 0.]]),
             converged=True, curvature=1., residual_norm=0., force_calls=0))
    rng = CountingRNG()
    result = run_ssw(Atoms('H', positions=[[0., 0., 0.]]), ASESurface(_Harmonic()),
                     steps=1, config=_config(), rng=rng, mc=NativeMCSettings(1e-3, 2),
                     checkpoint_path=tmp_path / "routing.pkl")
    assert quench_calls == [False, True, False]
    record = result.records[0]
    assert record.landing is not None
    if landing_converged:
        assert record.accepted == (landing_energy < 0)
        assert record.mc_telemetry.delta_energy_eV == landing_energy
        assert result.minima[-1].energy == landing_energy
        assert result.checkpoint.current_energy == (landing_energy if record.accepted else 0.)
        assert rng.draws == [record.mc_telemetry.uniform]
        assert record.mc_telemetry.state.nsame == 0
    else:
        assert record.status == 'true_quench_failed'
        assert not record.accepted
        assert record.mc_telemetry is None
        assert rng.draws == []


def test_native_ls_wrapper_forwards_mc(monkeypatch):
    from pamssw.standalone.ls_native_reference import NativeLSSettings, run_native_ls_ssw
    import pamssw.standalone.paper_reference as paper
    received = {}
    def capture(*args, **kwargs):
        received.update(kwargs)
        return 'sentinel'
    monkeypatch.setattr(paper, 'run_ssw', capture)
    settings = NativeMCSettings(.1, 99999)
    ls = NativeLSSettings(bond_energies={}, bond_lengths={})
    assert run_native_ls_ssw(_atoms(), None, steps=0, config=_config(),
                            rng=np.random.default_rng(3), ls=ls, mc=settings) == 'sentinel'
    assert received['mc'] is settings


def test_mc_failure_is_terminal_and_skips_native_ls_update(monkeypatch):
    import pamssw.standalone.paper_reference as paper
    from pamssw.standalone.ls_native_reference import NativeLSRuntime, NativeLSSettings

    updates = []
    original_update = NativeLSRuntime.update
    def observed_update(self, *args, **kwargs):
        updates.append(True)
        return original_update(self, *args, **kwargs)
    def failed_mc(*args, **kwargs):
        raise ValueError('synthetic native MC domain failure')

    monkeypatch.setattr(NativeLSRuntime, 'update', observed_update)
    monkeypatch.setattr(paper, 'native_metropolis', failed_mc)
    ls = NativeLSSettings(bond_energies={(29, 29): 1.}, bond_lengths={(29, 29): 3.},
                          target_mev_per_atom=1.)
    result = run_ssw(_atoms(), ASESurface(EMT()), steps=1, config=_config(),
                     rng=np.random.default_rng(19), ls=ls,
                     mc=NativeMCSettings(1e-3, 2))
    assert result.status == 'mc_failed'
    assert result.records[0].status == 'mc_failed'
    assert result.records[0].landing is not None and result.records[0].landing.converged
    assert result.records[0].mc_telemetry['error'].startswith('mc: ValueError:')
    assert updates == []
