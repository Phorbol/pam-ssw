import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.vc_softening import FrozenPeriodicCellSoftening
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart,ASEStressSurface
from pamssw.standalone.vc_reference import VCSSWConfig,run_vc_ssw
from pamssw.standalone.paper_reference import LSSettings

TABLE=dict(bond_energies={(29,29):1.},bond_lengths={(29,29):2.9})

def test_ls_stress_and_forces_pull_back_at_finite_strain():
    a=bulk('Cu','fcc',a=3.6,cubic=True)
    soft=FrozenPeriodicCellSoftening.from_atoms(a,**TABLE)
    chart=SymmetricLogStrainChart(a,strain_length=3.6);q=chart.pack(a)
    q[-6:]=[.12,-.07,.05,.03,-.04,.06];q[0]+=.03
    ev=chart.evaluate(q,soft.evaluate_stress)
    for k in range(len(q)):
        plus=q.copy();minus=q.copy();plus[k]+=1e-6;minus[k]-=1e-6
        fd=(chart.evaluate(plus,soft.evaluate_stress).energy-chart.evaluate(minus,soft.evaluate_stress).energy)/2e-6
        assert fd==pytest.approx(ev.gradient[k],abs=1e-8)
    single=bulk('Cu','fcc',a=3.6)
    e,f,s=FrozenPeriodicCellSoftening.from_atoms(single,**TABLE).evaluate_stress(single)
    np.testing.assert_array_equal(f,0.)
    assert np.trace(s)<0  # repulsive LS potential has compressive tensile-positive stress


def test_vc_periodic_stress_subclass_stays_full_pbc_only():
    a = bulk('Cu', 'fcc', a=3.6, cubic=True)
    a.pbc = [True, True, False]
    with pytest.raises(ValueError, match='full|PBC'):
        FrozenPeriodicCellSoftening.from_atoms(a, **TABLE)


def test_vc_ls_pipeline_discards_both_biases_and_preserves_cost():
    a=bulk('Cu','fcc',a=3.6,cubic=True);surface=ASEStressSurface(EMT())
    cfg=VCSSWConfig(strain_length=3.6,width=.1,rotation_bias=100,max_gaussians=1,
        relax_steps=200,rotation_hvp=40)
    r=run_vc_ssw(a,surface,steps=1,config=cfg,rng=np.random.default_rng(17),
        ls=LSSettings(**TABLE,target_per_atom=.001))
    assert r.records[-1]['ls']['energy_response'] is not None
    assert len(r.minima)==2
    assert r.requests==surface.requests==sum(x['requests'] for x in r.records)
    for ev in r.minima:
        e,f,s=ASEStressSurface(EMT()).evaluate(ev.atoms)
        assert e==pytest.approx(ev.energy,abs=1e-10)
        assert np.linalg.norm(f,axis=1).max()<=cfg.fmax
        assert np.abs(s).max()<=cfg.stress_tol


def test_ls_preparation_failure_stops_once_and_preserves_initial(monkeypatch):
    from pamssw.standalone import ls_cycle
    calls=[]
    def failed(atoms,surface,**kwargs):
        calls.append(1);surface.evaluate(atoms);raise ValueError('injected soft preparation failure')
    monkeypatch.setattr(ls_cycle,'prepare_ls_step',failed)
    a=bulk('Cu','fcc',a=3.6,cubic=True);s=ASEStressSurface(EMT())
    r=run_vc_ssw(a,s,steps=3,config=VCSSWConfig(strain_length=3.6,width=.1,rotation_bias=100),
        rng=np.random.default_rng(17),ls=LSSettings(**TABLE,target_per_atom=.001))
    assert len(calls)==1 and len(r.records)==2 and r.status=='ls_prequench_failed'
    assert len(r.minima)==1 and r.records[-1]['ls']['error']
    assert r.requests==s.requests==sum(x['requests'] for x in r.records)


def test_initial_ls_configuration_failure_retains_paid_quench():
    a=bulk('Cu','fcc',a=3.6,cubic=True);s=ASEStressSurface(EMT())
    r=run_vc_ssw(a,s,steps=2,config=VCSSWConfig(strain_length=3.6,width=.1,rotation_bias=100),
        rng=np.random.default_rng(17),ls=LSSettings(bond_energies={},bond_lengths={},target_per_atom=.001))
    assert r.status=='ls_initialization_failed' and len(r.minima)==1
    assert r.requests==s.requests==sum(x['requests'] for x in r.records)


def test_vc_ls_explicit_memory_reaches_soft_prequench(monkeypatch):
    from pamssw.standalone import ls_cycle
    original = ls_cycle.prepare_ls_step
    seen = []
    def capture(*args, **kwargs):
        seen.append(kwargs.get('lbfgs_memory'))
        return original(*args, **kwargs)
    monkeypatch.setattr(ls_cycle, 'prepare_ls_step', capture)
    a = bulk('Cu', 'fcc', a=3.6, cubic=True)
    cfg = VCSSWConfig(strain_length=3.6, width=.1, rotation_bias=100,
        max_gaussians=1, relax_steps=200, rotation_hvp=40, lbfgs_memory=23)
    result = run_vc_ssw(a, ASEStressSurface(EMT()), steps=1, config=cfg,
        rng=np.random.default_rng(17), ls=LSSettings(**TABLE, target_per_atom=.001))
    assert seen == [23]
    assert result.records[-1]['ls']['energy_response'] is not None
