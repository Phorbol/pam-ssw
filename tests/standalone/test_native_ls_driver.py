"""Native-derived controller integration on real Cu/EMT; no efficacy claim."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.paper_reference import SSWConfig,LSSettings,run_ls_ssw
from pamssw.standalone.ls_native_reference import NativeLSSettings,run_native_ls_ssw


def config():
    return SSWConfig(width=.2,rotation_bias=.5,max_gaussians=1,temperature_K=0.,fmax=.01,
        relax_steps=150,fd_step=1e-4,rotation_hvp=41,rotation_tol=.02,
        direction_sampling='global',rotation_solver='dimer',cluster_frame='translation_only',quench_optimizer='safe-lbfgs-total')


class Bounded(ASESurface):
    def evaluate(self,a):
        assert self.requests<500,'bounded integration exceeded declared test budget'
        return super().evaluate(a)


def test_real_cu_two_attempt_native_lifecycle_and_bare_landing():
    from ase.cluster.icosahedron import Icosahedron
    from dataclasses import replace
    atoms=Icosahedron('Cu',2);original=atoms.copy()
    c=replace(config(),cluster_frame='direction_only',rotation_solver='ritz')
    # Explicit fixture table for wiring, not a recovered Cu table or LS default.
    ls=NativeLSSettings({(29,29):3.},{(29,29):2.8},scale=.1)
    s=Bounded(EMT());r=run_native_ls_ssw(atoms,s,steps=2,config=c,rng=np.random.default_rng(7),ls=ls)
    assert r.status=='completed' and len(r.records)==2 and len(r.minima)==3
    assert r.evaluation_requests==s.requests==r.initial.evaluation_requests+sum(e.evaluation_requests for e in r.records)
    for i,event in enumerate(r.records):
        assert event.energy_response is not None
        assert event.ls_update['step']==i+1
        assert event.ls_update['caller_convention']=='completed_outer_attempts_selected_current'
        assert event.ls_update['observed_response_mev_per_atom']==pytest.approx(1000*event.energy_response)
    verify=ASESurface(EMT())
    for minimum in r.minima:
        e,f=verify.evaluate(minimum.atoms)
        assert e==pytest.approx(minimum.energy,abs=1e-12)
        assert np.linalg.norm(f,axis=1).max()<=config().fmax
    np.testing.assert_array_equal(atoms.positions,original.positions)
    np.testing.assert_array_equal(atoms.cell.array,original.cell.array)


@pytest.mark.parametrize('native',[False,True])
def test_initial_ls_table_failure_keeps_paid_initial_minimum(native):
    atoms=bulk('Cu','fcc',a=3.6,cubic=True);s=Bounded(EMT())
    # Missing Cu entries after a successful true quench.
    if native:
        ls=NativeLSSettings({(6,6):3.},{(6,6):1.6})
        r=run_native_ls_ssw(atoms,s,steps=2,config=config(),rng=np.random.default_rng(7),ls=ls)
    else:
        ls=LSSettings({(6,6):3.},{(6,6):1.6},target_per_atom=.02)
        r=run_ls_ssw(atoms,s,steps=2,config=config(),rng=np.random.default_rng(7),ls=ls)
    assert r.status=='ls_initialization_failed' and len(r.minima)==1
    assert r.evaluation_requests==s.requests==r.initial.evaluation_requests
    assert r.initial.converged and r.records[0].evaluation_requests==0
    assert 'missing' in r.records[0].error


def test_native_periodic_mic_prequench_failure_is_not_updated():
    # Native MIC geometry is deliberately distinct from the new paper-mode
    # periodic image potential. At Cu4 half-cell contacts this prequench fails;
    # preserve that failure instead of converting it into a valid LS response.
    atoms=bulk('Cu','fcc',a=3.6,cubic=True);s=Bounded(EMT())
    ls=NativeLSSettings({(29,29):3.},{(29,29):2.8},scale=.1)
    r=run_native_ls_ssw(atoms,s,steps=2,config=config(),rng=np.random.default_rng(7),ls=ls)
    assert r.status=='ls_prequench_failed' and len(r.records)==1
    assert r.records[0].ls_update is None and r.records[0].energy_response is None
    assert r.evaluation_requests==s.requests==r.initial.evaluation_requests+r.records[0].evaluation_requests


def test_true_quench_oracle_failure_keeps_completed_work_and_paid_cost(monkeypatch):
    from pamssw.standalone import paper_reference as module
    from dataclasses import replace
    from ase.cluster.icosahedron import Icosahedron
    s=Bounded(EMT());actual=module.quench;true_count=[0]
    def fail_final(atoms,surface,**kwargs):
        if not kwargs.get('terms'):
            true_count[0]+=1
            if true_count[0]==2:
                surface.evaluate(atoms) # Real paid request before backend failure.
                raise RuntimeError('intentional final quench cap')
        return actual(atoms,surface,**kwargs)
    monkeypatch.setattr(module,'quench',fail_final)
    c=replace(config(),cluster_frame='direction_only',rotation_solver='ritz')
    r=module.run_ssw(Icosahedron('Cu',2),s,steps=2,config=c,rng=np.random.default_rng(7))
    assert r.status=='evaluation_failed' and len(r.records)==1 and len(r.minima)==1
    event=r.records[0]
    assert event.status=='evaluation_failed' and event.climb
    assert event.landing is None and not event.accepted
    assert 'true_quench' in event.error
    assert r.evaluation_requests==s.requests==r.initial.evaluation_requests+event.evaluation_requests
    np.testing.assert_array_equal(r.current.positions,r.initial.atoms.positions)


def test_climb_budget_denial_returns_a_counted_partial_record():
    from pamssw.standalone.paper_reference import run_ssw
    class Capped(ASESurface):
        def evaluate(self,a):
            if self.requests>=2:raise RuntimeError('two-request regression cap')
            return super().evaluate(a)
    s=Capped(EMT());r=run_ssw(bulk('Cu','fcc',a=3.6,cubic=True),s,steps=2,
        config=config(),rng=np.random.default_rng(7))
    assert r.status=='evaluation_failed' and len(r.records)==1
    assert r.evaluation_requests==s.requests==2
    assert r.evaluation_requests==r.initial.evaluation_requests+r.records[0].evaluation_requests
    assert 'climb' in r.records[0].error and not r.records[0].accepted
    assert len(r.minima)==1
