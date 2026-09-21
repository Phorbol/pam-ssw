"""RC-VC lifecycle and certificate contracts, not material efficacy tests."""
from types import SimpleNamespace
import numpy as np
import pytest
from ase.build import bulk
from pamssw.standalone.rc_vc_reference import RCVCSSWConfig,run_rc_vc_ssw
from pamssw.standalone.vc_geometry import VCEvaluation


def inputs():
    a=bulk('Cu','fcc',a=3.6,cubic=True)
    return a,[dict(bodies=[tuple(range(4))],parents=(-1,),joints=(None,))]


class Flat:
    requests=0
    def evaluate(self,a):self.requests+=1;return 0.,np.zeros_like(a.positions),np.zeros((3,3))


def config(**kw):return RCVCSSWConfig(rotation_length=2.,torsion_length=2.,strain_length=4.,width=.2,rotation_bias=10.,max_gaussians=2,temperature_K=0.,**kw)


def test_full_path_enthalpy_mc_rejected_with_certificates_and_fixed_anchor(monkeypatch):
    import pamssw.standalone.rc_vc_reference as rc
    s=Flat();seen=[];anchors=[];original=rc.generalized_dimer
    def dimer(x,n,**kw):anchors.append(n.copy());return original(x,n,**kw)
    monkeypatch.setattr(rc,'generalized_dimer',dimer)
    def full(a,s,**kw):
        assert set(kw)=={'strain_length','pressure','fmax','stress_tol','max_step','maxiter'}
        seen.append(a.copy());s.evaluate(a);number=len(seen)-1
        ev=VCEvaluation(float(number),np.zeros(3*len(a)+6),a.copy(),-float(number),np.zeros_like(a.positions),np.zeros((3,3)),a.get_volume())
        return SimpleNamespace(converged=True,evaluation=ev,certificate=dict(certified=True,fmax=0.,stress_max=0.),requests=1)
    monkeypatch.setattr(rc,'cell_quench',full)
    a,t=inputs();r=run_rc_vc_ssw(a,s,trees=t,anchor=0,steps=1,config=config(),rng=np.random.default_rng(3))
    assert r.status=='completed' and len(r.minima)==2
    assert not r.records[1]['accepted'] and r.current is r.initial
    assert r.records[1]['delta']==1. and r.minima[-1].energy==-1. # MC uses objective, not E
    assert r.records[1]['certificate']['certified']
    assert r.requests==sum(e['requests'] for e in r.records)
    assert not np.allclose(seen[0].cell.array,seen[1].cell.array)
    assert len(anchors)==2
    np.testing.assert_array_equal(anchors[0],anchors[1])


def test_initial_uncertified_stops_without_proposal(monkeypatch):
    import pamssw.standalone.rc_vc_reference as rc
    a,t=inputs();s=Flat()
    def fail(a,s,**kw):s.evaluate(a);return SimpleNamespace(converged=False,evaluation=None,certificate=dict(certified=False),requests=1)
    monkeypatch.setattr(rc,'cell_quench',fail)
    r=run_rc_vc_ssw(a,s,trees=t,anchor=0,steps=2,config=config(),rng=np.random.default_rng(3))
    assert r.status=='initial_quench_failed' and r.requests==1 and not r.minima


def test_failed_rotation_cost_and_current_preserved(monkeypatch):
    import pamssw.standalone.rc_vc_reference as rc
    a,t=inputs();s=Flat()
    def full(a,s,**kw):
        s.evaluate(a);ev=VCEvaluation(0.,np.zeros(18),a.copy(),0.,np.zeros_like(a.positions),np.zeros((3,3)),a.get_volume())
        return SimpleNamespace(converged=True,evaluation=ev,certificate=dict(certified=True),requests=1)
    def fail(x,n,**kw):kw['evaluate'](x);raise RuntimeError('injected EFS cap')
    monkeypatch.setattr(rc,'cell_quench',full);monkeypatch.setattr(rc,'generalized_dimer',fail)
    r=run_rc_vc_ssw(a,s,trees=t,anchor=0,steps=2,config=config(),rng=np.random.default_rng(3))
    assert r.requests==3 and r.current is r.initial and len(r.minima)==1
    assert [e['requests'] for e in r.records]==[1,1,1]
    assert [e['status'] for e in r.records[1:]]==['evaluation_failed']*2


def test_nonperiodic_rejected_before_oracle():
    a,t=inputs();a.pbc=False;s=Flat()
    with pytest.raises(ValueError,match='periodic'):run_rc_vc_ssw(a,s,trees=t,anchor=0,steps=1,config=config(),rng=np.random.default_rng(3))
    assert s.requests==0
