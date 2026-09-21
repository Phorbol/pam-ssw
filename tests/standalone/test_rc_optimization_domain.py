"""Real molecular geometry counterexample; no search-efficiency claim."""
import numpy as np
import pytest
from ase.data.s22 import create_s22_system
from pamssw.standalone.rc_forest import RigidForestChart, ForestSurface
from pamssw.standalone.rc_vc_geometry import RigidForestCellChart
from pamssw.standalone.rc_optimization_domain import PrincipalRigidForestCellChart, RootRotationChartError
from pamssw.standalone.generalized_numerics import safe_lbfgs


def water():
    a=create_s22_system('Water_dimer')
    t=[dict(bodies=[(0,1,2)],parents=(-1,),joints=(None,)),dict(bodies=[(3,4,5)],parents=(-1,),joints=(None,))]
    return a,t


class NeverOracle:
    requests=0
    def evaluate(self,a):
        self.requests+=1
        raise AssertionError('domain must reject before physical oracle')


def test_s22_full_turn_raw_geometry_rank_loss_but_optimization_rejects():
    a,t=water();c=RigidForestChart(a,t);q=np.zeros(6);q[3]=2*np.pi
    raw,J=c.evaluate(q)
    np.testing.assert_allclose(raw.positions,a.positions,atol=3e-15)
    assert np.linalg.matrix_rank(J.reshape(-1,6))==4
    assert np.linalg.matrix_rank(c.evaluate(np.zeros(6))[1].reshape(-1,6))==6
    # A force along the lost y-rotation is nonzero but invisible at 2pi.
    f=c.evaluate(np.zeros(6))[1][:,:,4]
    surviving=J[:,:,3]
    f=f-surviving*np.sum(f*surviving)/np.sum(surviving**2)
    assert np.linalg.norm(f)>0.1
    assert np.linalg.norm(np.einsum('ijk,ij->k',J,f))<1e-14
    s=NeverOracle();ev=ForestSurface(c,s,rotation_length=2.3,torsion_length=1.)
    with pytest.raises(RootRotationChartError):ev.evaluate(q*ev.scales)
    assert s.requests==0
    r=safe_lbfgs(q*ev.scales,ev.evaluate,gradient_norm=np.linalg.norm,step_norm=np.linalg.norm,gtol=.01,max_step=.2,maxiter=3)
    assert r.status=='evaluation_failed' and not r.converged
    assert 'RootRotationChartError' in r.error and s.requests==0


def test_exact_principal_boundary_after_metric_unscaling():
    a,t=water();c=RigidForestChart(a,t);ev=ForestSurface(c,None,rotation_length=2.,torsion_length=1.)
    x=np.zeros(6);x[3]=2*np.nextafter(np.pi,0.)
    ev.coordinates(x)
    x[3]=2*np.pi
    with pytest.raises(RootRotationChartError):ev.coordinates(x)


def test_periodic_anchor_rotation_also_guarded_raw_map_unchanged():
    a,t=water();a.cell=np.eye(3)*12;a.pbc=True
    args=dict(rotation_length=2.,torsion_length=1.,strain_length=4.)
    raw=RigidForestCellChart(a,t,**args);opt=PrincipalRigidForestCellChart(a,t,**args)
    q=np.zeros(raw.dimension);i=raw.kinds.index('rotation');q[i]=4*np.pi
    out,*_=raw.geometry(q)
    np.testing.assert_allclose(out.positions,a.positions,atol=3e-15)
    with pytest.raises(RootRotationChartError):opt.geometry(q)
    with pytest.raises(RootRotationChartError):opt.unpack(q)


def test_forest_driver_domain_failure_retains_selected_minimum_and_cost(monkeypatch):
    import pamssw.standalone.rc_reference as rc
    from pamssw.standalone.rc_forest_reference import RCForestSSWConfig,run_rc_forest_ssw
    from pamssw.standalone.surface import QuenchResult
    class Flat:
        requests=0
        def evaluate(self,a):self.requests+=1;return 0.,np.zeros_like(a.positions)
    def full(a,s,**kw):
        s.evaluate(a);return QuenchResult(a.copy(),0.,0.,True,0,1,'true')
    def invalid(x,n,**kw):
        kw['evaluate'](x)
        bad=x.copy();bad[3]=4*np.pi
        kw['evaluate'](bad)
        raise AssertionError('unreachable')
    monkeypatch.setattr(rc,'quench',full);monkeypatch.setattr(rc,'generalized_dimer',invalid)
    a,t=water();s=Flat()
    cfg=RCForestSSWConfig(rotation_length=2.,torsion_length=2.,width=.2,rotation_bias=10.)
    r=run_rc_forest_ssw(a,s,trees=t,anchor=0,steps=2,config=cfg,rng=np.random.default_rng(3))
    assert r.current is r.initial and len(r.minima)==1 and r.requests==3
    assert [e['requests'] for e in r.records]==[1,1,1]
    for e in r.records[1:]:
        assert e['status']=='evaluation_failed' and 'principal log ball' in e['error']
        assert e['climb'][0]['status']=='evaluation_failed'


def test_rc_vc_driver_domain_failure_cost_and_current(monkeypatch):
    from types import SimpleNamespace
    import pamssw.standalone.rc_vc_reference as rc
    from pamssw.standalone.vc_geometry import VCEvaluation
    a,t=water();a.cell=np.eye(3)*12;a.pbc=True
    class Flat:
        requests=0
        def evaluate(self,a):self.requests+=1;return 0.,np.zeros_like(a.positions),np.zeros((3,3))
    def full(a,s,**kw):
        s.evaluate(a)
        ev=VCEvaluation(0.,np.zeros(24),a.copy(),0.,np.zeros_like(a.positions),np.zeros((3,3)),a.get_volume())
        return SimpleNamespace(converged=True,evaluation=ev,certificate=dict(certified=True),requests=1)
    def invalid(x,n,**kw):
        kw['evaluate'](x)
        bad=x.copy();bad[0]=4*np.pi # periodic anchor retains its root rotation
        kw['evaluate'](bad)
        raise AssertionError('unreachable')
    monkeypatch.setattr(rc,'cell_quench',full);monkeypatch.setattr(rc,'generalized_dimer',invalid)
    cfg=rc.RCVCSSWConfig(rotation_length=2.,torsion_length=2.,strain_length=4.,width=.2,rotation_bias=10.)
    s=Flat();r=rc.run_rc_vc_ssw(a,s,trees=t,anchor=0,steps=2,config=cfg,rng=np.random.default_rng(3))
    assert r.current is r.initial and len(r.minima)==1 and r.requests==3
    assert [e['requests'] for e in r.records]==[1,1,1]
    for e in r.records[1:]:assert e['status']=='evaluation_failed' and 'principal log ball' in e['error']
