from types import SimpleNamespace
import numpy as np
import pytest
from ase.build import molecule,bulk
from pamssw.standalone.surface import QuenchResult
from pamssw.standalone.vc_geometry import VCEvaluation


@pytest.mark.parametrize('periodic',[False,True])
def test_rc_second_rotation_failure_retains_paid_first_stage_and_bias(monkeypatch,periodic):
    if periodic:
        import pamssw.standalone.rc_vc_reference as rc
        a=bulk('Cu','fcc',a=3.6,cubic=True)
        trees=[dict(bodies=[tuple(range(4))],parents=(-1,),joints=(None,))]
        cfg=rc.RCVCSSWConfig(rotation_length=2.,torsion_length=2.,strain_length=4.,width=.2,rotation_bias=10.,max_gaussians=2)
    else:
        import pamssw.standalone.rc_reference as rc
        from pamssw.standalone.rc_forest_reference import run_rc_forest_ssw,RCForestSSWConfig
        a=molecule('H2O');b=a.copy();b.translate([3,0,0]);a+=b
        trees=[dict(bodies=[(0,1,2)],parents=(-1,),joints=(None,)),dict(bodies=[(3,4,5)],parents=(-1,),joints=(None,))]
        cfg=RCForestSSWConfig(rotation_length=2.,torsion_length=2.,width=.2,rotation_bias=10.,max_gaussians=2)
    class Flat:
        requests=0
        def evaluate(self,a):
            self.requests+=1
            return (0.,np.zeros_like(a.positions),np.zeros((3,3))) if periodic else (0.,np.zeros_like(a.positions))
    s=Flat()
    def full(a,s,**kw):
        s.evaluate(a)
        if not periodic:return QuenchResult(a.copy(),0.,0.,True,0,1,'true')
        ev=VCEvaluation(0.,np.zeros(3*len(a)+6),a.copy(),0.,np.zeros_like(a.positions),np.zeros((3,3)),a.get_volume())
        return SimpleNamespace(converged=True,evaluation=ev,certificate=dict(certified=True),requests=1)
    monkeypatch.setattr(rc,'cell_quench' if periodic else 'quench',full)
    original=rc.generalized_dimer;calls=[]
    def dimer(x,n,**kw):
        calls.append(x.copy())
        if len(calls)==2:
            kw['evaluate'](x)
            raise RuntimeError('second-stage cap')
        return original(x,n,**kw)
    monkeypatch.setattr(rc,'generalized_dimer',dimer)
    run=rc.run_rc_vc_ssw if periodic else run_rc_forest_ssw
    result=run(a,s,trees=trees,anchor=0,steps=1,config=cfg,rng=np.random.default_rng(3))
    e=result.records[1]
    assert e['status']=='evaluation_failed' and result.current is result.initial
    assert result.requests==sum(r['requests'] for r in result.records)
    assert len(e['frozen_gaussians'])==1
    term=e['frozen_gaussians'][0]
    assert term['width']==cfg.width and term['weight']==e['climb'][0]['weight']
    np.testing.assert_array_equal(term['center'],np.zeros_like(calls[0]))
    assert not np.allclose(e['last_work'].positions,a.positions)
    np.testing.assert_array_equal(e['chart_reference'].positions,a.positions)


def test_joint_vc_failure_keeps_frozen_objective_and_last_work(monkeypatch):
    import pamssw.standalone.vc_reference as vc
    a=bulk('Cu','fcc',a=3.6,cubic=True)
    class Flat:
        requests=0
        def evaluate(self,a):self.requests+=1;return 0.,np.zeros_like(a.positions),np.zeros((3,3))
    s=Flat();original=vc.generalized_dimer;calls=[]
    def dimer(x,n,**kw):
        calls.append(x.copy())
        if len(calls)==2:
            kw['evaluate'](x);raise RuntimeError('second-stage cap')
        return original(x,n,**kw)
    monkeypatch.setattr(vc,'generalized_dimer',dimer)
    cfg=vc.VCSSWConfig(strain_length=4.,width=.2,rotation_bias=10.,max_gaussians=2)
    r=vc.run_vc_ssw(a,s,steps=1,config=cfg,rng=np.random.default_rng(3));e=r.records[1]
    assert e['status']=='evaluation_failed' and r.current is r.initial
    assert r.requests==sum(item['requests'] for item in r.records)
    assert len(e['frozen_gaussians'])==1
    np.testing.assert_array_equal(e['frozen_gaussians'][0]['center'],calls[0])
    assert not np.allclose(e['last_work'].positions,a.positions)
    np.testing.assert_array_equal(e['chart_reference'].positions,a.positions)
