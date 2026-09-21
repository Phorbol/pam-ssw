"""RC stage/geometry contract checks; no scientific effectiveness claim."""
import numpy as np
import pytest
from ase.build import molecule
from pamssw.standalone.rc_reference import RCSSWConfig, run_rc_ssw, TorsionSurface
from pamssw.standalone.rc_geometry import RigidChainChart
from pamssw.standalone.surface import QuenchResult

BODIES=[(0,1,4,6,7),(0,1,2,10,11),(1,2,3,5,8,9,12,13)]
PARENTS=(-1,0,1);JOINTS=(None,(0,1),(1,2))

class Surface:
    requests=0
    def evaluate(self,a):
        self.requests+=1
        return float(np.sum(a.positions**2)/2),-a.positions.copy()

def test_torsion_metric_exact_pullback_no_root_dofs():
    a=molecule('trans-butane');s=Surface();c=RigidChainChart(a,BODIES,parents=PARENTS,joints=JOINTS)
    ev=TorsionSurface(c,s,2.3);x=np.array([.6,-.9]);e,g=ev.evaluate(x)
    for k in range(2):
        d=np.eye(2)[k]*1e-5
        fd=(ev.evaluate(x+d)[0]-ev.evaluate(x-d)[0])/2e-5
        assert g[k]==pytest.approx(fd,abs=2e-8)
    assert ev.atoms(x).positions.shape==(14,3)
    assert s.requests==5

def test_failed_rotation_preserves_current_and_cost(monkeypatch):
    import pamssw.standalone.rc_reference as rc
    a=molecule('trans-butane');s=Surface()
    def q(a,s,**kw):
        s.evaluate(a)
        return QuenchResult(a.copy(),0.,0.,True,0,1,'true')
    monkeypatch.setattr(rc,'quench',q)
    def fail(x,n,**kw):
        kw['evaluate'](x)
        raise RuntimeError('injected oracle error')
    monkeypatch.setattr(rc,'generalized_dimer',fail)
    r=run_rc_ssw(a,s,bodies=BODIES,parents=PARENTS,joints=JOINTS,steps=2,config=RCSSWConfig(torsion_length=2.,width=.2,rotation_bias=10.),rng=np.random.default_rng(3))
    assert r.requests==3 and len(r.minima)==1
    assert [e['status'] for e in r.records[1:]]==['evaluation_failed']*2
    np.testing.assert_allclose(r.current.atoms.positions,a.positions)
    assert [e['requests'] for e in r.records]==[1,1,1]

def test_no_internal_dofs_rejected_before_oracle():
    a=molecule('trans-butane');s=Surface()
    with pytest.raises(ValueError,match='torsion'):
        run_rc_ssw(a,s,bodies=[tuple(range(14))],parents=(-1,),joints=(None,),steps=1,config=RCSSWConfig(torsion_length=2.,width=.2,rotation_bias=10.),rng=np.random.default_rng(3))
    assert s.requests==0

def test_complete_biased_path_rejected_landing_stays_in_archive(monkeypatch):
    import pamssw.standalone.rc_reference as rc
    class Flat:
        requests=0
        def evaluate(self,a):
            self.requests+=1
            return 0.,np.zeros_like(a.positions)
    s=Flat();seen=[]
    from dataclasses import replace
    original_dimer=rc.generalized_dimer
    anchors=[]
    def rotated_mode(x,n,**kw):
        anchors.append(n.copy())
        mode=original_dimer(x,n,**kw)
        return replace(mode,direction=np.array([-n[1],n[0]]))
    monkeypatch.setattr(rc,'generalized_dimer',rotated_mode)
    def full(a,s,**kw):
        assert kw['optimizer']=='safe-lbfgs-total' and 'terms' not in kw
        seen.append(a.positions.copy());s.evaluate(a)
        return QuenchResult(a.copy(),float(len(seen)-1),0.,True,0,1,'true')
    monkeypatch.setattr(rc,'quench',full)
    a=molecule('trans-butane')
    r=run_rc_ssw(a,s,bodies=BODIES,parents=PARENTS,joints=JOINTS,steps=1,config=RCSSWConfig(torsion_length=2.,width=.2,rotation_bias=10.,temperature_K=0,max_gaussians=2),rng=np.random.default_rng(3))
    assert r.records[1]['status']=='valid_landing'
    assert not r.records[1]['accepted'] and len(r.minima)==2
    assert r.current is r.initial and r.requests==sum(x['requests'] for x in r.records)
    assert not np.allclose(seen[0],seen[1])
    assert r.records[1]['climb'][0]['relaxation'].converged
    assert len(anchors)==2
    np.testing.assert_array_equal(anchors[0],anchors[1])
