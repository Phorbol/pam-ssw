"""Finite-map tests; mathematical checks are not RC search efficacy evidence."""
import numpy as np
import pytest
from ase.build import molecule
from pamssw.standalone.rc_forest import RigidForestChart,ForestSurface
from pamssw.standalone.rc_forest_reference import RCForestSSWConfig,run_rc_forest_ssw
from pamssw.standalone.rc_geometry import RigidChainChart
from pamssw.standalone.surface import QuenchResult


def waters():
    a=molecule('H2O');b=a.copy();b.translate([3.,.2,.1]);a+=b
    return a,[dict(bodies=[(0,1,2)],parents=(-1,),joints=(None,)),dict(bodies=[(3,4,5)],parents=(-1,),joints=(None,))]


def test_exact_relative_finite_map_and_scaled_gradient():
    a,trees=waters();c=RigidForestChart(a,trees,anchor=0)
    assert c.dimension==6
    q=np.array([.3,-.2,.1,.7,-.4,1.1]);out,J=c.evaluate(q)
    np.testing.assert_allclose(out.positions[:3],a.positions[:3],atol=1e-14)
    assert not np.allclose(out.positions[3:],a.positions[3:])
    np.testing.assert_allclose(out.get_all_distances()[3:,3:],a.get_all_distances()[3:,3:],atol=2e-15)
    for k in range(6):
        d=np.eye(6)[k]*1e-6
        fd=(c.evaluate(q+d)[0].positions-c.evaluate(q-d)[0].positions)/2e-6
        np.testing.assert_allclose(fd,J[:,:,k],atol=1e-9)
    assert np.linalg.matrix_rank(J.reshape(-1,6))==6
    class Harmonic:
        def evaluate(self,a):return .5*np.sum(a.positions**2),-a.positions
    ev=ForestSurface(c,Harmonic(),rotation_length=2.3,torsion_length=1.7);_,g=ev.evaluate(q)
    for k in range(6):
        d=np.eye(6)[k]*1e-5
        assert g[k]==pytest.approx((ev.evaluate(q+d)[0]-ev.evaluate(q-d)[0])/2e-5,abs=2e-9)


def test_single_chain_equivalence_and_tree_mixture():
    a=molecule('trans-butane');tree=dict(bodies=[(0,1,4,6,7),(0,1,2,10,11),(1,2,3,5,8,9,12,13)],parents=(-1,0,1),joints=(None,(0,1),(1,2)))
    single=RigidChainChart(a,**tree);c=RigidForestChart(a,[tree]);q=np.array([.8,-1.1])
    aa,J=single.evaluate(np.r_[np.zeros(6),q]);bb,K=c.evaluate(q)
    np.testing.assert_allclose(aa.positions,bb.positions);np.testing.assert_allclose(J[:,:,6:],K)
    b=molecule('H2O');b.translate([6,0,0]);a+=b
    forest=RigidForestChart(a,[tree,dict(bodies=[(14,15,16)],parents=(-1,),joints=(None,))])
    q=np.arange(8)*.12;aa,J=forest.evaluate(q)
    for k in range(8):
        d=np.eye(8)[k]*1e-6
        np.testing.assert_allclose((forest.evaluate(q+d)[0].positions-forest.evaluate(q-d)[0].positions)/2e-6,J[:,:,k],atol=2e-9)


def test_invalid_roots_disjointness_and_zero_dofs():
    a,t=waters()
    with pytest.raises(ValueError,match='disjoint'):RigidForestChart(a,[t[0],t[0]])
    linear=a.copy();linear.positions[1]=linear.positions[0]+[1,0,0];linear.positions[2]=linear.positions[0]+[2,0,0]
    with pytest.raises(ValueError,match='noncollinear'):RigidForestChart(linear,t)
    c=RigidForestChart(a[:3],[t[0]])
    with pytest.raises(ValueError,match='no internal'):ForestSurface(c,None,rotation_length=2,torsion_length=2)


def test_forest_full_lifecycle_rejected_landing_and_cost(monkeypatch):
    import pamssw.standalone.rc_reference as rc
    class Flat:
        requests=0
        def evaluate(self,a):self.requests+=1;return 0.,np.zeros_like(a.positions)
    s=Flat();seen=[]
    def full(a,s,**kw):
        assert kw['optimizer']=='safe-lbfgs-total' and 'terms' not in kw
        seen.append(a.copy());s.evaluate(a)
        return QuenchResult(a.copy(),float(len(seen)-1),0.,True,0,1,'true')
    monkeypatch.setattr(rc,'quench',full)
    a,t=waters()
    result=run_rc_forest_ssw(a,s,trees=t,anchor=0,steps=1,config=RCForestSSWConfig(rotation_length=2.,torsion_length=2.,width=.2,rotation_bias=10.,temperature_K=0.,max_gaussians=1),rng=np.random.default_rng(3))
    assert result.records[1]['status']=='valid_landing'
    assert not result.records[1]['accepted'] and len(result.minima)==2
    assert result.current is result.initial
    assert result.requests==sum(r['requests'] for r in result.records)
    np.testing.assert_allclose(seen[0].positions[:3],seen[1].positions[:3],atol=1e-14)
    assert not np.allclose(seen[0].positions[3:],seen[1].positions[3:])
