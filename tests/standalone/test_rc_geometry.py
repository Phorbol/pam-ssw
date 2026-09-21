"""Actual ASE butane geometry: coordinate/derivative checks, not PES efficacy."""
import numpy as np
import pytest
from ase.build import molecule
from pamssw.standalone.rc_geometry import RigidChainChart

def chart():
    # Three articulated bodies; two actual C-C axes with shared endpoints.
    a=molecule('trans-butane')
    bodies=[(0,1,4,6,7),(0,1,2,10,11),(1,2,3,5,8,9,12,13)]
    return a,RigidChainChart(a,bodies,parents=(-1,0,1),joints=(None,(0,1),(1,2)))

def test_finite_chain_rigidity_shared_joints_and_exact_jacobian():
    a,c=chart();q=np.array([.2,-.1,.3,.7,-.4,.2,1.1,-.8]);b,J=c.evaluate(q)
    np.testing.assert_allclose(c.evaluate(np.zeros(8))[0].positions,a.positions,atol=1e-13)
    for group in c.bodies:
        ids=list(group)
        d=lambda x:np.linalg.norm(x[ids,None,:]-x[None,ids,:],axis=-1)
        np.testing.assert_allclose(d(b.positions),d(a.positions),atol=2e-13)
    for k in range(8):
        dq=np.eye(8)[k]*1e-6
        fd=(c.evaluate(q+dq)[0].positions-c.evaluate(q-dq)[0].positions)/2e-6
        np.testing.assert_allclose(J[:,:,k],fd,atol=2e-9)
    f=np.random.default_rng(7).normal(size=(14,3))
    np.testing.assert_allclose(c.pullback(q,f),np.einsum('ijk,ij->k',J,f),atol=1e-13)
    # Independent scalar work derivative at finite rotations, not just J reuse.
    for k in range(8):
        dq=np.eye(8)[k]*1e-6
        ep=-np.sum(f*c.evaluate(q+dq)[0].positions)
        em=-np.sum(f*c.evaluate(q-dq)[0].positions)
        np.testing.assert_allclose(c.pullback(q,f)[k],-(ep-em)/2e-6,atol=5e-9)
    np.testing.assert_allclose(a.positions,molecule('trans-butane').positions)

def test_zero_pi_and_full_rotation_have_finite_exact_derivatives():
    _,c=chart()
    for angle in [0.,np.pi,2*np.pi]:
        q=np.zeros(8);q[3]=angle;q[6]=angle
        _,j=c.evaluate(q);assert np.isfinite(j).all()
        d=np.zeros(8);d[4]=1e-6
        fd=(c.evaluate(q+d)[0].positions-c.evaluate(q-d)[0].positions)/2e-6
        np.testing.assert_allclose(j[:,:,4],fd,atol=3e-9)

def test_invalid_shared_joint_and_disconnected_membership_rejected():
    a,_=chart()
    with pytest.raises(ValueError):RigidChainChart(a,[(0,1),(1,2)],parents=(-1,0),joints=(None,(0,1)))
    a.pbc=True
    with pytest.raises(ValueError,match='nonperiodic'):RigidChainChart(a,[tuple(range(14))],parents=(-1,),joints=(None,))


def test_axis_only_subtree_rejected_as_identically_zero_torsion():
    from ase import Atoms
    a=Atoms('Cu3',positions=[[0,0,0],[1,0,0],[0,1,0]])
    with pytest.raises(ValueError,match='axis-only'):
        RigidChainChart(a,[(0,1,2),(1,2)],parents=(-1,0),joints=(None,(1,2)))
