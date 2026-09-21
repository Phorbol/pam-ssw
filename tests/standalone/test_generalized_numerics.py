"""Flat numerical-contract checks, not VC scientific validation."""
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from ase.cluster.icosahedron import Icosahedron
from pamssw.standalone.generalized_numerics import safe_lbfgs, generalized_dimer
from pamssw.standalone.dimer import paper_dimer_direction
from pamssw.standalone.surface import ASESurface
from pamssw.relax import Relaxer
from pamssw.state import State

def test_flat_dimension_and_last_accepted_on_failed_trial():
    seen=[]
    def evaluate(q):
        seen.append(q.copy())
        if len(seen)==3:raise RuntimeError('backend failed')
        return .5*np.dot(q,q),q
    q=np.ones(7)
    result=safe_lbfgs(q,evaluate,gradient_norm=np.linalg.norm,step_norm=np.linalg.norm,
                      gtol=1e-8,max_step=.2,maxiter=20)
    assert result.status=='evaluation_failed' and result.requests==3 and result.steps==1
    np.testing.assert_array_equal(result.q,seen[1])
    np.testing.assert_array_equal(result.gradient,result.q)
    np.testing.assert_array_equal(q,np.ones(7))

def test_request_limit_and_initial_failure():
    result=safe_lbfgs(np.ones(5),lambda q:(np.dot(q,q),2*q),gradient_norm=np.linalg.norm,
                      step_norm=np.linalg.norm,gtol=1e-8,max_step=.2,maxiter=20,max_requests=1)
    assert result.status=='request_limit' and result.requests==1 and result.steps==0
    def failed(q):raise ValueError('invalid geometry')
    result=safe_lbfgs(np.ones(5),failed,gradient_norm=np.linalg.norm,step_norm=np.linalg.norm,
                      gtol=1e-8,max_step=.2,maxiter=20)
    assert result.requests==1 and result.energy is None and result.gradient is None

def test_line_search_failure_returns_initial_and_counts_every_trial():
    calls=[]
    def evaluate(q):
        calls.append(q.copy());return (0. if len(calls)==1 else 1.),np.ones(5)
    r=safe_lbfgs(np.zeros(5),evaluate,gradient_norm=np.linalg.norm,step_norm=np.linalg.norm,
                 gtol=1e-8,max_step=.2,maxiter=5)
    assert r.status=='line_search_failed' and r.requests==21 and r.rejected_trials==20
    np.testing.assert_array_equal(r.q,np.zeros(5))

def test_safe_total_cartesian_emt_matches_existing_optimizer():
    atoms=Icosahedron('Cu',2);atoms.positions+=np.random.default_rng(8).normal(scale=.03,size=(13,3))
    a=ASESurface(EMT());b=ASESurface(EMT())
    def evaluate(q,surface):
        c=atoms.copy();c.positions=q.reshape(-1,3);e,f=surface.evaluate(c);return e,-f.ravel()
    old_trace=[]
    def old_eval(q,state):return evaluate(q,a)
    old=Relaxer(old_eval,optimizer='safe-lbfgs-total').relax(State(atoms.numbers,atoms.positions),fmax=.03,maxiter=100,
                                                          trajectory_callback=lambda s:old_trace.append(s.positions.copy()))
    measure=lambda x:float(np.linalg.norm(x.reshape(-1,3),axis=1).max())
    new=safe_lbfgs(atoms.positions.ravel(),lambda q:evaluate(q,b),gradient_norm=measure,step_norm=measure,
                   gtol=.03,max_step=.2,maxiter=100)
    assert new.converged and a.requests==b.requests==new.requests
    assert len(old_trace)==len(new.trace)
    for x,row in zip(old_trace,new.trace):np.testing.assert_allclose(x.ravel(),row['q'],rtol=0,atol=2e-12)

def test_generalized_plane_dimer_matches_cartesian_emt():
    atoms=Icosahedron('Cu',2);atoms.positions+=np.random.default_rng(7).normal(scale=.03,size=(13,3))
    anchor=np.random.default_rng(13).normal(size=(13,3));a=ASESurface(EMT());b=ASESurface(EMT())
    params=dict(rotation_bias=.2,fd_step=.001,max_hvp=15,tol=.01)
    old=paper_dimer_direction(atoms,anchor,evaluate=a.evaluate,**params)
    def evaluate(q):
        c=atoms.copy();c.positions=q.reshape(-1,3);e,f=b.evaluate(c);return e,-f.ravel()
    new=generalized_dimer(atoms.positions.ravel(),anchor.ravel(),evaluate=evaluate,**params)
    np.testing.assert_allclose(new.direction,old.direction.ravel(),rtol=0,atol=1e-13)
    np.testing.assert_allclose([new.curvature,new.residual_norm,new.projected_symmetry_error],
                               [old.curvature,old.residual_norm,old.projected_symmetry_error],rtol=0,atol=1e-13)
    assert new.force_calls==old.force_calls==a.requests==b.requests

def test_coordinate_dependent_stopping_never_uses_rejected_trial_state():
    seen=[];measured=[]
    def evaluate(q):
        seen.append(q.copy())
        # Reject full first direction; accept the half step. Then every trial
        # is rejected, so final norm must still see that accepted half step.
        return (0. if len(seen)==1 else -.001 if len(seen)==3 else 1.),np.ones(5)
    def measure(q,g):
        measured.append((q.copy(),g.copy()));return float(np.linalg.norm(g)+np.linalg.norm(q))
    result=safe_lbfgs(np.zeros(5),evaluate,gradient_norm=lambda g:0.,step_norm=np.linalg.norm,
                      convergence_norm=measure,gtol=.01,max_step=.2,maxiter=4)
    assert result.status=='line_search_failed' and result.steps==1
    assert result.requests==23 and result.rejected_trials==21
    accepted=seen[2]
    assert all(np.array_equal(q,np.zeros(5)) or np.array_equal(q,accepted) for q,g in measured)
    np.testing.assert_array_equal(measured[-1][0],accepted)
    np.testing.assert_array_equal(result.q,accepted)
    assert result.trace[-1]['gradient_norm']==np.linalg.norm(np.ones(5))+np.linalg.norm(accepted)
