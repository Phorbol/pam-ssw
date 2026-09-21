from dataclasses import replace
import numpy as np
from ase import Atoms
from pamssw.standalone import paper_reference as paper
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.minimal_angle_height import MinimalAngleHeightPolicy,MinimalAngleHeightResult
from test_paper_reference import configuration,Harmonic


def run(policy,calculator=None):
    s=ASESurface(Harmonic() if calculator is None else calculator)
    r=paper.run_ssw(Atoms('H',positions=[[.1,.2,.3]]),s,steps=1,
        config=replace(configuration(),quench_optimizer='safe-lbfgs-total'),rng=np.random.default_rng(9),height_policy=policy)
    assert r.evaluation_requests==s.requests
    return r


def test_nonpositive_maps_to_existing_failure_and_preserves_input():
    class Already(MinimalAngleHeightPolicy):
        def prepare(self,history,**kw):return MinimalAngleHeightResult(tuple(history),0.,'already_forward_satisfied',0.,1.,1.,0.)
    r=run(Already());assert r.records[0].status=='nonpositive_height' and len(r.minima)==1
    c=r.records[0].climb[0];assert c['weight']==0 and c['height_preparation'].status=='already_forward_satisfied'
    assert 'height_input' in c


def test_domain_failure_keeps_attempted_objective_and_cost():
    class Fail(MinimalAngleHeightPolicy):
        def prepare(self,history,**kw):raise ValueError('zero resultant boundary')
    r=run(Fail());assert r.status=='evaluation_failed' and len(r.minima)==1
    assert len(r.records[0].climb)==1
    c=r.records[0].climb[0];assert c['status']=='evaluation_failed' and 'zero resultant' in c['error']
    assert 'background_force' in c['height_input'] and c['requests']>0


def test_real_analytic_policy_prepared_stage_is_frozen():
    class Anisotropic(Harmonic):
        def calculate(self,atoms=None,*args,**kwargs):
            super().calculate(atoms,*args,**kwargs)
            stiffness=np.array([1.,2.,3.])
            self.results=dict(energy=float(np.sum(atoms.positions**2*stiffness)/2),forces=-atoms.positions*stiffness)
    r=run(MinimalAngleHeightPolicy(),Anisotropic())
    events=[c for s in r.records for c in s.climb if 'height_preparation' in c]
    assert events
    for c in events:
        p=c['height_preparation']
        if p.status=='prepared':
            assert p.weight>0 and abs(p.angle_degrees-87)<1e-9
            assert p.terms[-1].weight==c['weight']
