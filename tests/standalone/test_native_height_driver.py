"""Full conservative height-policy wiring, not efficiency validation."""
from dataclasses import replace
import numpy as np
import pytest
from ase import Atoms
from pamssw.standalone import paper_reference as paper
from pamssw.standalone.surface import ASESurface
from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy
from test_paper_reference import configuration, Harmonic


def policy():
    return ConservativeNativeHeightPolicy(2.,.2,1,10.,1.05,1.25)


def test_native_height_full_driver_freezes_terms_and_removes_rotation_curvature():
    s=ASESurface(Harmonic())
    config=replace(configuration(),quench_optimizer='safe-lbfgs-total')
    r=paper.run_ssw(Atoms('H',positions=[[.1,.2,.3]]),s,steps=1,
        config=config,rng=np.random.default_rng(9),height_policy=policy())
    assert r.evaluation_requests==s.requests
    events=[c for step in r.records for c in step.climb if 'height_preparation' in c]
    assert events
    first=events[0]['height_preparation']
    assert first.curvature_input==pytest.approx(1.,abs=1e-6)
    assert first.curvature_scope=='physical PES; rotation-only bias excluded'
    assert first.terms[0].weight>=5.6
    assert first.total_force_at_preparation.flags.writeable is False


def test_native_height_rejects_eckart_before_oracle():
    s=ASESurface(Harmonic())
    with pytest.raises(NotImplementedError,match='eckart'):
        paper.run_ssw(Atoms('H',positions=[[0,0,0]]),s,steps=1,
            config=replace(configuration(),cluster_frame='eckart'),
            rng=np.random.default_rng(9),height_policy=policy())
    assert s.requests==0


def test_prepared_height_survives_biased_quench_exception(monkeypatch):
    s=ASESurface(Harmonic());original=paper.quench
    def fail_biased(atoms,surface,**kwargs):
        if kwargs.get('terms'):
            surface.evaluate(atoms)
            raise RuntimeError('injected biased-quench oracle failure')
        return original(atoms,surface,**kwargs)
    monkeypatch.setattr(paper,'quench',fail_biased)
    r=paper.run_ssw(Atoms('H',positions=[[.1,.2,.3]]),s,steps=1,
        config=replace(configuration(),quench_optimizer='safe-lbfgs-total'),
        rng=np.random.default_rng(9),height_policy=policy())
    assert r.status=='evaluation_failed' and len(r.minima)==1
    assert len(r.records[0].climb)==1
    event=r.records[0].climb[0]
    assert event['status']=='evaluation_failed'
    assert 'height_preparation' in event and event['weight']==event['height_preparation'].final_weight
    assert event['requests']>event['rotation_force_requests']
    assert 'injected' in event['error'] and r.evaluation_requests==s.requests
