import numpy as np
import pytest
from ase import Atoms
import pamssw.standalone.surface_ga as ga


def test_failed_cubic_evaluation_includes_previous_starts_and_failed_call(monkeypatch):
    calls=[]
    def evaluate(*args):
        calls.append(1)
        if len(calls)==3:raise ValueError('injected synthetic evaluation failure')
        return 0.,np.zeros_like(args[0])
    monkeypatch.setattr(ga,'cubic_auxiliary_energy_gradient',evaluate)
    with pytest.raises(ValueError) as caught:
        ga.cubic_cluster([29],np.random.default_rng(1),atomic_radii={29:1.},space=(1,1,1),max_insertion_attempts=10,auxiliary_evaluations=1)
    assert caught.value.auxiliary_evaluations==3
    assert len(caught.value.auxiliary_runs)==3
    assert caught.value.auxiliary_runs[-1]['status']=='auxiliary_evaluation_failed'


def test_rebuild_adds_prior_family_cost_to_failed_family_cost(monkeypatch):
    a=Atoms('Cu4',positions=[[0,0,0],[1,0,0],[2,0,0],[3,0,1]],cell=[10,10,10],pbc=[True,True,False])
    cluster=Atoms('Cu2',positions=[[0,0,0],[1,0,0]])
    monkeypatch.setattr(ga,'triple_tangency_cluster',lambda *a,**kw:(cluster,(0,1),{}))
    count=[]
    def cubic(*args,**kwargs):
        count.append(1)
        if len(count)==2:
            e=ValueError('second family fails after paid calls');e.auxiliary_evaluations=3;raise e
        return cluster,(0,1),dict(auxiliary_evaluations=10)
    monkeypatch.setattr(ga,'cubic_cluster',cubic)
    with pytest.raises(ValueError) as caught:
        ga.rebuild_surface(a,(0,1),(2,3),np.random.default_rng(1),site_fractional=(.5,.5),atomic_radii={29:1.},max_face_attempts=1,max_insertion_attempts=1)
    assert caught.value.auxiliary_evaluations==13
