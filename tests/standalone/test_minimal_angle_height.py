import numpy as np
import pytest
from pamssw.standalone.minimal_angle_height import MinimalAngleHeightPolicy
from pamssw.standalone.native_height_policy import FrozenHeightGaussian


def args(f):return dict(history=(),center=np.zeros(3),direction=np.array([1.,0.,0.]),width=.4,point=np.array([.4,0,0]),background_force=np.array(f,dtype=float))


def test_exact_minimal_height_angle_and_strictly_smaller_fails():
    a=args([-2.,.6,.8]);r=MinimalAngleHeightPolicy().prepare(**a)
    assert r.status=='prepared' and r.weight>0
    assert r.angle_degrees==pytest.approx(87.,abs=1e-12)
    assert abs(r.criterion_residual)<1e-14
    lower=FrozenHeightGaussian(a['center'],a['direction'],.4,r.weight*.999)
    f=a['background_force']+lower.evaluate(a['point'])[1]
    assert f[0]/np.linalg.norm(f)<np.cos(np.deg2rad(87))


def test_already_satisfied_does_not_add_zero_bias_or_choose_walker_action():
    r=MinimalAngleHeightPolicy().prepare(**args([1.,.1,0.]))
    assert r.status=='already_forward_satisfied' and r.weight==0. and r.terms==()
    with pytest.raises(ValueError,match='unattained'):MinimalAngleHeightPolicy().prepare(**args([-1.,0.,0.]))
    with pytest.raises(ValueError,match='unattained'):MinimalAngleHeightPolicy().prepare(**args([0.,0.,0.]))


def test_history_unchanged_and_frozen_combined_energy_force():
    a=args([-2.,.6,.8]);old=FrozenHeightGaussian(np.zeros(3),np.array([0.,1.,0.]),.3,2.);a['history']=(old,)
    # physical linear potential has the declared constant background force;
    # the old Gaussian force happens to vanish at y=0 here.
    r=MinimalAngleHeightPolicy().prepare(**a);assert r.terms[0] is old and old.weight==2.
    q=a['point'];f=a['background_force']+sum((t.evaluate(q)[1] for t in r.terms),np.zeros(3))
    def energy(x):return -a['background_force']@x+sum(t.evaluate(x)[0] for t in r.terms)
    for k in range(3):
        d=np.eye(3)[k]*1e-5
        assert f[k]==pytest.approx(-(energy(q+d)-energy(q-d))/2e-5,abs=2e-9)
