"""Original initializer fixtures and conservative stage-frozen height contract."""
import json
from pathlib import Path
import numpy as np
import pytest
from pamssw.standalone.native_height_policy import ConservativeNativeHeightPolicy,FrozenHeightGaussian
from pamssw.standalone.gaussian import adjust_native_weight


def policy(**kw):
    args=dict(initial_weight=.6,negative_weight=.07,level=0,max_weight=10.,growth_step=1.2,growth_scale=1.5);args.update(kw)
    return ConservativeNativeHeightPolicy(**args)


def test_all_27_original_instruction_initialization_fixtures():
    path=Path(__file__).resolve().parents[2]/'research/ga_ssw/evidence/native-initial-gaussw-emulated/result.json'
    data=json.loads(path.read_text());assert len(data['cases'])==27
    for c in data['cases']:
        p=policy(level=c['level']);r=p.initialize_history(c['before'][:c['ng']-1],curvature=c['curv_real'])
        np.testing.assert_array_equal(r.weights,c['after'][:c['ng']])
        assert [p.growth_scale,p.max_weight,p.growth_step]==c['scale_maxw_step']


def test_single_stage_matches_recovered_87_degree_helper():
    n=np.array([1.,0.,0.]);x=n*.7;f=np.array([-2.,.6,.2]);p=policy()
    r=p.prepare([],center=np.zeros(3),direction=n,width=.7,point=x,background_force=f,curvature=1.,curvature_scope="physical_pes",max_updates=100)
    expected=adjust_native_weight(fa0=f.reshape(1,3),fa2=np.zeros((1,3)),n=n.reshape(1,3),d1=np.exp(-.5),d2=1/.7,e2=0,w=.6,maxw=10.,step=1.2,scalefact0=1.5)
    assert r.final_weight==pytest.approx(expected.weight,abs=1e-13)
    assert r.angle_degrees==pytest.approx(expected.angle_degrees,abs=1e-12)
    np.testing.assert_allclose(r.total_force_at_preparation,expected.force.ravel(),atol=1e-13)
    assert r.stop_reason==expected.stop_reason


def test_history_override_force_correction_and_conservative_frozen_fd():
    old=FrozenHeightGaussian(np.zeros(2),np.array([0.,1.]),.7,9.)
    point=np.array([1.,.5]);background=-point+old.evaluate(point)[1]
    r=policy(level=1).prepare([old],center=np.array([.3,.5]),direction=np.array([1.,0.]),width=.7,point=point,background_force=background,curvature=-1.,curvature_scope="physical_pes",max_updates=100)
    assert old.weight==9. and r.terms[0].weight==5.6
    assert len(r.changed_history)==1 and r.changed_history[0].before==9.
    force=-point+sum((t.evaluate(point)[1] for t in r.terms),np.zeros(2))
    np.testing.assert_allclose(force,r.total_force_at_preparation,atol=1e-13)
    def energy(q):return .5*q@q+sum(t.evaluate(q)[0] for t in r.terms)
    for k in range(2):
        d=np.eye(2)[k]*1e-5
        assert force[k]==pytest.approx(-(energy(point+d)-energy(point-d))/2e-5,abs=2e-9)
    assert not r.terms[0].center.flags.writeable


def test_maxw_is_postupdate_stop_and_budget_is_failure():
    args=dict(history=[],center=np.zeros(3),direction=np.array([1.,0.,0.]),width=1.,point=np.array([1.,0.,0.]),background_force=np.array([-100.,1.,0.]),curvature=1.,curvature_scope="physical_pes")
    r=policy(max_weight=.1).prepare(**args,max_updates=100)
    assert r.stop_reason=='maxw_exceeded' and r.final_weight>.1 and len(r.update_trace)==1
    with pytest.raises(RuntimeError,match='budget'):policy(max_weight=1000.).prepare(**args,max_updates=1)
    args['point']=np.zeros(3)
    with pytest.raises(ValueError,match='forward'):policy().prepare(**args,max_updates=100)
