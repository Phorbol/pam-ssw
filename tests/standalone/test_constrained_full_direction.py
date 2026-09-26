"""Public constrained direction contract, separate from search efficacy."""
import copy
import pickle
import numpy as np
import pytest
from ase.cluster import Icosahedron
from ase.constraints import FixAtoms
from ase.calculators.emt import EMT
from pamssw.standalone.constrained_reference import ConstrainedSSWConfig, run_constrained_ssw
from pamssw.standalone.recovered_direction import RecoveredDirectionSettings
from pamssw.standalone.surface import ASESurface


def settings():
    return RecoveredDirectionSettings(50,.5,.5,5,15,.2,.02,'euclidean',40)


def config():
    return ConstrainedSSWConfig(width=.6,rotation_bias=1.,max_gaussians=2,
        fmax=.03,gradient_tol=.1,relax_steps=1000,fd_step=.001,
        lbfgs_memory=500,rotation_exit_policy='force_or_budget')


def atoms():
    a=Icosahedron('Cu',2)
    a.set_constraint(FixAtoms(indices=[0]))
    return a


def test_initial_direction_checkpoint_roundtrip_and_fixed_identity():
    a=atoms()
    r=run_constrained_ssw(a,ASESurface(EMT()),steps=0,config=config(),
        rng=np.random.default_rng(41),recovered_direction=settings())
    assert r.status=='completed'
    cp=pickle.loads(pickle.dumps(r.checkpoint))
    assert cp.schema_version==2
    assert not cp.recovered_direction_state.active_mask[0]
    rng=np.random.default_rng(999)
    r2=run_constrained_ssw(a,ASESurface(EMT()),steps=0,config=config(),
        rng=rng,recovered_direction=settings(),checkpoint=cp)
    assert r2.requests==r.requests
    assert rng.bit_generator.state==cp.rng_state
    np.testing.assert_array_equal(r2.current.atoms.positions[0],a.positions[0])


def test_periodic_full_direction_rejected_before_pes_or_rng():
    a=atoms();a.cell=np.eye(3)*20;a.pbc=True
    s=ASESurface(EMT());rng=np.random.default_rng(41);state=copy.deepcopy(rng.bit_generator.state)
    with pytest.raises((ValueError,NotImplementedError),match='nonperiodic'):
        run_constrained_ssw(a,s,steps=0,config=config(),rng=rng,recovered_direction=settings())
    assert s.requests==0 and rng.bit_generator.state==state


@pytest.mark.parametrize('corrupt', ['missing','mask','settings'])
def test_direction_checkpoint_rejects_mismatch_before_pes_or_rng(corrupt):
    a=atoms();r=run_constrained_ssw(a,ASESurface(EMT()),steps=0,config=config(),
        rng=np.random.default_rng(41),recovered_direction=settings())
    cp=copy.deepcopy(r.checkpoint)
    if corrupt=='missing':cp.recovered_direction_state=None
    elif corrupt=='mask':
        from dataclasses import replace
        cp.recovered_direction_state=replace(cp.recovered_direction_state,active_mask=np.ones(len(a),bool))
    else:cp.recovered_direction=None
    s=ASESurface(EMT());rng=np.random.default_rng(99);state=copy.deepcopy(rng.bit_generator.state)
    with pytest.raises(ValueError,match='direction|active|schema'):
        run_constrained_ssw(a,s,steps=0,config=config(),rng=rng,
            recovered_direction=settings(),checkpoint=cp)
    assert s.requests==0 and rng.bit_generator.state==state


def test_qualified_landing_is_retained_when_next_axis_selection_fails(monkeypatch):
    from pamssw.standalone.constrained_direction import ConstrainedDirectionLifecycle
    def zero(self,current,reduced,work,*,first,rng):
        return np.zeros(reduced.dimension),True,{'selection_mode':'test_zero_release'}
    def fail(self,landing,rng):
        raise ValueError('active score band has no second pair candidate')
    monkeypatch.setattr(ConstrainedDirectionLifecycle,'propose',zero)
    monkeypatch.setattr(ConstrainedDirectionLifecycle,'observe',fail)
    r=run_constrained_ssw(atoms(),ASESurface(EMT()),steps=1,config=config(),
        rng=np.random.default_rng(41),recovered_direction=settings())
    assert r.status=='direction_selection_failed'
    assert r.records[-1]['landing'].converged
    assert len(r.minima)==2
    assert r.requests==sum(event['requests'] for event in r.records)
    with pytest.raises(ValueError,match='terminal'):
        run_constrained_ssw(atoms(),ASESurface(EMT()),steps=1,config=config(),
            rng=np.random.default_rng(99),recovered_direction=settings(),checkpoint=r.checkpoint)


def test_one_shot_direction_exclusions_match_rotation_and_checkpoint():
    r=run_constrained_ssw(atoms(),ASESurface(EMT()),steps=1,config=config(),
        rng=np.random.default_rng(41),recovered_direction=settings(),
        direction_fixed_indices=iter([1]))
    assert r.checkpoint.direction_fixed_indices==(1,)
    assert not r.checkpoint.recovered_direction_state.active_mask[1]
    event=r.records[1]
    np.testing.assert_array_equal(event['rotation_coordinate_indices'],np.arange(3,36))
    for stage in event['climb']:
        if 'mode' in stage:
            np.testing.assert_array_equal(stage['mode'].direction[:3],0.)


def test_in_memory_direction_checkpoint_does_not_copy_every_prefix(monkeypatch):
    import pamssw.standalone.constrained_reference as module
    from pamssw.standalone.constrained_direction import ConstrainedDirectionLifecycle
    original=module.ConstrainedCheckpoint
    snapshots=[]
    def capture(*args,**kwargs):
        cp=original(*args,**kwargs);snapshots.append(cp.next_index);return cp
    monkeypatch.setattr(module,'ConstrainedCheckpoint',capture)
    monkeypatch.setattr(ConstrainedDirectionLifecycle,'propose',
        lambda self,current,reduced,work,first,rng: (np.zeros(reduced.dimension),True,{}))
    monkeypatch.setattr(ConstrainedDirectionLifecycle,'observe',lambda self,landing,rng:None)
    r=run_constrained_ssw(atoms(),ASESurface(EMT()),steps=3,config=config(),
        rng=np.random.default_rng(41),recovered_direction=settings())
    assert r.status=='completed' and r.checkpoint.next_index==3
    assert len(snapshots)<=2
    assert set(snapshots)=={3}
