"""Constrained-gradient/certificate checks, separate from surface-search efficacy."""
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms,FixBondLengths
from pamssw.standalone.constrained_reference import (ReducedCartesianChart,ConstrainedSSWConfig,
    constrained_quench,run_constrained_ssw,_active_rotation_callback)
from pamssw.standalone.paper_reference import LSSettings
from pamssw.standalone.ls_native_reference import NativeLSSettings


class Harmonic:
    requests=0
    def evaluate(self,a):
        assert not a.constraints
        self.requests+=1
        return .5*float(np.sum(a.positions**2)),-a.positions.copy()


def atoms():return Atoms('Cu2',positions=[[3.,0.,0.],[.4,.3,.2]],cell=[8,8,10],pbc=[True,True,False],constraint=FixAtoms(indices=[0]))


def test_active_exact_gradient_and_fixed_cell_geometry():
    a=atoms();c=ReducedCartesianChart(a);s=Harmonic();q=np.array([.2,-.1,.3]);e,g=c.evaluate(q,s)
    assert c.dimension==3
    b=c.atoms(q)
    np.testing.assert_array_equal(b.positions[0],a.positions[0]);np.testing.assert_array_equal(b.cell.array,a.cell.array)
    assert not b.constraints and np.array_equal(b.pbc,a.pbc)
    for k in range(3):
        d=np.eye(3)[k]*1e-5
        assert g[k]==pytest.approx((c.evaluate(q+d,s)[0]-c.evaluate(q-d,s)[0])/2e-5,abs=1e-10)
    # Uniform active displacement is retained, not projected away as translation.
    np.testing.assert_allclose(b.positions[1],a.positions[1]+q)
    assert a.constraints


def test_active_certificate_does_not_hide_fixed_force():
    a=atoms();s=Harmonic();q=constrained_quench(a,s,fmax=1e-5,max_step=.2,maxiter=100)
    assert q.converged and q.active_fmax<=1e-5 and q.full_raw_fmax==3.
    assert q.certificate['scope']=='fixed_atom_manifold'
    assert q.certificate['fixed_and_cell_exact'] and q.atoms.constraints
    assert q.requests==s.requests
    np.testing.assert_array_equal(q.atoms.positions[0],a.positions[0])


def test_constraint_domain_rejections_before_oracle():
    a=atoms()
    with pytest.raises(ValueError,match='agree'):ReducedCartesianChart(a,fixed_indices=[1])
    a.set_constraint(FixBondLengths([[0,1]]))
    with pytest.raises(ValueError,match='only FixAtoms'):ReducedCartesianChart(a,fixed_indices=[0])
    a.set_constraint()
    with pytest.raises(ValueError,match='one fixed'):ReducedCartesianChart(a)


def test_constrained_memory_reaches_initial_biased_and_final(monkeypatch):
    import pamssw.standalone.constrained_reference as constrained
    import pamssw.standalone.rc_reference as shared
    original = constrained.safe_lbfgs
    original_shared = shared.safe_lbfgs
    seen = []
    def spy(*args, **kwargs):
        seen.append(kwargs.get('lbfgs_memory'))
        return original(*args, **kwargs)
    def shared_spy(*args, **kwargs):
        seen.append(kwargs.get('lbfgs_memory'))
        return original_shared(*args, **kwargs)
    monkeypatch.setattr(constrained, 'safe_lbfgs', spy)
    monkeypatch.setattr(shared, 'safe_lbfgs', shared_spy)
    r=run_constrained_ssw(atoms(), Harmonic(), steps=1,
        config=ConstrainedSSWConfig(width=.2,rotation_bias=10.,max_gaussians=1,lbfgs_memory=400),
        rng=np.random.default_rng(3))
    assert r.status=='completed' and len(r.minima)==2
    assert len(seen)==3 and seen==[400,400,400]


def test_constrained_memory_rejects_invalid_before_oracle():
    with pytest.raises(ValueError):
        ConstrainedSSWConfig(width=.2, rotation_bias=10., lbfgs_memory=0)


def test_constrained_default_path_does_not_snapshot(monkeypatch):
    import pamssw.standalone.constrained_reference as constrained
    monkeypatch.setattr(constrained, '_constrained_copy', lambda value: (_ for _ in ()).throw(AssertionError('unexpected snapshot')))
    result=run_constrained_ssw(atoms(), Harmonic(), steps=1,
        config=ConstrainedSSWConfig(width=.2,rotation_bias=10.,max_gaussians=1),
        rng=np.random.default_rng(3))
    assert result.status=='completed' and result.checkpoint is None


def test_constrained_checkpoint_resume_preserves_boundary_and_steps(tmp_path):
    from pamssw.standalone.constrained_reference import load_constrained_checkpoint
    cfg=ConstrainedSSWConfig(width=.2,rotation_bias=10.,max_gaussians=1)
    continuous=run_constrained_ssw(atoms(), Harmonic(), steps=2, config=cfg,
                                   rng=np.random.default_rng(3))
    path=tmp_path/'constrained.pkl'
    first_surface=Harmonic()
    first=run_constrained_ssw(atoms(), first_surface, steps=1, config=cfg,
                              rng=np.random.default_rng(3), checkpoint_path=path)
    cp=load_constrained_checkpoint(path)
    before=first_surface.requests
    zero=run_constrained_ssw(atoms(), first_surface, steps=0, config=cfg,
                             rng=np.random.default_rng(99), checkpoint=cp)
    assert first_surface.requests==before and zero.requests==cp.evaluation_requests
    resumed=run_constrained_ssw(atoms(), first_surface, steps=1, config=cfg,
                                rng=np.random.default_rng(99), checkpoint=cp,
                                checkpoint_path=path)
    assert resumed.requests==continuous.requests
    assert resumed.checkpoint.next_index==2
    assert len(resumed.records)==len(continuous.records)==3
    np.testing.assert_array_equal(resumed.current.atoms.positions, continuous.current.atoms.positions)
    assert resumed.records[1]['index']==0 and resumed.records[2]['index']==1


def test_constrained_terminal_checkpoint_is_loadable_diagnostic_not_resumable(tmp_path):
    class Failing:
        def __init__(self): self.requests=0
        def evaluate(self, atoms):
            self.requests += 1
            raise RuntimeError('paid physical failure')
    from pamssw.standalone.constrained_reference import load_constrained_checkpoint
    path=tmp_path/'terminal.pkl'; surface=Failing()
    result=run_constrained_ssw(atoms(), surface, steps=1,
        config=ConstrainedSSWConfig(width=.2,rotation_bias=10.,max_gaussians=1),
        rng=np.random.default_rng(3), checkpoint_path=path)
    cp=load_constrained_checkpoint(path)
    assert result.status=='initial_quench_failed' and cp.status=='initial_quench_failed'
    class Zero:
        requests=0
        def evaluate(self, atoms): raise AssertionError('resume called PES')
    with pytest.raises(ValueError, match='terminal'):
        run_constrained_ssw(atoms(), Zero(), steps=1,
            config=ConstrainedSSWConfig(width=.2,rotation_bias=10.,max_gaussians=1),
            rng=np.random.default_rng(3), checkpoint=cp)


def test_constrained_checkpoint_validates_all_fixed_positions_before_pes(tmp_path):
    from pamssw.standalone.constrained_reference import load_constrained_checkpoint
    base=Atoms('Cu4', positions=[[3.,0.,0.],[0.,3.,0.],[0.,0.,3.],[.4,.3,.2]],
               cell=[8,8,8], pbc=True, constraint=FixAtoms(indices=[0,1,2]))
    cfg=ConstrainedSSWConfig(width=.2,rotation_bias=10.,max_gaussians=1)
    path=tmp_path/'multi-fixed.pkl'
    run_constrained_ssw(base, Harmonic(), steps=0, config=cfg,
                        rng=np.random.default_rng(3), checkpoint_path=path,
                        fixed_indices=[0,1,2])
    cp=load_constrained_checkpoint(path)
    changed=base.copy(); changed.positions[1,1] += .25
    class Zero:
        requests=0
        def evaluate(self, atoms): raise AssertionError('validation called PES')
    with pytest.raises(ValueError, match='fixed coordinates'):
        run_constrained_ssw(changed, Zero(), steps=0, config=cfg,
            rng=np.random.default_rng(99), fixed_indices=[0,1,2], checkpoint=cp)


def test_constrained_paper_ls_checkpoint_preserves_runtime_state(tmp_path):
    from pamssw.standalone.constrained_reference import load_constrained_checkpoint
    a=Atoms('Cu2',positions=[[3.,0.,0.],[.4,.3,.2]],cell=[8,8,8],pbc=True,
            constraint=FixAtoms(indices=[0]))
    ls=LSSettings(bond_energies={(29,29):3.},bond_lengths={(29,29):5.},target_per_atom=.2)
    cfg=ConstrainedSSWConfig(width=.2,rotation_bias=10.,max_gaussians=1,relax_steps=30)
    continuous_path=tmp_path/'continuous.pkl'
    continuous_surface=Harmonic()
    run_constrained_ssw(a,continuous_surface,steps=2,config=cfg,rng=np.random.default_rng(3),
                        ls=ls,checkpoint_path=continuous_path)
    continuous_cp=load_constrained_checkpoint(continuous_path)
    split_path=tmp_path/'split.pkl'; split_surface=Harmonic()
    run_constrained_ssw(a,split_surface,steps=1,config=cfg,rng=np.random.default_rng(3),
                        ls=ls,checkpoint_path=split_path)
    first_cp=load_constrained_checkpoint(split_path)
    resumed=run_constrained_ssw(a,split_surface,steps=1,config=cfg,rng=np.random.default_rng(99),
                                ls=ls,checkpoint=first_cp,checkpoint_path=split_path)
    split_cp=load_constrained_checkpoint(split_path)
    assert resumed.requests==continuous_cp.evaluation_requests
    assert split_cp.next_index==continuous_cp.next_index==2
    assert split_cp.ls_state['response'].__dict__ == continuous_cp.ls_state['response'].__dict__
    assert split_cp.ls_state['softening'] == continuous_cp.ls_state['softening']


def test_constrained_native_ls_checkpoint_state_and_zero_step_resume(tmp_path):
    from pamssw.standalone.constrained_reference import load_constrained_checkpoint
    a = atoms()
    ls = NativeLSSettings(bond_energies={(29, 29): 3.},
                          bond_lengths={(29, 29): 5.})
    cfg = ConstrainedSSWConfig(width=.2, rotation_bias=10., max_gaussians=1,
                               relax_steps=30)
    path = tmp_path / 'native.pkl'
    surface = Harmonic()
    run_constrained_ssw(a, surface, steps=0, config=cfg,
                        rng=np.random.default_rng(3), ls=ls,
                        checkpoint_path=path)
    cp = load_constrained_checkpoint(path)
    assert cp.ls_state['kind'] == 'native'
    assert cp.ls_state['steps'] == 0
    requests = surface.requests
    class Zero:
        def __init__(self): self.requests = 0
        def evaluate(self, atoms): raise AssertionError('resume consumed PES')
    resumed = run_constrained_ssw(a, Zero(), steps=0, config=cfg,
                                  rng=np.random.default_rng(99), ls=ls,
                                  checkpoint=cp)
    assert resumed.requests == cp.evaluation_requests
    assert resumed.checkpoint is not None


def test_constrained_native_ls_records_isolated_update_event_and_units():
    a = atoms()
    ls = NativeLSSettings(bond_energies={(29, 29): 3.},
                          bond_lengths={(29, 29): 5.})
    cfg = ConstrainedSSWConfig(width=.2, rotation_bias=10., max_gaussians=1,
                               relax_steps=30)
    result = run_constrained_ssw(a, Harmonic(), steps=2, config=cfg,
                                 rng=np.random.default_rng(3), ls=ls)
    updated_records = [record for record in result.records[1:]
                       if record.get('ls_update') == 'updated']
    updates = [record['native_ls_update'] for record in updated_records]
    assert len(updates) == 2
    for record, update in zip(updated_records, updates):
        preparation = record['ls_preparation']
        expected = 1000. * (preparation.energy_after - preparation.energy_before) / len(a)
        assert update['response'] == pytest.approx(expected)
        assert isinstance(update['old_bond_count'], int)
        assert isinstance(update['table'], dict) and update['table']
    first_table = dict(updates[0]['table'])
    assert updates[0] is not updates[1]
    assert updates[0]['table'] is not updates[1]['table']
    changed_key = next(iter(updates[1]['table']))
    updates[1]['table'][changed_key] += 1.0
    assert updates[0]['table'] == first_table


def test_constrained_paper_ls_keeps_native_update_field_absent():
    a = atoms()
    ls = LSSettings(bond_energies={(29, 29): 3.},
                    bond_lengths={(29, 29): 5.}, target_per_atom=.2)
    result = run_constrained_ssw(
        a, Harmonic(), steps=1,
        config=ConstrainedSSWConfig(width=.2, rotation_bias=10., max_gaussians=1,
                                    relax_steps=30),
        rng=np.random.default_rng(3), ls=ls)
    assert all('native_ls_update' not in record for record in result.records)


def test_complete_lifecycle_actual_safe_and_fixed_coordinates():
    a=atoms();s=Harmonic()
    r=run_constrained_ssw(a,s,steps=1,config=ConstrainedSSWConfig(width=.2,rotation_bias=10.,max_gaussians=1),rng=np.random.default_rng(3))
    assert r.status=='completed' and len(r.minima)==2
    for landing in r.minima:
        assert landing.converged and landing.full_raw_fmax==3.
        np.testing.assert_array_equal(landing.atoms.positions[0],a.positions[0])
        np.testing.assert_array_equal(landing.atoms.cell.array,a.cell.array)
    assert r.requests==sum(event['requests'] for event in r.records)


def test_constrained_ls_entrypoint_two_steps_full_pbc():
    a=Atoms('Cu2',positions=[[3.,0.,0.],[.4,.3,.2]],cell=[8,8,8],pbc=True,
            constraint=FixAtoms(indices=[0]))
    s=Harmonic()
    ls=LSSettings(bond_energies={(29,29):3.},bond_lengths={(29,29):5.},target_per_atom=.2)
    r=run_constrained_ssw(a,s,steps=2,
        config=ConstrainedSSWConfig(width=.2,rotation_bias=10.,max_gaussians=1,
                                    fmax=.01,relax_steps=30),
        rng=np.random.default_rng(3),ls=ls)
    assert r.status in ('completed','completed_with_failures')
    assert r.records[1]['ls_preparation'] is not None
    assert r.records[2]['ls_preparation'] is not None
    assert r.requests==sum(event['requests'] for event in r.records)
    assert all(np.array_equal(m.atoms.positions[0],a.positions[0]) for m in r.minima)


def test_proposal_failure_preserves_certified_current(monkeypatch):
    import pamssw.standalone.rc_reference as shared
    a=atoms();s=Harmonic()
    def fail(x,n,**kw):kw['evaluate'](x);raise RuntimeError('injected cap')
    monkeypatch.setattr(shared,'generalized_dimer',fail)
    r=run_constrained_ssw(a,s,steps=2,config=ConstrainedSSWConfig(width=.2,rotation_bias=10.),rng=np.random.default_rng(3))
    assert r.status=='completed_with_failures' and len(r.minima)==1 and r.current is r.initial
    assert [e['requests'] for e in r.records[1:]]==[1,1]
    assert r.requests==sum(e['requests'] for e in r.records)


def test_direction_exclusion_does_not_freeze_relaxation():
    class Coupled:
        requests=0
        def evaluate(self,a):
            self.requests+=1
            x,y=a.positions[1:]
            forces=np.zeros((3,3))
            forces[1]=-2*x+y;forces[2]=-2*y+x
            return float(x@x+y@y-x@y),forces
    a=Atoms('Cu3',positions=[[4,0,0],[0,0,0],[0,0,0]],constraint=FixAtoms(indices=[0]))
    s=Coupled()
    r=run_constrained_ssw(a,s,steps=1,direction_fixed_indices=[0,1],
        config=ConstrainedSSWConfig(width=.2,rotation_bias=10.,max_gaussians=1),
        rng=np.random.default_rng(3))
    assert r.status=='completed'
    event=r.records[1]
    np.testing.assert_array_equal(event['climb'][0]['mode'].direction[:3],0)
    assert np.linalg.norm(event['last_work'].positions[1])>1e-3
    for landing in r.minima:
        assert landing.converged
        np.testing.assert_array_equal(landing.atoms.positions[0],a.positions[0])


def test_shared_ritz_active_adapter_matches_direct_atom_solver():
    from ase import Atoms
    from pamssw.standalone.direction import paper_biased_direction
    chart=ReducedCartesianChart(atoms()); surface=Harmonic(); q=np.array([.2,-.1,.3]); anchor=np.array([1.,2.,3.]); anchor/=np.linalg.norm(anchor)
    def evaluate_atom(candidate):
        e,g=chart.evaluate(candidate.positions.ravel(),surface)
        return e,-g.reshape(candidate.positions.shape)
    container=Atoms('H',positions=q.reshape(1,3))
    direct=paper_biased_direction(container,anchor.reshape(1,3),rotation_bias=10.,fd_step=1e-4,max_hvp=8,tol=.02,evaluate=evaluate_atom)
    adapted=_active_rotation_callback(q,anchor,evaluate=lambda x:chart.evaluate(x,surface),
        rotation_bias=10.,fd_step=1e-4,max_hvp=8,tol=.02,rotation_solver='ritz')
    np.testing.assert_allclose(adapted.direction,direct.direction.ravel())
    assert adapted.force_calls==direct.force_calls and adapted.residual_norm==pytest.approx(direct.residual_norm)


@pytest.mark.parametrize('solver', ['ritz','dimer','broyden-euclidean'])
def test_optin_shared_direction_solver_preserves_fixed_atoms(solver):
    a=atoms(); s=Harmonic()
    r=run_constrained_ssw(a,s,steps=1,config=ConstrainedSSWConfig(
        width=.2,rotation_bias=10.,max_gaussians=1,rotation_solver=solver),
        rng=np.random.default_rng(3))
    assert r.status=='completed' and r.records[1]['climb'][0]['mode'].force_calls >= 1
    for landing in r.minima:
        np.testing.assert_array_equal(landing.atoms.positions[0],a.positions[0])


def test_shared_direction_restriction_holds_with_coupled_hessian():
    class Coupled:
        requests=0
        def evaluate(self,a):
            self.requests+=1; x,y=a.positions[1:]
            f=np.zeros((3,3)); f[1]=-2*x+y; f[2]=-2*y+x
            return float(x@x+y@y-x@y),f
    a=Atoms('Cu3',positions=[[4,0,0],[.2,.1,0],[0,0,.3]],constraint=FixAtoms(indices=[0]))
    r=run_constrained_ssw(a,Coupled(),steps=1,direction_fixed_indices=[0,1],
        config=ConstrainedSSWConfig(width=.1,rotation_bias=10.,max_gaussians=1,rotation_solver='ritz'),
        rng=np.random.default_rng(4))
    direction=r.records[1]['climb'][0]['mode'].direction
    np.testing.assert_array_equal(direction[:3],0.)
    assert np.linalg.norm(r.records[1]['last_work'].positions[1]-a.positions[1])>1e-8


@pytest.mark.parametrize('kwargs', [
    dict(rotation_solver='generalized-dimer',pre_rotation_hvp=2,rotation_bias=10.),
    dict(rotation_solver='ritz',pre_rotation_hvp=2,rotation_bias=10.),
    dict(rotation_solver='ritz',pre_rotation_hvp=2,rotation_bias=None,rotation_hvp=3),
    dict(rotation_solver='ritz',rotation_bias=None),
])
def test_shared_staged_configuration_rejected_before_oracle(kwargs):
    base=dict(width=.2,rotation_bias=10.)
    base.update(kwargs)
    with pytest.raises(ValueError): ConstrainedSSWConfig(**base)


def test_shared_staged_callback_records_pre_and_main_budget():
    s=Harmonic(); cfg=ConstrainedSSWConfig(width=.1,rotation_bias=None,
        max_gaussians=1,rotation_solver='ritz',pre_rotation_hvp=2,rotation_hvp=6)
    r=run_constrained_ssw(atoms(),s,steps=1,config=cfg,rng=np.random.default_rng(5))
    mode=r.records[1]['climb'][0]['mode']
    assert mode.pre.hvp_calls <= 2 and mode.main.hvp_calls <= 3
    assert mode.pre.force_calls + mode.main.force_calls <= 1 + cfg.rotation_hvp


@pytest.mark.parametrize('excluded', [[1],[-1],[2],[True],[0,0]])
def test_direction_exclusion_validated_before_oracle(excluded):
    s=Harmonic()
    with pytest.raises(ValueError):
        run_constrained_ssw(atoms(),s,steps=1,direction_fixed_indices=excluded,
            config=ConstrainedSSWConfig(width=.2,rotation_bias=10.),rng=np.random.default_rng(3))
    assert s.requests==0
