"""Fixed-cell EMT integration; LS coefficients are explicit test inputs, not a Cu fit."""
from dataclasses import replace
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone.ls_prequench import LSPrequenchSettings
from pamssw.standalone.paper_reference import LSSettings, SSWConfig, run_ssw, _policy_signature
from pamssw.standalone.ls_native_reference import NativeLSSettings
from pamssw.standalone.constrained_reference import _constrained_signature, ReducedCartesianChart
from pamssw.standalone.constrained_ls import ConstrainedLSRuntime
from pamssw.standalone.ls_cycle import prepare_ls_step, LSCycleError
from pamssw.standalone.softening import FrozenBondSoftening
from pamssw.standalone.surface import ASESurface, quench

E={(29,29):1.}; L={(29,29):3.}
def settings(prequench=None):return LSSettings(E,L,.001,prequench=prequench)
def atoms():return Atoms('Cu2',positions=[[0,0,0],[2.7,0,0]])
def minimum():
    r=quench(atoms(),ASESurface(EMT()),fmax=1e-6,steps=100,optimizer='safe-lbfgs-total')
    assert r.converged
    return r.atoms

@pytest.mark.parametrize('f,s',[(0,50),(-1,50),(float('nan'),50),(True,50),(.1,-1),(.1,True),(.1,1.5)])
def test_invalid(f,s):
    with pytest.raises(ValueError):LSPrequenchSettings(f,s)

def test_exit_policy_is_explicit_and_defaults_to_force():
    assert LSPrequenchSettings(.1, 3).exit_policy == 'force'
    with pytest.raises(ValueError, match='exit_policy'):
        LSPrequenchSettings(.1, 3, 'backend_default')

def test_step_limit_policy_accepts_safe_maxiter_and_evaluates_true_response():
    start=minimum(); surf=ASESurface(EMT())
    soft=FrozenBondSoftening.from_atoms(start, bond_energies=E,
                                        bond_lengths={(29,29):3.5})
    prepared=prepare_ls_step(start, surf, softening=soft, fmax=1e-6, steps=1,
        optimizer='safe-lbfgs-total',
        prequench=LSPrequenchSettings(.001, 1, 'force_or_step_limit'))
    assert prepared.qualification == 'step_limit'
    assert not prepared.soft_quench.converged
    assert prepared.soft_quench.optimizer_telemetry.termination_reason == 'maxiter'
    assert np.isfinite(prepared.energy_response)
    assert surf.requests > prepared.soft_quench.evaluation_requests

def test_step_limit_rejects_non_safe_backend_before_pes():
    start=minimum(); surf=ASESurface(EMT())
    soft=FrozenBondSoftening.from_atoms(start, bond_energies=E,
                                        bond_lengths={(29,29):3.5})
    with pytest.raises(ValueError, match='safe-lbfgs-total'):
        prepare_ls_step(start, surf, softening=soft, fmax=1e-6, steps=1,
            optimizer='ase-lbfgs',
            prequench=LSPrequenchSettings(.001, 1, 'force_or_step_limit'))
    assert surf.requests == 0

def test_public_step_limit_rejects_backend_before_initial_pes():
    start=minimum(); surf=ASESurface(EMT())
    config=SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e-6, relax_steps=10, fd_step=1e-4,
        rotation_hvp=4, rotation_tol=1e-3, direction_sampling='global',
        quench_optimizer='ase-lbfgs')
    with pytest.raises(ValueError, match='safe-lbfgs-total'):
        run_ssw(start, surf, steps=1, config=config, rng=np.random.default_rng(2),
                ls=settings(LSPrequenchSettings(.001, 1, 'force_or_step_limit')))
    assert surf.requests == 0

def test_step_limit_does_not_accept_line_search_failure(monkeypatch):
    from dataclasses import replace
    import pamssw.standalone.ls_cycle as cycle
    start=minimum(); surf=ASESurface(EMT())
    soft=FrozenBondSoftening.from_atoms(start, bond_energies=E,
                                        bond_lengths={(29,29):3.5})
    original=cycle.quench
    def failed(*args, **kwargs):
        result=original(*args, **kwargs)
        telemetry=replace(result.optimizer_telemetry, termination_reason='line_search_failed')
        return replace(result, converged=False, optimizer_telemetry=telemetry)
    monkeypatch.setattr(cycle, 'quench', failed)
    with pytest.raises(LSCycleError, match='soft_quench'):
        prepare_ls_step(start, surf, softening=soft, fmax=1e-6, steps=1,
            optimizer='safe-lbfgs-total',
            prequench=LSPrequenchSettings(.001, 1, 'force_or_step_limit'))

def test_old_missing_fields_and_signature_comparison():
    for cls in (lambda:settings(),lambda:NativeLSSettings(E,L)):
        new=cls();old=cls();object.__delattr__(old,'prequench')
        for signature in (_policy_signature,_constrained_signature):
            assert signature(old)==signature(new)
            assert signature(old)!=signature(replace(new,prequench=LSPrequenchSettings(.1,50)))
    with pytest.raises(TypeError):settings({'fmax':.1,'steps':50})

def test_real_emt_soft_thresholds_and_true_start_not_relaxed():
    start=minimum(); soft=FrozenBondSoftening.from_atoms(start,bond_energies=E,bond_lengths=L)
    results=[]
    for tol in (.1,1e-6):
        r=prepare_ls_step(start,ASESurface(EMT()),softening=soft,fmax=1e-6,steps=0,
            optimizer='safe-lbfgs-total',prequench=LSPrequenchSettings(tol,100))
        assert r.soft_quench.converged and r.soft_quench.max_force<=tol
        results.append(r)
    assert results[0].soft_quench.optimizer_steps < results[1].soft_quench.optimizer_steps
    displaced=start.copy();displaced.positions[1,0]+=.02
    soft=FrozenBondSoftening.from_atoms(displaced,bond_energies=E,bond_lengths=L)
    with pytest.raises(LSCycleError,match='true_start'):
        prepare_ls_step(displaced,ASESurface(EMT()),softening=soft,fmax=1e-6,steps=100,
            optimizer='safe-lbfgs-total',prequench=LSPrequenchSettings(10.,100))

def test_zero_soft_budget_still_fails_strictly():
    start=minimum();soft=FrozenBondSoftening.from_atoms(start,bond_energies=E,bond_lengths=L)
    with pytest.raises(LSCycleError) as err:
        prepare_ls_step(start,ASESurface(EMT()),softening=soft,fmax=1e-6,steps=100,
            optimizer='safe-lbfgs-total',prequench=LSPrequenchSettings(1e-6,0))
    assert not err.value.result.converged

@pytest.mark.parametrize('tol',[.1,1e-6])
def test_full_fixed_cell_driver_forwarding_emt(monkeypatch,tol):
    from pamssw.standalone import ls_cycle
    original=ls_cycle.quench;seen=[]
    def track(*args,**kwargs):
        result=original(*args,**kwargs);seen.append((kwargs['fmax'],kwargs['steps'],result));return result
    monkeypatch.setattr(ls_cycle,'quench',track)
    config=SSWConfig(width=.1,rotation_bias=2.,max_gaussians=1,temperature_K=300.,
        fmax=1e-6,relax_steps=100,fd_step=1e-4,rotation_hvp=8,rotation_tol=1e-3,
        direction_sampling='global',quench_optimizer='safe-lbfgs-total')
    surf=ASESurface(EMT())
    result=run_ssw(minimum(),surf,steps=1,config=config,rng=np.random.default_rng(19),ls=settings(LSPrequenchSettings(tol,50)))
    assert len(seen)==1 and seen[0][:2]==(tol,50) and seen[0][2].converged
    assert result.records[0].status!='ls_prequench_failed'
    assert result.records[0].ls_preparation['qualification'] == 'force'
    assert result.records[0].ls_preparation['soft_quench'].optimizer_telemetry is not None
    assert all(m.max_force<=config.fmax for m in result.minima)
    assert result.evaluation_requests==surf.requests

def test_constrained_real_emt_override_and_start_certificate():
    start=minimum();surf=ASESurface(EMT())
    runtime=ConstrainedLSRuntime.initialize(start,surf,settings(LSPrequenchSettings(.1,50)),fixed_indices=[0])
    chart=ReducedCartesianChart(start,fixed_indices=[0])
    prepared=runtime.prepare(start,chart,fmax=1e-6,max_step=.1,maxiter=0)
    assert prepared.optimizer.converged and prepared.energy_after is not None
    displaced=start.copy();displaced.positions[1,0]+=.02
    runtime=ConstrainedLSRuntime.initialize(displaced,surf,settings(LSPrequenchSettings(10.,50)),fixed_indices=[0])
    with pytest.raises(LSCycleError,match='true_start'):
        runtime.prepare(displaced,ReducedCartesianChart(displaced,fixed_indices=[0]),fmax=1e-6,max_step=.1,maxiter=100)

@pytest.mark.parametrize('native',[False,True])
def test_constrained_public_driver_passes_separate_limits(monkeypatch,native):
    from pamssw.standalone import constrained_ls
    from pamssw.standalone.constrained_reference import run_constrained_ssw,ConstrainedSSWConfig
    original=constrained_ls.safe_lbfgs;seen=[]
    def track(*args,**kwargs):
        result=original(*args,**kwargs);seen.append((kwargs['gtol'],kwargs['maxiter'],result.status));return result
    monkeypatch.setattr(constrained_ls,'safe_lbfgs',track)
    policy=LSPrequenchSettings(.1,50)
    ls=NativeLSSettings(E,L,prequench=policy) if native else settings(policy)
    surf=ASESurface(EMT())
    config=ConstrainedSSWConfig(width=.1,rotation_bias=2.,max_gaussians=1,fmax=1e-6,relax_steps=100)
    result=run_constrained_ssw(minimum(),surf,steps=1,config=config,rng=np.random.default_rng(19),fixed_indices=[0],ls=ls)
    assert seen and seen[0][:2]==(.1,50)
    assert result.requests==surf.requests==sum(r['requests'] for r in result.records)
    assert result.initial.converged
    for row in result.records:
        if row.get('ls_preparation') is not None:
            assert row['ls_preparation'].optimizer.status==seen[0][2]
    if seen[0][2]!='converged':assert result.status=='ls_prequench_failed'

@pytest.mark.parametrize('native',[False,True])
def test_actual_checkpoint_resume_with_explicit_and_old_missing_field(tmp_path,monkeypatch,native):
    from pamssw.standalone import LSPrequenchSettings as PublicSettings
    from pamssw.standalone import ls_cycle
    from pamssw.standalone.paper_reference import load_ssw_checkpoint,save_ssw_checkpoint
    from pamssw.standalone.ls_native_reference import run_native_ls_ssw
    assert PublicSettings is LSPrequenchSettings
    entry=run_native_ls_ssw if native else run_ssw
    config=SSWConfig(width=.1,rotation_bias=2.,max_gaussians=1,temperature_K=300.,
        fmax=1e-6,relax_steps=100,fd_step=1e-4,rotation_hvp=8,rotation_tol=1e-3,
        direction_sampling='global',quench_optimizer='safe-lbfgs-total')
    base=NativeLSSettings(E,L) if native else settings()
    initial=minimum();path=tmp_path/'state.pkl'
    entry(initial,ASESurface(EMT()),steps=0,config=config,rng=np.random.default_rng(19),ls=base,checkpoint_path=path)
    old=load_ssw_checkpoint(path);object.__delattr__(old.ls,'prequench')
    save_ssw_checkpoint(path,old);old=load_ssw_checkpoint(path)
    legacy=entry(initial,ASESurface(EMT()),steps=0,config=config,rng=np.random.default_rng(20),ls=base,checkpoint=old)
    assert legacy.status=='completed'
    explicit=replace(base,prequench=LSPrequenchSettings(.1,50))
    entry(initial,ASESurface(EMT()),steps=0,config=config,rng=np.random.default_rng(19),ls=explicit,checkpoint_path=path)
    cp=load_ssw_checkpoint(path);surf=ASESurface(EMT())
    object.__delattr__(cp.ls.prequench, 'exit_policy')
    save_ssw_checkpoint(path, cp)
    cp=load_ssw_checkpoint(path)
    assert cp.ls.prequench.exit_policy == 'force'
    with pytest.raises(ValueError,match='LS settings'):
        entry(initial,surf,steps=1,config=config,rng=np.random.default_rng(20),ls=replace(explicit,prequench=LSPrequenchSettings(.2,50)),checkpoint=cp)
    assert surf.requests==0
    with pytest.raises(ValueError,match='LS settings'):
        entry(initial,surf,steps=1,config=config,rng=np.random.default_rng(20),
              ls=replace(explicit,prequench=LSPrequenchSettings(.1,50,'force_or_step_limit')),
              checkpoint=cp)
    assert surf.requests==0
    original=ls_cycle.quench;seen=[]
    def track(*args,**kw):
        out=original(*args,**kw);seen.append((kw['fmax'],kw['steps']));return out
    monkeypatch.setattr(ls_cycle,'quench',track)
    resumed=entry(initial,surf,steps=1,config=config,rng=np.random.default_rng(20),ls=explicit,checkpoint=cp)
    assert seen==[(.1,50)]
    assert resumed.evaluation_requests==cp.evaluation_requests+surf.requests


def test_vc_rejects_unimplemented_prequench_override_before_oracle():
    from ase.build import bulk
    from pamssw.standalone.vc_geometry import ASEStressSurface
    from pamssw.standalone.vc_reference import VCSSWConfig, run_vc_ssw
    surface = ASEStressSurface(EMT())
    with pytest.raises(ValueError, match='fixed-cell.*prequench'):
        run_vc_ssw(bulk('Cu', 'fcc', cubic=True), surface, steps=0,
                   config=VCSSWConfig(strain_length=1., width=.1, rotation_bias=2.), rng=np.random.default_rng(7),
                   ls=settings(LSPrequenchSettings(.1, 50)))
    assert surface.requests == 0

def test_constrained_nondefault_exit_policy_rejected_before_pes():
    from pamssw.standalone.constrained_reference import run_constrained_ssw, ConstrainedSSWConfig
    start=minimum(); surf=ASESurface(EMT())
    ls=NativeLSSettings(E, L, prequench=LSPrequenchSettings(.1, 2, 'force_or_step_limit'))
    with pytest.raises(ValueError, match='exit_policy'):
        run_constrained_ssw(start, surf, steps=0,
            config=ConstrainedSSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
                                        fmax=1e-6, relax_steps=10),
            rng=np.random.default_rng(5), fixed_indices=[0], ls=ls)
    assert surf.requests == 0

@pytest.mark.parametrize('native',[False,True])
def test_step_limit_diagnostic_survives_actual_checkpoint(tmp_path,native):
    from pamssw.standalone.paper_reference import load_ssw_checkpoint
    policy=LSPrequenchSettings(1e-6,1,'force_or_step_limit')
    ls=NativeLSSettings(E,L,prequench=policy) if native else settings(policy)
    cfg=SSWConfig(width=.1,rotation_bias=2.,max_gaussians=1,temperature_K=300.,
        fmax=1e-6,relax_steps=100,fd_step=1e-4,rotation_hvp=8,rotation_tol=1e-3,
        direction_sampling='global',quench_optimizer='safe-lbfgs-total')
    path=tmp_path/'cap.pkl'
    r=run_ssw(minimum(),ASESurface(EMT()),steps=1,config=cfg,
              rng=np.random.default_rng(19),ls=ls,checkpoint_path=path)
    saved=load_ssw_checkpoint(path)
    for rec in (r.records[0],saved.records[0]):
        assert rec.ls_preparation['qualification']=='step_limit'
        assert not rec.ls_preparation['soft_quench'].converged
        assert rec.ls_preparation['soft_quench'].optimizer_telemetry.termination_reason=='maxiter'
        if native: assert rec.ls_update['prequench_qualification']=='step_limit'
    assert all(m.max_force<=cfg.fmax for m in r.minima)


def test_direct_constrained_preparation_rejects_policy_before_pes():
    start=minimum();surface=ASESurface(EMT())
    runtime=ConstrainedLSRuntime.initialize(start,surface,
        settings(LSPrequenchSettings(.1,1,'force_or_step_limit')),fixed_indices=[0])
    before=surface.requests
    with pytest.raises(ValueError,match='exit_policy'):
        runtime.prepare(start,ReducedCartesianChart(start,fixed_indices=[0]),
                        fmax=1e-6,max_step=.1,maxiter=100)
    assert surface.requests==before
