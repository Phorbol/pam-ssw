"""Periodic atomic climbing extracted from paper_reference's existing baseline.

No initial/final true quench, cell change, LS or MC is performed here.
"""
from dataclasses import dataclass
from copy import deepcopy
import math
import numpy as np
from .paper_reference import sample_initial_direction
from .periodic_geometry import FixedCellTranslationFrame
from .surface import SurfaceCalculator, quench
from .gaussian import ProjectedGaussian
from .dimer import paper_dimer_direction
from .direction import paper_biased_direction

@dataclass(frozen=True)
class AtomicClimbCheckpoint:
    """Completed Gaussian boundary; pending work is diagnostic, never committed.

    Replay redoes the pending Gaussian from this boundary, including direction
    refinement. Optimizer history is not saved. previous_requests retains all
    earlier work, including truncated attempts; replay requests are additional.
    """
    next_index: int
    atoms: object
    initial_anchor: np.ndarray
    climb: tuple
    reference_energy: float
    config: object
    previous_requests: int = 0
    pending: dict | None = None
    terminal_status: str | None = None
    gaussian_policy: object | None = None


@dataclass(frozen=True)
class AtomicClimbResult:
    atoms: object
    status: str
    climb: tuple
    requests: int
    initial_direction: np.ndarray
    error: str | None = None
    checkpoint: AtomicClimbCheckpoint | None = None

    @property
    def total_requests(self):
        return self.checkpoint.previous_requests if self.checkpoint else self.requests

def atomic_climb(atoms,surface,*,reference_energy,config,rng,max_completed_gaussians=None,gaussian_policy=None):
    """Existing fixed-cell climb, with explicit outer starting energy threshold.

    Fully periodic translation_only/global/Safe-total only. Position lift and
    cell remain unchanged in convention. Failures return the latest available
    work geometry and charged surface-request count; no fallback is attempted.
    """
    if config.rotation_solver == 'broyden-euclidean':
        raise NotImplementedError('broyden-euclidean is supported by run_ssw, not atomic_climb')
    if config.pre_rotation_hvp is not None:
        raise NotImplementedError('staged rotation is supported by run_ssw, not atomic_climb')
    if config.rotation_exit_policy != 'force':
        raise NotImplementedError('rotation_exit_policy is supported by run_ssw, not atomic_climb')
    if gaussian_policy is not None:
        from .pam_gaussian import PAMCurvatureGaussian
        if not isinstance(gaussian_policy, PAMCurvatureGaussian): raise TypeError('gaussian_policy must be PAMCurvatureGaussian')
    if (not atoms.pbc.all() or config.cluster_frame!='translation_only'
        or config.direction_sampling!='global' or config.quench_optimizer!='safe-lbfgs-total'):
        raise ValueError('atomic_climb requires full PBC, translation_only/global and safe-lbfgs-total')
    FixedCellTranslationFrame(atoms)
    if not np.isfinite(reference_energy):raise ValueError('finite reference_energy required')
    masses=atoms.get_masses()
    if not len(atoms) or not np.isfinite(masses).all() or np.any(masses<=0):raise ValueError('finite positive masses required')
    anchor=sample_initial_direction(atoms,rng,mode=config.direction_sampling)
    checkpoint=AtomicClimbCheckpoint(0,atoms.copy(),anchor.copy(),(),float(reference_energy),deepcopy(config),gaussian_policy=gaussian_policy)
    return resume_atomic_climb(checkpoint,surface,config,max_completed_gaussians=max_completed_gaussians)


def resume_atomic_climb(checkpoint,surface,config=None,*,max_completed_gaussians=None):
    """Resume at a completed boundary, redoing any pending work at new cost.

    No initial direction resampling, initial/final quench or MC is performed.
    Optional max_completed_gaussians pauses after that many NEW completed terms.
    Terminal successful checkpoints are returned without new oracle calls.
    """
    config=checkpoint.config if config is None else config
    if config != checkpoint.config:raise ValueError('checkpoint config must match replay config')
    if config.rotation_solver == 'broyden-euclidean':
        raise NotImplementedError('broyden-euclidean is supported by run_ssw, not atomic_climb')
    if config.pre_rotation_hvp is not None:
        raise NotImplementedError('staged rotation is supported by run_ssw, not atomic_climb')
    if config.rotation_exit_policy != 'force':
        raise NotImplementedError('rotation_exit_policy is supported by run_ssw, not atomic_climb')
    if max_completed_gaussians is not None and (isinstance(max_completed_gaussians,bool) or not isinstance(max_completed_gaussians,int) or max_completed_gaussians<1):
        raise ValueError('max_completed_gaussians must be a positive integer')
    if checkpoint.next_index != len(checkpoint.climb) or not 0<=checkpoint.next_index<=config.max_gaussians:
        raise ValueError('checkpoint completed index/events mismatch')
    if any(e.get('index')!=i or 'true_energy' not in e for i,e in enumerate(checkpoint.climb)):
        raise ValueError('checkpoint requires contiguous completed Gaussian events')
    if (not checkpoint.atoms.pbc.all() or config.cluster_frame!='translation_only'
        or config.direction_sampling!='global' or config.quench_optimizer!='safe-lbfgs-total'):
        raise ValueError('replay requires full PBC, translation_only/global and safe-lbfgs-total')
    if checkpoint.previous_requests<0:raise ValueError('negative checkpoint request count')
    FixedCellTranslationFrame(checkpoint.atoms)
    anchor=np.asarray(checkpoint.initial_anchor,dtype=float).copy()
    if anchor.shape!=checkpoint.atoms.positions.shape or not np.isfinite(anchor).all() or np.linalg.norm(anchor)==0:
        raise ValueError('invalid checkpoint initial anchor')
    reference_energy=checkpoint.reference_energy
    if not np.isfinite(reference_energy):raise ValueError('finite reference energy required')
    before=surface.requests;work=checkpoint.atoms.copy();events=list(deepcopy(checkpoint.climb))
    completed=list(deepcopy(checkpoint.climb));boundary=work.copy();next_index=checkpoint.next_index
    terms=[ProjectedGaussian(np.asarray(e['center']),np.asarray(e['direction']),e['width'],e['weight']) for e in completed]
    gaussian_policy=checkpoint.gaussian_policy
    if gaussian_policy is not None:
        from .pam_gaussian import PAMCurvatureGaussian
        if not isinstance(gaussian_policy, PAMCurvatureGaussian):
            raise TypeError('checkpoint gaussian_policy must be PAMCurvatureGaussian')
    status=checkpoint.terminal_status or 'gaussian_limit';error=None;pending=None
    rotation_frame=None
    def rotation_surface(candidate):
        candidate=candidate.copy();candidate.positions=rotation_frame.positions(candidate.positions)
        energy,forces=surface.evaluate(candidate)
        return energy,rotation_frame.project(forces)
    solver=paper_dimer_direction if config.rotation_solver=='dimer' else paper_biased_direction
    try:
        for index in range(checkpoint.next_index, config.max_gaussians if checkpoint.terminal_status is None else checkpoint.next_index):
            pending=dict(index=index,stage="rotation")
            rotation_frame=FixedCellTranslationFrame(work)
            rotation_anchor=rotation_frame.project(anchor);norm=np.linalg.norm(rotation_anchor)
            if not np.isfinite(norm) or norm<=np.finfo(float).eps*rotation_anchor.size:raise ValueError('anchor has no resolvable internal component')
            rotation_anchor/=norm
            mode=solver(work,rotation_anchor,rotation_bias=config.rotation_bias,fd_step=config.fd_step,
                        max_hvp=config.rotation_hvp,tol=config.rotation_tol,evaluate=rotation_surface)
            if not mode.converged:
                status='rotation_failed'
                events.append(dict(index=index,rotation_solver=config.rotation_solver,
                    residual=mode.residual_norm,force_requests=mode.force_calls,
                    rotation_stop_reason=getattr(mode, 'stop_reason', 'unspecified'),
                    rotation_converged=False,rotation_budget_released=False))
                break
            center=work.positions.copy()
            pending.update(stage="height",center=center.tolist(),direction=mode.direction.tolist(),width=config.width)
            force_parallel=None
            if gaussian_policy is None:
                displaced=work.copy();displaced.positions+=config.width*mode.direction
                pending['displaced']=displaced.copy();displaced.calc=SurfaceCalculator(surface,terms=terms)
                force_parallel=float(np.sum(displaced.get_forces()*mode.direction))
                weight=(config.forward_force-force_parallel)*config.width*math.exp(.5)
                if not np.isfinite(weight) or weight<=0:
                    status='nonpositive_height';events.append(dict(index=index,weight=weight,background_forward_force=force_parallel));break
            pending.update(stage="biased_quench",weight=(None if gaussian_policy is not None else weight),background_forward_force=force_parallel)
            if gaussian_policy is None:
                policy_data=None; width=config.width; weight=(config.forward_force-force_parallel)*config.width*math.exp(.5)
            else:
                policy_data=gaussian_policy.choose(mode=mode,anchor=rotation_anchor,center=center,terms=terms,base_width=config.width,rotation_bias=config.rotation_bias)
                width=policy_data['width']; weight=policy_data['weight']
            displaced=work.copy(); displaced.positions=center + width*mode.direction
            displaced.calc=SurfaceCalculator(surface,terms=terms)
            terms.append(ProjectedGaussian(center,mode.direction,width,weight))
            if gaussian_policy is not None:
                pending.update(width=width,weight=weight,displaced=displaced.copy(),policy=policy_data)
            stage_steps = config.relax_steps if config.bias_stage_steps is None else config.bias_stage_steps
            bias_fmax = config.fmax if config.bias_fmax is None else config.bias_fmax
            relaxed=quench(displaced,surface,fmax=bias_fmax,steps=stage_steps,terms=terms,
                           optimizer=config.quench_optimizer,lbfgs_memory=config.lbfgs_memory)
            event=dict(index=index,rotation_solver=config.rotation_solver,cluster_frame=config.cluster_frame,center=center.tolist(),direction=mode.direction.tolist(),weight=weight,width=width,biased_energy=relaxed.energy,force_certificate=relaxed.surface,max_force=relaxed.max_force,rotation_residual=mode.residual_norm,rotation_force_requests=mode.force_calls,quench_requests=relaxed.evaluation_requests)
            event.update(rotation_stop_reason=getattr(mode, 'stop_reason', 'unspecified'),
                         rotation_converged=bool(mode.converged),
                         rotation_budget_released=False,
                         actual_anchor=rotation_anchor.tolist(),
                         actual_rotation_bias=float(config.rotation_bias))
            if policy_data is not None: event['gaussian_policy']=policy_data
            if config.bias_fmax is not None: event['bias_fmax']=config.bias_fmax
            event['optimizer_telemetry'] = relaxed.optimizer_telemetry
            event['termination_reason'] = (None if relaxed.optimizer_telemetry is None
                                           else relaxed.optimizer_telemetry.termination_reason)
            budget_stop = (config.bias_stage_steps is not None and
                           relaxed.optimizer_telemetry is not None and
                           relaxed.optimizer_telemetry.termination_reason == 'maxiter' and
                           np.isfinite(relaxed.energy) and np.isfinite(relaxed.max_force) and
                           np.isfinite(relaxed.atoms.positions).all())
            if config.bias_stage_steps is not None:
                event['stage_stop_reason'] = 'iteration_budget' if budget_stop else ('force_converged' if relaxed.converged else 'failure')
                event['status'] = 'stage_budget' if budget_stop else ('converged' if relaxed.converged else 'biased_quench_failed')
            events.append(event);work=relaxed.atoms.copy()
            if not relaxed.converged and not budget_stop:status='biased_quench_failed';break
            pending['stage']='true_energy'
            energy,_=surface.evaluate(work);event['true_energy']=energy
            completed.append(deepcopy(event));boundary=work.copy();next_index=index+1;pending=None
            if energy<reference_energy:status='lower_true_energy';break
            if max_completed_gaussians is not None and next_index-checkpoint.next_index>=max_completed_gaussians and next_index<config.max_gaussians:
                status='checkpoint_boundary';break
    except Exception as exc:
        status='evaluation_failed';error=f'{type(exc).__name__}: {exc}'
        events.append(dict(error=error))
    requests=surface.requests-before
    terminal=status if status in ('lower_true_energy','gaussian_limit') else None
    saved=AtomicClimbCheckpoint(next_index,boundary.copy(),anchor.copy(),tuple(completed),reference_energy,
        deepcopy(config),checkpoint.previous_requests+requests,deepcopy(pending),terminal,gaussian_policy)
    return AtomicClimbResult(work.copy(),status,tuple(events),requests,anchor.copy(),error,saved)
