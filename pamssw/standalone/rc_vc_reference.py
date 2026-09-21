"""Independent periodic rigid-forest SSW with bare all-DOF cell landing quench."""
from dataclasses import dataclass
import math
import numpy as np
from ase import units
from .rc_forest_reference import RCForestSSWConfig
from .rc_optimization_domain import PrincipalRigidForestCellChart
from .generalized_numerics import generalized_dimer,safe_lbfgs
from .cell_relax import cell_quench
from .vc_reference import VCSSWResult


@dataclass(frozen=True,kw_only=True)
class RCVCSSWConfig(RCForestSSWConfig):
    strain_length: float  # Angstrom per dimensionless symmetric strain
    pressure: float = 0.  # eV/Angstrom^3, positive compression
    stress_tol: float = .001  # eV/Angstrom^3, maximum absolute residual stress

    def __post_init__(self):
        super().__post_init__()
        for name in ('strain_length','stress_tol'):
            value=getattr(self,name)
            if not np.isscalar(value) or not np.isfinite(value) or value<=0:raise ValueError(f'positive finite {name} required')
        if not np.isscalar(self.pressure) or not np.isfinite(self.pressure):raise ValueError('finite scalar pressure required')


def run_rc_vc_ssw(atoms,surface,*,trees,anchor,steps,config,rng,
                  direction_solver=None, rotation_force_calls=None):
    """Rigid-center/cell escape -> unrestricted E+pV quench -> enthalpy MC.

    Inputs must have explicit continuous lifted molecular coordinates and a
    translation-invariant periodic PES. No wrapping/lifting inference occurs.
    Each chart is frozen throughout a proposal and rebuilt from selected true
    minimum only. All full-force/stress certified candidates, including rejected
    ones, remain visible. Native Kabsch/lambda parity is not claimed.
    """
    if not isinstance(config,RCVCSSWConfig):raise TypeError('RCVCSSWConfig required')
    if isinstance(steps,(bool,np.bool_)) or not isinstance(steps,(int,np.integer)) or steps<0:raise ValueError('nonnegative integer steps required')
    if direction_solver is None:
        if rotation_force_calls is not None:
            raise ValueError('rotation_force_calls requires direction_solver')
        solver = generalized_dimer
    else:
        if not callable(direction_solver):
            raise ValueError('direction_solver must be callable')
        if isinstance(rotation_force_calls,(bool,np.bool_)) or not isinstance(rotation_force_calls,(int,np.integer)) or rotation_force_calls<1:
            raise ValueError('rotation_force_calls must be a positive integer with direction_solver')
        solver = direction_solver
    trees=tuple(dict(bodies=tuple(tuple(g) for g in t['bodies']),parents=tuple(t['parents']),joints=tuple(None if a is None else tuple(a) for a in t['joints'])) for t in trees)
    def chart_for(a):return PrincipalRigidForestCellChart(a,trees,anchor=anchor,rotation_length=config.rotation_length,torsion_length=config.torsion_length,strain_length=config.strain_length)
    chart_for(atoms)  # input-domain errors must precede oracle work
    start=surface.requests;initial=current=best=None;records=[];minima=[]
    def finish(status):return VCSSWResult(initial,current,best,minima,records,surface.requests-start,status)
    def true_quench(a):
        kwargs = dict(strain_length=config.strain_length, pressure=config.pressure,
                      fmax=config.fmax, stress_tol=config.stress_tol,
                      max_step=config.max_step, maxiter=config.relax_steps)
        if config.lbfgs_memory is not None:
            kwargs['lbfgs_memory'] = config.lbfgs_memory
        return cell_quench(a, surface, **kwargs)
    try:
        first=true_quench(atoms);initial=first.evaluation
        records.append(dict(stage='initial',status='converged' if first.converged else 'quench_failed',quench=first,certificate=first.certificate,requests=surface.requests-start))
        if not first.converged:return finish('initial_quench_failed')
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
        records.append(dict(stage='initial',status='evaluation_failed',error=str(error),requests=surface.requests-start));return finish('initial_quench_failed')
    current=best=initial;minima.append(initial)
    for index in range(steps):
        before=surface.requests;event=dict(index=index,status='running',accepted=False,landing=None,climb=[],requests=0);records.append(event)
        try:
            chart=chart_for(current.atoms);work=np.zeros(chart.dimension)
            anchor_direction=rng.normal(size=chart.dimension);anchor_direction/=np.linalg.norm(anchor_direction)
            terms=[]
            event.update(chart_reference=current.atoms.copy(),last_work=current.atoms.copy(),frozen_gaussians=[])
            def bare(x):
                ev=chart.evaluate(x,surface.evaluate,pressure=config.pressure)
                return ev.objective,ev.gradient
            def biased(x):
                energy,g=bare(x)
                for center,n,w in terms:
                    z=float((x-center)@n);v=w*math.exp(-.5*(z/config.width)**2)
                    energy+=v;g-=v*z/config.width**2*n
                return energy,g
            status='gaussian_limit'
            for j in range(config.max_gaussians):
                stage_before=surface.requests;stage=dict(index=j,status='running');event['climb'].append(stage)
                solver_kwargs=dict(rotation_bias=config.rotation_bias,fd_step=config.fd_step,tol=config.rotation_tol,evaluate=bare)
                if direction_solver is None: solver_kwargs['max_hvp']=config.rotation_hvp
                else: solver_kwargs['max_force_calls']=rotation_force_calls
                mode=solver(work,anchor_direction,**solver_kwargs)
                stage['mode']=mode
                if not mode.converged:
                    stage.update(status='rotation_failed',requests=surface.requests-stage_before);status='rotation_failed';break
                displaced=work+config.width*mode.direction;_,g=biased(displaced)
                weight=float((config.forward_force+g@mode.direction)*config.width*math.exp(.5));stage['weight']=weight
                if not np.isfinite(weight) or weight<=0:
                    stage.update(status='nonpositive_height',requests=surface.requests-stage_before);status='nonpositive_height';break
                terms.append((work.copy(),mode.direction.copy(),weight))
                event['frozen_gaussians'].append(dict(center=work.copy(),direction=mode.direction.copy(),weight=weight,width=config.width))
                relaxed=safe_lbfgs(displaced,biased,gradient_norm=np.linalg.norm,step_norm=np.linalg.norm,gtol=config.gradient_tol,max_step=config.max_step,maxiter=config.relax_steps,lbfgs_memory=config.lbfgs_memory)
                work=relaxed.q.copy();stage.update(status=relaxed.status,relaxation=relaxed,scaled_coordinates=work.copy(),requests=surface.requests-stage_before)
                event['last_work']=chart.unpack(work)
                if not relaxed.converged:status='biased_quench_failed';break
                enthalpy,_=bare(work);stage.update(true_objective=enthalpy,requests=surface.requests-stage_before)
                if enthalpy<current.objective:status='lower_true_objective';break
            event['last_work']=chart.unpack(work)
            if status in ('gaussian_limit','lower_true_objective'):
                landing=true_quench(event['last_work']);event.update(quench=landing,certificate=landing.certificate,landing=landing.evaluation)
                if not landing.converged:status='true_quench_failed'
                else:
                    candidate=landing.evaluation;minima.append(candidate);delta=candidate.objective-current.objective
                    accepted=delta<=0 or (config.temperature_K>0 and rng.random()<math.exp(-delta/(units.kB*config.temperature_K)))
                    if candidate.objective<best.objective:best=candidate
                    if accepted:current=candidate
                    event.update(accepted=bool(accepted),delta=delta);status='valid_landing'
            event['status']=status
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            event.update(status='evaluation_failed',error=str(error))
            if event['climb'] and event['climb'][-1]['status']=='running':stage.update(status='evaluation_failed',error=str(error),requests=surface.requests-stage_before)
        event['requests']=surface.requests-before
    return finish('completed_with_failures' if any(e['status']!='valid_landing' for e in records[1:]) else 'completed')
