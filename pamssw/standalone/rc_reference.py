"""Independent isolated single-chain RC-SSW with exact torsional geometry.

No native lambda transmission, native coordinate-gauge parity or cell coupling.
The physical PES must be invariant under whole-molecule translation/rotation.
"""
from dataclasses import dataclass,replace
import copy
import math
import numpy as np
from ase import units
from .rc_geometry import RigidChainChart
from .generalized_numerics import generalized_dimer,safe_lbfgs
from .surface import quench


@dataclass(frozen=True)
class RCSSWConfig:
    torsion_length: float  # Angstrom/radian; x = torsion_length * theta
    width: float  # Angstrom in scaled torsion coordinates
    rotation_bias: float  # eV/Angstrom^2
    temperature_K: float = 300.
    forward_force: float = .1
    max_gaussians: int = 14
    gradient_tol: float = .005
    fmax: float = .01
    max_step: float = .2
    relax_steps: int = 300
    fd_step: float = 1e-4
    rotation_hvp: int = 100
    rotation_tol: float = .02
    lbfgs_memory: int | None = None

    def __post_init__(self):
        from pamssw.relax import _validate_lbfgs_memory
        _validate_lbfgs_memory(self.lbfgs_memory, 'safe-lbfgs-total')
        for k in ('torsion_length','width','forward_force','gradient_tol','fmax','max_step','fd_step','rotation_tol'):
            v=getattr(self,k)
            if not np.isscalar(v) or not np.isfinite(v) or v<=0:raise ValueError(f'{k} must be positive finite')
        for k in ('rotation_bias','temperature_K'):
            v=getattr(self,k)
            if not np.isscalar(v) or not np.isfinite(v) or v<0:raise ValueError(f'{k} must be nonnegative finite')
        for k in ('max_gaussians','relax_steps','rotation_hvp'):
            v=getattr(self,k)
            if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<1:raise ValueError(f'{k} must be positive integer')


class TorsionSurface:
    """Exact reduced objective: root pose fixed; internal angles unwrapped.

    Free global pose is excluded from this invariant-PES proposal chart, rather
    than presented to the dimer as six chemically unproductive soft modes.
    """
    def __init__(self,chart,surface,torsion_length):
        if not np.isfinite(torsion_length) or torsion_length<=0:raise ValueError('positive torsion length required')
        self.chart=chart;self.surface=surface;self.length=torsion_length
        self.dimension=chart.dimension-6
        if self.dimension<1:raise ValueError('at least one internal torsion required')

    def coordinates(self,x):
        x=np.asarray(x,dtype=float)
        if x.shape!=(self.dimension,) or not np.isfinite(x).all():raise ValueError('finite torsion vector required')
        return np.r_[np.zeros(6),x/self.length]

    def atoms(self,x):return self.chart.evaluate(self.coordinates(x))[0]

    def evaluate(self,x):
        a,J=self.chart.evaluate(self.coordinates(x));e,f=self.surface.evaluate(a)
        f=np.asarray(f,dtype=float)
        if not np.isfinite(e) or f.shape!=a.positions.shape or not np.isfinite(f).all():raise ValueError('invalid physical E/F')
        return float(e),-np.einsum('ijk,ij->k',J[:,:,6:],f)/self.length


@dataclass
class RCSSWResult:
    initial: object
    current: object
    best: object
    minima: list
    records: list
    requests: int
    status: str


def run_rc_ssw(atoms,surface,*,bodies,parents,joints,steps,config,rng):
    """Torsion-space SSW -> unrestricted true Cartesian quench -> outer MC.

    surface supplies counted evaluate(Atoms)->E,F. Config step/angle metric is
    explicit. A reference chart is rebuilt from each selected full-atom minimum,
    never during a proposal. All certified rejected landings remain in minima;
    these are force-certified candidates, not Hessian/chemical certificates.
    """
    if not isinstance(config,RCSSWConfig):raise TypeError('RCSSWConfig required')
    if isinstance(steps,(bool,np.bool_)) or not isinstance(steps,(int,np.integer)) or steps<0:raise ValueError('nonnegative integer steps required')
    def factory(a):
        return TorsionSurface(RigidChainChart(a,bodies,parents=parents,joints=joints),surface,config.torsion_length)
    return _run_reduced_ssw(atoms,surface,steps=steps,config=config,rng=rng,factory=factory)


def _run_reduced_ssw(atoms,surface,*,steps,config,rng,factory,coordinate_label="scaled_torsions",quench_callback=None,rotation_indices=None,ls_runtime=None,rotation_callback=None, gaussian_policy=None, resume_state=None, boundary_callback=None, result_factory=RCSSWResult, direction_lifecycle=None):
    """Shared exact-coordinate climbing lifecycle; factory rebuilds each chart."""
    factory(atoms)  # reject invalid geometry before spending oracle requests
    rotation_settings = (direction_lifecycle.rotation_settings if direction_lifecycle is not None
                         else getattr(config, "recovered_rotation", None))
    begin=surface.requests;records=[];minima=[];initial=current=best=None;run_status='completed';next_index=0
    prior_requests=0
    if resume_state is not None:
        initial=resume_state['initial']; current=resume_state['current']; best=resume_state['best']
        minima=list(resume_state['minima']); records=list(resume_state['records'])
        next_index=int(resume_state['next_index']); prior_requests=int(resume_state['evaluation_requests'])
    def finish(status):
        if boundary_callback is not None:
            boundary_callback(initial,current,best,minima,records,next_index,status)
        return result_factory(initial,current,best,minima,records,prior_requests + surface.requests-begin,status)
    def full_quench(a):
        if quench_callback is not None:return quench_callback(a)
        return quench(a,surface,fmax=config.fmax,steps=config.relax_steps,
                      optimizer='safe-lbfgs-total',
                      lbfgs_memory=getattr(config, 'lbfgs_memory', None))
    if resume_state is None:
      try:initial=full_quench(atoms)
      except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as e:
        records.append(dict(stage='initial',status='evaluation_failed',error=str(e),requests=surface.requests-begin));return finish('initial_quench_failed')
    if resume_state is None:
      records.append(dict(stage='initial',status='converged' if initial.converged else 'quench_failed',landing=initial,requests=surface.requests-begin))
      if not initial.converged:return finish('initial_quench_failed')
      current=best=initial;minima.append(initial)
    if direction_lifecycle is not None and resume_state is None:
        try:
            direction_lifecycle.initialize(atoms, initial.atoms, rng)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            records.append(dict(stage='direction_initialize', status='direction_initialization_failed',
                                error=str(error), requests=0))
            return finish('direction_initialization_failed')
    if ls_runtime is not None and resume_state is None:
        try:
            ls_runtime.initialize_at(initial.atoms)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            records.append(dict(stage='ls_initialize',status='ls_initialization_failed',error=str(error),requests=0))
            return finish('ls_initialization_failed')
    for index in range(next_index,next_index+steps):
        before=surface.requests;event=dict(index=index,status='running',accepted=False,landing=None,climb=[],requests=0,
            chart_reference=current.atoms.copy(),last_work=current.atoms.copy())
        records.append(event)
        prepared=None
        try:
            reduced=factory(current.atoms)
            if ls_runtime is not None:
                try:
                    prepared=ls_runtime.prepare(current.atoms, reduced.chart,
                        fmax=config.fmax, max_step=config.max_step,
                        maxiter=config.relax_steps,
                        lbfgs_memory=getattr(config, 'lbfgs_memory', None))
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                    event.update(status='ls_prequench_failed',error=str(error),requests=surface.requests-before)
                    event['ls_preparation']=getattr(error,'result',None)
                    if event['ls_preparation'] is not None:
                        event['last_work']=event['ls_preparation'].atoms.copy()
                    event['requests']=surface.requests-before
                    next_index=index+1; return finish('ls_prequench_failed')
                event['ls_preparation']=prepared
                event['last_work']=prepared.atoms.copy()
                if not prepared.optimizer.converged:
                    event.update(status='ls_prequench_failed',requests=surface.requests-before)
                    next_index=index+1; return finish('ls_prequench_failed')
            work=np.zeros(reduced.dimension)
            if prepared is not None and hasattr(reduced, 'chart') and hasattr(reduced.chart, 'active_indices'):
                active=np.asarray(reduced.chart.active_indices, dtype=int)
                work=(prepared.atoms.positions[active]-current.atoms.positions[active]).ravel()
            if direction_lifecycle is None:
                anchor=rng.normal(size=reduced.dimension);anchor/=np.linalg.norm(anchor)
            else:
                anchor=np.zeros(reduced.dimension)
            terms=[]
            policy_terms=[]
            event.update(chart_reference=current.atoms.copy(),last_work=(current.atoms.copy() if prepared is None else prepared.atoms.copy()),frozen_gaussians=[])
            if rotation_indices is not None:
                if direction_lifecycle is None:
                    anchor_masked=np.zeros_like(anchor)
                    anchor_masked[rotation_indices]=anchor[rotation_indices]
                    anchor=anchor_masked/np.linalg.norm(anchor_masked)
                event['rotation_coordinate_indices']=rotation_indices.copy()
            def biased(x):
                e,g=reduced.evaluate(x)
                if gaussian_policy is None:
                    for center,n,w in terms:
                        z=float((x-center)@n);v=w*math.exp(-.5*(z/config.width)**2)
                        e+=v;g-=v*z/config.width**2*n
                else:
                    for term in policy_terms:
                        center=term.center.ravel(); n=term.direction.ravel(); width=term.sigma
                        z=float((x-center)@n);v=term.weight*math.exp(-.5*(z/width)**2)
                        e+=v;g-=v*z/width**2*n
                return e,g
            status='gaussian_limit'
            for j in range(config.max_gaussians):
                stage=dict(index=j,status='running');event['climb'].append(stage);stage_before=surface.requests
                if direction_lifecycle is not None:
                    anchor, release, diagnostic = direction_lifecycle.propose(
                        current.atoms, reduced, work, first=(j == 0), rng=rng)
                    stage['recovered_direction'] = diagnostic
                    stage['initial_direction'] = anchor.copy()
                    if release:
                        stage.update(status='direction_zero_release', requests=surface.requests-stage_before)
                        status='stage_release'
                        break
                rotation_kwargs=dict(rotation_bias=config.rotation_bias,fd_step=config.fd_step,max_hvp=config.rotation_hvp,tol=config.rotation_tol)
                if rotation_callback is not None:
                    if rotation_indices is None:
                        mode=rotation_callback(work,anchor,evaluate=reduced.evaluate,
                                               rotation_bias=config.rotation_bias,fd_step=config.fd_step,
                                               max_hvp=config.rotation_hvp,tol=config.rotation_tol,
                                               pre_rotation_hvp=getattr(config,'pre_rotation_hvp',None),
                                               rotation_solver=getattr(config,'rotation_solver',None),
                                               recovered_rotation=rotation_settings)
                    else:
                        def restricted_evaluate(z):
                            lifted=work.copy(); lifted[rotation_indices]=z
                            energy, gradient=reduced.evaluate(lifted)
                            return energy, gradient[rotation_indices]
                        submode=rotation_callback(work[rotation_indices], anchor[rotation_indices],
                            evaluate=restricted_evaluate, rotation_bias=config.rotation_bias,
                            fd_step=config.fd_step, max_hvp=config.rotation_hvp,
                            tol=config.rotation_tol,
                            pre_rotation_hvp=getattr(config,'pre_rotation_hvp',None),
                            rotation_solver=getattr(config,'rotation_solver',None),
                            recovered_rotation=rotation_settings)
                        lifted=np.zeros_like(work); lifted[rotation_indices]=submode.direction
                        mode=replace(submode,direction=lifted)
                        if rotation_settings is not None:
                            lifted_reference=np.zeros_like(work)
                            lifted_reference[rotation_indices]=submode.bias_reference
                            mode=replace(mode,bias_reference=lifted_reference)
                        mode_info='selected_coordinate_subspace'
                        # Preserve the existing diagnostic schema used below.
                        if event is not None: event['rotation_coordinate_scope']=mode_info
                elif rotation_indices is None:
                    mode=generalized_dimer(work,anchor,evaluate=reduced.evaluate,**rotation_kwargs)
                else:
                    # Only the mode problem is restricted. Biased relaxation
                    # below continues to use the full conservative gradient.
                    def rotation_evaluate(z):
                        lifted=work.copy();lifted[rotation_indices]=z
                        e,g=reduced.evaluate(lifted)
                        return e,g[rotation_indices]
                    submode=generalized_dimer(work[rotation_indices],anchor[rotation_indices],evaluate=rotation_evaluate,**rotation_kwargs)
                    lifted=np.zeros_like(work);lifted[rotation_indices]=submode.direction
                    mode=replace(submode,direction=lifted)
                    stage['rotation_residual_scope']='selected_coordinate_subspace'
                stage['mode']=mode
                rotation_stop = getattr(mode, 'stop_reason', 'unspecified')
                recovered = rotation_settings is not None
                budget_released = (not mode.converged and
                    getattr(config, 'rotation_exit_policy', 'force') == 'force_or_budget' and
                    (rotation_stop == 'budget_exhausted' or
                     (recovered and rotation_stop in ('rotation_limit', 'force_budget'))))
                stage.update(rotation_stop_reason=rotation_stop,
                             rotation_converged=bool(mode.converged),
                             rotation_budget_released=bool(budget_released))
                if recovered:
                    stage['recovered_rotation'] = dict(
                        stage=getattr(mode, 'stage', None),
                        stage_complete=getattr(mode, 'stage_complete', None),
                        real_curvature=float(mode.real_curvature),
                        trace=getattr(mode, 'trace', ()),
                        force_calls=int(mode.force_calls),
                        bias_reference=np.asarray(mode.bias_reference).copy(),
                        rotation_weight=float(mode.rotation_weight))
                if not mode.converged and not budget_released:
                    stage.update(status='rotation_failed',requests=surface.requests-stage_before);status='rotation_failed';break
                if budget_released:
                    direction=np.asarray(mode.direction)
                    if (direction.shape != work.shape or not np.isfinite(direction).all() or
                            np.linalg.norm(direction) == 0 or not np.isfinite(mode.curvature) or
                            not np.isfinite(mode.residual_norm) or mode.residual_norm < 0):
                        raise ValueError('budget rotation returned invalid evaluated direction or certificate')
                actual_anchor=anchor
                actual_bias=config.rotation_bias
                if recovered:
                    actual_anchor=np.asarray(mode.bias_reference).copy()
                    actual_bias=(float(mode.rotation_weight)
                                 if getattr(mode, 'stage', None) == 'CBD_biasedRot' else 0.)
                if recovered:
                    stage['actual_anchor']=np.asarray(actual_anchor).copy()
                    stage['actual_rotation_bias']=float(actual_bias)
                if gaussian_policy is not None and getattr(config, 'pre_rotation_hvp', None) is not None:
                    actual_anchor=np.asarray(mode.pre.direction, dtype=float)
                    if rotation_indices is not None:
                        lifted=np.zeros_like(work);lifted[rotation_indices]=actual_anchor.ravel()
                        actual_anchor=lifted
                    actual_bias=float(mode.bias_curvature)
                if gaussian_policy is not None:
                    shape=(work.size//3,3)
                    policy_mode=replace(mode,direction=np.asarray(mode.direction).reshape(shape))
                    policy_anchor=np.asarray(actual_anchor).reshape(shape)
                    policy_center=np.asarray(work).reshape(shape)
                    policy_data=gaussian_policy.choose(mode=policy_mode, anchor=policy_anchor,
                        center=policy_center, terms=tuple(policy_terms), base_width=config.width,
                        rotation_bias=actual_bias)
                    if not isinstance(policy_data, dict): raise TypeError('gaussian policy must return a dict')
                    width=float(policy_data['width']); w=float(policy_data['weight'])
                    if not np.isfinite(width) or width<=0 or not np.isfinite(w) or w<0:
                        raise ValueError('gaussian policy returned invalid width/weight')
                    stage['gaussian_policy']=policy_data
                else:
                    displaced=work+config.width*mode.direction;_,g=biased(displaced)
                    width=config.width; w=float((config.forward_force+g@mode.direction)*width*math.exp(.5))
                stage['weight']=w
                if gaussian_policy is None and (not np.isfinite(w) or w<=0):
                    stage.update(status='nonpositive_height',requests=surface.requests-stage_before);status='nonpositive_height';break
                if gaussian_policy is None:
                    terms.append((work.copy(),mode.direction.copy(),w))
                else:
                    from .gaussian import ProjectedGaussian
                    term=ProjectedGaussian(work.reshape(-1,3).copy(),mode.direction.reshape(-1,3).copy(),width,w)
                    policy_terms.append(term)
                if direction_lifecycle is not None:
                    direction_lifecycle.save_center(reduced, work)
                event['frozen_gaussians'].append(dict(center=work.copy(),direction=mode.direction.copy(),weight=w,width=width))
                displaced=work+width*mode.direction
                relaxed=safe_lbfgs(displaced,biased,gradient_norm=np.linalg.norm,step_norm=np.linalg.norm,gtol=config.gradient_tol,max_step=config.max_step,maxiter=config.relax_steps,lbfgs_memory=getattr(config, 'lbfgs_memory', None))
                work=relaxed.q.copy();stage.update(status=relaxed.status,relaxation=relaxed,requests=surface.requests-stage_before);stage[coordinate_label]=work.copy()
                event['last_work']=reduced.atoms(work)
                if not relaxed.converged:status='biased_quench_failed';break
                landing_atoms=reduced.atoms(work)
                e,_=(reduced.evaluate(work) if ls_runtime is None else surface.evaluate(landing_atoms))
                stage.update(true_energy=e,requests=surface.requests-stage_before)
                if e<current.energy:status='lower_true_energy';break
            event['last_work']=reduced.atoms(work)
            if status in ('gaussian_limit','lower_true_energy','stage_release'):
                landing=full_quench(event['last_work']);event['landing']=landing
                if not landing.converged:status='true_quench_failed'
                else:
                    if direction_lifecycle is not None:
                        # Qualified structures survive failure of the next-axis
                        # selection. That failure is terminal, not an MC reject.
                        minima.append(landing)
                        if landing.energy < best.energy:
                            best = landing
                        try:
                            direction_lifecycle.observe(landing.atoms, rng)
                        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
                            event.update(status='direction_selection_failed', error=str(error),
                                         requests=surface.requests-before)
                            next_index=index+1
                            return finish('direction_selection_failed')
                    else:
                        minima.append(landing)
                    delta=landing.energy-current.energy
                    accepted=delta<=0 or (config.temperature_K>0 and rng.random()<math.exp(-delta/(units.kB*config.temperature_K)))
                    if landing.energy<best.energy:best=landing
                    if accepted:current=landing
                    event.update(accepted=bool(accepted),delta=delta);status='valid_landing'
            if ls_runtime is not None and prepared is not None:
                try:
                    ls_runtime.update(current.atoms,
                        energy_before=prepared.energy_before,
                        energy_after=prepared.energy_after)
                    event['ls_update']='updated'
                    native = getattr(ls_runtime, 'native', None)
                    if native is not None and native.last_update is not None:
                        event['native_ls_update'] = copy.deepcopy(native.last_update)
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                    event.update(ls_update='ls_update_failed',ls_update_error=str(error))
                    status='ls_update_failed';run_status='ls_update_failed'
            event['status']=status
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as e:
            event.update(status='evaluation_failed',error=str(e))
            if event['climb'] and event['climb'][-1]['status']=='running':
                event['climb'][-1].update(status='evaluation_failed',requests=surface.requests-stage_before,error=str(e))
            if ls_runtime is not None and prepared is not None:
                try:
                    ls_runtime.update(current.atoms, energy_before=prepared.energy_before,
                                      energy_after=prepared.energy_after)
                    event['ls_update']='updated_after_exception'
                    native = getattr(ls_runtime, 'native', None)
                    if native is not None and native.last_update is not None:
                        event['native_ls_update'] = copy.deepcopy(native.last_update)
                except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
                    event.update(ls_update='ls_update_failed',ls_update_error=str(error))
                    run_status='ls_update_failed'
        event['requests']=surface.requests-before
        next_index=index+1
        if boundary_callback is not None:
            boundary_callback(initial,current,best,minima,records,next_index,
                             run_status if run_status != 'completed' else 'completed')
        if run_status != 'completed':
            return finish(run_status)
    return finish('completed_with_failures' if any(r['status']!='valid_landing' for r in records[1:]) else 'completed')
