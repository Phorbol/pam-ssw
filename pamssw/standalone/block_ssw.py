"""Cell/atomic block SSW inspired by 2014 SSW-crystal and native geometry.

Not an exact native CBD/Broyden or release scheduling reproduction. Explicit
cycle counts and conventional derivatives replace ambiguous printed notation.
"""
from dataclasses import dataclass
import math
import numpy as np
from ase import units
from .cbd_cell import CellChart, cell_direction
from .atomic_climb import atomic_climb
from .cell_relax import cell_quench
from .generalized_numerics import safe_lbfgs


@dataclass(frozen=True)
class BlockSSWConfig:
    atomic: object
    quench_length: float
    cell_cycles: int = 5
    atomic_period: int = 2
    cell_step_fraction: float = .15
    cell_step_metric: str = 'lattice_frobenius'
    cell_fd_step: float = .005
    cell_rotation_requests: int = 6
    cell_rotation_force_tol: float = .1
    partial_atom_steps: int = 25
    pressure: float = 0.
    stress_tol: float = .001
    max_step: float = .2
    atomic_gaussian_policy: object | None = None
    partial_atom_fmax: float | None = None
    def __post_init__(self):
        if self.partial_atom_fmax is not None and (not np.isfinite(self.partial_atom_fmax) or self.partial_atom_fmax <= 0):
            raise ValueError('partial_atom_fmax must be positive and finite when set')
        if self.atomic_gaussian_policy is not None:
            from .pam_gaussian import PAMCurvatureGaussian
            if not isinstance(self.atomic_gaussian_policy, PAMCurvatureGaussian):
                raise TypeError('atomic_gaussian_policy must be PAMCurvatureGaussian')
        for name in ('quench_length','cell_step_fraction','cell_fd_step',
                     'cell_rotation_force_tol','stress_tol','max_step'):
            value=getattr(self,name)
            if not np.isfinite(value) or value<=0:raise ValueError(f'{name} must be positive')
        for name in ('cell_cycles','atomic_period','cell_rotation_requests','partial_atom_steps'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,int) or value<1:raise ValueError(f'{name} must be a positive integer')
        if self.cell_rotation_requests<2:raise ValueError('cell rotation needs center and one image')
        if self.cell_step_metric not in ('lattice_frobenius', 'deformation_rms'):
            raise ValueError('cell_step_metric must be lattice_frobenius or deformation_rms')
        if not np.isfinite(self.pressure):raise ValueError('finite pressure required')
        if (self.atomic.cluster_frame!='translation_only' or self.atomic.direction_sampling!='global'
            or self.atomic.quench_optimizer!='safe-lbfgs-total'):
            raise ValueError('block atomic kernel requires translation_only/global/Safe-total')


@dataclass
class BlockSSWResult:
    initial: object
    current: object
    best: object
    minima: list
    records: list
    requests: int
    status: str


class FixedCellSurface:
    def __init__(self, surface):self.surface=surface
    @property
    def requests(self):return self.surface.requests
    def evaluate(self, atoms):
        e,f,_=self.surface.evaluate(atoms)
        return e,f


def _cell_displacement(lattice, direction, fraction, metric):
    """Return a cell displacement for an explicit entry or relative metric.

    ``direction`` is expected to be a unit nine-entry mode from
    ``cell_direction``; the zero case is rejected to avoid an undefined
    relative scale.
    """
    lattice = np.asarray(lattice, dtype=float)
    direction = np.asarray(direction, dtype=float).reshape(-1)
    if lattice.shape != (3, 3) or direction.shape != (9,):
        raise ValueError('lattice and direction shapes are invalid')
    if not np.isfinite(lattice).all() or not np.isfinite(direction).all():
        raise ValueError('lattice and direction must be finite')
    if np.linalg.norm(direction) == 0:
        raise ValueError('direction must be nonzero')
    if not np.isfinite(fraction) or fraction <= 0:
        raise ValueError('fraction must be finite and positive')
    if metric == 'lattice_frobenius':
        distance = fraction * np.linalg.norm(lattice)
    elif metric == 'deformation_rms':
        relative_direction = np.linalg.solve(lattice, direction.reshape(3, 3))
        distance = np.sqrt(3.) * fraction / np.linalg.norm(relative_direction)
    else:
        raise ValueError('cell_step_metric must be lattice_frobenius or deformation_rms')
    return distance * direction


def run_block_ssw(atoms, surface, *, steps, config, rng):
    """Sequential cell block -> optional atomic climb -> joint quench -> MC.

    One-based outer steps divisible by atomic_period include atomic climbing;
    cell_cycles is the actual number performed, not an array-index endpoint.
    A cell mode at its request budget is usable and explicitly marked approximate.
    Partial atomic relaxation may reach maxiter; numerical failures are rejected.
    At fixed cell the climbing threshold is H_initial-p*V_cell, so its stopping
    comparison is consistent with the outer enthalpy reference even at p!=0.
    """
    if isinstance(steps,bool) or not isinstance(steps,int) or steps<0:raise ValueError('nonnegative integer steps required')
    CellChart(atoms)  # Validate before requesting the oracle.
    begin=surface.requests
    fixed=FixedCellSurface(surface)
    def full_quench(a):
        return cell_quench(a,surface,strain_length=config.quench_length,
            pressure=config.pressure,fmax=config.atomic.fmax,stress_tol=config.stress_tol,
            max_step=config.max_step,maxiter=config.atomic.relax_steps,
            lbfgs_memory=config.atomic.lbfgs_memory)
    initial=full_quench(atoms)
    records=[dict(stage='initial',quench=initial,requests=surface.requests-begin)]
    current=best=None
    minima=[]
    if not initial.converged:
        return BlockSSWResult(initial,current,best,minima,records,surface.requests-begin,'initial_quench_failed')
    current=best=initial.evaluation
    minima.append(current)
    for i in range(steps):
        before=surface.requests
        work=current.atoms.copy()
        event=dict(index=i,atomic_scheduled=(i+1)%config.atomic_period==0,
                   cell_cycles=[],atomic=None,landing=None,accepted=False,status='running')
        try:
            for j in range(config.cell_cycles):
                cycle_before=surface.requests
                cycle=dict(index=j,status='running',requests=0,
                           direction_source='fresh_random_each_cycle_unverified_native_policy')
                event['cell_cycles'].append(cycle)
                chart=CellChart(work);q=chart.pack(work)
                mode=cell_direction(chart,q,rng.normal(size=9),evaluate=surface.evaluate,
                    pressure=config.pressure,fd_step=config.cell_fd_step,
                    max_hvp=config.cell_rotation_requests-1,
                    rotation_force_tol=config.cell_rotation_force_tol)
                cycle['mode']=mode
                displacement = _cell_displacement(work.cell.array, mode.direction,
                    config.cell_step_fraction, config.cell_step_metric)
                distance = float(np.linalg.norm(displacement))
                displaced=chart.unpack(q+displacement)
                trial=displaced.copy()
                def evaluate(x):
                    trial.positions=x.reshape(-1,3)
                    e,f=fixed.evaluate(trial)
                    return e,-f.ravel()
                norm=lambda x:float(np.linalg.norm(x.reshape(-1,3),axis=1).max())
                partial_fmax = config.atomic.fmax if config.partial_atom_fmax is None else config.partial_atom_fmax
                relaxed=safe_lbfgs(displaced.positions.ravel(),evaluate,
                    gradient_norm=norm,step_norm=norm,gtol=partial_fmax,
                    max_step=config.max_step,maxiter=config.partial_atom_steps,
                    lbfgs_memory=config.atomic.lbfgs_memory)
                work=displaced.copy();work.positions=relaxed.q.reshape(-1,3)
                delta_cell = displaced.cell.array - chart.reference.cell.array
                relative_cell = np.linalg.solve(chart.reference.cell.array, delta_cell)
                cycle.update(index=j,status='completed',distance=distance,
                    cell_step_metric=config.cell_step_metric,
                    deformation_rms=float(np.linalg.norm(relative_cell)/np.sqrt(3.)),
                    principal_stretches=np.linalg.svd(
                        np.eye(3) + relative_cell, compute_uv=False).tolist(),
                    cell_before=chart.reference.cell.array.copy(),cell_after=work.cell.array.copy(),
                    mode=mode,partial_status=relaxed.status,partial_steps=relaxed.steps,
                    partial_error=relaxed.error,requests=surface.requests-cycle_before)
                if config.partial_atom_fmax is not None:
                    cycle['partial_atom_fmax'] = partial_fmax
                if relaxed.status not in ('converged','maxiter'):
                    cycle['status']='partial_atom_relax_failed'
                    event['status']='partial_atom_relax_failed';break
            if event['status']=='running' and event['atomic_scheduled']:
                atomic_kwargs = ({'gaussian_policy': config.atomic_gaussian_policy}
                                 if config.atomic_gaussian_policy is not None else {})
                result=atomic_climb(work,fixed,
                    reference_energy=current.objective-config.pressure*work.get_volume(),
                    config=config.atomic,rng=rng,**atomic_kwargs)
                event['atomic']=result;work=result.atoms.copy()
                if result.status not in ('gaussian_limit','lower_true_energy'):
                    event['status']='atomic_'+result.status
            if event['status']=='running':
                landing=full_quench(work);event['landing']=landing
                if not landing.converged:event['status']='true_quench_failed'
                else:
                    ev=landing.evaluation;minima.append(ev)
                    delta=ev.objective-current.objective
                    temperature=config.atomic.temperature_K
                    accepted=delta<=0 or (temperature>0 and rng.random()<math.exp(-delta/(units.kB*temperature)))
                    if ev.objective<best.objective:best=ev
                    if accepted:current=ev
                    event.update(status='valid_landing',accepted=bool(accepted),delta=delta)
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:
            if event['cell_cycles'] and event['cell_cycles'][-1]['status']=='running':
                event['cell_cycles'][-1].update(status='evaluation_failed',
                    error=str(error),requests=surface.requests-cycle_before)
            event.update(status='evaluation_failed',error=str(error))
        event['last_work']=work.copy();event['requests']=surface.requests-before;records.append(event)
    return BlockSSWResult(initial,current,best,minima,records,surface.requests-begin,'completed')
