"""Fixed-substrate Cartesian SSW with explicit constrained certificates."""
from dataclasses import dataclass, replace, fields, is_dataclass, MISSING
from copy import deepcopy
import os
from pathlib import Path
import pickle
import tempfile
import numpy as np
from ase.constraints import FixAtoms
from .ase_constraints import normalize_constraints, bind_hookean_surface, HookeanSurface
from .generalized_numerics import safe_lbfgs
from .rc_reference import _run_reduced_ssw
from .paper_reference import LSSettings
from .recovered_rotation import RecoveredRotationSettings
from .constrained_ls import (ConstrainedLSRuntime, ConstrainedNativeLSRuntime,
                              ConstrainedLSSurface)


@dataclass(frozen=True)
class ConstrainedSSWConfig:
    width: float  # Angstrom in active Cartesian displacement coordinates
    rotation_bias: float | None  # eV/Angstrom^2; None only for staged presweep
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
    rotation_solver: str = 'generalized-dimer'
    pre_rotation_hvp: int | None = None
    recovered_rotation: RecoveredRotationSettings | None = None
    rotation_exit_policy: str = 'force'

    def __post_init__(self):
        from pamssw.relax import _validate_lbfgs_memory
        _validate_lbfgs_memory(self.lbfgs_memory, 'safe-lbfgs-total')
        for k in ('width','forward_force','gradient_tol','fmax','max_step','fd_step','rotation_tol'):
            v=getattr(self,k)
            if not np.isscalar(v) or not np.isfinite(v) or v<=0:raise ValueError(f'{k} must be positive finite')
        for k in ('temperature_K',):
            v=getattr(self,k)
            if not np.isscalar(v) or not np.isfinite(v) or v<0:raise ValueError(f'{k} must be nonnegative finite')
        if self.rotation_bias is not None and (not np.isscalar(self.rotation_bias) or
                not np.isfinite(self.rotation_bias) or self.rotation_bias < 0):
            raise ValueError('rotation_bias must be nonnegative finite or None')
        for k in ('max_gaussians','relax_steps','rotation_hvp'):
            v=getattr(self,k)
            if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<1:raise ValueError(f'{k} must be positive integer')
        if self.rotation_solver not in ('generalized-dimer','ritz','dimer','broyden-euclidean'):
            raise ValueError('rotation_solver must be generalized-dimer, ritz, dimer or broyden-euclidean')
        if self.pre_rotation_hvp is not None and (isinstance(self.pre_rotation_hvp,(bool,np.bool_)) or
                not isinstance(self.pre_rotation_hvp,(int,np.integer)) or self.pre_rotation_hvp < 1):
            raise ValueError('pre_rotation_hvp must be a positive integer or None')
        if self.pre_rotation_hvp is not None and self.rotation_solver == 'generalized-dimer':
            raise ValueError('pre_rotation_hvp requires an explicit shared rotation solver')
        if self.pre_rotation_hvp is not None and self.rotation_bias is not None:
            raise ValueError('pre_rotation_hvp requires rotation_bias=None')
        if self.pre_rotation_hvp is None and self.rotation_bias is None:
            raise ValueError('rotation_bias=None requires pre_rotation_hvp')
        if self.rotation_exit_policy not in ('force', 'force_or_budget'):
            raise ValueError('rotation_exit_policy must be force or force_or_budget')
        if self.recovered_rotation is not None:
            if not isinstance(self.recovered_rotation, RecoveredRotationSettings):
                raise TypeError('recovered_rotation must be RecoveredRotationSettings')
            if self.pre_rotation_hvp is not None:
                raise ValueError('recovered_rotation and pre_rotation_hvp are mutually exclusive')
        minimum_hvp = 2 if self.rotation_solver == 'ritz' else 1
        if self.rotation_hvp < minimum_hvp:
            raise ValueError('rotation_hvp is too small for selected solver')
        if self.pre_rotation_hvp is not None and (self.rotation_hvp < 2 + minimum_hvp or
                self.pre_rotation_hvp > self.rotation_hvp - minimum_hvp - 1):
            raise ValueError('rotation_hvp does not cover presweep and main solver budgets')


def _active_rotation_callback(work, anchor, *, evaluate, rotation_bias, fd_step,
                              max_hvp, tol, pre_rotation_hvp=None,
                              rotation_solver=None, recovered_rotation=None):
    """Adapt active q vectors to the public atom-shaped direction solvers."""
    from ase import Atoms
    container = Atoms('H' * len(np.asarray(work).reshape(-1, 3)),
                      positions=np.asarray(work, dtype=float).reshape(-1, 3))
    def atom_evaluate(candidate):
        energy, gradient = evaluate(np.asarray(candidate.positions, dtype=float).reshape(-1))
        return energy, -np.asarray(gradient, dtype=float).reshape(candidate.positions.shape)
    anchor = np.asarray(anchor, dtype=float).reshape(container.positions.shape)
    if recovered_rotation is not None:
        from .recovered_cbd import recovered_cbd_direction
        result = recovered_cbd_direction(container, anchor, fd_step=fd_step,
            max_force_calls=recovered_rotation.max_force_calls,
            pre_rotmax=recovered_rotation.pre_rotmax,
            rotmax=recovered_rotation.rotmax,
            pre_ftol=recovered_rotation.pre_ftol,
            ftol=recovered_rotation.ftol,
            metric=recovered_rotation.metric, evaluate=atom_evaluate)
        return replace(result, direction=result.direction.ravel(),
                       bias_reference=result.bias_reference.ravel())
    if pre_rotation_hvp is not None:
        from .staged_direction import two_stage_dimer_direction
        result = two_stage_dimer_direction(container, anchor, fd_step=fd_step,
            max_hvp=max_hvp, pre_rotation_hvp=pre_rotation_hvp, tol=tol,
            evaluate=atom_evaluate, main_solver=rotation_solver or 'ritz')
        return replace(result, direction=result.direction.ravel())
    if rotation_solver == 'dimer':
        from .dimer import paper_dimer_direction
        solver = paper_dimer_direction
    elif rotation_solver == 'broyden-euclidean':
        from .broyden_direction import paper_broyden_direction
        solver = paper_broyden_direction
    else:
        from .direction import paper_biased_direction
        solver = paper_biased_direction
    result = solver(container, anchor, rotation_bias=rotation_bias, fd_step=fd_step,
                  max_hvp=max_hvp, tol=tol, evaluate=atom_evaluate)
    return replace(result, direction=result.direction.ravel())


class ReducedCartesianChart:
    """Active Cartesian displacements; fixed atoms and cell remain exact.

    Supports full, partial or no PBC. FixAtoms and Hookean are normalized at the
    driver boundary; oracle Atoms are constraint-free so raw forces remain visible. The chart
    itself receives cleaned Atoms. Explicit
    fixed_indices must agree with any existing FixAtoms, not override them.
    No global translation/rotation removal is valid for a fixed substrate.
    """
    def __init__(self,atoms,*,fixed_indices=None,allow_no_fixed=False):
        if not len(atoms) or not np.isfinite(atoms.positions).all() or not np.isfinite(atoms.cell.array).all():
            raise ValueError('finite nonempty reference required')
        present=set()
        for constraint in atoms.constraints:
            if not isinstance(constraint,FixAtoms):raise ValueError('only FixAtoms constraints supported')
            present.update(map(int,constraint.get_indices()))
        if fixed_indices is None:fixed=sorted(present)
        else:
            fixed=list(fixed_indices)
            if any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or i<0 or i>=len(atoms) for i in fixed) or len(set(fixed))!=len(fixed):raise ValueError('unique valid fixed indices required')
            if present and set(fixed)!=present:raise ValueError('explicit fixed indices must agree with FixAtoms')
            fixed=sorted(fixed)
        if any(i<0 or i>=len(atoms) for i in fixed):raise ValueError('valid fixed indices required')
        if len(fixed)>=len(atoms) or (not fixed and not allow_no_fixed):raise ValueError('at least one fixed and one active atom required')
        self.fixed_indices=np.array(fixed,dtype=int)
        self.active_indices=np.array([i for i in range(len(atoms)) if i not in fixed],dtype=int)
        self.reference=atoms.copy();self.reference.set_constraint();self.reference.calc=None
        self.dimension=3*len(self.active_indices)

    def atoms(self,q):
        q=np.asarray(q,dtype=float)
        if q.shape!=(self.dimension,) or not np.isfinite(q).all():raise ValueError('finite active displacement vector required')
        a=self.reference.copy();a.positions[self.active_indices]+=q.reshape(-1,3)
        return a

    def evaluate(self,q,surface):
        a=self.atoms(q);e,f=surface.evaluate(a);f=np.asarray(f,dtype=float)
        if not np.isfinite(e) or f.shape!=a.positions.shape or not np.isfinite(f).all():raise ValueError('invalid raw E/F')
        return float(e),-f[self.active_indices].ravel().copy()


class _ReducedSurface:
    def __init__(self,chart,surface):self.chart=chart;self.surface=surface;self.dimension=chart.dimension
    def atoms(self,q):return self.chart.atoms(q)
    def evaluate(self,q):return self.chart.evaluate(q,self.surface)


@dataclass
class ConstrainedQuenchResult:
    atoms: object
    energy: float | None
    active_fmax: float | None
    full_raw_fmax: float | None
    optimizer: object
    certificate: dict
    requests: int

    @property
    def converged(self):return self.optimizer.converged and self.certificate.get('certified',False)


@dataclass
class ConstrainedCheckpoint:
    initial: object
    current: object
    best: object
    minima: tuple
    records: tuple
    chart_reference: object
    fixed_indices: tuple
    direction_fixed_indices: tuple
    config: ConstrainedSSWConfig
    ls_settings: object
    ls_state: object
    rng_state: object
    evaluation_requests: int
    next_index: int
    status: str
    schema_version: int = 1
    hookean_specs: tuple = ()
    gaussian_policy: object = None


@dataclass(frozen=True)
class ConstrainedSSWResult:
    initial: object
    current: object
    best: object
    minima: list
    records: list
    requests: int
    status: str
    checkpoint: object = None


def _constrained_copy(value):
    """Copy checkpoint data while stripping calculators from every Atoms."""
    from ase import Atoms
    if isinstance(value, Atoms):
        result=value.copy(); result.calc=None; return result
    if isinstance(value, dict): return {k:_constrained_copy(v) for k,v in value.items()}
    if isinstance(value, list): return [_constrained_copy(v) for v in value]
    if isinstance(value, tuple): return tuple(_constrained_copy(v) for v in value)
    if is_dataclass(value):
        return replace(value, **{f.name:_constrained_copy(getattr(value,f.name)) for f in fields(value) if f.init})
    return deepcopy(value)


def _restore_constraints(value, constraints):
    """Restore user constraints on known Atoms nested in result records."""
    from ase import Atoms
    if isinstance(value, Atoms):
        return constraints.attach(value)
    if isinstance(value, dict):
        return {k: _restore_constraints(v, constraints) for k, v in value.items()}
    if isinstance(value, list):
        return [_restore_constraints(v, constraints) for v in value]
    if isinstance(value, tuple):
        return tuple(_restore_constraints(v, constraints) for v in value)
    if is_dataclass(value):
        return replace(value, **{f.name: _restore_constraints(getattr(value, f.name), constraints)
                                 for f in fields(value) if f.init})
    return value


def save_constrained_checkpoint(path, checkpoint):
    if not isinstance(checkpoint, ConstrainedCheckpoint):
        raise TypeError('checkpoint must be ConstrainedCheckpoint')
    target=Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.tmp', dir=target.parent)
    try:
        with os.fdopen(fd,'wb') as handle:
            pickle.dump(_constrained_copy(checkpoint),handle,protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp,target)
    except BaseException:
        try: os.unlink(tmp)
        except FileNotFoundError: pass
        raise


def _constrained_signature(value):
    from .ls_prequench import normalize_prequench_settings
    return pickle.dumps(normalize_prequench_settings(value), protocol=4)


def _constrained_config_equal(saved, requested):
    """Compare config fields while allowing schema-1 pickles to lack new defaults."""
    if not isinstance(saved, ConstrainedSSWConfig) or not isinstance(requested, ConstrainedSSWConfig):
        return False
    for field in fields(ConstrainedSSWConfig):
        default = None if field.default is MISSING else field.default
        if getattr(saved, field.name, default) != getattr(requested, field.name, default):
            return False
    return True


def _validate_constrained_checkpoint(checkpoint):
    if not isinstance(checkpoint,ConstrainedCheckpoint): raise TypeError('checkpoint must be ConstrainedCheckpoint')
    if checkpoint.schema_version != 1: raise ValueError(f'unsupported constrained checkpoint schema {checkpoint.schema_version!r}')
    if not isinstance(checkpoint.rng_state,dict) or checkpoint.rng_state.get('bit_generator') is None:
        raise ValueError('checkpoint RNG state is invalid')
    if checkpoint.next_index < 0: raise ValueError('checkpoint next_index must be nonnegative')
    attempts=[r.get('index') for r in checkpoint.records if isinstance(r,dict) and 'index' in r]
    if tuple(attempts) != tuple(range(checkpoint.next_index)):
        raise ValueError('checkpoint records are not contiguous from index zero')
    accounted=sum(int(r.get('requests',0)) for r in checkpoint.records if isinstance(r,dict))
    if checkpoint.evaluation_requests < 0 or checkpoint.evaluation_requests != accounted:
        raise ValueError('checkpoint evaluation request count disagrees with records')
    if checkpoint.status not in ('completed','completed_with_failures'):
        return
    if checkpoint.evaluation_requests < 0: raise ValueError('checkpoint evaluation requests must be nonnegative')


def load_constrained_checkpoint(path):
    with Path(path).open('rb') as handle: checkpoint=pickle.load(handle)
    _validate_constrained_checkpoint(checkpoint)
    return checkpoint


def constrained_quench(atoms,surface,*,fixed_indices=None,fmax,max_step,maxiter,lbfgs_memory=None):
    """True quench on the same fixed-atom manifold, with fresh raw E/F check."""
    from pamssw.relax import _validate_lbfgs_memory
    _validate_lbfgs_memory(lbfgs_memory, 'safe-lbfgs-total')
    if not np.isscalar(fmax) or not np.isfinite(fmax) or fmax<=0:raise ValueError('positive finite fmax required')
    constraints=normalize_constraints(atoms, fixed_indices=fixed_indices)
    clean=constraints.clean_atoms(atoms)
    objective=bind_hookean_surface(surface, constraints.hookean_specs)
    chart=ReducedCartesianChart(clean,fixed_indices=constraints.fixed_indices,allow_no_fixed=True);before=objective.requests
    norm=lambda v:float(np.linalg.norm(v.reshape(-1,3),axis=1).max())
    relaxed=safe_lbfgs(np.zeros(chart.dimension),lambda x:chart.evaluate(x,objective),gradient_norm=norm,step_norm=norm,gtol=fmax,max_step=max_step,maxiter=maxiter,lbfgs_memory=lbfgs_memory)
    a=chart.atoms(relaxed.q)
    scope='physical_plus_hookean' if constraints.hookean_specs else 'fixed_atom_manifold'
    certificate=dict(certified=False,scope=scope,objective=scope,fixed_indices=chart.fixed_indices.tolist())
    energy=active=full=None
    if relaxed.energy is not None:
        try:
            energy,f=objective.evaluate(a);f=np.asarray(f,dtype=float)
            if not np.isfinite(energy) or f.shape!=a.positions.shape or not np.isfinite(f).all():raise ValueError('invalid raw certificate E/F')
            active=float(np.linalg.norm(f[chart.active_indices],axis=1).max());full=float(np.linalg.norm(f,axis=1).max())
            invariant=np.array_equal(a.positions[chart.fixed_indices],chart.reference.positions[chart.fixed_indices]) and np.array_equal(a.cell.array,chart.reference.cell.array)
            certificate.update(certified=active<=fmax and invariant,active_fmax=active,full_raw_fmax=full,fixed_and_cell_exact=bool(invariant))
            last=getattr(objective, 'last_evaluation', None)
            if isinstance(objective, HookeanSurface) and last is not None:
                certificate['physical_energy']=last['physical_energy']
                certificate['physical_forces']=np.asarray(last['physical_forces']).copy()
                certificate['hookean_energy']=last['hookean_energy']
                certificate['hookean_forces']=np.asarray(last['hookean_forces']).copy()
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError) as error:certificate['error']=str(error)
    a=constraints.attach(a)
    return ConstrainedQuenchResult(a,energy,active,full,relaxed,certificate,objective.requests-before)


def run_constrained_ssw(atoms,surface,*,steps,config,rng,fixed_indices=None,direction_fixed_indices=None,ls=None,gaussian_policy=None,checkpoint=None,checkpoint_path=None):
    """Complete fixed-cell SSW with explicit fixed substrate and active forces.

    True landing relaxation stays on the constrained manifold. A certified
    active gradient does not certify full unconstrained stationarity. Variable
    cell and unsupported ASE constraints are rejected; optional LS is analytic
    on the same active chart and is removed for true landing quench. No global
    rigid-motion projection is used.
    Optional direction_fixed_indices excludes atoms only from initial directions
    and soft-mode refinement. They can still move in biased and true quenches
    unless also physically fixed. Rotation residuals then refer to that subspace.
    """
    if not isinstance(config,ConstrainedSSWConfig):raise TypeError('ConstrainedSSWConfig required')
    if isinstance(steps,(bool,np.bool_)) or not isinstance(steps,(int,np.integer)) or steps<0:raise ValueError('nonnegative integer steps required')
    from .pam_gaussian import PAMCurvatureGaussian
    if gaussian_policy is not None and not isinstance(gaussian_policy, PAMCurvatureGaussian):
        raise TypeError('gaussian_policy must be PAMCurvatureGaussian')
    if config.recovered_rotation is not None and gaussian_policy is not None:
        raise ValueError('recovered_rotation requires a compatible explicit anchor; PAM Gaussian is unsupported')
    constraints=normalize_constraints(atoms, fixed_indices=fixed_indices)
    atoms=constraints.clean_atoms(atoms)
    surface=bind_hookean_surface(surface, constraints.hookean_specs)
    reference=ReducedCartesianChart(atoms,fixed_indices=constraints.fixed_indices,allow_no_fixed=True);fixed=reference.fixed_indices
    if checkpoint is not None:
        _validate_constrained_checkpoint(checkpoint)
        if checkpoint.status not in ('completed','completed_with_failures'):
            raise ValueError(f'cannot resume terminal constrained checkpoint with status {checkpoint.status!r}')
        cp_atoms=checkpoint.chart_reference
        if tuple(cp_atoms.numbers)!=tuple(atoms.numbers) or tuple(cp_atoms.pbc)!=tuple(atoms.pbc):
            raise ValueError('checkpoint composition/PBC does not match input atoms')
        if not np.array_equal(cp_atoms.cell.array,atoms.cell.array) or not np.array_equal(cp_atoms.get_masses(),atoms.get_masses()):
            raise ValueError('checkpoint cell/masses do not match input atoms')
        if tuple(checkpoint.fixed_indices)!=tuple(fixed): raise ValueError('checkpoint fixed_indices do not match input')
        if tuple(getattr(checkpoint, 'hookean_specs', ())) != tuple(constraints.hookean_specs):
            raise ValueError('checkpoint Hookean constraints do not match input')
        cp_policy=getattr(checkpoint, 'gaussian_policy', None)
        requested_policy=None if gaussian_policy is None else gaussian_policy.parameters()
        saved_policy=None if cp_policy is None else cp_policy.parameters()
        if saved_policy != requested_policy:
            raise ValueError('checkpoint Gaussian policy does not match requested policy')
        requested_direction=tuple() if direction_fixed_indices is None else tuple(sorted(map(int,direction_fixed_indices)))
        if requested_direction!=tuple(checkpoint.direction_fixed_indices): raise ValueError('checkpoint direction_fixed_indices do not match input')
        if not _constrained_config_equal(checkpoint.config, config): raise ValueError('checkpoint constrained config does not match requested config')
        if not np.array_equal(checkpoint.current.atoms.positions[fixed], cp_atoms.positions[fixed]):
            raise ValueError('checkpoint current fixed coordinates do not match chart reference')
        if not np.array_equal(atoms.positions[fixed], cp_atoms.positions[fixed]):
            raise ValueError('input fixed coordinates do not match checkpoint chart reference')
        if _constrained_signature(checkpoint.ls_settings)!=_constrained_signature(ls): raise ValueError('checkpoint LS settings do not match requested settings')
        bit_name=checkpoint.rng_state.get('bit_generator')
        if bit_name != type(rng.bit_generator).__name__: raise ValueError('checkpoint RNG bit generator does not match requested RNG')
    from .ls_native_reference import NativeLSSettings
    if ls is not None and not isinstance(ls, (LSSettings, NativeLSSettings)):
        raise TypeError('ls must be LSSettings or NativeLSSettings')
    if ls is not None:
        from .ls_prequench import validate_prequench
        validate_prequench(getattr(ls, 'prequench', None))
        if (getattr(getattr(ls, 'prequench', None), 'exit_policy', 'force') != 'force'):
            raise ValueError("constrained driver does not support LS prequench exit_policy other than 'force'")
    runtime_type = (ConstrainedNativeLSRuntime if isinstance(ls, NativeLSSettings)
                    else ConstrainedLSRuntime)
    ls_runtime = (runtime_type.from_settings(atoms, surface, ls,
        fixed_indices=fixed) if ls is not None else None)
    resume_state=None
    if checkpoint is not None:
        resume_state=dict(initial=_constrained_copy(checkpoint.initial),current=_constrained_copy(checkpoint.current),
            best=_constrained_copy(checkpoint.best),minima=_constrained_copy(checkpoint.minima),
            records=_constrained_copy(checkpoint.records),next_index=checkpoint.next_index,
            evaluation_requests=checkpoint.evaluation_requests)
        if ls_runtime is not None:
            state=checkpoint.ls_state
            if state is None: raise ValueError('checkpoint is missing LS runtime state')
            state_kind = state.get('kind', 'paper')
            expected_kind = ('native' if isinstance(ls, NativeLSSettings) else 'paper')
            if state_kind != expected_kind:
                raise ValueError('checkpoint LS runtime kind does not match requested settings')
            ls_runtime.reference=_constrained_copy(state['reference'])
            ls_runtime.softening=_constrained_copy(state['softening'])
            ls_runtime.soft_surface=ConstrainedLSSurface(surface,ls_runtime.softening)
            if state.get('kind', 'paper') == 'native':
                from .ls_native_reference import NativeLSRuntime
                native=NativeLSRuntime.__new__(NativeLSRuntime)
                native.settings=_constrained_copy(ls)
                native.lengths=_constrained_copy(state['lengths'])
                native.frozen=_constrained_copy(state['softening'])
                native.state=_constrained_copy(state['state'])
                native.steps=int(state['steps'])
                native.last_update=_constrained_copy(state.get('last_update'))
                ls_runtime.native=native
            else:
                ls_runtime.response=_constrained_copy(state['response'])
        rng.bit_generator.state=_constrained_copy(checkpoint.rng_state)
    rotation_indices=None
    if direction_fixed_indices is not None:
        excluded=list(direction_fixed_indices)
        if any(isinstance(i,(bool,np.bool_)) or not isinstance(i,(int,np.integer)) or i<0 or i>=len(atoms) for i in excluded) or len(set(excluded))!=len(excluded):
            raise ValueError('direction_fixed_indices requires unique valid atom indices')
        selected=np.array([i not in excluded for i in reference.active_indices])
        rotation_indices=np.flatnonzero(np.repeat(selected,3))
        if not len(rotation_indices):raise ValueError('direction subspace must contain a mobile atom')
    def factory(a):
        if not np.array_equal(a.positions[fixed],reference.reference.positions[fixed]) or not np.array_equal(a.cell.array,reference.reference.cell.array):raise ValueError('fixed coordinates/cell changed')
        chart=ReducedCartesianChart(constraints.clean_atoms(a),fixed_indices=fixed,allow_no_fixed=True)
        return _ReducedSurface(chart,ls_runtime.soft_surface if ls_runtime is not None else surface)
    def quench(a):return constrained_quench(constraints.attach(a),surface,fixed_indices=fixed,fmax=config.fmax,max_step=config.max_step,maxiter=config.relax_steps,lbfgs_memory=config.lbfgs_memory)
    callback = (_active_rotation_callback if (config.recovered_rotation is not None or
                                              config.rotation_solver != 'generalized-dimer') else None)
    latest=None
    def ls_state():
        if ls_runtime is None:return None
        if isinstance(ls_runtime, ConstrainedNativeLSRuntime):
            native=ls_runtime.native
            if native is None: return None
            return dict(kind='native', reference=_constrained_copy(ls_runtime.reference),
                softening=_constrained_copy(native.frozen),
                lengths=_constrained_copy(native.lengths),
                state=_constrained_copy(native.state), steps=native.steps,
                last_update=_constrained_copy(native.last_update))
        return dict(reference=_constrained_copy(ls_runtime.reference),softening=_constrained_copy(ls_runtime.softening),
                    response=_constrained_copy(ls_runtime.response))
    def boundary(initial,current,best,minima,records,next_index,status):
        nonlocal latest
        latest=ConstrainedCheckpoint(_constrained_copy(initial),_constrained_copy(current),_constrained_copy(best),
            tuple(_constrained_copy(_restore_constraints(minima, constraints))),
            tuple(_constrained_copy(_restore_constraints(records, constraints))),_constrained_copy(reference.reference),
            tuple(map(int,fixed)),tuple() if direction_fixed_indices is None else tuple(sorted(map(int,direction_fixed_indices))),
            config,_constrained_copy(ls),ls_state(),_constrained_copy(rng.bit_generator.state),
            (resume_state['evaluation_requests'] if resume_state is not None else 0)+surface.requests-begin_requests,
            next_index,status,hookean_specs=tuple(constraints.hookean_specs),
            gaussian_policy=None if gaussian_policy is None else _constrained_copy(gaussian_policy))
        if checkpoint_path is not None: save_constrained_checkpoint(checkpoint_path,latest)
    begin_requests=surface.requests
    def result_factory(initial,current,best,minima,records,requests,status):
        return ConstrainedSSWResult(initial,current,best,
            _restore_constraints(minima, constraints),
            _restore_constraints(records, constraints),requests,status,latest)
    boundary_hook = boundary if (checkpoint is not None or checkpoint_path is not None) else None
    return _run_reduced_ssw(atoms,surface,steps=steps,config=config,rng=rng,factory=factory,
        coordinate_label='active_displacements',quench_callback=quench,
        rotation_indices=rotation_indices,ls_runtime=ls_runtime,
        rotation_callback=callback,gaussian_policy=gaussian_policy,
        resume_state=resume_state,boundary_callback=boundary_hook,
        result_factory=result_factory)
