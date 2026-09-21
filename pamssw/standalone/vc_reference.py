"""Experimental joint atomic/log-strain SSW; independent of the LASP runtime.

This is a consistent generalized-coordinate extension, not a claim of native
VC schedule parity. The explicit strain_length defines the search metric.
"""
from dataclasses import dataclass
import math
import numpy as np
from ase import units
from .vc_geometry import SymmetricLogStrainChart
from .generalized_numerics import safe_lbfgs, generalized_dimer


@dataclass(frozen=True)
class VCSSWConfig:
    strain_length: float
    width: float
    rotation_bias: float
    pressure: float = 0.0
    temperature_K: float = 300.0
    forward_force: float = .1
    max_gaussians: int = 14
    gradient_tol: float = .005
    fmax: float = .01
    stress_tol: float = .001
    max_step: float = .2
    relax_steps: int = 300
    fd_step: float = 1e-4
    rotation_hvp: int = 100
    rotation_tol: float = .02
    lbfgs_memory: int | None = None
    bias_release: str = 'strict'

    def __post_init__(self):
        if self.bias_release not in ('strict', 'numerical_stop'):
            raise ValueError('bias_release must be strict or numerical_stop')
        from pamssw.relax import _validate_lbfgs_memory
        _validate_lbfgs_memory(self.lbfgs_memory, 'safe-lbfgs-total')
        for name in ('strain_length', 'width', 'forward_force', 'gradient_tol',
                     'fmax', 'stress_tol', 'max_step', 'fd_step', 'rotation_tol'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        for name in ('rotation_bias', 'temperature_K'):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not np.isfinite(self.pressure):
            raise ValueError('finite pressure required')
        for name in ('max_gaussians', 'relax_steps', 'rotation_hvp'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f'{name} must be a positive integer')


@dataclass
class VCSSWResult:
    initial: object
    current: object
    best: object
    minima: list
    records: list
    requests: int
    status: str


def run_vc_ssw(atoms, surface, *, steps, config, rng, ls=None,
               direction_solver=None, rotation_force_calls=None, height_policy=None,
               height_update_budget=1000, ls_prequench='fixed_cell'):
    """Run joint 3N+6 SSW with physical force AND stress landing certificates.

    surface is an ASEStressSurface or equivalent counted E/F/stress oracle.
    The PES must be invariant under global atomic translation (no external
    position-dependent fields); atomic updates remain in a fixed mean-X chart.
    All stress thresholds/pressure are eV/Angstrom^3. Width and max_step are
    lengths in the explicitly scaled q chart. Failed proposals retain current.
    Initial certification failure returns a result with no certified minima.
    Optional MinimalAngleHeightPolicy or ConservativeNativeHeightPolicy uses
    the same scaled joint-coordinate force metric; these are explicit policy
    choices and are not a claim of native VC height/schedule parity.
    ls_prequench='joint' relaxes the frozen softened atomic/cell objective
    before climbing and updates LS using the true enthalpy response. The
    existing fixed-cell preparation remains the default explicit baseline.
    bias_release='numerical_stop' permits finite accepted biased iterates after
    maxiter/native_stop to enter true quench. This experimental proposal policy
    never substitutes numerical stopping for a physical landing certificate.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    if ls_prequench not in ('fixed_cell', 'joint'):
        raise ValueError('ls_prequench must be fixed_cell or joint')
    if ls_prequench == 'joint' and ls is None:
        raise ValueError('joint prequench requires LS settings')
    if height_policy is not None:
        from .minimal_angle_height import MinimalAngleHeightPolicy
        from .native_height_policy import ConservativeNativeHeightPolicy, FrozenHeightGaussian
        if not isinstance(height_policy, (MinimalAngleHeightPolicy, ConservativeNativeHeightPolicy)):
            raise TypeError('supported explicit joint VC height_policy required')
        if (isinstance(height_update_budget, (bool, np.bool_)) or
                not isinstance(height_update_budget, (int, np.integer)) or height_update_budget < 1):
            raise ValueError('positive integer height-update budget required')
    if direction_solver is None:
        if rotation_force_calls is not None:
            raise ValueError('rotation_force_calls requires direction_solver')
        solver = generalized_dimer
    else:
        if not callable(direction_solver):
            raise ValueError('direction_solver must be callable')
        if isinstance(rotation_force_calls, (bool, np.bool_)) or not isinstance(rotation_force_calls, (int, np.integer)) or rotation_force_calls < 1:
            raise ValueError('rotation_force_calls must be a positive integer with direction_solver')
        solver = direction_solver
    if ls is not None:
        from .paper_reference import LSSettings
        from .softening import LSResponseState
        from .vc_softening import FrozenPeriodicCellSoftening, FixedAtomicSurface
        from .ls_cycle import prepare_ls_step
        if not isinstance(ls, LSSettings):raise TypeError('ls requires explicit LSSettings')
        if getattr(ls, 'prequench', None) is not None:
            raise ValueError('fixed-cell LS prequench overrides are not supported by the VC driver')
    softening = None
    chart = SymmetricLogStrainChart(atoms, strain_length=config.strain_length)
    start_requests = surface.requests
    def norm(v):
        return max(float(np.linalg.norm(v[:-6].reshape(-1, 3), axis=1).max()),
                   float(np.linalg.norm(v[-6:])))
    def physical(q):
        return chart.evaluate(q, surface.evaluate, pressure=config.pressure)
    def bare(q):
        if softening is None:
            ev = physical(q)
        else:
            def softened(atoms):
                e,f,s = surface.evaluate(atoms)
                de,df,ds = softening.evaluate_stress(atoms)
                return e+de,f+df,s+ds
            ev = chart.evaluate(q, softened, pressure=config.pressure)
        return ev.objective, chart.project(ev.gradient)
    def minimize(q, evaluate):
        return safe_lbfgs(q, evaluate, gradient_norm=norm, step_norm=norm,
                          gtol=config.gradient_tol, max_step=config.max_step,
                          maxiter=config.relax_steps, lbfgs_memory=config.lbfgs_memory)
    def true_minimize(q):
        from .cell_relax import relax_cell_coordinates
        return relax_cell_coordinates(chart, q, surface, pressure=config.pressure,
            fmax=config.fmax, stress_tol=config.stress_tol,
            max_step=config.max_step, maxiter=config.relax_steps,
            lbfgs_memory=config.lbfgs_memory)
    def certify(q):
        ev = physical(q)
        f = float(np.linalg.norm(ev.forces, axis=1).max())
        s = float(np.abs(ev.stress + config.pressure * np.eye(3)).max())
        return ev, dict(fmax=f, stress_max=s,
                        certified=f <= config.fmax and s <= config.stress_tol)
    q = chart.pack(atoms)
    initial_relax = true_minimize(q)
    if initial_relax.energy is None:
        return VCSSWResult(None, None, None, [],
            [dict(stage='initial', status=initial_relax.status, error=initial_relax.error,
                  requests=surface.requests-start_requests)],
            surface.requests-start_requests, 'initial_quench_failed')
    try:
        initial, certificate = certify(initial_relax.q)
    except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
        return VCSSWResult(None, None, None, [],
            [dict(stage='initial', status='evaluation_failed', error=str(error),
                  requests=surface.requests-start_requests)],
            surface.requests-start_requests, 'initial_quench_failed')
    minima = []
    records = [dict(stage='initial', status=initial_relax.status,
                    certificate=certificate, objective=initial.objective,
                    requests=surface.requests-start_requests)]
    if not initial_relax.converged or not certificate['certified']:
        return VCSSWResult(initial, initial, initial, minima, records,
                           surface.requests-start_requests, 'initial_quench_failed')
    q = initial_relax.q.copy()
    current = best = initial
    minima.append(initial)
    if ls is not None:
        try:
            softening = FrozenPeriodicCellSoftening.from_atoms(current.atoms,
                bond_energies=ls.bond_energies,bond_lengths=ls.bond_lengths,
                initial_fraction=ls.initial_fraction,xi=ls.xi,
                energy_filter=ls.energy_filter)
            response = LSResponseState(ls.target_per_atom,learning_rate=ls.learning_rate)
        except (ValueError,RuntimeError,FloatingPointError) as error:
            records.append(dict(stage='ls_initialization',status='ls_initialization_failed',
                error=str(error),requests=0))
            return VCSSWResult(initial,current,best,minima,records,
                surface.requests-start_requests,'ls_initialization_failed')
    run_status = 'completed'
    for index in range(steps):
        before = surface.requests
        # Begin a new local chart only after all previous biases are discarded.
        # Never rebase within a climbing path or rewrite Gaussian history.
        chart = SymmetricLogStrainChart(current.atoms, strain_length=config.strain_length)
        q = chart.pack(current.atoms)
        work = q.copy()
        anchor = chart.project(rng.normal(size=work.size))
        anchor /= np.linalg.norm(anchor)
        terms = []
        frozen_gaussians = []
        chart_reference = current.atoms.copy()
        last_work = current.atoms.copy()
        frozen_softening = softening
        climb = []
        accepted = False
        status = 'gaussian_limit'
        landing = None
        cert = None
        landing_optimizer = None
        prepared = None
        ls_record = None
        def biased(x):
            energy, gradient = bare(x)
            for center, direction, weight in terms:
                projection = float((x-center) @ direction)
                bias = weight * math.exp(-.5*(projection/config.width)**2)
                energy += bias
                gradient -= bias*projection/config.width**2 * direction
            return energy, gradient
        try:
            if ls is not None:
                if ls_prequench == 'fixed_cell':
                    prepared = prepare_ls_step(current.atoms,FixedAtomicSurface(surface),
                        softening=softening,fmax=config.fmax,steps=config.relax_steps,
                        optimizer='safe-lbfgs-total', lbfgs_memory=config.lbfgs_memory)
                    work = chart.pack(prepared.atoms)
                    response_before, response_after = prepared.energy_before, prepared.energy_after
                    ls_record = dict(energy_response=prepared.energy_response,
                        energy_before=prepared.energy_before,energy_after=prepared.energy_after,
                        prequench_requests=prepared.evaluation_requests,
                        prequench='fixed_cell_atoms_only',bond_count=len(softening.pairs))
                else:
                    from .joint_ls import prepare_joint_ls_step
                    prepared = prepare_joint_ls_step(current.atoms,surface,
                        softening=softening,config=config)
                    work = prepared.q.copy()
                    response_before, response_after = prepared.true_enthalpy_before, prepared.true_enthalpy_after
                    ls_record = dict(prequench='joint_atoms_cell', response_quantity='physical_enthalpy',
                        energy_before=prepared.true_energy_before,energy_after=prepared.true_energy_after,
                        enthalpy_before=response_before,enthalpy_after=response_after,
                        energy_response=(prepared.true_energy_after-prepared.true_energy_before)/len(current.atoms),
                        enthalpy_response=(response_after-response_before)/len(current.atoms),
                        ls_energy_before=prepared.ls_energy_before,ls_energy_after=prepared.ls_energy_after,
                        volume_before=prepared.volume_before,volume_after=prepared.volume_after,
                        soft_joint_gradient_norm=norm(prepared.post_soft_gradient),
                        soft_fmax=prepared.post_soft_fmax,soft_stress_max=prepared.post_soft_stress_max,
                        prequench_requests=prepared.requests,optimizer=prepared.optimizer,
                        bond_count=len(softening.pairs))
                last_work = prepared.atoms.copy()
            for j in range(config.max_gaussians):
                stage_before = surface.requests
                event = dict(index=j, status='running', rotation_requests=0,
                    height_requests=0, biased_quench_requests=0, true_check_requests=0)
                climb.append(event)
                def measured(name, operation):
                    start = surface.requests
                    try:
                        return operation()
                    finally:
                        event[name] += surface.requests-start
                try:
                    solver_kwargs = dict(rotation_bias=config.rotation_bias,
                        fd_step=config.fd_step, tol=config.rotation_tol, evaluate=bare)
                    if direction_solver is None:
                        solver_kwargs['max_hvp'] = config.rotation_hvp
                    else:
                        solver_kwargs['max_force_calls'] = rotation_force_calls
                    mode = measured('rotation_requests', lambda: solver(work, anchor, **solver_kwargs))
                    if not mode.converged:
                        status = 'rotation_failed'
                        event.update(status=status, residual=mode.residual_norm)
                        break
                    displaced = work + config.width*mode.direction
                    _, background_g = measured('height_requests', lambda: biased(displaced))
                    if height_policy is None:
                        weight = (config.forward_force + background_g @ mode.direction)*config.width*math.exp(.5)
                    else:
                        history = tuple(FrozenHeightGaussian(c, n, config.width, w)
                                        for c, n, w in terms)
                        event['height_input'] = dict(history=history, center=work.copy(),
                            direction=mode.direction.copy(), width=config.width,
                            point=displaced.copy(), background_force=-background_g.copy())
                        if isinstance(height_policy, MinimalAngleHeightPolicy):
                            preparation = height_policy.prepare(history, center=work,
                                direction=mode.direction, width=config.width, point=displaced,
                                background_force=-background_g)
                        else:
                            # generalized_dimer includes the analytic
                            # rotation-only rank-one curvature. Restore its
                            # physical (or physical+LS) directional curvature
                            # before applying the native-derived policy.
                            curvature = mode.curvature + config.rotation_bias * float(np.sum(mode.direction * anchor))**2
                            curvature_scope = ('physical enthalpy E+pV; rotation-only bias excluded' if softening is None
                                               else 'physical enthalpy E+pV plus frozen LS; rotation-only bias excluded')
                            event['curvature_scope'] = curvature_scope
                            event['height_input']['curvature'] = float(curvature)
                            event['height_update_budget'] = height_update_budget
                            preparation = height_policy.prepare(history, center=work,
                                direction=mode.direction, width=config.width, point=displaced,
                                background_force=-background_g, curvature=curvature,
                                curvature_scope=curvature_scope,
                                max_updates=height_update_budget)
                        event['height_preparation'] = preparation
                        weight = (preparation.weight if isinstance(height_policy, MinimalAngleHeightPolicy)
                                  else preparation.final_weight)
                    if not np.isfinite(weight) or weight <= 0:
                        status = 'nonpositive_height'
                        event['status'] = status
                        break
                    if height_policy is None:
                        terms.append((work.copy(), mode.direction.copy(), float(weight)))
                    else:
                        # The policy returns the complete history. In
                        # particular, ConservativeNativeHeightPolicy may
                        # rewrite prior weights; appending would resurrect the
                        # pre-rewrite terms and double-count the bias.
                        terms = [(t.center.copy(), t.direction.copy(), float(t.weight))
                                 for t in preparation.terms]
                    if height_policy is None:
                        frozen_gaussians.append(dict(center=work.copy(), direction=mode.direction.copy(),
                            weight=float(weight), width=config.width))
                    else:
                        frozen_gaussians = [dict(center=t.center.copy(), direction=t.direction.copy(),
                            weight=float(t.weight), width=float(t.width))
                            for t in preparation.terms]
                    relaxed = measured('biased_quench_requests', lambda: minimize(displaced, biased))
                    work = relaxed.q.copy()
                    last_work = chart.unpack(work)
                    event.update(weight=float(weight), status=relaxed.status,
                        q=work.tolist(), cell=chart.unpack(work).cell.array.tolist(),
                        direction=mode.direction.tolist(), rotation_residual=mode.residual_norm,
                        relaxation=dict(steps=relaxed.steps, attempted_requests=relaxed.requests,
                            rejected_trials=relaxed.rejected_trials, error=relaxed.error,
                            gradient_norm=None if relaxed.gradient is None else float(np.linalg.norm(relaxed.gradient))))
                    if height_policy is not None:
                        event['height_update_budget'] = height_update_budget
                    if not relaxed.converged:
                        releasable = (config.bias_release == 'numerical_stop'
                            and relaxed.status in ('maxiter', 'native_stop')
                            and relaxed.error is None
                            and relaxed.energy is not None and np.isfinite(relaxed.energy)
                            and relaxed.gradient is not None
                            and np.isfinite(relaxed.gradient).all()
                            and np.isfinite(work).all())
                        status = ('biased_numerical_stop' if releasable
                                  else 'biased_quench_failed')
                        break
                    true_ev = measured('true_check_requests', lambda: physical(work))
                    event['objective'] = true_ev.objective
                    if true_ev.objective < current.objective:
                        status = 'lower_true_enthalpy'
                        break
                except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                    event['status'] = 'evaluation_failed'
                    raise
                finally:
                    event['requests'] = surface.requests-stage_before
            if status in ('gaussian_limit', 'lower_true_enthalpy', 'biased_numerical_stop'):
                relaxed = true_minimize(work)
                landing_optimizer = dict(status=relaxed.status, steps=relaxed.steps,
                    requests=relaxed.requests, error=relaxed.error)
                landing, cert = certify(relaxed.q)
                if not relaxed.converged or not cert['certified']:
                    status = 'true_quench_failed'
                else:
                    minima.append(landing)
                    delta = landing.objective-current.objective
                    accepted = delta <= 0 or (config.temperature_K > 0 and
                        rng.random() < math.exp(-delta/(units.kB*config.temperature_K)))
                    if landing.objective < best.objective:
                        best = landing
                    if accepted:
                        q = relaxed.q.copy()
                        current = landing
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            status = 'evaluation_failed'
            climb.append(dict(error=str(error)))
            if ls is not None and prepared is None:
                status = run_status = 'ls_prequench_failed'
                failed_preparation = getattr(error,'result',None)
                if getattr(failed_preparation,'atoms',None) is not None:
                    last_work = failed_preparation.atoms.copy()
                ls_record = dict(error=str(error),stage=getattr(error,'stage','prepare'),
                    failed_result=failed_preparation,requests=surface.requests-before,
                    prequench=('joint_atoms_cell' if ls_prequench == 'joint' else 'fixed_cell_atoms_only'))
        if prepared is not None:
            try:
                softening = response.update(softening,current.atoms,
                    energy_before=response_before,energy_after=response_after,
                    bond_energies=ls.bond_energies,bond_lengths=ls.bond_lengths)
            except (ValueError,RuntimeError,FloatingPointError) as error:
                status = run_status = 'ls_update_failed'
                ls_record['update_error'] = str(error)
        records.append(dict(index=index, status=status, accepted=bool(accepted),
            climb=climb, landing=landing, certificate=cert,
            chart_reference=chart_reference,last_work=last_work,
            frozen_gaussians=frozen_gaussians,frozen_softening=frozen_softening,
            landing_optimizer=landing_optimizer, requests=surface.requests-before,
            **({'ls':ls_record} if ls is not None else {})))
        if run_status != 'completed':break
    return VCSSWResult(initial, current, best, minima, records,
                       surface.requests-start_requests, run_status)
