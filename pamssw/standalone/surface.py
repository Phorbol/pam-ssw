"""Calculator-independent true/modified surfaces and fixed-cell quenching.

ASE units: energy eV, positions Angstrom, forces eV/Angstrom. Requests
count API evaluations, not SCF iterations or backend internal force calls.
One ASESurface owns one calculator and is intended for serial use; parallel
walkers must receive separate calculators (and directories for file backends).
"""
from dataclasses import dataclass

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.optimize import BFGS, LBFGSLineSearch
from pamssw.result import RelaxTelemetry


@dataclass(frozen=True)
class SciPyQuenchTelemetry(RelaxTelemetry):
    """Local extension; global RelaxTelemetry schema remains unchanged."""
    message: str | None = None
    nit: int | None = None
    nfev: int | None = None


class ASESurface:
    """Select the calculator objective explicitly when electronic smearing matters.

Default energy follows ASE's ordinary get_potential_energy convention.
For calculators whose forces differentiate free_energy instead, pass
force_consistent=True; missing support is propagated, never silently
replaced with a different objective. This interface cannot itself prove
that a backend's supplied energy and forces are mathematically consistent.
"""
    def __init__(self, calculator, *, force_consistent=False):
        if calculator is None:
            raise ValueError('an ASE calculator is required')
        self.calculator = calculator
        self.force_consistent = bool(force_consistent)
        self.requests = 0

    def evaluate(self, atoms):
        if atoms.constraints:
            raise NotImplementedError('constraint geometry is not implemented by this surface')
        if not len(atoms) or not np.isfinite(atoms.positions).all():
            raise ValueError('finite nonempty geometry required')
        work = atoms.copy()
        work.calc = self.calculator
        self.requests += 1  # Failed requests count too.
        energy = float(work.get_potential_energy(force_consistent=self.force_consistent))
        forces = np.array(work.get_forces(), dtype=float, copy=True)
        if not np.isfinite(energy) or forces.shape != (len(work), 3) or not np.isfinite(forces).all():
            raise ValueError('calculator returned invalid energy or forces')
        return energy, forces


class SurfaceCalculator(Calculator):
    """Evaluate a fixed collection of additive terms on a true ASE surface.

Terms provide evaluate(atoms)->(energy, forces). Construct a new calculator
when terms change; mutating a term behind ASE's cache is unsupported.
free_energy names the SAME selected objective as energy, so an optimizer
cannot silently switch from the user's explicit true-surface choice.
"""
    implemented_properties = ['energy', 'free_energy', 'forces']

    def __init__(self, surface, *, terms=()):
        super().__init__()
        self.surface = surface
        self.terms = tuple(terms)

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy, forces = self.surface.evaluate(self.atoms)
        true_energy, true_forces = energy, forces.copy()
        for term in self.terms:
            de, df = term.evaluate(self.atoms)
            df = np.asarray(df, dtype=float)
            if not np.isfinite(de) or df.shape != forces.shape or not np.isfinite(df).all():
                raise ValueError('bias returned invalid energy or forces')
            energy += float(de)
            forces += df
        self.results = dict(energy=energy, free_energy=energy, forces=forces,
                            true_energy=true_energy, true_forces=true_forces)


@dataclass(frozen=True)
class QuenchResult:
    atoms: object
    energy: float
    max_force: float
    converged: bool
    optimizer_steps: int
    evaluation_requests: int
    surface: str
    optimizer_telemetry: object | None = None


def quench(atoms, surface, *, fmax, steps, terms=(), optimizer=BFGS, frame=None, lbfgs_memory=None):
    """Relax at fixed cell, returning an explicit force certificate.

With terms=() this certifies only force stationarity on the true surface,
not a positive Hessian or physical stability. With terms supplied this is
a modified-surface quench; it must not be archived as a true minimum.
Caller geometry/calculator are untouched, and output has no calculator.
optimizer may be an ASE optimizer class (existing default), or the explicit
'safe-lbfgs-total' backend from PAM's numerical Relaxer. The latter optimizes
the same complete objective with existing default numerical settings; no PAM
walker or bias-separated secants are used. Its iteration budget does not cap
line-search E/F requests. Frame coordinates are not supported by that adapter.
lbfgs_memory=None preserves10 pairs; a positive integer opts into a different
Safe-total history size with O(memory * 3N) storage. Other backends reject it.
"""
    from pamssw.relax import _validate_lbfgs_memory
    _validate_lbfgs_memory(lbfgs_memory, optimizer)
    if not np.isfinite(fmax) or fmax <= 0:
        raise ValueError('fmax must be positive and finite')
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    terms = tuple(terms)
    if isinstance(optimizer, str) and optimizer not in ('safe-lbfgs-total', 'scipy-lbfgsb'):
        if optimizer != 'ase-lbfgs-linesearch':
            raise ValueError('unknown quench optimizer: ' + optimizer)
    if optimizer == 'safe-lbfgs-total' and frame is not None:
        raise NotImplementedError('safe-lbfgs-total does not support frame coordinates')
    if optimizer == 'scipy-lbfgsb' and frame is not None:
        raise NotImplementedError('scipy-lbfgsb does not support frame coordinates')
    if frame is not None and not terms:
        raise ValueError("frame quench is only a modified-surface certificate; true quench must be unrestricted")
    work = atoms.copy()
    work.calc = SurfaceCalculator(surface, terms=terms)
    before = surface.requests
    target = work
    if frame is not None:
        from .cluster_frame import ClusterFrameFilter
        target = ClusterFrameFilter(work, frame)
    optimizer_telemetry = None
    if optimizer == 'safe-lbfgs-total':
        from pamssw.relax import Relaxer
        from pamssw.state import State

        # The numerical variable is the unwrapped Cartesian array. Physical
        # cell/PBC and all ASE atom metadata stay on work, used by every oracle
        # call. Do not let Relaxer wrap this array: a fixed Gaussian center is
        # defined in the same continuous coordinates, not modulo a unit cell.
        state = State(work.numbers.copy(), work.positions.copy())

        def evaluate(flat, template):
            work.set_positions(np.asarray(flat).reshape(-1, 3))
            energy = float(work.get_potential_energy())
            return energy, -work.get_forces().ravel()

        result = Relaxer(evaluate, optimizer='safe-lbfgs-total', lbfgs_memory=lbfgs_memory).relax(
            state, fmax=fmax, maxiter=int(steps))
        work.set_positions(result.state.positions)
        optimizer_steps = result.n_iter
        optimizer_telemetry = result.telemetry
    elif optimizer == 'scipy-lbfgsb':
        from scipy.optimize import minimize
        x0 = work.get_positions().reshape(-1).copy()
        def objective(flat):
            work.set_positions(np.asarray(flat, dtype=float).reshape(-1, 3))
            return (float(work.get_potential_energy()), -work.get_forces().reshape(-1))
        if steps == 0:
            objective(x0)
            result = None
            optimizer_steps = 0
            scipy_result = {
                'backend': 'scipy-lbfgsb', 'success': False, 'message': 'zero steps',
                'nit': 0, 'nfev': 1,
            }
        else:
            result = minimize(objective, x0, method='L-BFGS-B', jac=True,
                              options={'maxiter': int(steps), 'gtol': fmax / np.sqrt(3.0)})
            work.set_positions(np.asarray(result.x).reshape(-1, 3))
            optimizer_steps = int(getattr(result, 'nit', 0))
            scipy_result = {
                'backend': 'scipy-lbfgsb', 'success': bool(result.success),
                'message': str(result.message), 'nit': int(result.nit),
                'nfev': int(result.nfev),
            }
    else:
        optimizer_cls = LBFGSLineSearch if optimizer == 'ase-lbfgs-linesearch' else optimizer
        opt = optimizer_cls(target, logfile=None)
        opt.run(fmax=fmax, steps=int(steps))
        optimizer_steps = opt.nsteps
    forces = target.get_forces()
    max_force = float(np.linalg.norm(forces, axis=1).max())
    energy = float(work.get_potential_energy())
    if optimizer == 'scipy-lbfgsb':
        optimizer_telemetry = SciPyQuenchTelemetry(
            backend='scipy-lbfgsb', evaluator_calls=surface.requests - before,
            backend_evaluations=surface.requests - before, gradient_measure='raw_max_force',
            converged=max_force <= fmax,
            termination_reason='converged' if max_force <= fmax else (
                'maxiter' if scipy_result['nit'] >= int(steps) else 'scipy_termination'),
            optimizer_success=scipy_result['success'], message=scipy_result['message'],
            nit=scipy_result['nit'], nfev=scipy_result['nfev'])
    output = work.copy()
    return QuenchResult(output, energy, max_force, max_force <= fmax,
                        optimizer_steps, surface.requests - before,
                        'modified_cluster_section' if frame is not None else ('modified' if terms else 'true'),
                        optimizer_telemetry)
