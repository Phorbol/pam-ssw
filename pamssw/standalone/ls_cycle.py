"""Preparation and true-surface finishing for independent, fixed-cell LS steps.

These components implement only the soft-only pre-relaxation and removal of
biases described in Guan et al., JCTC (2024), DOI 10.1021/acs.jctc.4c01081,
sections 2.3–2.4. They are NOT a complete LS-SSW walk: direction selection,
Gaussian climbing, Metropolis acceptance and cross-step response updates
remain the caller's responsibility. No scientific performance claim.
"""
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.optimize import BFGS

from .ls_prequench import validate_prequench, validate_prequench_exit_policy
from .softening import FrozenBondSoftening
from .periodic_softening import FrozenPeriodicBondSoftening
from .surface import ASESurface, QuenchResult, quench


class LSCycleError(RuntimeError):
    """Explicit failed stage, optionally retaining the failed force certificate."""
    def __init__(self, stage, message, *, result=None):
        super().__init__(f'{stage}: {message}')
        self.stage = stage
        self.result = result


@dataclass(frozen=True)
class PreparedLSStep:
    """Softened geometry and true-energy response, with no attached calculator.

    `energy_response` = (energy_after - energy_before)/N in eV/atom, evaluated
    on the original ASESurface, not the energy of E+V_LS. `soft_quench.energy`
    is instead on the modified surface. Keep `softening` frozen throughout
    the ensuing walk. `atoms` is an owned, mutable ASE geometry snapshot.
    """
    atoms: Atoms
    softening: FrozenBondSoftening
    energy_before: float
    energy_after: float
    energy_response: float
    start_max_force: float
    soft_quench: QuenchResult
    evaluation_requests: int
    qualification: str = 'force'


def _validate_budget(fmax, steps):
    if not np.isfinite(fmax) or fmax <= 0:
        raise ValueError('fmax must be finite and positive')
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')


def prepare_ls_step(atoms: Atoms, surface: ASESurface, *,
                    softening: FrozenBondSoftening, fmax: float, steps: int,
                    optimizer=BFGS, lbfgs_memory=None, prequench=None) -> PreparedLSStep:
    """Check a true stationary start, then quench only E+V_LS at fixed cell.

    Build `softening` from this step's true starting atoms using explicit bond
    tables, or obtain it from LSResponseState.update for this same geometry.
    Stale reference distances are rejected. A nonstationary start is rejected,
    not silently repaired. The explicit fmax (eV/Angstrom) applies to both the
    start's true force and, by default, the pre-quench's modified force.
    An explicit prequench overrides only the softened force/iteration limits;
    strict convergence and the true-start threshold remain unchanged. The optimizer has
    an explicit iteration budget, which is not a bound on backend SCF work.

    This function accepts no Gaussian terms; any calculator attached to atoms
    is ignored. The supplied `surface` must represent the physical objective.
    All constraints are currently rejected by ASESurface. Small forces do not
    prove a positive Hessian, chemical stability or calculator applicability.
    """
    from pamssw.relax import _validate_lbfgs_memory
    _validate_lbfgs_memory(lbfgs_memory, optimizer)
    _validate_budget(fmax, steps)
    validate_prequench_exit_policy(prequench, optimizer=optimizer)
    if not isinstance(softening, (FrozenBondSoftening, FrozenPeriodicBondSoftening)):
        raise TypeError('softening must be a FrozenBondSoftening')
    softening._validate_atoms(atoms)
    before = surface.requests
    energy_before, true_forces = surface.evaluate(atoms)
    start_max_force = float(np.linalg.norm(true_forces, axis=1).max())
    if start_max_force > fmax:
        raise LSCycleError('true_start',
                           f'maximum true force {start_max_force} exceeds fmax={fmax}')
    # Numerical equality allowance only (32 floating-point ulps of the scale),
    # not a chemical bond threshold or a configurable search heuristic.
    for distance, reference in zip(softening.pair_distances(atoms), softening.reference_distances):
        allowance = 32*np.finfo(float).eps*max(1., reference)
        if abs(float(distance)-reference) > allowance:
            raise ValueError('frozen reference distances do not describe this starting geometry')
    result = quench(atoms, surface,
                    fmax=fmax if prequench is None else prequench.fmax,
                    steps=steps if prequench is None else prequench.steps,
                    terms=(softening,), optimizer=optimizer, lbfgs_memory=lbfgs_memory)
    qualification = 'force'
    if not result.converged:
        telemetry = result.optimizer_telemetry
        if (prequench is None or prequench.exit_policy != 'force_or_step_limit' or
                optimizer != 'safe-lbfgs-total' or telemetry is None or
                telemetry.backend != 'safe-lbfgs-total' or
                telemetry.termination_reason != 'maxiter' or
                result.optimizer_steps < int(prequench.steps) or
                not np.isfinite(result.energy) or not np.isfinite(result.max_force)):
            raise LSCycleError('soft_quench', 'modified-surface force criterion not reached', result=result)
        qualification = 'step_limit'
    energy_after, _ = surface.evaluate(result.atoms)
    response = (energy_after-energy_before)/len(atoms)
    return PreparedLSStep(result.atoms.copy(), softening, energy_before, energy_after,
                          response, start_max_force, result, surface.requests-before,
                          qualification)


def finish_ls_step(atoms: Atoms, surface: ASESurface, *, fmax: float, steps: int,
                   optimizer=BFGS, lbfgs_memory=None) -> QuenchResult:
    """Remove every additive term and quench on the supplied original surface.

    `atoms.calc` is ignored. Passing terms is deliberately unsupported: LS and
    Gaussian contributions cannot leak in from the candidate calculator.
    Nonconvergence raises LSCycleError(stage='true_finish', result=certificate).
    There is no fallback and no MC/archive decision here. The returned force
    certificate is stationarity only, not a proof of physical minimum identity.
    """
    result = quench(atoms, surface, fmax=fmax, steps=steps, terms=(), optimizer=optimizer, lbfgs_memory=lbfgs_memory)
    if not result.converged:
        raise LSCycleError('true_finish', 'true-surface force criterion not reached', result=result)
    return result
