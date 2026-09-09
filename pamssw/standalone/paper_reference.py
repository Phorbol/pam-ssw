"""Independent cluster SSW/LS-SSW, following the published outer algorithm.

SSW: Shang & Liu, JCTC 2013, DOI 10.1021/ct301010b, p1840 steps 1-8.
Height: BP-CBD, DOI 10.1021/ct300250h, p2218 forward-force equation.
LS: DOI 10.1021/acs.jctc.4c01081, eqs 11-15.

Explicit numerical differences: single-sided finite-difference Ritz rotation
instead of native Broyden; ASE LBFGS; conventional Metropolis without the
release's NSAME trapping schedule. Climb lower-energy exit uses true energy
at the modified minimum (an interpretation of SSW step 5). This module is
a paper-level reference, NOT execution parity with the uploaded release.
"""
from dataclasses import dataclass
import math

import numpy as np
from ase import units
from ase.optimize import LBFGS

from .direction import paper_biased_direction
from .gaussian import ProjectedGaussian
from .ls_cycle import LSCycleError, prepare_ls_step
from .softening import FrozenBondSoftening, LSResponseState
from .surface import SurfaceCalculator, QuenchResult, quench


@dataclass(frozen=True)
class SSWConfig:
    width: float                 # Angstrom, Gaussian width and translation.
    rotation_bias: float         # eV/Angstrom², negative rank-one curvature.
    max_gaussians: int           # H, hard climbing budget from SSW step 5.
    temperature_K: float         # MC strategy temperature, not dynamics.
    fmax: float                  # eV/Angstrom, largest atom-force norm.
    relax_steps: int             # Per-quench optimizer iteration budget.
    fd_step: float               # Angstrom, dimer separation.
    rotation_hvp: int            # Finite-difference rotation budget.
    rotation_tol: float          # eV/Angstrom², numerical eigen residual.
    forward_force: float = .1    # eV/Angstrom, BP-CBD 2012 p2218 setting.
    direction_sampling: str = 'paper'  # 'global' is an explicit reference variant.

    def __post_init__(self):
        if self.direction_sampling not in ('paper', 'global'):
            raise ValueError('direction_sampling must be paper or global')
        for key in ('width', 'rotation_bias', 'fmax', 'fd_step',
                    'rotation_tol', 'forward_force'):
            value = getattr(self, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be positive and finite')
        if not np.isfinite(self.temperature_K) or self.temperature_K < 0:
            raise ValueError('temperature_K must be finite and nonnegative')
        for key, lower in (('max_gaussians', 1), ('relax_steps', 0), ('rotation_hvp', 2)):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < lower:
                raise ValueError(f'{key} must be an integer >= {lower}')


@dataclass(frozen=True)
class LSSettings:
    bond_energies: dict          # Explicit element-pair standard bond energies, eV.
    bond_lengths: dict           # Explicit pair neighbor cutoffs, Angstrom.
    target_per_atom: float       # Target true pre-quench response, eV/atom.
    initial_fraction: float = .03
    xi: float = .2              # Dimensionless fraction of frozen bond length.
    learning_rate: float = 1.8  # Paper eq15, not universal optimal parameters.


@dataclass(frozen=True)
class SSWStep:
    index: int
    status: str
    accepted: bool
    climb: tuple
    landing: object
    energy_response: object
    evaluation_requests: int
    error: object = None
    last_atoms: object = None
    initial_direction: object = None


@dataclass(frozen=True)
class SSWResult:
    initial: QuenchResult
    current: object
    best: object
    minima: tuple
    records: tuple
    evaluation_requests: int
    status: str


class InitialQuenchError(RuntimeError):
    def __init__(self, result):
        super().__init__('initial true quench did not reach requested force tolerance')
        self.result = result


def sample_initial_direction(atoms, rng, *, mode):
    """2013 eqs 1-2: normalized Maxwell direction plus lambda bond-formation.

Paper constants: lambda uniform [0.1,1.5], pair separation >3 Angstrom.
The local vector is the unnormalized coordinate swap of eq2, evaluated in
the paper's Angstrom convention. 'global' omits the pair term explicitly;
it is useful for systems with no eligible pair and is not a silent fallback.
"""
    global_direction = rng.normal(size=(len(atoms), 3)) / np.sqrt(atoms.get_masses()[:, None])
    global_direction /= np.linalg.norm(global_direction)
    if mode == 'global':
        return global_direction
    if mode != 'paper':
        raise ValueError('unknown direction sampling mode')
    eligible = [(i, j) for i in range(len(atoms)) for j in range(i+1, len(atoms))
                if np.linalg.norm(atoms.positions[j]-atoms.positions[i]) > 3.]
    if not eligible:
        raise ValueError('paper direction requires an atom pair separated by more than 3 Angstrom')
    i, j = eligible[int(rng.integers(len(eligible)))]
    local_direction = np.zeros_like(global_direction)
    local_direction[i] = atoms.positions[j] - atoms.positions[i]
    local_direction[j] = -local_direction[i]
    direction = global_direction + rng.uniform(.1, 1.5) * local_direction
    return direction / np.linalg.norm(direction)


def run_ssw(atoms, surface, *, steps, config, rng, ls=None):
    """Run independent fixed-cell nonperiodic SSW, optionally with paper LS.

No native program is called. Atoms/its calculator are not changed. The
returned minima include rejected MC landings, all force-converged on the
true surface; they are NOT deduplicated or positive-Hessian certified.
    Failed rotations/quenches consume an outer step and stay in records,
    never enter minima. A failed LS strength update retains any already valid
    landing. Calculator exceptions propagate; there is no alternate potential.
Random velocity directions use normal components divided by sqrt(mass),
then normalize; the Maxwell temperature factor cancels. Global rigid modes
are not projected out (external ASE potentials may break these symmetries).
"""
    if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError('steps must be a nonnegative integer')
    if atoms.pbc.any() or atoms.constraints:
        raise NotImplementedError('paper reference currently requires nonperiodic unconstrained atoms')
    masses = atoms.get_masses()
    if not len(atoms) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError('finite positive masses required')
    begin = surface.requests
    initial = quench(atoms, surface, fmax=config.fmax,
                     steps=config.relax_steps, optimizer=LBFGS)
    if not initial.converged:
        raise InitialQuenchError(initial)
    current = initial.atoms.copy()
    current_energy = initial.energy
    best = initial
    minima = [initial]
    records = []
    frozen = response = None
    if ls is not None:
        frozen = FrozenBondSoftening.from_atoms(current,
            bond_energies=ls.bond_energies, bond_lengths=ls.bond_lengths,
            initial_fraction=ls.initial_fraction, xi=ls.xi)
        response = LSResponseState(ls.target_per_atom, learning_rate=ls.learning_rate)
    run_status = 'completed'
    for index in range(steps):
        before = surface.requests
        work = current.copy()
        climb = []
        landing = prepared = None
        status = 'gaussian_limit'
        error_message = None
        accepted = False
        soft_terms = () if frozen is None else (frozen,)
        if frozen is not None:
            try:
                prepared = prepare_ls_step(work, surface, softening=frozen,
                    fmax=config.fmax, steps=config.relax_steps, optimizer=LBFGS)
                work = prepared.atoms.copy()
            except LSCycleError as error:
                records.append(SSWStep(index, error.stage + '_failed', False, (),
                    error.result, None, surface.requests - before, str(error),
                    work.copy() if error.result is None else error.result.atoms.copy()))
                continue
        anchor = sample_initial_direction(current, rng, mode=config.direction_sampling)
        terms = list(soft_terms)

        def rotation_surface(candidate):
            energy, forces = surface.evaluate(candidate)
            for term in soft_terms:
                de, df = term.evaluate(candidate)
                energy += de
                forces += df
            return energy, forces

        for gaussian_index in range(config.max_gaussians):
            mode = paper_biased_direction(work, anchor,
                rotation_bias=config.rotation_bias, fd_step=config.fd_step,
                max_hvp=config.rotation_hvp, tol=config.rotation_tol,
                evaluate=rotation_surface)
            if not mode.converged:
                status = 'rotation_failed'
                climb.append(dict(index=gaussian_index, residual=mode.residual_norm,
                                  force_requests=mode.force_calls))
                break
            center = work.positions.copy()
            displaced = work.copy()
            displaced.positions += config.width * mode.direction
            background = SurfaceCalculator(surface, terms=terms)
            displaced.calc = background
            force_parallel = float(np.sum(displaced.get_forces() * mode.direction))
            weight = (config.forward_force - force_parallel) * config.width * math.exp(.5)
            if not np.isfinite(weight) or weight <= 0:
                status = 'nonpositive_height'
                climb.append(dict(index=gaussian_index, weight=weight,
                                  background_forward_force=force_parallel))
                break
            terms.append(ProjectedGaussian(center, mode.direction, config.width, weight))
            relaxed = quench(displaced, surface, fmax=config.fmax,
                             steps=config.relax_steps, terms=terms, optimizer=LBFGS)
            event = dict(index=gaussian_index, center=center.tolist(),
                         direction=mode.direction.tolist(), weight=weight,
                         width=config.width, biased_energy=relaxed.energy,
                         max_force=relaxed.max_force, rotation_residual=mode.residual_norm,
                         rotation_force_requests=mode.force_calls,
                         quench_requests=relaxed.evaluation_requests)
            climb.append(event)
            work = relaxed.atoms.copy()
            if not relaxed.converged:
                status = 'biased_quench_failed'
                break
            true_energy, _ = surface.evaluate(work)
            event['true_energy'] = true_energy
            if true_energy < current_energy:
                status = 'lower_true_energy'
                break
        if status in ('gaussian_limit', 'lower_true_energy'):
            landing = quench(work, surface, fmax=config.fmax,
                             steps=config.relax_steps, optimizer=LBFGS)
            if not landing.converged:
                status = 'true_quench_failed'
            else:
                minima.append(landing)
                delta = landing.energy - current_energy
                accepted = delta <= 0 or (config.temperature_K > 0 and
                    rng.random() < math.exp(-delta / (units.kB * config.temperature_K)))
                if landing.energy < best.energy:
                    best = landing
                if accepted:
                    current = landing.atoms.copy()
                    current_energy = landing.energy
        energy_response = None if prepared is None else prepared.energy_response
        if prepared is not None:
            try:
                frozen = response.update(frozen, current,
                    energy_before=prepared.energy_before, energy_after=prepared.energy_after,
                    bond_energies=ls.bond_energies, bond_lengths=ls.bond_lengths)
            except ValueError as error:
                # The paper finite domain has ended, e.g. negative amplitudes.
                # Preserve the completed work and stop, without amplitude clipping.
                status = 'ls_update_failed'
                run_status = 'ls_update_failed'
                error_message = str(error)
        records.append(SSWStep(index, status, bool(accepted), tuple(climb), landing,
                              energy_response, surface.requests - before,
                              error_message, work.copy(), anchor.copy()))
        if run_status != 'completed':
            break
    return SSWResult(initial, current.copy(), best.atoms.copy(), tuple(minima),
                     tuple(records), surface.requests - begin, run_status)


def run_ls_ssw(atoms, surface, *, steps, config, rng, ls):
    """Explicit LS entry point, sharing the independent paper SSW lifecycle."""
    if not isinstance(ls, LSSettings):
        raise TypeError('ls must be LSSettings with explicit bond tables and response target')
    return run_ssw(atoms, surface, steps=steps, config=config, rng=rng, ls=ls)
