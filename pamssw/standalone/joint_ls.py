"""Research-only joint atomic/cell periodic LS preparation.

This helper is an explicit alternative to fixed-cell ``prepare_ls_step``.  It
optimizes physical E + frozen periodic LS + pV in the existing log-strain
metric. The VC walker selects it with ``ls_prequench="joint"``; this is
an independent experimental option, not native LASP parity.
"""
from dataclasses import dataclass
import numpy as np
from .vc_geometry import SymmetricLogStrainChart
from .generalized_numerics import safe_lbfgs
from .ls_cycle import LSCycleError

@dataclass(frozen=True)
class JointLSFailureSnapshot:
    q: object
    atoms: object
    optimizer: object
    requests: int
    joint_gradient_norm: object = None
    true_start_fmax: object = None
    true_start_stress_max: object = None
    softened_fmax: object = None
    softened_stress_max: object = None

@dataclass(frozen=True)
class PreparedJointLSStep:
    q: object
    atoms: object
    true_energy_before: float
    true_energy_after: float
    true_enthalpy_before: float
    true_enthalpy_after: float
    ls_energy_before: float
    ls_energy_after: float
    volume_before: float
    volume_after: float
    optimizer: object
    requests: int
    post_soft_gradient: object
    post_soft_fmax: float
    post_soft_stress_max: float


def prepare_joint_ls_step(atoms, surface, *, softening, config):
    """Prepare one frozen periodic LS state on the full 3N+6 objective.

    Existing VC tolerances and Safe-total controls are read from ``config``;
    this function introduces no numerical parameters.  All failed requests
    count, and failures raise :class:`LSCycleError` with a last-state snapshot.
    """
    chart = SymmetricLogStrainChart(atoms, strain_length=config.strain_length)
    softening._validate_atoms(atoms)
    for distance, reference in zip(softening.pair_distances(atoms), softening.reference_distances):
        allowance = 32 * np.finfo(float).eps * max(1., reference)
        if abs(float(distance) - reference) > allowance:
            raise ValueError('frozen reference distances do not describe this starting geometry')
    start = surface.requests
    q0 = chart.pack(atoms)
    # This stores only convergence metadata for the currently accepted q.
    # Evaluations themselves are never memoized: every Safe-total request is a
    # fresh physical surface request, including repeated coordinates.
    convergence = {}
    try:
        physical_before = surface.evaluate(atoms)
    except Exception as error:
        snapshot = JointLSFailureSnapshot(q0.copy(), chart.unpack(q0), None,
                                          surface.requests-start)
        raise LSCycleError('true_start', str(error), result=snapshot) from error
    physical_before_energy, physical_before_forces, physical_before_stress = physical_before
    physical_before_fmax = float(np.linalg.norm(physical_before_forces, axis=1).max())
    physical_before_stress_max = float(np.abs(physical_before_stress + config.pressure*np.eye(3)).max())
    if (physical_before_fmax > config.fmax
            or physical_before_stress_max > config.stress_tol):
        snapshot = JointLSFailureSnapshot(q0.copy(), chart.unpack(q0), None,
                                          surface.requests-start, None,
                                          physical_before_fmax, physical_before_stress_max)
        raise LSCycleError('true_start', 'physical force or stress exceeds tolerance',
                           result=snapshot)

    def evaluate(q):
        key = np.asarray(q, dtype=float).tobytes()
        def combined(trial):
            e, f, stress = surface.evaluate(trial)
            le, lf, ls = softening.evaluate_stress(trial)
            return e + le, f + lf, stress + ls
        ev = chart.evaluate(q, combined, pressure=config.pressure)
        fmax = float(np.linalg.norm(ev.forces, axis=1).max())
        stress_max = float(np.abs(ev.stress + config.pressure*np.eye(3)).max())
        ev_gradient = chart.project(ev.gradient)
        joint_norm = max(float(np.linalg.norm(ev_gradient[:-6].reshape(-1, 3), axis=1).max()),
                         float(np.linalg.norm(ev_gradient[-6:])))
        convergence[key] = (
            max(joint_norm/config.gradient_tol,
                fmax/config.fmax, stress_max/config.stress_tol),
            joint_norm, fmax, stress_max,
        )
        return ev.objective, ev_gradient

    def norm(g):
        return max(float(np.linalg.norm(g[:-6].reshape(-1, 3), axis=1).max()),
                   float(np.linalg.norm(g[-6:])))

    try:
        result = safe_lbfgs(q0, evaluate, gradient_norm=norm, step_norm=norm,
            convergence_norm=lambda q, g: convergence[np.asarray(q).tobytes()][0],
            gtol=1., max_step=config.max_step, maxiter=config.relax_steps,
            lbfgs_memory=config.lbfgs_memory)
    except LSCycleError:
        raise
    except Exception as error:
        snapshot=JointLSFailureSnapshot(q0.copy(), chart.unpack(q0), None,
                                        surface.requests-start)
        raise LSCycleError('joint_soft_quench', str(error), result=snapshot) from error

    qlast = result.q.copy() if result.q is not None else q0.copy()
    try:
        last_atoms = chart.unpack(qlast)
    except Exception as error:
        snapshot = JointLSFailureSnapshot(qlast, None, result,
                                          surface.requests-start)
        raise LSCycleError('joint_soft_quench', str(error), result=snapshot) from error
    if result.energy is None or not result.converged:
        metrics = convergence.get(qlast.tobytes())
        snapshot=JointLSFailureSnapshot(qlast, last_atoms, result, surface.requests-start,
            None if metrics is None else metrics[1], None, None,
            None if metrics is None else metrics[2],
            None if metrics is None else metrics[3])
        raise LSCycleError('joint_soft_quench', 'joint objective did not converge', result=snapshot)
    # Obtain the final physical certificate independently from the accepted
    # objective evaluation.  This is deliberately another charged request.
    try:
        physical_after_energy, physical_after_forces, physical_after_stress = surface.evaluate(last_atoms)
        ls_after, ls_after_forces, ls_after_stress = softening.evaluate_stress(last_atoms)
    except Exception as error:
        snapshot = JointLSFailureSnapshot(qlast, last_atoms, result,
                                          surface.requests-start)
        raise LSCycleError('true_finish', str(error), result=snapshot) from error
    softened_forces = physical_after_forces + ls_after_forces
    softened_stress = physical_after_stress + ls_after_stress
    softened_fmax = float(np.linalg.norm(softened_forces, axis=1).max())
    softened_stress_max = float(np.abs(softened_stress + config.pressure*np.eye(3)).max())
    fresh_ev = chart.evaluate(
        qlast,
        lambda trial: (physical_after_energy + ls_after,
                       physical_after_forces + ls_after_forces,
                       physical_after_stress + ls_after_stress),
        pressure=config.pressure)
    post_soft_gradient = chart.project(fresh_ev.gradient)
    post_joint_norm = max(
        float(np.linalg.norm(post_soft_gradient[:-6].reshape(-1, 3), axis=1).max()),
        float(np.linalg.norm(post_soft_gradient[-6:])),
    )
    if (post_joint_norm > config.gradient_tol
            or softened_fmax > config.fmax
            or softened_stress_max > config.stress_tol):
        snapshot = JointLSFailureSnapshot(qlast, last_atoms, result,
                                          surface.requests-start, post_joint_norm, None, None,
                                          softened_fmax, softened_stress_max)
        raise LSCycleError('true_finish', 'softened force or stress exceeds tolerance',
                           result=snapshot)
    volume_before = float(atoms.get_volume())
    volume_after = float(last_atoms.get_volume())
    e_before = float(physical_before_energy)
    ls_before, _, _ = softening.evaluate_stress(atoms)
    return PreparedJointLSStep(qlast, last_atoms, e_before, float(physical_after_energy),
        float(e_before + config.pressure*volume_before),
        float(physical_after_energy + config.pressure*volume_after),
        float(ls_before), float(ls_after),
        volume_before, volume_after,
        result, surface.requests-start, post_soft_gradient,
        softened_fmax, softened_stress_max)
