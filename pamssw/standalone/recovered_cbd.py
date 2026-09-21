"""Experimental fixed-cell CBD stage composition, independent of LASP binaries.

This composes recovered Broyden arithmetic and caller transitions. It is not
the complete native direction generator or a production SSW policy. In
particular ASE constraints and Run_type-specific parameter
selection are excluded. All tolerances and budgets are explicit. Euclidean
history is an intentional alternative to the recovered block-sum arithmetic.
No native-equivalence or material-search performance claim follows from the
numerical tests of this wrapper.
"""
from dataclasses import dataclass

import numpy as np

from ._broyden_state import BroydenState
from .broyden_direction import (
    NATIVE_WEIGHT, NATIVE_HISTORY_LIMIT, NATIVE_SPECTRAL_LIMIT, _integer, _scalar,
)
from .native_rotation_control import cap_rotation, retry_factor, finish_rotation


@dataclass(frozen=True)
class RecoveredCBDResult:
    direction: np.ndarray
    curvature: float
    real_curvature: float
    residual_norm: float
    force_calls: int
    stage: str
    stage_complete: bool
    converged: bool
    stop_reason: str
    rotation_weight: float
    trace: tuple
    bias_reference: np.ndarray


def recovered_cbd_direction(atoms, anchor, *, fd_step, max_force_calls,
                            pre_rotmax, rotmax, pre_ftol, ftol, metric, evaluate,
                            project=None):
    """Run PreRot → biased/unbiased rotation using one center force snapshot.

    ``evaluate(atoms)`` supplies energy and forces on the SAME rotation
    surface throughout, including frozen LS terms when present. ``ftol`` and
    ``pre_ftol`` apply to the recovered force quantity ``10*||tangent_force||``
    (eV/Å), not the HVP residual. ``rotmax`` uses the native strict ``>`` test;
    reaching it ends a stage without certifying force convergence.

    The first center and each endpoint count toward ``max_force_calls``.
    Algebraic FACT1 retries and same-endpoint stage transitions cost no extra
    evaluations. The external budget always returns an evaluated direction.
    Only unconstrained Cartesian coordinates are currently implemented.
    Optional ``project`` removes rigid components using the fixed center's
    geometry, after angular capping and before endpoint requests, following
    rotate_dimer 0x6e780e. It does not alter the supplied physical forces or
    turn the recovered stopping quantity into a projected-Hessian residual.
    """
    if atoms.constraints:
        raise ValueError('recovered CBD currently requires unconstrained atoms')
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError('requires finite nonempty atom positions')
    dr = _scalar(fd_step, 'fd_step')
    pre_ftol = _scalar(pre_ftol, 'pre_ftol')
    ftol = _scalar(ftol, 'ftol')
    max_force_calls = _integer(max_force_calls, 'max_force_calls', 2)
    pre_rotmax = _integer(pre_rotmax, 'pre_rotmax', 0)
    rotmax = _integer(rotmax, 'rotmax', 0)
    if metric not in ('euclidean', 'native_block_sum'):
        raise ValueError('unsupported Broyden metric')
    if not callable(evaluate):
        raise ValueError('evaluate callback required')
    if project is not None and not callable(project):
        raise ValueError('project must be callable or None')
    shape = atoms.positions.shape
    direction = np.asarray(anchor, dtype=float)
    if direction.shape != shape or not np.isfinite(direction).all():
        raise ValueError('anchor requires finite shape (N, 3)')

    def projected_unit(vector):
        value = np.asarray(project(vector.reshape(shape)), dtype=float)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('project returned invalid direction')
        norm = np.linalg.norm(value)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError('projected direction must be finite and nonzero')
        return value.ravel()/norm

    if project is not None:
        direction = projected_unit(direction).reshape(shape)
    length = np.linalg.norm(direction)
    if not np.isfinite(length) or length == 0:
        raise ValueError('anchor must be nonzero and finite')
    direction = direction.ravel().copy()/length
    center = atoms.positions.ravel().copy()
    force_calls = 0
    trace = []

    def force_at(x):
        nonlocal force_calls
        candidate = atoms.copy()
        candidate.calc = atoms.calc
        candidate.set_positions(x.reshape(shape), apply_constraint=False)
        force_calls += 1
        energy, force = evaluate(candidate)
        force = np.asarray(force, dtype=float)
        if np.ndim(energy) or not np.isfinite(energy):
            raise ValueError('evaluator returned invalid energy')
        if force.shape != shape or not np.isfinite(force).all():
            raise ValueError('evaluator returned invalid force')
        return force.ravel().copy()

    def new_history():
        return BroydenState(np.ones(center.size), weight=NATIVE_WEIGHT,
                            metric=metric, history_limit=NATIVE_HISTORY_LIMIT,
                            spectral_limit=NATIVE_SPECTRAL_LIMIT)

    center_force = force_at(center)
    raw_force = force_at(center + dr*direction)
    state = new_history()
    stage = 'CBD_PreRot'
    rotnum = 1
    factor = .05  # Recovered modelevel=0 FACT; not a tuned recommendation.
    reference = direction.copy()
    weight = 0.

    while True:
        hv = (center_force-raw_force)/dr
        real_curvature = float(np.dot(direction, hv))
        if stage == 'CBD_biasedRot' and real_curvature < -1e-6:
            stage, rotnum, factor, state = 'CBD_UnbiasedRot', 1, .05, new_history()
        biased_hv = hv.copy()
        if stage == 'CBD_biasedRot':
            biased_hv -= weight*np.dot(reference, direction)*reference
        curvature = float(np.dot(direction, biased_hv))
        residual_vector = biased_hv-curvature*direction
        residual = float(np.linalg.norm(residual_vector))
        tangent = -dr*residual_vector
        before = state.history_size
        endpoint = center+dr*direction
        update = state.step(endpoint, factor*tangent)
        retries = 0
        while True:
            displacement = (update.x-center)/dr
            reduced = retry_factor(rotnum, retries+1, float(np.linalg.norm(displacement)), factor)
            if reduced is None:
                break
            retries += 1
            factor = reduced
            state = new_history()
            update = state.step(endpoint, factor*tangent)
        candidate_direction = cap_rotation(direction.reshape(shape), displacement.reshape(shape)).ravel()
        if project is not None:
            candidate_direction = projected_unit(candidate_direction)
        finish = finish_rotation(
            direction.reshape(shape), candidate_direction.reshape(shape),
            rotnum=rotnum, rotmax=pre_rotmax if stage == 'CBD_PreRot' else rotmax,
            reported_force=10.*float(np.linalg.norm(tangent)),
            ftol=pre_ftol if stage == 'CBD_PreRot' else ftol,
            infor=stage, curv_real=real_curvature,
        )
        trace.append(dict(event='rotation', stage=stage, rotnum=rotnum,
                          force_calls=force_calls, curvature=curvature,
                          real_curvature=real_curvature, residual_norm=residual,
                          history_size_before=before, retries=retries,
                          factor1=factor, force_converged=finish.force_converged,
                          rotation_limit=finish.budget_exceeded,
                          prerot_override=finish.prerot_override))
        if stage == 'CBD_PreRot' and curvature < -1e-6:
            stage = 'CBD_UnbiasedRot'
        if finish.stop:
            if stage == 'CBD_PreRot' and curvature > -1e-6:
                # Native caller restores work1, saves evaluated n as anchor,
                # and enters biasedrot on the SAME endpoint without a PES call.
                reference = direction.copy()
                weight = curvature
                stage, rotnum, factor, state = 'CBD_biasedRot', 1, .05, new_history()
                continue
            return RecoveredCBDResult(
                direction.reshape(shape).copy(), curvature, real_curvature, residual,
                force_calls, stage, True, finish.force_converged,
                'force_tolerance' if finish.force_converged else 'rotation_limit', weight, tuple(trace), reference.reshape(shape).copy())
        if force_calls >= max_force_calls:
            return RecoveredCBDResult(
                direction.reshape(shape).copy(), curvature, real_curvature, residual,
                force_calls, stage, False, False, 'force_budget', weight, tuple(trace), reference.reshape(shape).copy())
        direction = candidate_direction
        rotnum = finish.next_rotnum
        raw_force = force_at(center+dr*direction)
