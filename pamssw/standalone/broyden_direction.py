"""Public experimental ASE-facing direction runner built from recovered state.

Design boundary: this runner follows a *root-selection* direction problem.  It
seeks a local stationary direction from the supplied anchor within a bounded
iteration, and makes no lowest-eigenmode claim. A returned result can be
unconverged when its endpoint budget is exhausted.
The native-control boundary is also explicit: the recurrence, FACT1 retry
algebra, 40-degree cap, and rank-one anchor bias are useful numerical
comparisons, while the stopping test here is a directly measured finite-
endpoint HVP residual.  It is therefore an independent comparison, not a
native ``CBD_PreRot`` or native ``ftol`` implementation.

The public wrapper chooses a rotation-equivariant Euclidean metric and fixes
FACT to the recovered level-0 value0.05; no ASE
calculator is run unless the caller invokes the solver.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from pamssw.standalone.native_rotation_control import cap_rotation, retry_factor
from ._broyden_state import BroydenState


NATIVE_WEIGHT = 1000.0
NATIVE_HISTORY_LIMIT = 50
NATIVE_SPECTRAL_LIMIT = 1.0e7


def paper_broyden_direction(atoms, anchor, *, rotation_bias, fd_step, max_hvp, tol, evaluate):
    """Public Euclidean Broyden direction with recovered native FACT=0.05.

    This is an explicit numerical alternative to Ritz/dimer and uses direct
    endpoint residual certification; it does not claim native CBD stopping
    parity.
    """
    return broyden_direction(atoms, anchor, rotation_bias=rotation_bias,
                             fd_step=fd_step, max_hvp=max_hvp, tol=tol,
                             initial_factor=0.05, metric="euclidean",
                             evaluate=evaluate)


@dataclass(frozen=True)
class BroydenDirectionResult:
    direction: np.ndarray
    curvature: float
    residual_norm: float
    hvp_calls: int
    force_calls: int
    converged: bool
    projected_symmetry_error: float
    trace: tuple
    stop_reason: str = 'unspecified'


def _scalar(value, name, positive=True):
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite scalar") from exc
    if not np.isfinite(out) or (positive and out <= 0):
        qualifier = "positive" if positive else "finite"
        raise ValueError(f"{name} must be a finite {qualifier} scalar")
    return out


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def broyden_direction(
    atoms,
    anchor,
    *,
    rotation_bias,
    fd_step,
    max_hvp,
    tol,
    initial_factor,
    metric,
    evaluate,
):
    """Return a bounded independent CBD-style direction.

    ``evaluate(atoms)`` must return ``(energy, forces)`` in ASE units.  One
    center force is evaluated, then at most ``max_hvp`` endpoint forces are
    requested.  Each endpoint is ``center + fd_step * n``.  The raw endpoint
    force difference gives ``Hn``; the rank-one bias uses the original anchor:
    ``H_b n = Hn - a*(anchor.n)*anchor``.  The scaled tangent response passed
    to :class:`BroydenState` is
    ``FACT1 * (-fd_step * (H_b n - (n.H_b n)*n))``.

    The Broyden proposal is normalized and passed through the recovered
    40-degree cap.  On the first rotation only, proposals with norm above the
    recovered 1.02 threshold retry algebraically with ``FACT1 *= .8``; the
    current endpoint force is reused and the state is reset.  No PES request
    is made by a retry.  Returned directions are always endpoint-evaluated.
    """
    if atoms.constraints:
        raise ValueError("broyden direction requires unconstrained atoms")
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError("requires finite nonempty atom positions")
    dr = _scalar(fd_step, "fd_step")
    tol = _scalar(tol, "tol")
    factor = _scalar(initial_factor, "initial_factor")
    bias = _scalar(rotation_bias, "rotation_bias", positive=False)
    if bias < 0:
        raise ValueError("rotation_bias must be nonnegative")
    max_hvp = _integer(max_hvp, "max_hvp", 1)
    if metric not in {"native_block_sum", "euclidean"}:
        raise ValueError("metric must be 'native_block_sum' or 'euclidean'")
    if not callable(evaluate):
        raise ValueError("evaluate callback required")

    shape = atoms.positions.shape
    anchor_array = np.asarray(anchor, dtype=float)
    if anchor_array.shape != shape or not np.isfinite(anchor_array).all():
        raise ValueError("anchor requires finite shape (N, 3)")
    anchor_flat = anchor_array.ravel().copy()
    anchor_norm = np.linalg.norm(anchor_flat)
    if not np.isfinite(anchor_norm) or anchor_norm == 0:
        raise ValueError("anchor requires finite nonzero norm")
    anchor_flat /= anchor_norm
    center = atoms.positions.copy()
    center_flat = center.ravel()
    force_calls = 0
    hvp_calls = 0
    trace = []

    def force_at(positions):
        nonlocal force_calls
        candidate = atoms.copy()
        candidate.calc = atoms.calc
        candidate.set_positions(np.asarray(positions, dtype=float).reshape(shape), apply_constraint=False)
        force_calls += 1
        energy, forces = evaluate(candidate)
        forces = np.asarray(forces, dtype=float)
        if np.ndim(energy) != 0 or not np.isfinite(energy):
            raise ValueError("evaluator returned invalid energy")
        if forces.shape != shape or not np.isfinite(forces).all():
            raise ValueError("evaluator returned invalid forces")
        return forces.ravel().copy()

    center_force = force_at(center)
    state = BroydenState(
        np.ones(center_flat.size), weight=NATIVE_WEIGHT, metric=metric,
        history_limit=NATIVE_HISTORY_LIMIT, spectral_limit=NATIVE_SPECTRAL_LIMIT,
    )
    direction = anchor_flat.copy()
    endpoint = center_flat + dr * direction
    last = None

    while hvp_calls < max_hvp:
        endpoint_force = force_at(endpoint)
        hvp_calls += 1
        hessian_direction = (center_force - endpoint_force) / dr
        biased_hessian_direction = hessian_direction - bias * np.dot(anchor_flat, direction) * anchor_flat
        curvature = float(np.dot(direction, biased_hessian_direction))
        residual_vector = biased_hessian_direction - curvature * direction
        residual = float(np.linalg.norm(residual_vector))
        last = (direction.copy(), curvature, residual)
        trace.append({
            "event": "endpoint",
            "hvp": hvp_calls,
            "force_calls": force_calls,
            "factor1": factor,
            "curvature": curvature,
            "residual_norm": residual,
            "history_size": state.history_size,
            "endpoint": endpoint.copy().tolist(),
        })
        if not np.isfinite(curvature) or not np.isfinite(residual):
            raise ValueError("nonfinite biased HVP certificate")
        if residual <= tol:
            return BroydenDirectionResult(direction.reshape(shape), curvature, residual,
                                          hvp_calls, force_calls, True, 0., tuple(trace),
                                          'residual_converged')
        if hvp_calls >= max_hvp:
            break

        unscaled_response = -dr * residual_vector
        proposal = state.step(endpoint, factor * unscaled_response)
        retries = 0
        while True:
            candidate_displacement = (proposal.x - center_flat) / dr
            candidate_norm = float(np.linalg.norm(candidate_displacement))
            reduced = retry_factor(hvp_calls, retries + 1, candidate_norm, factor)
            if reduced is None:
                break
            retries += 1
            factor = reduced
            state = BroydenState(
                np.ones(center_flat.size), weight=NATIVE_WEIGHT, metric=metric,
                history_limit=NATIVE_HISTORY_LIMIT, spectral_limit=NATIVE_SPECTRAL_LIMIT,
            )
            proposal = state.step(endpoint, factor * unscaled_response)
        candidate = cap_rotation(direction.reshape(shape),
                                 candidate_displacement.reshape(shape)).ravel()
        trace.append({
            "event": "proposal",
            "hvp": hvp_calls,
            "retries": retries,
            "factor1": factor,
            "candidate_norm_before_cap": candidate_norm,
            "history_size": state.history_size,
            "dropped": (None if proposal.dropped == 0 else proposal.dropped),
            "restarted": bool(proposal.restarted),
            "candidate": candidate.tolist(),
        })
        direction = candidate
        endpoint = center_flat + dr * direction

    direction, curvature, residual = last
    return BroydenDirectionResult(direction.reshape(shape), curvature, residual,
                                  hvp_calls, force_calls, False, 0., tuple(trace),
                                  'budget_exhausted')
