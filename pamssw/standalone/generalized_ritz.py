"""Optional flat-coordinate symmetric Ritz soft-mode solver.

This is a coordinate-level adaptation of ``standalone.direction``.  It has
no ASE or calculator dependency and is not connected to any native SSW driver.  The
callback returns ``(energy, gradient)`` in consistent units.  The optional
rank-one rotation term is the analytic anchor projector
``-rotation_bias * (q-q0).anchor**2 / 2``.
"""

import numpy as np

from pamssw.standalone.direction import SoftModeResult


def _flat(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty flat vector")
    return array.copy()


def solve(q0, anchor, evaluate, rotation_bias, fd_step, tol,
          max_force_calls, finite_difference):
    """Solve a low-curvature Ritz mode under an explicit force-call budget.

    ``q0`` is the shared center and ``anchor`` is the initial Krylov vector;
    both are arbitrary-dimensional flat vectors.  ``evaluate(q)`` returns
    ``(energy, gradient)`` and is called only at the finite-difference points.
    Forward differences use one shared center gradient and one endpoint per
    HVP.  Central differences use two endpoint gradients per HVP.  The last
    HVP is always reserved for the returned direction's direct residual, so
    ``force_calls`` never exceeds ``max_force_calls``.
    """
    center = _flat(q0, "q0")
    initial = _flat(anchor, "anchor")
    if initial.shape != center.shape:
        raise ValueError("q0 and anchor shapes differ")
    norm = np.linalg.norm(initial)
    if norm == 0:
        raise ValueError("anchor must have nonzero norm")
    initial /= norm
    try:
        bias = float(rotation_bias)
    except (TypeError, ValueError):
        raise ValueError("rotation_bias must be finite and nonnegative") from None
    if not np.isfinite(bias) or bias < 0:
        raise ValueError("rotation_bias must be finite and nonnegative")
    try:
        step = float(fd_step)
        tolerance = float(tol)
    except (TypeError, ValueError):
        raise ValueError("fd_step and tol must be finite and positive") from None
    if not np.isfinite(step) or step <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("fd_step and tol must be finite and positive")
    if isinstance(max_force_calls, (bool, np.bool_)):
        raise ValueError("max_force_calls must be an integer force budget")
    try:
        budget = int(max_force_calls)
    except (TypeError, ValueError):
        raise ValueError("max_force_calls must be an integer force budget") from None
    if budget != max_force_calls:
        raise ValueError("max_force_calls must be an integer force budget")
    if finite_difference not in ("forward", "central"):
        raise ValueError("finite_difference must be forward or central")
    minimum = 3 if finite_difference == "forward" else 4
    if budget < minimum:
        raise ValueError(f"max_force_calls must be at least {minimum}")
    if not callable(evaluate):
        raise ValueError("evaluate callback is required")

    force_calls = 0

    def call(q):
        nonlocal force_calls
        if force_calls >= budget:
            raise RuntimeError("maximum force-call budget exhausted")
        force_calls += 1
        energy, gradient = evaluate(np.array(q, copy=True))
        gradient = np.asarray(gradient, dtype=float)
        if gradient.shape != center.shape or not np.isfinite(gradient).all():
            raise ValueError("evaluator returned invalid gradient")
        if not np.isfinite(energy):
            raise ValueError("evaluator returned invalid energy")
        projection = float(np.dot(q - center, initial))
        return (float(energy) - .5 * bias * projection * projection,
                gradient - bias * projection * initial)

    center_gradient = call(center)[1] if finite_difference == "forward" else None
    h_calls = 0

    def hessian_vector(vector):
        nonlocal h_calls
        if finite_difference == "forward":
            endpoint = call(center + step * vector)[1]
            value = (endpoint - center_gradient) / step
        else:
            plus = call(center + step * vector)[1]
            minus = call(center - step * vector)[1]
            value = (plus - minus) / (2 * step)
        h_calls += 1
        return value

    per_hvp = 1 if finite_difference == "forward" else 2
    iterative = (budget - (1 if finite_difference == "forward" else 0)) // per_hvp - 1
    iterative = max(1, iterative)
    basis = []
    images = []
    vector = initial.copy()
    symmetry_error = 0.0
    direction = initial.copy()
    for _ in range(min(iterative, center.size)):
        basis.append(vector.copy())
        images.append(hessian_vector(vector))
        q = np.column_stack(basis)
        hq = np.column_stack(images)
        projected = q.T @ hq
        symmetry_error = float(np.linalg.norm(projected - projected.T))
        values, coefficients = np.linalg.eigh((projected + projected.T) / 2)
        direction = q @ coefficients[:, 0]
        surrogate = hq @ coefficients[:, 0] - values[0] * direction
        if np.linalg.norm(surrogate) <= tolerance:
            break
        vector = images[-1].copy()
        for _ in range(2):
            for previous in basis:
                vector -= np.dot(previous, vector) * previous
        next_norm = np.linalg.norm(vector)
        if next_norm <= np.finfo(float).eps * max(1., np.linalg.norm(images[-1])):
            break
        vector /= next_norm

    direction /= np.linalg.norm(direction)
    if np.dot(direction, initial) < 0:
        direction *= -1
    direct_hvp = hessian_vector(direction)
    curvature = float(np.dot(direction, direct_hvp))
    residual = float(np.linalg.norm(direct_hvp - curvature * direction))
    return SoftModeResult(direction, curvature, residual, h_calls, force_calls,
                          residual <= tolerance, symmetry_error)
