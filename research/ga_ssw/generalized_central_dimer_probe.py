"""Research-only flat central-difference generalized dimer probe."""

import numpy as np

from pamssw.standalone.direction import SoftModeResult


def _flat(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite nonempty flat vector")
    return array.copy()


def solve(q0, anchor, *, rotation_bias, fd_step, max_force_calls, tol, evaluate):
    """Return a two-vector central-difference dimer mode.

    The callback returns ``(energy, gradient)`` for a flat coordinate vector.
    Each HVP evaluates only ``q0 +/- fd_step*n``; there is no center-gradient
    request.  The first HVP costs two requests, and every rotation costs one
    tangent HVP plus one proposed-direction HVP (four requests).  The current
    direction's residual is already available from its proposed HVP, so no
    extra direct check is made.  Requests are charged before callback entry,
    including a callback that raises.  This is a numerical substitution probe,
    not native CBD parity or a production solver.
    """
    center = _flat(q0, "q0")
    n0 = _flat(anchor, "anchor")
    if center.shape != n0.shape:
        raise ValueError("anchor and q0 shapes differ")
    if isinstance(max_force_calls, (bool, np.bool_)) or not isinstance(max_force_calls, (int, np.integer)):
        raise ValueError("max_force_calls must be an integer budget")
    budget = int(max_force_calls)
    if budget < 2:
        raise ValueError("max_force_calls must allow one central HVP")
    try:
        beta = float(rotation_bias)
        step = float(fd_step)
        tolerance = float(tol)
    except (TypeError, ValueError):
        raise ValueError("rotation_bias, fd_step, and tol must be finite scalars") from None
    if not np.isfinite(beta) or beta < 0:
        raise ValueError("rotation_bias must be finite and nonnegative")
    if not np.isfinite(step) or step <= 0 or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("fd_step and tol must be finite and positive")
    if not callable(evaluate):
        raise ValueError("evaluate callback required")
    norm = np.linalg.norm(n0)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("anchor requires finite nonzero norm")
    n0 /= norm
    force_calls = 0
    hvp_calls = 0

    def gradient_at(q):
        nonlocal force_calls
        force_calls += 1
        energy, gradient = evaluate(np.array(q, copy=True))
        gradient = np.asarray(gradient, dtype=float)
        if np.ndim(energy) != 0 or not np.isfinite(energy) or gradient.shape != center.shape or not np.isfinite(gradient).all():
            raise ValueError("evaluator returned invalid energy/gradient")
        return gradient

    def hvp(direction):
        nonlocal hvp_calls
        plus = gradient_at(center + step * direction)
        minus = gradient_at(center - step * direction)
        hvp_calls += 1
        return (plus - minus) / (2 * step) - beta * np.dot(n0, direction) * n0

    n = n0.copy()
    hn = hvp(n)
    symmetry_error = 0.0
    while True:
        curvature = float(np.dot(n, hn))
        residual = hn - curvature * n
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tolerance or force_calls + 4 > budget:
            break
        tangent = -residual
        tangent -= np.dot(tangent, n) * n
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm <= np.finfo(float).eps * max(1., np.linalg.norm(hn)):
            break
        tangent /= tangent_norm
        ht = hvp(tangent)
        projected = np.array([[curvature, np.dot(n, ht)],
                              [np.dot(tangent, hn), np.dot(tangent, ht)]])
        symmetry_error = max(symmetry_error,
                             float(np.linalg.norm(projected - projected.T)))
        _, vectors = np.linalg.eigh((projected + projected.T) / 2)
        proposed = vectors[0, 0] * n + vectors[1, 0] * tangent
        proposed /= np.linalg.norm(proposed)
        if np.dot(proposed, n0) < 0:
            proposed = -proposed
        n = proposed
        hn = hvp(n)

    curvature = float(np.dot(n, hn))
    residual_norm = float(np.linalg.norm(hn - curvature * n))
    return SoftModeResult(n, curvature, residual_norm, hvp_calls, force_calls,
                          residual_norm <= tolerance, symmetry_error)
