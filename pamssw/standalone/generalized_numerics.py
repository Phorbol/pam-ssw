"""Flat-coordinate Safe-total and plane dimer numerics; no Atoms surrogate.

The caller owns coordinate/gradient consistency and physical norm definitions.
Safe-total constants and secant algebra are reused from pamssw.relax. This module
adds no VC metric, bias, strain representation or physical acceptance criterion.
"""
from dataclasses import dataclass
from typing import Callable
import numpy as np
from pamssw.relax import (_lbfgs_inverse_product, _accept_lbfgs_curvature,
    _validate_lbfgs_memory,
    _SAFE_LBFGS_MEMORY, _SAFE_LBFGS_ARMIJO_C1, _SAFE_LBFGS_BACKTRACK,
    _SAFE_LBFGS_MAX_LINE_TRIALS, _SAFE_LBFGS_MIN_ALPHA)
from .direction import SoftModeResult


@dataclass(frozen=True)
class GeneralizedRelaxResult:
    q: np.ndarray
    energy: float | None
    gradient: np.ndarray | None
    status: str
    steps: int
    requests: int
    trace: tuple
    accepted_secants: int
    rejected_secants: int
    rejected_trials: int
    error: str | None = None

    @property
    def converged(self):
        return self.status == 'converged'


def _flat(value, name):
    q = np.asarray(value, dtype=float)
    if q.ndim != 1 or not q.size or not np.isfinite(q).all():
        raise ValueError(f'{name} must be a finite nonempty flat vector')
    return q.copy()


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be integer >= {minimum}')


def _positive(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')


def _evaluation(evaluate, q):
    energy, gradient = evaluate(q.copy())
    gradient = np.asarray(gradient, dtype=float)
    if np.ndim(energy) != 0 or not np.isfinite(energy) or gradient.shape != q.shape or not np.isfinite(gradient).all():
        raise ValueError('evaluator returned invalid energy/gradient')
    return float(energy), gradient.copy()


def safe_lbfgs(q0, evaluate, *, gradient_norm, step_norm, gtol, max_step,
               maxiter, max_requests=None, convergence_norm=None, lbfgs_memory=None):
    """Existing Safe-total Armijo/positive-curvature policy in flat coordinates.

    evaluate(q)->(E,dE/dq); gradient_norm(g) and step_norm(dq) return finite
    nonnegative scalars in caller-declared units. gtol and max_step use those
    units. Step limiting multiplies the entire direction by a single scalar,
    preserving descent and the existing Safe-total policy. No adaptive scaling,
    restart or invalid-trial recovery is added. max_requests counts attempted
    evaluator calls, including a call raising an exception. trace contains only
    initial/accepted points. Optional convergence_norm(q,g) overrides the stopping
    norm using only the current accepted pair, including after rejected trials;
    it must not perform extra evaluator requests. Errors and failed line searches return that last
    accepted point. If the very first call fails, energy/gradient are None.
    """
    _validate_lbfgs_memory(lbfgs_memory, 'safe-lbfgs-total')
    history_limit = _SAFE_LBFGS_MEMORY if lbfgs_memory is None else int(lbfgs_memory)
    q = _flat(q0, 'q0');_positive(gtol, 'gtol');_positive(max_step, 'max_step')
    _integer(maxiter, 'maxiter', 0)
    if max_requests is not None:_integer(max_requests, 'max_requests', 1)
    if convergence_norm is not None and not callable(convergence_norm):
        raise ValueError('convergence_norm must be callable when supplied')
    if not all(callable(f) for f in (evaluate, gradient_norm, step_norm)):
        raise ValueError('evaluate and both norm callbacks required')
    energy = gradient = None; requests = steps = rejected_trials = accepted_secants = rejected_secants = 0
    history = []; trace = []; status = 'maxiter'; error = None
    def norm(callback, vector):
        value = callback(vector.copy())
        if np.ndim(value) != 0 or not np.isfinite(value) or value < 0:
            raise ValueError('norm callback must return a finite nonnegative scalar')
        return float(value)
    def convergence_measure():
        if convergence_norm is None:
            return norm(gradient_norm, gradient)
        value = convergence_norm(q.copy(), gradient.copy())
        if np.ndim(value) != 0 or not np.isfinite(value) or value < 0:
            raise ValueError('convergence_norm must return a finite nonnegative scalar')
        return float(value)
    def snapshot():
        trace.append(dict(q=q.copy(), energy=energy, gradient=gradient.copy(),
                          gradient_norm=convergence_measure(), requests=requests, step=steps))
    try:
        requests += 1;energy, gradient = _evaluation(evaluate, q);snapshot()
        while steps < maxiter:
            if convergence_measure() <= gtol:status = 'converged';break
            direction = -_lbfgs_inverse_product(gradient, history)
            length = norm(step_norm, direction)
            if length > max_step:direction *= max_step / length
            derivative = float(gradient @ direction)
            if not np.isfinite(direction).all() or not np.isfinite(derivative):status = 'nonfinite_direction';break
            if derivative >= 0:status = 'non_descent_direction';break
            accepted = False; alpha = 1.
            for _ in range(_SAFE_LBFGS_MAX_LINE_TRIALS):
                if max_requests is not None and requests >= max_requests:status = 'request_limit';break
                trial = q + alpha * direction
                requests += 1
                trial_energy, trial_gradient = _evaluation(evaluate, trial)
                if trial_energy <= energy + _SAFE_LBFGS_ARMIJO_C1 * alpha * derivative:
                    accepted = True;break
                rejected_trials += 1;alpha *= _SAFE_LBFGS_BACKTRACK
                if alpha < _SAFE_LBFGS_MIN_ALPHA:break
            if not accepted:
                if status != 'request_limit':status = 'line_search_failed'
                break
            s = trial - q;y = trial_gradient - gradient
            if _accept_lbfgs_curvature(s, y):
                history.append((s.copy(), y.copy(), 1. / float(s @ y)))
                history = history[-history_limit:];accepted_secants += 1
            else:rejected_secants += 1
            q = trial.copy();energy = trial_energy;gradient = trial_gradient.copy();steps += 1;snapshot()
        if convergence_measure() <= gtol:status = 'converged'
    except Exception as exc:
        status = 'evaluation_failed';error = f'{type(exc).__name__}: {exc}'
    return GeneralizedRelaxResult(q.copy(), energy, None if gradient is None else gradient.copy(),
                                  status, steps, requests, tuple(trace), accepted_secants,
                                  rejected_secants, rejected_trials, error)


def generalized_dimer(q0, anchor, *, rotation_bias, fd_step, max_hvp, tol, evaluate):
    """Flat equivalent of paper_dimer_direction, using gradient differences.

    Coordinates have the caller's Euclidean metric. No metric or constraints are
    inferred. force_calls in the shared SoftModeResult means E/g requests here.
    A failed evaluator raises; the caller's evaluator counter records its cost.
    """
    center = _flat(q0, 'q0');n0 = _flat(anchor, 'anchor')
    if n0.shape != center.shape:raise ValueError('anchor and q0 shapes differ')
    _positive(fd_step, 'fd_step');_positive(tol, 'tol');_integer(max_hvp, 'max_hvp', 1)
    if not np.isfinite(rotation_bias) or rotation_bias < 0:raise ValueError('rotation_bias must be finite and nonnegative')
    if not callable(evaluate):raise ValueError('evaluate callback required')
    magnitude = np.linalg.norm(n0)
    if magnitude == 0 or not np.isfinite(magnitude):raise ValueError('anchor requires finite nonzero norm')
    n0 /= magnitude;force_calls = hvp_calls = 0
    def gradient_at(q):
        nonlocal force_calls
        force_calls += 1
        return _evaluation(evaluate, q)[1]
    g0 = gradient_at(center)
    def hvp(n):
        nonlocal hvp_calls
        g1 = gradient_at(center + fd_step * n);hvp_calls += 1
        return (g1 - g0) / fd_step - rotation_bias * np.dot(n0, n) * n0
    n = n0.copy();hn = hvp(n);symmetry_error = 0.
    while True:
        curvature = float(n @ hn);residual = hn - curvature * n
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tol or hvp_calls + 2 > max_hvp:break
        t = -residual;t -= np.dot(t, n) * n;t /= np.linalg.norm(t)
        ht = hvp(t);projected = np.array([[curvature, n @ ht], [t @ hn, t @ ht]])
        symmetry_error = max(symmetry_error, float(np.linalg.norm(projected - projected.T)))
        _, vectors = np.linalg.eigh((projected + projected.T) / 2)
        proposed = vectors[0, 0] * n + vectors[1, 0] * t;proposed /= np.linalg.norm(proposed)
        if np.dot(proposed, n0) < 0:proposed = -proposed
        n = proposed;hn = hvp(n)
    return SoftModeResult(n, curvature, residual_norm, hvp_calls, force_calls,
                          residual_norm <= tol, symmetry_error)


def generalized_central_ritz(q0, anchor, *, rotation_bias, fd_step,
                             max_force_calls, tol, evaluate):
    """Central finite-difference Ritz mode with an explicit E/g budget.

    This is an optional research solver.  Its budget counts evaluator calls,
    unlike :func:`generalized_dimer`, whose ``max_hvp`` counts HVPs.
    """
    from .generalized_ritz import solve
    return solve(q0, anchor, evaluate=evaluate, rotation_bias=rotation_bias,
                 fd_step=fd_step, tol=tol, max_force_calls=max_force_calls,
                 finite_difference='central')
