"""Independent plane-minimizing dimer rotation; not native CBD reproduction.

See docs/research/standard-dimer-direction.md for the source/derivation and
finite-separation limitations. No ASE dimer code or executable is invoked.
"""
import numpy as np
from .direction import SoftModeResult


def paper_dimer_direction(atoms, anchor, *, rotation_bias, fd_step, max_hvp,
                          tol, evaluate):
    """Rotate at fixed R on H-a*n0*n0.T using direct forward force differences.

    Each rotation minimizes the local quadratic curvature in the plane of n
    and its negative tangent residual. Unlike the unrestarted Ritz alternative,
    no growing subspace/history is retained. Every returned direction has its
    own directly evaluated endpoint force; no extrapolated residual is accepted.

    fd_step: Angstrom; rotation_bias and tol: eV/Angstrom²; max_hvp: positive
    integer endpoint budget. Total force requests <= 1+max_hvp, including center.
    The original anchor n0 is normalized once and never updated. Rotations and
    translations are not projected out internally. Fixed-cell, unconstrained only;
    a periodic caller supplies a translation-projected evaluator and anchor.

    Convergence means small finite-secant tangent residual, not proof of the
    globally lowest eigenmode, negative true-PES curvature, or physical stability.
    projected_symmetry_error reports the largest Frobenius antisymmetry norm
    of any sampled 2D operator; zero when no rotation plane was sampled.
    """
    if atoms.constraints:
        raise ValueError('dimer requires unconstrained atoms')
    shape = atoms.positions.shape
    if not len(atoms) or not np.isfinite(atoms.positions).all():
        raise ValueError('requires finite nonempty atom positions')
    for name, value in [('fd_step', fd_step), ('tol', tol)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} must be finite and positive')
    if not np.isfinite(rotation_bias) or rotation_bias < 0:
        raise ValueError('rotation_bias must be finite and nonnegative')
    if (isinstance(max_hvp, (bool, np.bool_)) or
            not isinstance(max_hvp, (int, np.integer)) or max_hvp < 1):
        raise ValueError('max_hvp must be a positive integer')
    n0 = np.asarray(anchor, dtype=float)
    if n0.shape != shape or not np.isfinite(n0).all():
        raise ValueError('anchor requires finite shape (N, 3)')
    norm = np.linalg.norm(n0)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError('anchor requires finite nonzero norm')
    n0 = n0.ravel().copy()/norm
    if not callable(evaluate):
        raise ValueError('evaluate callback required')
    center = atoms.positions.copy()
    force_calls = 0
    hvp_calls = 0

    def force_at(positions):
        nonlocal force_calls
        trial = atoms.copy()
        trial.calc = atoms.calc
        trial.set_positions(positions, apply_constraint=False)
        force_calls += 1
        energy, force = evaluate(trial)
        force = np.asarray(force, dtype=float)
        if (np.ndim(energy) != 0 or not np.isfinite(energy) or
                force.shape != shape or not np.isfinite(force).all()):
            raise ValueError('evaluator returned invalid energy/forces')
        return force.ravel().copy()

    f0 = force_at(center)

    def hvp(n):
        nonlocal hvp_calls
        f1 = force_at(center+fd_step*n.reshape(shape))
        hvp_calls += 1
        return (f0-f1)/fd_step-rotation_bias*np.dot(n0, n)*n0

    n = n0.copy()
    hn = hvp(n)
    symmetry_error = 0.
    stop_reason = 'unspecified'
    while True:
        curvature = float(n@hn)
        residual = hn-curvature*n
        residual_norm = float(np.linalg.norm(residual))
        if residual_norm <= tol:
            stop_reason = 'residual_converged'
            break
        if hvp_calls+2 > max_hvp:
            stop_reason = 'budget_exhausted'
            break
        # Negative tangent gradient, with roundoff leakage explicitly removed.
        t = -residual
        t -= np.dot(t, n)*n
        t /= np.linalg.norm(t)
        ht = hvp(t)
        projected = np.array([[curvature, n@ht], [t@hn, t@ht]])
        symmetry_error = max(symmetry_error,
                            float(np.linalg.norm(projected-projected.T)))
        _, vectors = np.linalg.eigh((projected+projected.T)/2)
        proposed = vectors[0, 0]*n+vectors[1, 0]*t
        proposed /= np.linalg.norm(proposed)
        if np.dot(proposed, n0) < 0:
            proposed = -proposed
        n = proposed
        hn = hvp(n)  # Direct check: secants on nonlinear PES are not linear.
    return SoftModeResult(n.reshape(shape), curvature, residual_norm, hvp_calls,
                          force_calls, residual_norm <= tol, symmetry_error,
                          stop_reason)
