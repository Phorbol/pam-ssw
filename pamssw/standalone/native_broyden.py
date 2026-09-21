"""Recovered BRZERO4 arithmetic primitives, NOT a full Broyden optimizer.

Evidence: docs/research/native-broyden-prefix-recovery.md. The executable's
block-sum bilinear form is degenerate and is not a physical Cartesian metric.
These functions are named native to identify legacy behavior, not endorse it.
No search driver uses them by default. Full matrix/history update remains open.
The verified domain has finite intermediate arithmetic as well as finite inputs.
"""
from dataclasses import dataclass
import numpy as np


def _vectors(*values):
    arrays = [np.asarray(v, dtype=float) for v in values]
    if not arrays or arrays[0].size == 0 or arrays[0].size % 3:
        raise ValueError('requires nonempty complete three-component blocks')
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError('vector shapes must agree')
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise ValueError('vectors must be finite')
    return arrays


def native_block_sum_product(a, b):
    """Sum over XYZ blocks of sum(a_block)*sum(b_block), ELF 0x700f20.

The native helper ignores an incomplete trailing block; this independent API
instead requires complete blocks. It is positive semidefinite, not an inner
product: for instance (1, -1, 0) has zero squared length.
    """
    a, b = _vectors(a, b)
    return float(np.dot(a.reshape(-1, 3).sum(axis=1), b.reshape(-1, 3).sum(axis=1)))


def initial_step(x, f, g0):
    """BRZERO4 initial update X <- X + G0*F (diagonal G0).

BRIONS4 supplies a constant G0 array, equal to its saved STEP (initially 1).
The supplied force has already been scaled by rotate_dimer's FACT1. G0 here
is an elementwise preconditioner, not a dense inverse Hessian.
    """
    x, f, g0 = _vectors(x, f, g0)
    return x + g0*f


@dataclass(frozen=True)
class SecantPrefix:
    force_difference: np.ndarray
    displacement: np.ndarray
    u: np.ndarray
    normalizer: float


def secant_prefix(x, f, x_last, f_last, g0):
    """Recover source lines 854–881, before matrix/history calculations.

DF = (F-F_last)/s, U = G0*DF + (X-X_last)/s. Here s is the
native block-sum seminorm. The physical units follow the caller's scaled force
workspace. Zero s produces NaN/Inf natively; this API raises explicitly instead
of propagating that result. No epsilon, Euclidean substitution or fallback is
introduced. This function does not advance or truncate optimizer history.
    """
    x, f, x_last, f_last, g0 = _vectors(x, f, x_last, f_last, g0)
    df = f-f_last
    s = float(np.sqrt(native_block_sum_product(df, df)))
    if s == 0:
        raise ValueError('zero block-sum seminorm: native update is undefined')
    displacement = x-x_last
    normalized = df/s
    u = g0*normalized + displacement/s
    return SecantPrefix(normalized, displacement, u, s)
