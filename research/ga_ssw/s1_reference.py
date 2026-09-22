"""Private analytic S1 radial reference; independent of the production API.

This is a bounded nonperiodic primitive for a supplied center and explicit
relative neighbor vectors. It does not choose species, modes, images, or the
native ``neigb_atom`` metadata field, and it does not claim float32/libm cache
parity with LASP.
"""
from __future__ import annotations

import math
import numpy as np


def radial_value_derivative(r, cutoff, exponent, epsilon):
    """Return ``r**exponent * tanh(1-r/cutoff)**3`` and ``d/dr``.

    The recovered native guard skips a pair when ``cutoff-r <= epsilon``.
    ``cutoff``, ``exponent`` and ``epsilon`` are intentionally required.
    """
    if isinstance(exponent, (bool, np.bool_)) or not isinstance(exponent, (int, np.integer)):
        raise ValueError("exponent must be an integer")
    r = float(r); cutoff = float(cutoff); epsilon = float(epsilon)
    if (not all(math.isfinite(x) for x in (r, cutoff, epsilon))
            or r <= 0.0 or cutoff <= 0.0 or epsilon < 0.0):
        raise ValueError("r must be positive, cutoff positive, epsilon nonnegative")
    if cutoff - r <= epsilon:
        return 0.0, 0.0
    t = math.tanh(1.0 - r / cutoff)
    value = r ** exponent * t ** 3
    derivative = (exponent * r ** (exponent - 1) * t ** 3
                  + r ** exponent * (-3.0 * t * t * (1.0 - t * t) / cutoff))
    return value, derivative


def center_neighbors_value_gradient(relative_neighbors, cutoff, exponent, epsilon):
    """Return S1 value, center gradient, and explicit neighbor gradients.

    ``relative_neighbors[j]`` is the vector from the center to the supplied
    neighbor ``j``. The caller owns atom indices/species and periodic images.
    """
    vectors = np.asarray(relative_neighbors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.isfinite(vectors).all():
        raise ValueError("relative_neighbors must be a finite (M,3) array")
    neighbors_gradient = np.zeros_like(vectors)
    center_gradient = np.zeros(3, dtype=float)
    value = 0.0
    for index, vector in enumerate(vectors):
        r = float(np.linalg.norm(vector))
        pair_value, radial_derivative = radial_value_derivative(
            r, cutoff, exponent, epsilon)
        value += pair_value
        pair_gradient = (radial_derivative / r) * vector
        neighbors_gradient[index] += pair_gradient
        center_gradient -= pair_gradient
    return value, center_gradient, neighbors_gradient
