"""Paper-derived nonperiodic S2/Q-value analytic reference.

This is an independent implementation of the addition-theorem form in
``docs/research/2026-09-26-q-s2-evidence.md``. It uses the recovered S1 radial
envelope, but does not claim native S2 parameter-slot, species, image, or
floating-point parity. The caller supplies explicit center-to-neighbor vectors.
"""
from __future__ import annotations

import math
import numpy as np

from .s1_reference import radial_value_derivative


def _legendre_value_derivative(degree: int, x: np.ndarray):
    """Evaluate P_degree and dP_degree/dx by the three-term recurrence."""
    previous = np.ones_like(x)
    previous_derivative = np.zeros_like(x)
    if degree == 0:
        return previous, previous_derivative
    current = x.copy()
    current_derivative = np.ones_like(x)
    for order in range(1, degree):
        following = ((2 * order + 1) * x * current - order * previous) / (order + 1)
        following_derivative = (
            (2 * order + 1) * (current + x * current_derivative)
            - order * previous_derivative
        ) / (order + 1)
        previous, current = current, following
        previous_derivative, current_derivative = current_derivative, following_derivative
    return current, current_derivative


def center_neighbors_value_gradient(relative_neighbors, degree, *, exponent, cutoff, epsilon):
    """Return S2 value, center gradient, and explicit neighbor gradients.

    ``relative_neighbors[j]`` is the vector from the center to neighbor ``j``.
    The degree ``L`` and radial exponent are explicit integer parameters. The
    radial function is the S1 envelope ``r**exponent*tanh(1-r/cutoff)**3`` with
    its recovered ``cutoff-r <= epsilon`` guard. The caller owns species and
    periodic-image selection/mapping.

    The value is
    ``sqrt((2L+1)/(4*pi) * sum_jk phi_j*phi_k*P_L(e_j dot e_k))``.
    Zero descriptors are outside this derivative's domain and raise
    ``ValueError``; no epsilon regularization or fallback direction is added.
    """
    if (isinstance(degree, (bool, np.bool_))
            or not isinstance(degree, (int, np.integer)) or degree < 0):
        raise ValueError("degree must be a nonnegative integer")
    if (isinstance(exponent, (bool, np.bool_))
            or not isinstance(exponent, (int, np.integer))):
        raise ValueError("exponent must be an integer")
    cutoff = float(cutoff)
    epsilon = float(epsilon)
    if (not math.isfinite(cutoff) or cutoff <= 0.0
            or not math.isfinite(epsilon) or epsilon < 0.0):
        raise ValueError("cutoff must be positive and epsilon nonnegative")

    vectors = np.asarray(relative_neighbors, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] != 3 or not np.isfinite(vectors).all():
        raise ValueError("relative_neighbors must be a finite (M,3) array")
    neighbor_count = len(vectors)
    center_gradient = np.zeros(3, dtype=float)
    neighbor_gradients = np.zeros_like(vectors)
    if neighbor_count == 0:
        raise ValueError("zero S2 descriptor is outside the derivative domain")

    radii = np.linalg.norm(vectors, axis=1)
    if not np.isfinite(radii).all() or np.any(radii <= 0.0):
        raise ValueError("neighbor vectors must have finite nonzero norms")
    try:
        radial = [radial_value_derivative(r, cutoff, exponent, epsilon) for r in radii]
    except (OverflowError, FloatingPointError) as error:
        raise ValueError("nonfinite S2 radial value or derivative") from error
    phi = np.asarray([item[0] for item in radial], dtype=float)
    dphi = np.asarray([item[1] for item in radial], dtype=float)
    if not np.isfinite(phi).all() or not np.isfinite(dphi).all():
        raise ValueError("nonfinite S2 radial value or derivative")

    unit = vectors / radii[:, None]
    cosine = unit @ unit.T
    legendre, legendre_derivative = _legendre_value_derivative(int(degree), cosine)
    normalization = (2 * int(degree) + 1) / (4.0 * math.pi)
    squared_value = float(normalization * np.sum(phi[:, None] * phi[None, :] * legendre))
    if not math.isfinite(squared_value):
        raise ValueError("nonfinite S2 descriptor")
    if squared_value < 0.0:
        raise ValueError("negative S2 squared norm is outside the numerical domain")
    if squared_value == 0.0:
        raise ValueError("zero S2 descriptor is outside the derivative domain")
    value = math.sqrt(squared_value)

    radial_sum = legendre @ phi
    for j in range(neighbor_count):
        angular_vectors = unit - cosine[j, :, None] * unit[j]
        angular_sum = np.sum(
            (phi * legendre_derivative[j])[:, None] * angular_vectors, axis=0)
        neighbor_gradients[j] = (normalization / value) * (
            dphi[j] * unit[j] * radial_sum[j]
            + (phi[j] / radii[j]) * angular_sum
        )
    center_gradient = -neighbor_gradients.sum(axis=0)
    if not np.isfinite(center_gradient).all() or not np.isfinite(neighbor_gradients).all():
        raise ValueError("nonfinite S2 gradient")
    return value, center_gradient, neighbor_gradients
