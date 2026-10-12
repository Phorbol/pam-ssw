"""Explicit, paper-inspired S1/S2 descriptor gradient directions.

This adapter makes species and periodic-image selection explicit. It is not a
reconstruction of LASP's complete Q selector or floating-point implementation.
"""
from dataclasses import dataclass
import math

import numpy as np
from ase.neighborlist import neighbor_list


class DescriptorDirectionUnavailable(ValueError):
    """An explicitly selected mode has no differentiable direction here."""

    def __init__(self, message, image_count=0):
        super().__init__(message)
        self.image_count = int(image_count)


@dataclass(frozen=True)
class DescriptorMode:
    family: str
    center: int
    neighbor_species: tuple
    cutoff: float
    exponent: int
    epsilon: float
    degree: int | None = None

    def __post_init__(self):
        if self.family not in ('S1', 'S2'):
            raise ValueError('family must be S1 or S2')
        if isinstance(self.center, (bool, np.bool_)) or not isinstance(self.center, (int, np.integer)) or self.center < 0:
            raise ValueError('center must be a nonnegative atom index')
        species = tuple(self.neighbor_species)
        if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < 1 or x > 118 for x in species):
            raise ValueError('neighbor_species must contain atomic numbers in [1,118]')
        object.__setattr__(self, 'neighbor_species', tuple(int(x) for x in species))
        if not math.isfinite(self.cutoff) or self.cutoff <= 0 or not math.isfinite(self.epsilon) or self.epsilon < 0:
            raise ValueError('cutoff must be positive and epsilon nonnegative')
        if isinstance(self.exponent, (bool, np.bool_)) or not isinstance(self.exponent, (int, np.integer)):
            raise ValueError('exponent must be an integer')
        if self.family == 'S2':
            if isinstance(self.degree, (bool, np.bool_)) or not isinstance(self.degree, (int, np.integer)) or self.degree < 0:
                raise ValueError('S2 degree must be a nonnegative integer')
        elif self.degree is not None:
            raise ValueError('degree applies only to S2')


def _radial(r, mode):
    if mode.cutoff - r <= mode.epsilon:
        return 0.0, 0.0
    t = math.tanh(1.0 - r / mode.cutoff)
    value = r ** int(mode.exponent) * t ** 3
    derivative = (mode.exponent * r ** (mode.exponent - 1) * t ** 3
                  - 3 * r ** mode.exponent * t * t * (1 - t * t) / mode.cutoff)
    return value, derivative


def _legendre(degree, x):
    p0 = np.ones_like(x); d0 = np.zeros_like(x)
    if degree == 0:
        return p0, d0
    p1 = x.copy(); d1 = np.ones_like(x)
    for n in range(1, degree):
        p2 = ((2*n+1)*x*p1-n*p0)/(n+1)
        d2 = ((2*n+1)*(p1+x*d1)-n*d0)/(n+1)
        p0, p1, d0, d1 = p1, p2, d1, d2
    return p1, d1


def descriptor_value_gradient(atoms, mode):
    """Return descriptor value, all-atom gradient, and selected image count."""
    if not isinstance(mode, DescriptorMode):
        raise TypeError('mode must be DescriptorMode')
    n_atoms = len(atoms)
    if mode.center >= n_atoms:
        raise ValueError('descriptor center is outside the structure')
    positions = np.asarray(atoms.positions, dtype=float)
    numbers = np.asarray(atoms.numbers)
    if positions.shape != (n_atoms, 3) or not np.isfinite(positions).all():
        raise ValueError('finite atom positions required')
    # ASE returns directed center-neighbor pairs and image offsets. Self images
    # are retained; mapped image gradients accumulate back onto their atom.
    i, j, shifts = neighbor_list('ijS', atoms, mode.cutoff, self_interaction=True)
    selected = i == int(mode.center)
    if mode.neighbor_species:
        selected &= np.isin(numbers[j], mode.neighbor_species)
    i, j, shifts = i[selected], j[selected], shifts[selected]
    zero_self = (i == j) & np.all(shifts == 0, axis=1)
    i, j, shifts = i[~zero_self], j[~zero_self], shifts[~zero_self]
    vectors = positions[j] + shifts @ np.asarray(atoms.cell.array) - positions[i]
    if np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError('distinct descriptor atoms overlap')
    grad = np.zeros((n_atoms, 3), dtype=float)
    count = len(vectors)
    if mode.family == 'S1':
        value = 0.0
        for atom, vector in zip(j, vectors):
            r = float(np.linalg.norm(vector)); phi, dphi = _radial(r, mode)
            g = (dphi / r) * vector
            value += phi; grad[atom] += g; grad[mode.center] -= g
        return value, grad, count
    if count == 0:
        raise DescriptorDirectionUnavailable('zero S2 descriptor', count)
    radii = np.linalg.norm(vectors, axis=1)
    radial = [_radial(float(r), mode) for r in radii]
    phi = np.asarray([x[0] for x in radial]); dphi = np.asarray([x[1] for x in radial])
    unit = vectors / radii[:, None]
    cosine = unit @ unit.T
    legendre, dlegendre = _legendre(int(mode.degree), cosine)
    norm_factor = (2 * int(mode.degree) + 1) / (4 * math.pi)
    squared = float(norm_factor * np.sum(phi[:, None] * phi[None, :] * legendre))
    if not math.isfinite(squared):
        raise ValueError('nonfinite S2 descriptor')
    if squared == 0:
        raise DescriptorDirectionUnavailable('zero S2 descriptor', count)
    if squared < 0:
        raise ValueError('negative S2 squared norm is outside the numerical domain')
    value = math.sqrt(squared)
    radial_sum = legendre @ phi
    image_grad = np.zeros_like(vectors)
    for a in range(count):
        tangent = unit - cosine[a, :, None] * unit[a]
        angular = np.sum((phi*dlegendre[a])[:, None]*tangent, axis=0)
        image_grad[a] = norm_factor/value * (dphi[a]*unit[a]*radial_sum[a] + phi[a]/radii[a]*angular)
    for atom, g in zip(j, image_grad):
        grad[atom] += g
    grad[mode.center] -= image_grad.sum(axis=0)
    return value, grad, count


def normalized_descriptor_direction(atoms, mode, active_mask=None):
    try:
        value, gradient, image_count = descriptor_value_gradient(atoms, mode)
    except DescriptorDirectionUnavailable as error:
        return None, 0.0, error.image_count, 0.0
    if active_mask is not None:
        active = np.asarray(active_mask)
        if active.dtype != np.bool_ or active.shape != (len(atoms),) or not np.any(active):
            raise ValueError('active_mask must be a nonempty boolean N-vector')
        gradient[~active] = 0.0
    norm = float(np.linalg.norm(gradient))
    if not math.isfinite(norm) or norm == 0.0:
        return None, value, image_count, norm
    return gradient / norm, value, image_count, norm
