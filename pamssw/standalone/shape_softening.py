"""Experimental fixed-radius shape pullback of frozen bond softening.

The bias acts on centered coordinates rescaled to the immutable reference
radius. It is limited to free, isolated clusters; it has no calculator or
optimizer integration and adds no Hessian implementation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from ase import Atoms

from .softening import BondTable, EnergyFilter, FrozenBondSoftening


def _centered_radius(atoms: Atoms) -> float:
    if not isinstance(atoms, Atoms):
        raise TypeError('atoms must be an ASE Atoms object')
    if len(atoms) < 2 or atoms.pbc.any() or atoms.constraints:
        raise ValueError('shape bias requires a free nonperiodic unconstrained cluster')
    if not np.isfinite(atoms.positions).all():
        raise ValueError('positions must be finite')
    relative = atoms.positions - atoms.positions.mean(axis=0)
    radius = float(np.linalg.norm(relative))
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('centered radius must be finite and positive')
    return radius


@dataclass(frozen=True)
class FrozenShapeBondSoftening(FrozenBondSoftening):
    """Frozen LS pair bias evaluated at ``R0 * centered(x) / ||centered(x)||``.

    ``radius0`` is the centered Cartesian norm at the structure where this
    frozen bias was built. Pair identities, reference distances, strengths,
    and this radius remain fixed for the lifetime of one search step.
    """

    radius0: float = field(default=0.0, kw_only=True)

    def __post_init__(self):
        super().__post_init__()
        if any(self.pbc):
            raise ValueError('shape bias requires a nonperiodic cluster')
        radius0 = float(self.radius0)
        if not math.isfinite(radius0) or radius0 <= 0:
            raise ValueError('radius0 must be finite and positive')
        object.__setattr__(self, 'radius0', radius0)

    @classmethod
    def _build(cls, atoms, bond_energies, bond_lengths, xi, *,
               initial_fraction=None, total_strength=None, energy_filter=None):
        radius0 = _centered_radius(atoms)
        base = FrozenBondSoftening._build(
            atoms, bond_energies, bond_lengths, xi,
            initial_fraction=initial_fraction, total_strength=total_strength,
            energy_filter=energy_filter,
        )
        return cls(base.numbers, base.cell, base.pbc, base.pairs,
                   base.reference_distances, base.strengths, base.xi,
                   base.energy_filter, radius0=radius0)

    @classmethod
    def from_atoms(cls, atoms: Atoms, *, bond_energies: BondTable,
                   bond_lengths: BondTable, initial_fraction: float = 0.03,
                   xi: float = 0.2,
                   energy_filter: EnergyFilter | None = None) -> FrozenShapeBondSoftening:
        """Freeze LS pairs and the current structure's centered radius."""
        return super().from_atoms(
            atoms, bond_energies=bond_energies, bond_lengths=bond_lengths,
            initial_fraction=initial_fraction, xi=xi, energy_filter=energy_filter,
        )

    def evaluate(self, atoms: Atoms) -> tuple[float, np.ndarray]:
        radius = _centered_radius(atoms)
        relative = atoms.positions - atoms.positions.mean(axis=0)
        mapped = atoms.copy()
        mapped.positions[:] = (self.radius0 / radius) * relative
        energy, mapped_forces = super().evaluate(mapped)

        # Pull back the gradient through y = R0 * Cx / ||Cx||, where C
        # removes translation by the geometric center.
        gradient_y = -mapped_forces.reshape(-1)
        n = len(atoms)
        gradient_y = gradient_y.reshape(n, 3)
        gradient_centered = gradient_y - gradient_y.mean(axis=0)
        direction = relative.reshape(-1) / radius
        gradient_centered = gradient_centered.reshape(-1)
        alpha = self.radius0 / radius
        gradient_x = alpha * (
            gradient_centered - direction * float(direction @ gradient_centered)
        )
        return float(energy), -gradient_x.reshape(n, 3)
