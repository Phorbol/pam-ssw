"""Untruncated nonperiodic LJ cluster oracle, for published model benchmarks.

V = 4 epsilon sum_{i<j} [(sigma/r)^12 - (sigma/r)^6].
epsilon is eV; sigma is Angstrom, not the equilibrium pair distance.
No element-dependent interactions, shifts, cutoff, or periodic images.
"""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class FullPairLJ(Calculator):
    implemented_properties = ['energy', 'free_energy', 'forces']

    def __init__(self, *, epsilon=1., sigma=2.7):
        super().__init__()
        if not np.isfinite([epsilon, sigma]).all() or min(epsilon, sigma) <= 0:
            raise ValueError('positive finite LJ scales required')
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        if self.atoms.pbc.any():
            raise ValueError('FullPairLJ is a nonperiodic cluster oracle')
        positions = self.atoms.positions
        i, j = np.triu_indices(len(positions), 1)
        displacement = positions[i] - positions[j]
        r2 = np.einsum('ij,ij->i', displacement, displacement)
        if not np.isfinite(r2).all() or (r2 <= 0).any():
            raise ValueError('finite, distinct atom positions required')
        s6 = (self.sigma ** 2 / r2) ** 3
        energy = float(4 * self.epsilon * np.sum(s6 * (s6 - 1)))
        pair_force = (24 * self.epsilon * s6 * (2 * s6 - 1) / r2)[:, None] * displacement
        forces = np.zeros_like(positions)
        np.add.at(forces, i, pair_force)
        np.add.at(forces, j, -pair_force)
        self.results = dict(energy=energy, free_energy=energy, forces=forces)
