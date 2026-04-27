from __future__ import annotations

import numpy as np

from .state import State


class DoubleWell2D:
    def energy_gradient(self, flat_positions: np.ndarray, state: State) -> tuple[float, np.ndarray]:
        positions = np.asarray(flat_positions, dtype=float).reshape(state.n_atoms, 3)
        x = float(positions[0, 0])
        y = float(positions[0, 1])
        z = float(positions[0, 2])
        energy = (x * x - 1.0) ** 2 + 0.5 * y * y + 0.25 * z * z
        gradient = np.zeros_like(positions)
        gradient[0, 0] = 4.0 * x * (x * x - 1.0)
        gradient[0, 1] = y
        gradient[0, 2] = 0.5 * z
        return float(energy), gradient.reshape(-1)


class CoupledPairWell:
    def __init__(self, center_stiffness: float = 1.5, bond_stiffness: float = 24.0) -> None:
        self.center_stiffness = center_stiffness
        self.bond_stiffness = bond_stiffness
        self.left = 0.7
        self.right = 1.2

    def energy_gradient(self, flat_positions: np.ndarray, state: State) -> tuple[float, np.ndarray]:
        positions = np.asarray(flat_positions, dtype=float).reshape(state.n_atoms, 3)
        x1 = float(positions[0, 0])
        x2 = float(positions[1, 0])
        y = positions[:, 1:]
        center = 0.5 * (x1 + x2)
        distance = x2 - x1
        f = (distance - self.left) * (distance - self.right)
        center_term = self.center_stiffness * center * center
        bond = self.bond_stiffness * f * f
        transverse = 0.2 * float(np.sum(y * y))
        energy = center_term + bond + transverse

        d_energy_dcenter = 2.0 * self.center_stiffness * center
        dfd_distance = 2.0 * distance - (self.left + self.right)
        d_energy_ddistance = 2.0 * self.bond_stiffness * f * dfd_distance

        gradient = np.zeros_like(positions)
        gradient[0, 0] = 0.5 * d_energy_dcenter - d_energy_ddistance
        gradient[1, 0] = 0.5 * d_energy_dcenter + d_energy_ddistance
        gradient[:, 1:] = 0.4 * y
        return float(energy), gradient.reshape(-1)
