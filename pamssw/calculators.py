from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from ase import Atoms

from .state import State


class Calculator(Protocol):
    def evaluate(self, state: State) -> tuple[float, np.ndarray]:
        ...

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        ...


@dataclass
class AnalyticCalculator:
    potential: object

    def evaluate(self, state: State) -> tuple[float, np.ndarray]:
        energy, flat_gradient = self.evaluate_flat(state.flatten_positions(), state)
        return energy, flat_gradient.reshape(state.n_atoms, 3)

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        energy, gradient = self.potential.energy_gradient(flat_positions, template)
        return float(energy), np.asarray(gradient, dtype=float).reshape(-1)


@dataclass
class ASECalculator:
    calculator: object

    def evaluate(self, state: State) -> tuple[float, np.ndarray]:
        atoms = self._to_atoms(state)
        energy = float(atoms.get_potential_energy())
        gradient = -atoms.get_forces()
        return energy, gradient

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        state = template.with_flat_positions(flat_positions)
        energy, gradient = self.evaluate(state)
        return energy, gradient.reshape(-1)

    def _to_atoms(self, state: State) -> Atoms:
        atoms = Atoms(
            numbers=state.numbers,
            positions=state.positions,
            cell=state.cell,
            pbc=state.pbc,
        )
        atoms.calc = self.calculator
        return atoms
