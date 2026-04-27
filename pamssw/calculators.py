from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from ase import Atoms

from .state import State


class Calculator(Protocol):
    def evaluate(self, state: State) -> EnergyResult:
        ...

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        ...


@dataclass(frozen=True)
class EnergyResult:
    energy: float
    gradient: np.ndarray
    stress: np.ndarray | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "energy", float(self.energy))
        object.__setattr__(self, "gradient", np.asarray(self.gradient, dtype=float))
        if self.stress is not None:
            object.__setattr__(self, "stress", np.asarray(self.stress, dtype=float))

    def __iter__(self):
        yield self.energy
        yield self.gradient


@dataclass
class AnalyticCalculator:
    potential: object

    def evaluate(self, state: State) -> EnergyResult:
        energy, flat_gradient = self.evaluate_flat(state.flatten_positions(), state)
        return EnergyResult(energy=energy, gradient=flat_gradient.reshape(state.n_atoms, 3))

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        energy, gradient = self.potential.energy_gradient(flat_positions, template)
        return float(energy), np.asarray(gradient, dtype=float).reshape(-1)


@dataclass
class ASECalculator:
    calculator: object

    def evaluate(self, state: State) -> EnergyResult:
        atoms = self._to_atoms(state)
        energy = float(atoms.get_potential_energy())
        gradient = -atoms.get_forces()
        stress = None
        if state.cell is not None and any(state.pbc):
            try:
                stress = atoms.get_stress(voigt=False)
            except Exception:
                stress = None
        return EnergyResult(energy=energy, gradient=gradient, stress=stress)

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        state = template.with_flat_positions(flat_positions)
        result = self.evaluate(state)
        return result.energy, result.gradient.reshape(-1)

    def _to_atoms(self, state: State) -> Atoms:
        atoms = Atoms(
            numbers=state.numbers,
            positions=state.positions,
            cell=state.cell,
            pbc=state.pbc,
        )
        atoms.calc = self.calculator
        return atoms
