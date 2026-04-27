from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import State


@dataclass(frozen=True)
class PairSofteningTerm:
    atom_i: int
    atom_j: int
    reference_distance: float
    width: float
    strength: float


class LocalSofteningModel:
    def __init__(self, terms: list[PairSofteningTerm]) -> None:
        self.terms = terms

    @classmethod
    def from_state(
        cls,
        state: State,
        pairs: list[tuple[int, int]],
        strength: float,
    ) -> LocalSofteningModel:
        terms: list[PairSofteningTerm] = []
        for atom_i, atom_j in pairs:
            distance = np.linalg.norm(state.positions[atom_j] - state.positions[atom_i])
            width = max(0.15, 0.25 * distance)
            terms.append(
                PairSofteningTerm(
                    atom_i=atom_i,
                    atom_j=atom_j,
                    reference_distance=distance,
                    width=width,
                    strength=strength,
                )
            )
        return cls(terms)

    def evaluate(self, flat_positions: np.ndarray) -> tuple[float, np.ndarray]:
        positions = np.asarray(flat_positions, dtype=float).reshape(-1, 3)
        gradient = np.zeros_like(positions)
        total_energy = 0.0
        for term in self.terms:
            delta = positions[term.atom_j] - positions[term.atom_i]
            distance = float(np.linalg.norm(delta))
            if distance < 1e-12:
                continue
            deviation = distance - term.reference_distance
            exponent = np.exp(-0.5 * (deviation / term.width) ** 2)
            energy = term.strength * exponent
            d_energy_d_distance = -(energy * deviation) / (term.width**2)
            direction = delta / distance
            grad_i = -d_energy_d_distance * direction
            grad_j = -grad_i
            gradient[term.atom_i] += grad_i
            gradient[term.atom_j] += grad_j
            total_energy += energy
        return float(total_energy), gradient.reshape(-1)
