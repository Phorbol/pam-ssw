from __future__ import annotations

from dataclasses import dataclass

from ase.data import covalent_radii
import numpy as np

from .pbc import mic_displacement, mic_distance_matrix
from .state import State


@dataclass(frozen=True)
class PairSofteningTerm:
    atom_i: int
    atom_j: int
    reference_distance: float
    width: float
    strength: float


class LocalSofteningModel:
    def __init__(
        self,
        terms: list[PairSofteningTerm],
        cell: np.ndarray | None = None,
        pbc: tuple[bool, bool, bool] = (False, False, False),
    ) -> None:
        self.terms = terms
        self.cell = None if cell is None else np.asarray(cell, dtype=float).copy()
        if len(pbc) != 3:
            raise ValueError("pbc must contain three booleans")
        self.pbc = tuple(bool(axis) for axis in pbc)

    @classmethod
    def from_state(
        cls,
        state: State,
        pairs: list[tuple[int, int]] | None,
        strength: float,
        mode: str = "manual",
        cutoff_scale: float = 1.25,
        active_indices: np.ndarray | None = None,
    ) -> LocalSofteningModel:
        if mode == "manual":
            selected_pairs = pairs or []
        elif mode in {"neighbor_auto", "active_neighbors"}:
            selected_pairs = automatic_neighbor_pairs(
                state,
                cutoff_scale=cutoff_scale,
                active_indices=active_indices if mode == "active_neighbors" else None,
            )
        else:
            raise ValueError("mode must be manual, neighbor_auto, or active_neighbors")
        terms: list[PairSofteningTerm] = []
        for pair in selected_pairs:
            atom_i, atom_j = _validate_pair(pair, state.n_atoms)
            delta = mic_displacement(
                state.positions[atom_j : atom_j + 1],
                state.positions[atom_i : atom_i + 1],
                state.cell,
                state.pbc,
            )[0]
            distance = float(np.linalg.norm(delta))
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
        return cls(terms, cell=state.cell, pbc=state.pbc)

    def evaluate(self, flat_positions: np.ndarray) -> tuple[float, np.ndarray]:
        positions = np.asarray(flat_positions, dtype=float).reshape(-1, 3)
        gradient = np.zeros_like(positions)
        total_energy = 0.0
        for term in self.terms:
            delta = mic_displacement(
                positions[term.atom_j : term.atom_j + 1],
                positions[term.atom_i : term.atom_i + 1],
                self.cell,
                self.pbc,
            )[0]
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


def automatic_neighbor_pairs(
    state: State,
    cutoff_scale: float,
    active_indices: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Return covalent-radius neighbor pairs using MIC for periodic axes."""
    if cutoff_scale <= 0:
        raise ValueError("cutoff_scale must be positive")
    if state.n_atoms < 2:
        return []
    active_set = None
    if active_indices is not None:
        active_set = {int(index) for index in np.asarray(active_indices, dtype=int)}
    distances = mic_distance_matrix(state.positions, state.cell, state.pbc)
    pairs: list[tuple[int, int]] = []
    for atom_i in range(state.n_atoms - 1):
        radius_i = covalent_radii[int(state.numbers[atom_i])]
        for atom_j in range(atom_i + 1, state.n_atoms):
            if active_set is not None and atom_i not in active_set and atom_j not in active_set:
                continue
            radius_j = covalent_radii[int(state.numbers[atom_j])]
            cutoff = cutoff_scale * float(radius_i + radius_j)
            if distances[atom_i, atom_j] <= cutoff:
                pairs.append((atom_i, atom_j))
    return pairs


def _validate_pair(pair: tuple[int, int], n_atoms: int) -> tuple[int, int]:
    if len(pair) != 2:
        raise ValueError("pair must contain exactly two atom indices")
    atom_i = int(pair[0])
    atom_j = int(pair[1])
    if atom_i == atom_j:
        raise ValueError("pair atom indices must be distinct")
    if atom_i < 0 or atom_j < 0 or atom_i >= n_atoms or atom_j >= n_atoms:
        raise ValueError("pair atom index out of range")
    return atom_i, atom_j
