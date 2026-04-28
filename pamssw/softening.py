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
        penalty: str = "gaussian_well",
        xi: float = 0.5,
        cutoff: float | None = 3.0,
        adaptive_strength: bool = False,
        max_strength_scale: float = 3.0,
        deviation_scale: float = 0.25,
    ) -> None:
        self.terms = terms
        self.cell = None if cell is None else np.asarray(cell, dtype=float).copy()
        if len(pbc) != 3:
            raise ValueError("pbc must contain three booleans")
        self.pbc = tuple(bool(axis) for axis in pbc)
        if penalty not in {"gaussian_well", "buckingham_repulsive"}:
            raise ValueError("penalty must be gaussian_well or buckingham_repulsive")
        if xi <= 0:
            raise ValueError("xi must be positive")
        if cutoff is not None and cutoff <= 0:
            raise ValueError("cutoff must be positive when set")
        if max_strength_scale < 1.0:
            raise ValueError("max_strength_scale must be at least 1")
        if deviation_scale <= 0:
            raise ValueError("deviation_scale must be positive")
        self.penalty = penalty
        self.xi = float(xi)
        self.cutoff = None if cutoff is None else float(cutoff)
        self.adaptive_strength = bool(adaptive_strength)
        self.max_strength_scale = float(max_strength_scale)
        self.deviation_scale = float(deviation_scale)

    @classmethod
    def from_state(
        cls,
        state: State,
        pairs: list[tuple[int, int]] | None,
        strength: float,
        mode: str = "manual",
        cutoff_scale: float = 1.25,
        active_indices: np.ndarray | None = None,
        penalty: str = "gaussian_well",
        xi: float = 0.5,
        cutoff: float | None = 3.0,
        adaptive_strength: bool = False,
        max_strength_scale: float = 3.0,
        deviation_scale: float = 0.25,
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
        return cls(
            terms,
            cell=state.cell,
            pbc=state.pbc,
            penalty=penalty,
            xi=xi,
            cutoff=cutoff,
            adaptive_strength=adaptive_strength,
            max_strength_scale=max_strength_scale,
            deviation_scale=deviation_scale,
        )

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
            strength, d_strength_d_distance = self._effective_strength(term, deviation)
            if self.penalty == "gaussian_well":
                exponent = np.exp(-0.5 * (deviation / term.width) ** 2)
                energy = strength * exponent
                d_energy_d_distance = (
                    d_strength_d_distance * exponent
                    - strength * exponent * deviation / (term.width**2)
                )
            else:
                if self.cutoff is not None and distance > term.reference_distance + self.cutoff:
                    continue
                exponent = np.exp(-deviation / self.xi)
                energy = strength * exponent
                d_energy_d_distance = d_strength_d_distance * exponent - strength * exponent / self.xi
            direction = delta / distance
            grad_i = -d_energy_d_distance * direction
            grad_j = -grad_i
            gradient[term.atom_i] += grad_i
            gradient[term.atom_j] += grad_j
            total_energy += energy
        return float(total_energy), gradient.reshape(-1)

    def _effective_strength(self, term: PairSofteningTerm, deviation: float) -> tuple[float, float]:
        if not self.adaptive_strength:
            return term.strength, 0.0
        denominator = max(self.deviation_scale * term.reference_distance, 1e-12)
        raw_extra = abs(deviation) / denominator
        capped_extra = min(self.max_strength_scale - 1.0, raw_extra)
        scale = 1.0 + capped_extra
        if raw_extra >= self.max_strength_scale - 1.0 or abs(deviation) <= 1e-12:
            d_scale_d_distance = 0.0
        else:
            d_scale_d_distance = np.sign(deviation) / denominator
        return term.strength * scale, term.strength * d_scale_d_distance


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
