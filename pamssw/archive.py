from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import State


@dataclass
class MinimaEntry:
    entry_id: int
    state: State
    energy: float
    parent_id: int | None
    visits: int = 0


class MinimaArchive:
    def __init__(self, energy_tol: float, rmsd_tol: float) -> None:
        self.energy_tol = energy_tol
        self.rmsd_tol = rmsd_tol
        self.entries: list[MinimaEntry] = []

    def add(self, state: State, energy: float, parent_id: int | None) -> MinimaEntry:
        for entry in self.entries:
            if abs(entry.energy - energy) > self.energy_tol:
                continue
            if self._rmsd(entry.state, state) <= self.rmsd_tol:
                entry.visits += 1
                return entry

        entry = MinimaEntry(
            entry_id=len(self.entries),
            state=state,
            energy=float(energy),
            parent_id=parent_id,
            visits=1,
        )
        self.entries.append(entry)
        return entry

    def next_seed(self) -> MinimaEntry:
        return min(self.entries, key=lambda entry: (entry.visits, entry.energy, entry.entry_id))

    @staticmethod
    def _rmsd(lhs: State, rhs: State) -> float:
        if lhs.n_atoms != rhs.n_atoms:
            return float("inf")
        if not np.array_equal(lhs.numbers, rhs.numbers):
            return float("inf")
        if lhs.n_atoms <= 1 or lhs.cell is not None or rhs.cell is not None or any(lhs.pbc) or any(rhs.pbc):
            diff = lhs.positions - rhs.positions
            return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
        if MinimaArchive._distance_signature_delta(lhs, rhs) > 2.0 * lhs.n_atoms * 1e-1:
            return float("inf")

        lhs_centered = lhs.positions - lhs.positions.mean(axis=0, keepdims=True)
        rhs_centered = rhs.positions - rhs.positions.mean(axis=0, keepdims=True)
        rotation = MinimaArchive._kabsch_rotation(lhs_centered, rhs_centered)
        aligned = lhs_centered @ rotation
        diff = aligned - rhs_centered
        return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))

    @staticmethod
    def _distance_signature_delta(lhs: State, rhs: State) -> float:
        lhs_distances = MinimaArchive._pair_distance_signature(lhs.positions)
        rhs_distances = MinimaArchive._pair_distance_signature(rhs.positions)
        if lhs_distances.shape != rhs_distances.shape:
            return float("inf")
        return float(np.linalg.norm(lhs_distances - rhs_distances))

    @staticmethod
    def _pair_distance_signature(positions: np.ndarray) -> np.ndarray:
        n_atoms = positions.shape[0]
        signature: list[float] = []
        for atom_i in range(n_atoms):
            delta = positions[atom_i + 1 :] - positions[atom_i]
            if delta.size:
                signature.extend(np.linalg.norm(delta, axis=1).tolist())
        return np.sort(np.asarray(signature, dtype=float))

    @staticmethod
    def _kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        covariance = source.T @ target
        left, _, right_t = np.linalg.svd(covariance)
        determinant = np.linalg.det(right_t.T @ left.T)
        correction = np.diag([1.0, 1.0, np.sign(determinant) if determinant != 0.0 else 1.0])
        return left @ correction @ right_t
