from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fingerprint import descriptor_distance, structural_descriptor
from .state import State


@dataclass
class MinimaEntry:
    entry_id: int
    state: State
    energy: float
    parent_id: int | None
    visits: int = 0
    descriptor: np.ndarray | None = None
    node_trials: int = 0
    node_successes: int = 0
    frontier_value: float = 1.0
    duplicate_hits: int = 0


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
                entry.duplicate_hits += 1
                return entry

        entry = MinimaEntry(
            entry_id=len(self.entries),
            state=state,
            energy=float(energy),
            parent_id=parent_id,
            visits=1,
            descriptor=structural_descriptor(state),
        )
        self.entries.append(entry)
        self._refresh_frontier_values()
        return entry

    def next_seed(self) -> MinimaEntry:
        return min(self.entries, key=lambda entry: (entry.visits, entry.energy, entry.entry_id))

    def select_seed(self, selector, rng: np.random.Generator) -> MinimaEntry:
        entry = selector.select(self, rng)
        entry.node_trials += 1
        entry.visits += 1
        return entry

    def normalized_energy(self, entry: MinimaEntry) -> float:
        energies = np.asarray([item.energy for item in self.entries], dtype=float)
        span = float(np.ptp(energies))
        if span <= 1e-12:
            return 0.0
        return float((entry.energy - energies.min()) / span)

    def descriptor_density(self, entry: MinimaEntry | np.ndarray) -> float:
        descriptor = entry if isinstance(entry, np.ndarray) else entry.descriptor
        if descriptor is None or not self.entries:
            return 0.0
        distances = np.asarray(
            [
                descriptor_distance(descriptor, other.descriptor)
                for other in self.entries
                if other.descriptor is not None and (not isinstance(entry, MinimaEntry) or other.entry_id != entry.entry_id)
            ],
            dtype=float,
        )
        if distances.size == 0:
            return 0.0
        bandwidth = max(0.25, float(np.median(distances)) + 1e-6)
        return float(np.exp(-0.5 * (distances / bandwidth) ** 2).sum())

    def novelty(self, entry: MinimaEntry | np.ndarray) -> float:
        return float(1.0 / (1.0 + self.descriptor_density(entry)))

    def coverage_gain(self, descriptor: np.ndarray) -> float:
        return self.novelty(descriptor)

    def duplicate_rate(self) -> float:
        hits = sum(entry.duplicate_hits for entry in self.entries)
        total = hits + len(self.entries)
        return hits / total if total else 0.0

    def descriptor_degeneracy_rate(self, bin_width: float = 0.05) -> float:
        if len(self.entries) < 2:
            return 0.0
        bins: dict[tuple[int, ...], set[int]] = {}
        for entry in self.entries:
            if entry.descriptor is None:
                continue
            key = tuple(np.floor(entry.descriptor / bin_width).astype(int).tolist())
            bins.setdefault(key, set()).add(entry.entry_id)
        if not bins:
            return 0.0
        degenerate_bins = sum(1 for entry_ids in bins.values() if len(entry_ids) > 1)
        return degenerate_bins / len(bins)

    def record_success(self, entry: MinimaEntry, reward: float) -> None:
        if reward > 0.0:
            entry.node_successes += 1
        entry.frontier_value = 0.8 * entry.frontier_value + 0.2 * max(0.0, reward)

    def _refresh_frontier_values(self) -> None:
        for entry in self.entries:
            success_rate = entry.node_successes / max(1, entry.node_trials)
            entry.frontier_value = max(entry.frontier_value, self.novelty(entry) * (1.0 + success_rate))

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
