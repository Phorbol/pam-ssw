from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fingerprint import descriptor_distance, structural_descriptor, variable_cell_structural_descriptor
from .pbc import mic_displacement
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
    node_duplicate_failures: int = 0
    frontier_score: float = 1.0
    is_frontier: bool = True
    is_dead: bool = False


@dataclass
class ArchivePrototype:
    descriptor: np.ndarray
    representative_entry_id: int
    weight: int = 1

    def merge(self, descriptor: np.ndarray) -> None:
        total = self.weight + 1
        self.descriptor = (self.descriptor * self.weight + descriptor) / total
        self.weight = total


class MinimaArchive:
    def __init__(
        self,
        energy_tol: float,
        rmsd_tol: float,
        max_prototypes: int = 1000,
        variable_cell: bool = False,
        cell_tol: float = 0.1,
        lattice_descriptor_weight: float = 1.0,
    ) -> None:
        if max_prototypes <= 0:
            raise ValueError("max_prototypes must be positive")
        self.energy_tol = energy_tol
        self.rmsd_tol = rmsd_tol
        self.max_prototypes = max_prototypes
        self.variable_cell = bool(variable_cell)
        self.cell_tol = float(cell_tol)
        self.lattice_descriptor_weight = float(lattice_descriptor_weight)
        self.entries: list[MinimaEntry] = []
        self.prototypes: list[ArchivePrototype] = []

    def add(self, state: State, energy: float, parent_id: int | None) -> MinimaEntry:
        for entry in self.entries:
            if abs(entry.energy - energy) > self.energy_tol:
                continue
            if self._is_duplicate(entry.state, state):
                entry.visits += 1
                entry.duplicate_hits += 1
                return entry

        entry = MinimaEntry(
            entry_id=len(self.entries),
            state=state,
            energy=float(energy),
            parent_id=parent_id,
            visits=1,
            descriptor=self._descriptor(state),
        )
        self.entries.append(entry)
        self._update_prototypes(entry)
        self.refresh_frontier_status()
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
        if descriptor is None or not self.prototypes:
            return 0.0
        distances = np.asarray(
            [
                descriptor_distance(descriptor, prototype.descriptor)
                for prototype in self.prototypes
                if not isinstance(entry, MinimaEntry) or prototype.representative_entry_id != entry.entry_id
            ],
            dtype=float,
        )
        finite_mask = np.isfinite(distances)
        if distances.size == 0 or not np.any(finite_mask):
            return 0.0
        distances = distances[finite_mask]
        weights = np.asarray(
            [
                prototype.weight
                for prototype in self.prototypes
                if not isinstance(entry, MinimaEntry) or prototype.representative_entry_id != entry.entry_id
            ],
            dtype=float,
        )
        weights = weights[finite_mask]
        bandwidth = max(0.25, float(np.median(distances)) + 1e-6)
        return float((weights * np.exp(-0.5 * (distances / bandwidth) ** 2)).sum())

    def novelty(self, entry: MinimaEntry | np.ndarray) -> float:
        return float(1.0 / (1.0 + self.descriptor_density(entry)))

    def coverage_gain(self, descriptor: np.ndarray) -> float:
        return self.novelty(descriptor)

    def descriptor_for(self, state: State) -> np.ndarray:
        return self._descriptor(state)

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

    def prototype_occupancy(self) -> dict[str, float | int]:
        if not self.prototypes:
            return {
                "n_prototypes": 0,
                "max_prototypes": self.max_prototypes,
                "max_prototype_weight": 0,
                "mean_prototype_weight": 0.0,
            }
        weights = np.asarray([prototype.weight for prototype in self.prototypes], dtype=float)
        return {
            "n_prototypes": len(self.prototypes),
            "max_prototypes": self.max_prototypes,
            "max_prototype_weight": int(weights.max()),
            "mean_prototype_weight": float(weights.mean()),
        }

    def frontier_diagnostics(self) -> dict[str, float | int]:
        if not self.entries:
            return {
                "frontier_nodes": 0,
                "dead_nodes": 0,
                "mean_frontier_score": 0.0,
                "mean_node_duplicate_failure_rate": 0.0,
                "max_node_duplicate_failure_rate": 0.0,
            }
        scores = np.asarray([entry.frontier_score for entry in self.entries], dtype=float)
        duplicate_failure_rates = np.asarray(
            [entry.node_duplicate_failures / max(1, entry.node_trials) for entry in self.entries],
            dtype=float,
        )
        return {
            "frontier_nodes": sum(1 for entry in self.entries if entry.is_frontier),
            "dead_nodes": sum(1 for entry in self.entries if entry.is_dead),
            "mean_frontier_score": float(scores.mean()),
            "mean_node_duplicate_failure_rate": float(duplicate_failure_rates.mean()),
            "max_node_duplicate_failure_rate": float(duplicate_failure_rates.max()),
        }

    def record_success(self, entry: MinimaEntry, reward: float, duplicate_failures: int = 0) -> None:
        entry.node_duplicate_failures += duplicate_failures
        if reward > 0.0:
            entry.node_successes += 1
        self.refresh_frontier_status()

    def refresh_frontier_status(self) -> None:
        if not self.entries:
            return
        best_energy = min(entry.energy for entry in self.entries)
        energies = np.asarray([entry.energy for entry in self.entries], dtype=float)
        energy_window = max(self.energy_tol, float(np.median(energies - best_energy)) + self.energy_tol)
        max_trials = max((entry.node_trials for entry in self.entries), default=0)
        for entry in self.entries:
            duplicate_failure_rate = entry.node_duplicate_failures / max(1, entry.node_trials)
            success_rate = entry.node_successes / max(1, entry.node_trials)
            low_visit = entry.node_trials <= max(1, int(0.5 * max_trials))
            low_energy = entry.energy <= best_energy + energy_window
            sparse = self.novelty(entry) >= 0.4
            recently_successful = entry.node_successes > 0 and success_rate >= 0.1
            entry.is_dead = entry.node_trials >= 8 and duplicate_failure_rate >= 0.75 and success_rate <= 0.05
            if entry.is_dead:
                entry.frontier_score = 0.0
                entry.frontier_value = 0.0
                entry.is_frontier = False
                continue
            entry.is_frontier = bool(low_visit and low_energy and (sparse or recently_successful or entry.node_trials == 0))
            visit_score = 1.0 / (1.0 + entry.node_trials)
            energy_score = 1.0 / (1.0 + max(0.0, entry.energy - best_energy) / max(energy_window, self.energy_tol))
            observable_score = 0.4 * visit_score + 0.3 * self.novelty(entry) + 0.2 * energy_score + 0.1 * success_rate
            entry.frontier_score = float(np.clip(observable_score if entry.is_frontier else 0.5 * observable_score, 0.0, 1.0))
            entry.frontier_value = entry.frontier_score

    def _update_prototypes(self, entry: MinimaEntry) -> None:
        if entry.descriptor is None:
            return
        if not self.prototypes:
            self.prototypes.append(ArchivePrototype(entry.descriptor.copy(), entry.entry_id))
            return
        nearest_index, nearest_distance = self._nearest_prototype(entry.descriptor)
        if len(self.prototypes) < self.max_prototypes:
            self.prototypes.append(ArchivePrototype(entry.descriptor.copy(), entry.entry_id))
            return
        spread = self._prototype_distance_scale()
        if nearest_distance <= spread:
            self.prototypes[nearest_index].merge(entry.descriptor)
            return
        replace_index = min(range(len(self.prototypes)), key=lambda index: (self.prototypes[index].weight, index))
        self.prototypes[replace_index] = ArchivePrototype(entry.descriptor.copy(), entry.entry_id)

    def _nearest_prototype(self, descriptor: np.ndarray) -> tuple[int, float]:
        distances = [descriptor_distance(descriptor, prototype.descriptor) for prototype in self.prototypes]
        nearest_index = int(np.argmin(distances))
        return nearest_index, float(distances[nearest_index])

    def _prototype_distance_scale(self) -> float:
        if len(self.prototypes) < 2:
            return 0.25
        distances: list[float] = []
        for index, prototype in enumerate(self.prototypes):
            for other in self.prototypes[index + 1 :]:
                distance = descriptor_distance(prototype.descriptor, other.descriptor)
                if np.isfinite(distance):
                    distances.append(distance)
        if not distances:
            return 0.25
        return max(0.25, float(np.median(np.asarray(distances, dtype=float))))

    @staticmethod
    def _rmsd(lhs: State, rhs: State) -> float:
        if lhs.n_atoms != rhs.n_atoms:
            return float("inf")
        if not np.array_equal(lhs.numbers, rhs.numbers):
            return float("inf")
        if lhs.n_atoms <= 1 or any(lhs.pbc) or any(rhs.pbc):
            if lhs.cell is not None and rhs.cell is not None and lhs.pbc == rhs.pbc and np.allclose(lhs.cell, rhs.cell):
                diff = mic_displacement(lhs.positions, rhs.positions, lhs.cell, lhs.pbc)
            else:
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

    def _descriptor(self, state: State) -> np.ndarray:
        if self.variable_cell:
            return variable_cell_structural_descriptor(state, lattice_weight=self.lattice_descriptor_weight)
        return structural_descriptor(state)

    def _is_duplicate(self, lhs: State, rhs: State) -> bool:
        if not self.variable_cell:
            return self._rmsd(lhs, rhs) <= self.rmsd_tol
        lhs_descriptor = variable_cell_structural_descriptor(lhs, lattice_weight=self.lattice_descriptor_weight)
        rhs_descriptor = variable_cell_structural_descriptor(rhs, lattice_weight=self.lattice_descriptor_weight)
        lattice_delta = descriptor_distance(lhs_descriptor[-7:], rhs_descriptor[-7:])
        if lattice_delta > self.cell_tol:
            return False
        atomic_delta = descriptor_distance(lhs_descriptor[:-7], rhs_descriptor[:-7])
        if atomic_delta > max(self.rmsd_tol, 0.25):
            return False
        if lhs.cell is not None and rhs.cell is not None and np.allclose(lhs.cell, rhs.cell, atol=self.cell_tol):
            return self._rmsd(lhs, rhs) <= self.rmsd_tol
        return True
