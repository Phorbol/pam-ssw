from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class SearchMode(str, Enum):
    GLOBAL_MINIMUM = "global_minimum"
    REACTION_NETWORK = "reaction_network"
    CRYSTAL_SEARCH = "crystal_search"


@dataclass(frozen=True)
class AcquisitionPolicy:
    archive_density_weight: float = 0.5
    novelty_weight: float = 1.0
    frontier_weight: float = 0.5
    exploration_weight: float = 0.75
    baseline_probability: float = 0.15
    beta_energy: float = 1.0

    @classmethod
    def for_mode(cls, mode: SearchMode | str) -> AcquisitionPolicy:
        mode = SearchMode(mode)
        if mode == SearchMode.GLOBAL_MINIMUM:
            return cls(
                archive_density_weight=0.5,
                novelty_weight=0.5,
                frontier_weight=0.35,
                exploration_weight=0.5,
                baseline_probability=0.15,
                beta_energy=2.0,
            )
        if mode == SearchMode.REACTION_NETWORK:
            return cls(
                archive_density_weight=0.5,
                novelty_weight=1.0,
                frontier_weight=1.0,
                exploration_weight=0.75,
                baseline_probability=0.15,
                beta_energy=0.5,
            )
        if mode == SearchMode.CRYSTAL_SEARCH:
            return cls(
                archive_density_weight=0.75,
                novelty_weight=0.75,
                frontier_weight=0.5,
                exploration_weight=0.75,
                baseline_probability=0.15,
                beta_energy=1.5,
            )
        raise ValueError(f"unsupported search mode: {mode}")

    def effective(self, duplicate_rate: float, descriptor_degeneracy_rate: float) -> AcquisitionPolicy:
        health = float(np.clip(1.0 - max(duplicate_rate, descriptor_degeneracy_rate), 0.0, 1.0))
        return AcquisitionPolicy(
            archive_density_weight=self.archive_density_weight * health,
            novelty_weight=self.novelty_weight,
            frontier_weight=self.frontier_weight * health,
            exploration_weight=self.exploration_weight,
            baseline_probability=self.baseline_probability,
            beta_energy=self.beta_energy,
        )


@dataclass(frozen=True)
class ProposalOutcome:
    energy: float
    previous_best_energy: float
    is_new_minimum: bool
    is_duplicate: bool
    descriptor_coverage_gain: float
    is_new_edge: bool = False


@dataclass(frozen=True)
class ProposalScorer:
    mode: SearchMode = SearchMode.GLOBAL_MINIMUM
    near_energy_window: float = 0.5

    @classmethod
    def for_mode(cls, mode: SearchMode | str) -> ProposalScorer:
        return cls(mode=SearchMode(mode))

    def rank_key(self, outcome: ProposalOutcome) -> tuple[float, ...]:
        best_gain = self._best_gain(outcome)
        near_low_energy = float(outcome.energy <= outcome.previous_best_energy + self.near_energy_window)
        not_duplicate = 1.0 - float(outcome.is_duplicate)
        if self.mode == SearchMode.GLOBAL_MINIMUM:
            return (
                best_gain,
                near_low_energy,
                float(outcome.is_new_minimum),
                float(outcome.descriptor_coverage_gain),
                not_duplicate,
            )
        if self.mode == SearchMode.REACTION_NETWORK:
            return (
                float(outcome.is_new_edge),
                float(outcome.is_new_minimum),
                float(outcome.descriptor_coverage_gain),
                best_gain,
                not_duplicate,
            )
        if self.mode == SearchMode.CRYSTAL_SEARCH:
            return (
                near_low_energy,
                float(outcome.is_new_minimum),
                float(outcome.descriptor_coverage_gain),
                best_gain,
                not_duplicate,
            )
        raise ValueError(f"unsupported search mode: {self.mode}")

    def score(self, outcome: ProposalOutcome) -> float:
        best_gain = self._best_gain(outcome)
        return (
            best_gain
            + float(outcome.is_new_minimum)
            + float(outcome.is_new_edge)
            + float(outcome.descriptor_coverage_gain)
            - float(outcome.is_duplicate)
        )

    @staticmethod
    def _best_gain(outcome: ProposalOutcome) -> float:
        return max(0.0, outcome.previous_best_energy - outcome.energy)


@dataclass(frozen=True)
class BanditSelector:
    policy: AcquisitionPolicy = AcquisitionPolicy()

    def select(self, archive, rng: np.random.Generator):
        if not archive.entries:
            raise ValueError("cannot select from an empty archive")
        live_entries = [entry for entry in archive.entries if not getattr(entry, "is_dead", False)]
        candidates = live_entries or archive.entries
        effective_policy = self.policy.effective(
            duplicate_rate=archive.duplicate_rate(),
            descriptor_degeneracy_rate=archive.descriptor_degeneracy_rate(),
        )
        if rng.random() < effective_policy.baseline_probability:
            if rng.random() < 0.7:
                return min(candidates, key=lambda entry: (entry.energy, entry.entry_id))
            return candidates[int(rng.integers(0, len(candidates)))]

        total_trials = sum(entry.node_trials for entry in candidates)
        return max(
            candidates,
            key=lambda entry: (
                self.score_entry(archive, entry, total_trials, effective_policy),
                -entry.entry_id,
            ),
        )

    def score_entry(
        self,
        archive,
        entry,
        total_trials: int | None = None,
        policy: AcquisitionPolicy | None = None,
    ) -> float:
        policy = self.policy if policy is None else policy
        total = sum(item.node_trials for item in archive.entries) if total_trials is None else total_trials
        density_penalty = np.log1p(archive.descriptor_density(entry))
        dead_penalty = 1.0 if getattr(entry, "is_dead", False) else 0.0
        frontier_score = getattr(entry, "frontier_score", entry.frontier_value)
        return (
            -policy.beta_energy * archive.normalized_energy(entry)
            + policy.novelty_weight * archive.novelty(entry)
            - policy.archive_density_weight * density_penalty
            + policy.exploration_weight * np.sqrt(np.log1p(total + 1.0) / (1.0 + entry.node_trials))
            + policy.frontier_weight * frontier_score
            - 10.0 * dead_penalty
        )
