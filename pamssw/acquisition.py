from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AcquisitionPolicy:
    archive_density_weight: float = 0.5
    novelty_weight: float = 1.0
    frontier_weight: float = 0.5
    exploration_weight: float = 0.75
    baseline_probability: float = 0.15
    beta_energy: float = 1.0

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


@dataclass(frozen=True)
class ProposalScorer:
    w_new: float = 1.0
    w_best: float = 5.0
    w_cov: float = 0.5
    w_dup: float = 0.25

    def score(self, outcome: ProposalOutcome) -> float:
        best_gain = max(0.0, outcome.previous_best_energy - outcome.energy)
        return (
            self.w_new * float(outcome.is_new_minimum)
            + self.w_best * best_gain
            + self.w_cov * outcome.descriptor_coverage_gain
            - self.w_dup * float(outcome.is_duplicate)
        )


@dataclass(frozen=True)
class BanditSelector:
    policy: AcquisitionPolicy = AcquisitionPolicy()

    def select(self, archive, rng: np.random.Generator):
        if not archive.entries:
            raise ValueError("cannot select from an empty archive")
        effective_policy = self.policy.effective(
            duplicate_rate=archive.duplicate_rate(),
            descriptor_degeneracy_rate=archive.descriptor_degeneracy_rate(),
        )
        if rng.random() < effective_policy.baseline_probability:
            if rng.random() < 0.7:
                return min(archive.entries, key=lambda entry: (entry.energy, entry.entry_id))
            return archive.entries[int(rng.integers(0, len(archive.entries)))]

        total_trials = sum(entry.node_trials for entry in archive.entries)
        return max(
            archive.entries,
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
        return (
            -policy.beta_energy * archive.normalized_energy(entry)
            + policy.novelty_weight * archive.novelty(entry)
            - policy.archive_density_weight * density_penalty
            + policy.exploration_weight * np.sqrt(np.log1p(total + 1.0) / (1.0 + entry.node_trials))
            + policy.frontier_weight * entry.frontier_value
        )
