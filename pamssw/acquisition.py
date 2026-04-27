from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
    archive_density_weight: float
    novelty_weight: float
    frontier_weight: float
    exploration_weight: float
    baseline_probability: float
    beta_energy: float = 1.0

    def select(self, archive, rng: np.random.Generator):
        if not archive.entries:
            raise ValueError("cannot select from an empty archive")
        if rng.random() < self.baseline_probability:
            if rng.random() < 0.7:
                return min(archive.entries, key=lambda entry: (entry.energy, entry.entry_id))
            return archive.entries[int(rng.integers(0, len(archive.entries)))]

        total_trials = sum(entry.node_trials for entry in archive.entries)
        return max(
            archive.entries,
            key=lambda entry: (
                self.score_entry(archive, entry, total_trials),
                -entry.entry_id,
            ),
        )

    def score_entry(self, archive, entry, total_trials: int | None = None) -> float:
        total = sum(item.node_trials for item in archive.entries) if total_trials is None else total_trials
        return (
            -self.beta_energy * archive.normalized_energy(entry)
            + self.novelty_weight * archive.novelty(entry)
            - self.archive_density_weight * archive.descriptor_density(entry)
            + self.exploration_weight * np.sqrt(np.log1p(total + 1.0) / (1.0 + entry.node_trials))
            + self.frontier_weight * entry.frontier_value
        )
