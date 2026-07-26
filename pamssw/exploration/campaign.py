from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from ..accounting import EvaluationCounts
from ..archive import MinimaArchive
from .policies import SUPPORTED_POLICIES
from .posterior import StarterProductivityPosterior


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _stripped_nonempty(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be nonempty")
    return normalized


@dataclass(frozen=True)
class PosteriorExplorationConfig:
    """Fixed-fidelity configuration for a posterior exploration campaign."""

    policy_name: str
    batch_size: int
    max_workers: int
    action_force_budget: int
    total_force_budget: int
    master_seed: int
    calculator_label: str
    calculator_fingerprint: str
    run_directory: Path
    mode: str = "new"

    def __post_init__(self) -> None:
        if not isinstance(self.policy_name, str):
            raise TypeError("policy_name must be a string")
        if self.policy_name not in SUPPORTED_POLICIES:
            raise ValueError(f"unsupported policy: {self.policy_name}")
        _positive_int(self.batch_size, "batch_size")
        _positive_int(self.max_workers, "max_workers")
        if self.max_workers > self.batch_size:
            raise ValueError("max_workers must not exceed batch_size")
        _positive_int(self.action_force_budget, "action_force_budget")
        _positive_int(self.total_force_budget, "total_force_budget")
        _nonnegative_int(self.master_seed, "master_seed")
        object.__setattr__(
            self, "calculator_label", _stripped_nonempty(self.calculator_label, "calculator_label")
        )
        object.__setattr__(
            self,
            "calculator_fingerprint",
            _stripped_nonempty(self.calculator_fingerprint, "calculator_fingerprint"),
        )
        if not isinstance(self.run_directory, (str, os.PathLike)):
            raise TypeError("run_directory must be path-like")
        object.__setattr__(self, "run_directory", Path(self.run_directory))
        if self.mode not in {"new", "resume"}:
            raise ValueError("mode must be 'new' or 'resume'")


class CampaignStopReason(str, Enum):
    BUDGET_TAIL = "budget_tail"
    ZERO_COST_STALL = "zero_cost_stall"


@dataclass(frozen=True)
class CampaignBudgetSnapshot:
    total: int
    action_force_budget: int
    bootstrap_counts: EvaluationCounts
    action_counts: EvaluationCounts
    committed_batches: int
    committed_attempts: int
    bootstrap_recorded: bool
    stop_reason: CampaignStopReason | None
    last_batch_spend: int | None


@dataclass
class CampaignBudget:
    """Budget ledger that charges observed evaluations, not reserved capacity."""

    total: int
    action_force_budget: int
    bootstrap_counts: EvaluationCounts = field(default_factory=EvaluationCounts.zero, init=False)
    action_counts: EvaluationCounts = field(default_factory=EvaluationCounts.zero, init=False)
    committed_batches: int = field(default=0, init=False)
    committed_attempts: int = field(default=0, init=False)
    bootstrap_recorded: bool = field(default=False, init=False)
    stop_reason: CampaignStopReason | None = field(default=None, init=False)
    last_batch_spend: int | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _positive_int(self.total, "total")
        _positive_int(self.action_force_budget, "action_force_budget")

    @property
    def spent(self) -> int:
        return self.bootstrap_counts.total + self.action_counts.total

    @property
    def remaining(self) -> int:
        return self.total - self.spent

    @property
    def unused(self) -> int:
        return self.remaining

    def record_bootstrap(self, counts: EvaluationCounts) -> None:
        if self.bootstrap_recorded:
            raise RuntimeError("bootstrap has already been recorded")
        if not isinstance(counts, EvaluationCounts):
            raise TypeError("bootstrap counts must be an EvaluationCounts")
        if counts.total > self.total:
            raise ValueError("bootstrap counts exceed total budget")
        self.bootstrap_counts = EvaluationCounts(tuple(counts.values))
        self.bootstrap_recorded = True
        if self.remaining < self.action_force_budget:
            self.stop_reason = CampaignStopReason.BUDGET_TAIL

    def next_batch_size(self, batch_size: int) -> int:
        _positive_int(batch_size, "batch_size")
        if not self.bootstrap_recorded:
            raise RuntimeError("bootstrap must be recorded before scheduling")
        if self.stop_reason is not None:
            return 0
        width = min(batch_size, self.remaining // self.action_force_budget)
        if width == 0:
            self.stop_reason = CampaignStopReason.BUDGET_TAIL
        return width

    def commit_batch(self, counts: tuple[EvaluationCounts, ...]) -> None:
        if self.stop_reason is not None:
            raise RuntimeError("cannot commit after campaign termination")
        if not self.bootstrap_recorded:
            raise RuntimeError("bootstrap must be recorded before committing")
        if not isinstance(counts, tuple):
            raise TypeError("batch counts must be a tuple")
        if not counts:
            raise ValueError("batch counts must not be empty")
        for count in counts:
            if not isinstance(count, EvaluationCounts):
                raise TypeError("batch counts must contain EvaluationCounts")
            if count.total > self.action_force_budget:
                raise ValueError("an action exceeded its force-evaluation budget")
        if len(counts) * self.action_force_budget > self.remaining:
            raise ValueError("batch reservation exceeds remaining budget")
        merged = EvaluationCounts.sum(counts)
        self.action_counts = self.action_counts + merged
        self.committed_batches += 1
        self.committed_attempts += len(counts)
        self.last_batch_spend = merged.total
        if merged.total == 0:
            self.stop_reason = CampaignStopReason.ZERO_COST_STALL
        elif self.remaining < self.action_force_budget:
            self.stop_reason = CampaignStopReason.BUDGET_TAIL

    def snapshot(self) -> CampaignBudgetSnapshot:
        return CampaignBudgetSnapshot(
            total=self.total,
            action_force_budget=self.action_force_budget,
            bootstrap_counts=EvaluationCounts(tuple(self.bootstrap_counts.values)),
            action_counts=EvaluationCounts(tuple(self.action_counts.values)),
            committed_batches=self.committed_batches,
            committed_attempts=self.committed_attempts,
            bootstrap_recorded=self.bootstrap_recorded,
            stop_reason=self.stop_reason,
            last_batch_spend=self.last_batch_spend,
        )

    @classmethod
    def from_snapshot(
        cls, snapshot: CampaignBudgetSnapshot, *, action_force_budget: int
    ) -> CampaignBudget:
        if not isinstance(snapshot, CampaignBudgetSnapshot):
            raise TypeError("snapshot must be a CampaignBudgetSnapshot")
        action_force_budget = _positive_int(action_force_budget, "action_force_budget")
        snapshot_action_force_budget = _positive_int(
            snapshot.action_force_budget, "snapshot action_force_budget"
        )
        if action_force_budget != snapshot_action_force_budget:
            raise ValueError("manifest action_force_budget does not match campaign identity")
        budget = cls(snapshot.total, snapshot_action_force_budget)
        if not isinstance(snapshot.bootstrap_counts, EvaluationCounts):
            raise TypeError("snapshot bootstrap_counts must be an EvaluationCounts")
        if not isinstance(snapshot.action_counts, EvaluationCounts):
            raise TypeError("snapshot action_counts must be an EvaluationCounts")
        if not isinstance(snapshot.bootstrap_recorded, bool):
            raise TypeError("snapshot bootstrap_recorded must be a boolean")
        if snapshot.stop_reason is not None and not isinstance(snapshot.stop_reason, CampaignStopReason):
            raise TypeError("snapshot stop_reason must be a CampaignStopReason or None")
        _nonnegative_int(snapshot.committed_batches, "snapshot committed_batches")
        _nonnegative_int(snapshot.committed_attempts, "snapshot committed_attempts")
        if snapshot.committed_batches > snapshot.committed_attempts or (
            snapshot.committed_batches == 0
        ) != (snapshot.committed_attempts == 0):
            raise ValueError("snapshot batches must match committed attempts")
        if snapshot.last_batch_spend is None:
            if snapshot.committed_batches != 0:
                raise ValueError("snapshot last batch spend is required after a committed batch")
        else:
            _nonnegative_int(snapshot.last_batch_spend, "snapshot last_batch_spend")
            if snapshot.committed_batches == 0:
                raise ValueError("snapshot last batch spend requires a committed batch")
            if snapshot.last_batch_spend > snapshot.action_counts.total:
                raise ValueError("snapshot last batch spend exceeds action counts")
            if (
                snapshot.committed_batches == 1
                and snapshot.last_batch_spend != snapshot.action_counts.total
            ):
                raise ValueError("single-batch snapshot last spend must equal action counts")
        if not snapshot.bootstrap_recorded:
            if (
                snapshot.bootstrap_counts != EvaluationCounts.zero()
                or snapshot.action_counts != EvaluationCounts.zero()
                or snapshot.committed_batches != 0
                or snapshot.committed_attempts != 0
                or snapshot.stop_reason is not None
                or snapshot.last_batch_spend is not None
            ):
                raise ValueError("unrecorded bootstrap requires an empty budget snapshot")
        elif snapshot.bootstrap_counts.total + snapshot.action_counts.total > budget.total:
            raise ValueError("snapshot counts exceed total budget")
        if snapshot.action_counts.total > 0 and snapshot.committed_attempts == 0:
            raise ValueError("snapshot action counts require committed attempts")
        expected_stop_reason: CampaignStopReason | None
        if not snapshot.bootstrap_recorded:
            expected_stop_reason = None
        elif snapshot.last_batch_spend == 0:
            expected_stop_reason = CampaignStopReason.ZERO_COST_STALL
        elif budget.total - snapshot.bootstrap_counts.total - snapshot.action_counts.total < budget.action_force_budget:
            expected_stop_reason = CampaignStopReason.BUDGET_TAIL
        else:
            expected_stop_reason = None
        if snapshot.stop_reason is not expected_stop_reason:
            raise ValueError("snapshot stop reason does not match manifest-derived terminal state")
        budget.bootstrap_counts = EvaluationCounts(tuple(snapshot.bootstrap_counts.values))
        budget.action_counts = EvaluationCounts(tuple(snapshot.action_counts.values))
        budget.committed_batches = snapshot.committed_batches
        budget.committed_attempts = snapshot.committed_attempts
        budget.bootstrap_recorded = snapshot.bootstrap_recorded
        budget.stop_reason = expected_stop_reason
        budget.last_batch_spend = snapshot.last_batch_spend
        return budget

    @classmethod
    def restore(
        cls, snapshot: CampaignBudgetSnapshot, *, action_force_budget: int
    ) -> CampaignBudget:
        """Restore a ledger from a validated immutable snapshot."""
        return cls.from_snapshot(snapshot, action_force_budget=action_force_budget)


@dataclass(frozen=True)
class PosteriorExplorationResult:
    """Immutable campaign summary.

    ``completed_attempts`` and ``failed_attempts`` are the terminal-status
    partition of every committed attempt represented by the summary.
    """

    archive: MinimaArchive
    posterior: StarterProductivityPosterior
    policy_name: str
    completed_batches: int
    completed_attempts: int
    failed_attempts: int
    posterior_observed_attempts: int
    bootstrap_evaluations: int
    action_evaluations: int
    total_evaluations: int
    purpose_counts: EvaluationCounts
    total_force_budget: int
    unused_force_budget: int
    stop_reason: CampaignStopReason
    benchmark_eligible: bool
    benchmark_ineligibility_reasons: tuple[str, ...]
    run_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.archive, MinimaArchive):
            raise TypeError("archive must be a MinimaArchive")
        if not isinstance(self.posterior, StarterProductivityPosterior):
            raise TypeError("posterior must be a StarterProductivityPosterior")
        if not isinstance(self.policy_name, str):
            raise TypeError("policy_name must be a string")
        if self.policy_name not in SUPPORTED_POLICIES:
            raise ValueError(f"unsupported policy: {self.policy_name}")
        for name in (
            "completed_batches",
            "completed_attempts",
            "failed_attempts",
            "posterior_observed_attempts",
            "bootstrap_evaluations",
            "action_evaluations",
            "total_evaluations",
            "total_force_budget",
            "unused_force_budget",
        ):
            _nonnegative_int(getattr(self, name), name)
        if not isinstance(self.purpose_counts, EvaluationCounts):
            raise TypeError("purpose_counts must be an EvaluationCounts")
        if not isinstance(self.stop_reason, CampaignStopReason):
            raise TypeError("stop_reason must be a CampaignStopReason")
        if not isinstance(self.benchmark_eligible, bool):
            raise TypeError("benchmark_eligible must be a boolean")
        if not isinstance(self.benchmark_ineligibility_reasons, tuple):
            raise TypeError("benchmark_ineligibility_reasons must be a tuple")
        reasons: list[str] = []
        for reason in self.benchmark_ineligibility_reasons:
            if not isinstance(reason, str):
                raise TypeError("benchmark ineligibility reasons must be strings")
            normalized = reason.strip()
            if not normalized:
                raise ValueError("benchmark ineligibility reasons must be nonempty")
            reasons.append(normalized)
        if self.benchmark_eligible != (not reasons):
            raise ValueError("benchmark eligibility must match ineligibility reasons")
        if self.stop_reason is CampaignStopReason.ZERO_COST_STALL and (
            self.benchmark_eligible or not reasons
        ):
            raise ValueError("zero-cost stalls are benchmark-ineligible and require a reason")
        attempts = self.completed_attempts + self.failed_attempts
        if self.completed_batches > attempts or (self.completed_batches == 0) != (attempts == 0):
            raise ValueError("completed batches must match represented terminal attempts")
        if self.posterior_observed_attempts > attempts:
            raise ValueError("posterior observations cannot exceed terminal attempts")
        if self.total_evaluations != self.bootstrap_evaluations + self.action_evaluations:
            raise ValueError("total evaluations must equal bootstrap plus action evaluations")
        if self.total_evaluations != self.purpose_counts.total:
            raise ValueError("total evaluations must equal the purpose-count total")
        if self.total_evaluations + self.unused_force_budget != self.total_force_budget:
            raise ValueError("total and unused evaluations must equal the force budget")
        if not isinstance(self.run_directory, (str, os.PathLike)):
            raise TypeError("run_directory must be path-like")
        object.__setattr__(self, "archive", self.archive.clone())
        object.__setattr__(self, "posterior", self.posterior.clone())
        object.__setattr__(self, "purpose_counts", EvaluationCounts(tuple(self.purpose_counts.values)))
        object.__setattr__(self, "benchmark_ineligibility_reasons", tuple(reasons))
        object.__setattr__(self, "run_directory", Path(self.run_directory))


__all__ = [
    "CampaignBudget",
    "CampaignBudgetSnapshot",
    "CampaignStopReason",
    "PosteriorExplorationConfig",
    "PosteriorExplorationResult",
]
