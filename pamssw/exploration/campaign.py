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
    batch_attempt_counts: tuple[int, ...]
    batch_evaluation_counts: tuple[EvaluationCounts, ...]
    bootstrap_recorded: bool
    stop_reason: CampaignStopReason | None

    @property
    def action_counts(self) -> EvaluationCounts:
        return EvaluationCounts.sum(self.batch_evaluation_counts)

    @property
    def committed_batches(self) -> int:
        return len(self.batch_attempt_counts)

    @property
    def committed_attempts(self) -> int:
        return sum(self.batch_attempt_counts)

    @property
    def last_batch_spend(self) -> int | None:
        if not self.batch_evaluation_counts:
            return None
        return self.batch_evaluation_counts[-1].total


@dataclass
class CampaignBudget:
    """Budget ledger that charges observed evaluations, not reserved capacity."""

    total: int
    action_force_budget: int
    bootstrap_counts: EvaluationCounts = field(default_factory=EvaluationCounts.zero, init=False)
    _batch_attempt_counts: list[int] = field(default_factory=list, init=False, repr=False)
    _batch_evaluation_counts: list[EvaluationCounts] = field(
        default_factory=list, init=False, repr=False
    )
    bootstrap_recorded: bool = field(default=False, init=False)
    stop_reason: CampaignStopReason | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        _positive_int(self.total, "total")
        _positive_int(self.action_force_budget, "action_force_budget")

    @property
    def spent(self) -> int:
        return self.bootstrap_counts.total + self.action_counts.total

    @property
    def action_counts(self) -> EvaluationCounts:
        return EvaluationCounts.sum(self._batch_evaluation_counts)

    @property
    def committed_batches(self) -> int:
        return len(self._batch_attempt_counts)

    @property
    def committed_attempts(self) -> int:
        return sum(self._batch_attempt_counts)

    @property
    def last_batch_spend(self) -> int | None:
        if not self._batch_evaluation_counts:
            return None
        return self._batch_evaluation_counts[-1].total

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
        self._batch_attempt_counts.append(len(counts))
        self._batch_evaluation_counts.append(EvaluationCounts(tuple(merged.values)))
        if merged.total == 0:
            self.stop_reason = CampaignStopReason.ZERO_COST_STALL
        elif self.remaining < self.action_force_budget:
            self.stop_reason = CampaignStopReason.BUDGET_TAIL

    def snapshot(self) -> CampaignBudgetSnapshot:
        return CampaignBudgetSnapshot(
            total=self.total,
            action_force_budget=self.action_force_budget,
            bootstrap_counts=EvaluationCounts(tuple(self.bootstrap_counts.values)),
            batch_attempt_counts=tuple(self._batch_attempt_counts),
            batch_evaluation_counts=tuple(
                EvaluationCounts(tuple(counts.values))
                for counts in self._batch_evaluation_counts
            ),
            bootstrap_recorded=self.bootstrap_recorded,
            stop_reason=self.stop_reason,
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
        if not isinstance(snapshot.batch_attempt_counts, tuple):
            raise TypeError("snapshot batch_attempt_counts must be a tuple")
        if not isinstance(snapshot.batch_evaluation_counts, tuple):
            raise TypeError("snapshot batch_evaluation_counts must be a tuple")
        if len(snapshot.batch_attempt_counts) != len(snapshot.batch_evaluation_counts):
            raise ValueError("snapshot batch history lengths must match")
        if not isinstance(snapshot.bootstrap_recorded, bool):
            raise TypeError("snapshot bootstrap_recorded must be a boolean")
        if snapshot.stop_reason is not None and not isinstance(snapshot.stop_reason, CampaignStopReason):
            raise TypeError("snapshot stop_reason must be a CampaignStopReason or None")
        if not snapshot.bootstrap_recorded:
            if (
                snapshot.bootstrap_counts != EvaluationCounts.zero()
                or snapshot.batch_attempt_counts
                or snapshot.batch_evaluation_counts
                or snapshot.stop_reason is not None
            ):
                raise ValueError("unrecorded bootstrap requires an empty budget snapshot")
            return budget
        if snapshot.bootstrap_counts.total > budget.total:
            raise ValueError("snapshot bootstrap counts exceed total budget")
        remaining = budget.total - snapshot.bootstrap_counts.total
        restored_widths: list[int] = []
        restored_counts: list[EvaluationCounts] = []
        for index, (width, counts) in enumerate(
            zip(snapshot.batch_attempt_counts, snapshot.batch_evaluation_counts)
        ):
            width = _positive_int(width, "snapshot batch attempt count")
            if not isinstance(counts, EvaluationCounts):
                raise TypeError("snapshot batch evaluation counts must be EvaluationCounts")
            if width * budget.action_force_budget > remaining:
                raise ValueError("snapshot batch reservation exceeds remaining budget")
            if counts.total > width * budget.action_force_budget:
                raise ValueError("snapshot batch counts exceed reserved action capacity")
            if counts.total == 0 and index != len(snapshot.batch_evaluation_counts) - 1:
                raise ValueError("zero-spend batch must be the final batch")
            remaining -= counts.total
            restored_widths.append(width)
            restored_counts.append(EvaluationCounts(tuple(counts.values)))
        if restored_counts and restored_counts[-1].total == 0:
            expected_stop_reason: CampaignStopReason | None = CampaignStopReason.ZERO_COST_STALL
        elif remaining < budget.action_force_budget:
            expected_stop_reason = CampaignStopReason.BUDGET_TAIL
        else:
            expected_stop_reason = None
        if snapshot.stop_reason is not expected_stop_reason:
            raise ValueError("snapshot stop reason does not match manifest-derived terminal state")
        budget.bootstrap_counts = EvaluationCounts(tuple(snapshot.bootstrap_counts.values))
        budget._batch_attempt_counts = restored_widths
        budget._batch_evaluation_counts = restored_counts
        budget.bootstrap_recorded = snapshot.bootstrap_recorded
        budget.stop_reason = expected_stop_reason
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
