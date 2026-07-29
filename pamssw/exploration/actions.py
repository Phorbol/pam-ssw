from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from math import fsum, isfinite
from numbers import Integral

import numpy as np

from ..accounting import EvaluationCounts
from ..state import State


class AttemptStatus(str, Enum):
    """Terminal status reported for one dispatched exploration action."""

    COMPLETED = "completed"
    INVALID = "invalid"
    FRAGMENTED = "fragmented"
    BUDGET_EXHAUSTED = "budget_exhausted"
    WORKER_ERROR = "worker_error"


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_float(name: str, value: object) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _failure_reason(name: str, value: object) -> str:
    return _nonempty_string(name, value)


def _status(value: object) -> AttemptStatus:
    if not isinstance(value, AttemptStatus):
        raise ValueError("status must be an AttemptStatus")
    return value


def _terminal_evaluation_counts(
    force_evaluations: int, evaluation_counts: EvaluationCounts | None
) -> EvaluationCounts:
    if evaluation_counts is None:
        return EvaluationCounts.unattributed(force_evaluations)
    if not isinstance(evaluation_counts, EvaluationCounts):
        raise ValueError("evaluation_counts must be an EvaluationCounts")
    canonical_counts = EvaluationCounts(tuple(evaluation_counts.values))
    if canonical_counts.total != force_evaluations:
        raise ValueError("evaluation_counts total must equal force_evaluations")
    return canonical_counts


def _strict_boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def should_observe_posterior(status: AttemptStatus, evaluation_counts: EvaluationCounts) -> bool:
    """Return whether one terminal attempt is a valid posterior observation."""
    _status(status)
    if not isinstance(evaluation_counts, EvaluationCounts):
        raise ValueError("evaluation_counts must be an EvaluationCounts")
    if status is AttemptStatus.WORKER_ERROR or evaluation_counts.total == 0:
        return False
    return status in {
        AttemptStatus.COMPLETED,
        AttemptStatus.BUDGET_EXHAUSTED,
        AttemptStatus.FRAGMENTED,
        AttemptStatus.INVALID,
    }


@dataclass(frozen=True)
class PolicySnapshot:
    """Immutable starter-policy distribution used for one batch dispatch."""

    version: int
    archive_version: int
    policy_name: str
    eligible_starter_ids: tuple[int, ...]
    probabilities: tuple[float, ...]
    support_complete: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _nonnegative_int("version", self.version))
        object.__setattr__(self, "archive_version", _nonnegative_int("archive_version", self.archive_version))
        object.__setattr__(self, "policy_name", _nonempty_string("policy_name", self.policy_name))
        if not isinstance(self.support_complete, bool):
            raise ValueError("support_complete must be a boolean")

        try:
            starter_ids = tuple(self.eligible_starter_ids)
        except TypeError as exc:
            raise ValueError("eligible_starter_ids must be a nonempty sequence") from exc
        if not starter_ids:
            raise ValueError("eligible_starter_ids cannot be empty")
        starter_ids = tuple(_nonnegative_int("eligible_starter_ids", starter_id) for starter_id in starter_ids)
        if len(set(starter_ids)) != len(starter_ids):
            raise ValueError("eligible_starter_ids must be unique")

        try:
            raw_probabilities = tuple(self.probabilities)
        except TypeError as exc:
            raise ValueError("probabilities must be a sequence") from exc
        if len(starter_ids) != len(raw_probabilities):
            raise ValueError("probabilities must align with eligible_starter_ids")
        probabilities = tuple(_finite_float("probabilities", value) for value in raw_probabilities)
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("probabilities must be in [0, 1]")
        if abs(fsum(probabilities) - 1.0) > 1e-12:
            raise ValueError("probabilities must sum to one")
        if self.support_complete and any(value <= 0.0 for value in probabilities):
            raise ValueError("support_complete requires positive probability for every starter")

        object.__setattr__(self, "eligible_starter_ids", starter_ids)
        object.__setattr__(self, "probabilities", probabilities)

    def probability_for(self, starter_id: int) -> float:
        try:
            index = self.eligible_starter_ids.index(starter_id)
        except ValueError as exc:
            raise KeyError(f"unknown starter_id: {starter_id!r}") from exc
        return self.probabilities[index]


@dataclass(frozen=True)
class StarterAction:
    """One deterministic starter selection dispatched to a worker."""

    action_id: str
    batch_id: int
    slot_id: int
    policy_name: str
    policy_version: int
    archive_version: int
    starter_id: int
    selection_probability: float
    random_seed: int
    force_budget: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _nonempty_string("action_id", self.action_id))
        object.__setattr__(self, "batch_id", _nonnegative_int("batch_id", self.batch_id))
        object.__setattr__(self, "slot_id", _nonnegative_int("slot_id", self.slot_id))
        object.__setattr__(self, "policy_name", _nonempty_string("policy_name", self.policy_name))
        object.__setattr__(self, "policy_version", _nonnegative_int("policy_version", self.policy_version))
        object.__setattr__(self, "archive_version", _nonnegative_int("archive_version", self.archive_version))
        object.__setattr__(self, "starter_id", _nonnegative_int("starter_id", self.starter_id))

        selection_probability = _finite_float("selection_probability", self.selection_probability)
        if not 0.0 < selection_probability <= 1.0:
            raise ValueError("selection_probability must be in (0, 1]")
        object.__setattr__(self, "selection_probability", selection_probability)
        object.__setattr__(self, "random_seed", _nonnegative_int("random_seed", self.random_seed))
        if self.force_budget is not None:
            object.__setattr__(self, "force_budget", _positive_int("force_budget", self.force_budget))


@dataclass(frozen=True)
class AttemptResult:
    """Field-frozen worker result that captures an independent State snapshot.

    The captured State remains mutable internally; only this record's fields are frozen.
    """

    action: StarterAction
    landing_state: State | None
    landing_energy: float | None
    force_evaluations: int
    status: AttemptStatus
    failure_reason: str | None
    evaluation_counts: EvaluationCounts | None = None
    cost_is_exact: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.action, StarterAction):
            raise ValueError("action must be a StarterAction")
        force_evaluations = _nonnegative_int("force_evaluations", self.force_evaluations)
        if self.action.force_budget is not None and force_evaluations > self.action.force_budget:
            raise ValueError("force_evaluations cannot exceed action force_budget")
        object.__setattr__(self, "force_evaluations", force_evaluations)
        object.__setattr__(
            self,
            "evaluation_counts",
            _terminal_evaluation_counts(force_evaluations, self.evaluation_counts),
        )
        object.__setattr__(self, "cost_is_exact", _strict_boolean("cost_is_exact", self.cost_is_exact))
        _status(self.status)

        if self.status is AttemptStatus.COMPLETED:
            if not isinstance(self.landing_state, State):
                raise ValueError("completed attempts require landing_state")
            if self.landing_energy is None:
                raise ValueError("completed attempts require landing_energy")
            object.__setattr__(self, "landing_energy", _finite_float("landing_energy", self.landing_energy))
            if self.failure_reason is not None:
                raise ValueError("completed attempts cannot have failure_reason")
            try:
                landing_snapshot = deepcopy(self.landing_state)
            except Exception as exc:
                raise ValueError("landing_state must be deepcopyable to capture a snapshot") from exc
            if not np.isfinite(landing_snapshot.positions).all() or (
                landing_snapshot.cell is not None and not np.isfinite(landing_snapshot.cell).all()
            ):
                raise ValueError("completed attempts require finite landing geometry")
            object.__setattr__(self, "landing_state", landing_snapshot)
            return

        if self.landing_state is not None or self.landing_energy is not None:
            raise ValueError("failed attempts cannot carry landing data")
        object.__setattr__(self, "failure_reason", _failure_reason("failure_reason", self.failure_reason))


@dataclass(frozen=True)
class CreditedOutcome:
    """Committed archive credit for a completed or failed dispatched action."""

    action_id: str
    starter_id: int
    discovered_against_snapshot: bool
    inserted_into_archive: bool
    within_batch_collision: bool
    force_evaluations: int
    status: AttemptStatus
    landing_entry_id: int | None
    landing_energy: float | None
    failure_reason: str | None
    evaluation_counts: EvaluationCounts | None = None
    cost_is_exact: bool = True
    posterior_observed: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_id", _nonempty_string("action_id", self.action_id))
        object.__setattr__(self, "starter_id", _nonnegative_int("starter_id", self.starter_id))
        force_evaluations = _nonnegative_int("force_evaluations", self.force_evaluations)
        object.__setattr__(self, "force_evaluations", force_evaluations)
        object.__setattr__(
            self,
            "evaluation_counts",
            _terminal_evaluation_counts(force_evaluations, self.evaluation_counts),
        )
        object.__setattr__(self, "cost_is_exact", _strict_boolean("cost_is_exact", self.cost_is_exact))
        object.__setattr__(
            self,
            "posterior_observed",
            _strict_boolean("posterior_observed", self.posterior_observed),
        )
        _status(self.status)
        if self.posterior_observed != should_observe_posterior(
            self.status, self.evaluation_counts
        ):
            raise ValueError("posterior_observed must match terminal observation predicate")
        for name in (
            "discovered_against_snapshot",
            "inserted_into_archive",
            "within_batch_collision",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")

        if self.status is AttemptStatus.COMPLETED:
            if self.landing_entry_id is None or self.landing_energy is None:
                raise ValueError("completed outcomes require landing data")
            object.__setattr__(
                self,
                "landing_entry_id",
                _nonnegative_int("landing_entry_id", self.landing_entry_id),
            )
            object.__setattr__(self, "landing_energy", _finite_float("landing_energy", self.landing_energy))
            if self.failure_reason is not None:
                raise ValueError("completed outcomes cannot have failure_reason")
            if self.inserted_into_archive and not self.discovered_against_snapshot:
                raise ValueError("inserted outcomes must be discovered against the dispatch snapshot")
            if self.within_batch_collision != (
                self.discovered_against_snapshot and not self.inserted_into_archive
            ):
                raise ValueError("within_batch_collision must describe an uninserted discovery")
            return

        if self.landing_entry_id is not None or self.landing_energy is not None:
            raise ValueError("failed outcomes cannot carry landing data")
        if self.discovered_against_snapshot or self.inserted_into_archive or self.within_batch_collision:
            raise ValueError("failed outcomes cannot carry discovery or insertion credit")
        object.__setattr__(self, "failure_reason", _failure_reason("failure_reason", self.failure_reason))


__all__ = [
    "AttemptResult",
    "AttemptStatus",
    "CreditedOutcome",
    "PolicySnapshot",
    "StarterAction",
    "should_observe_posterior",
]
