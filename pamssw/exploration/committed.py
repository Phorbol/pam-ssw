"""Immutable, fully reconciled facts for one exploration batch."""

from __future__ import annotations

from dataclasses import dataclass

from .actions import (
    AttemptResult,
    CreditedOutcome,
    PolicySnapshot,
    StarterAction,
    should_observe_posterior,
)


def _validate_action_against_snapshot(
    action: StarterAction, snapshot: PolicySnapshot, batch_id: int
) -> None:
    if action.batch_id != batch_id:
        raise ValueError("all actions must share one batch_id")
    if action.policy_name != snapshot.policy_name:
        raise ValueError("action policy_name must match snapshot")
    if action.policy_version != snapshot.version:
        raise ValueError("action policy_version must match snapshot")
    if action.archive_version != snapshot.archive_version:
        raise ValueError("action archive_version must match snapshot")
    if action.starter_id not in snapshot.eligible_starter_ids:
        raise ValueError("action starter_id must be eligible in snapshot")
    if action.selection_probability != snapshot.probability_for(action.starter_id):
        raise ValueError("action selection_probability must match snapshot")


@dataclass(frozen=True)
class CommittedExplorationBatch:
    """The immutable result and credit facts for one committed batch.

    ``AttemptResult`` retains the canonical landing geometry.  This batch only
    validates scalar agreement between a result and its archive credit.
    """

    snapshot: PolicySnapshot
    actions: tuple[StarterAction, ...]
    results: tuple[AttemptResult, ...]
    outcomes: tuple[CreditedOutcome, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, PolicySnapshot):
            raise ValueError("snapshot must be a PolicySnapshot")
        for name in ("actions", "results", "outcomes"):
            if not isinstance(getattr(self, name), tuple):
                raise ValueError(f"{name} must be a tuple")
        if not self.actions:
            raise ValueError("actions, results, and outcomes must be nonempty")
        if len(self.actions) != len(self.results) or len(self.actions) != len(self.outcomes):
            raise ValueError("actions, results, and outcomes must have equal lengths")
        if not all(isinstance(action, StarterAction) for action in self.actions):
            raise ValueError("actions must contain StarterAction values")
        if not all(isinstance(result, AttemptResult) for result in self.results):
            raise ValueError("results must contain AttemptResult values")
        if not all(isinstance(outcome, CreditedOutcome) for outcome in self.outcomes):
            raise ValueError("outcomes must contain CreditedOutcome values")

        batch_id = self.actions[0].batch_id
        action_ids: set[str] = set()
        slot_ids: set[int] = set()
        for action in self.actions:
            _validate_action_against_snapshot(action, self.snapshot, batch_id)
            if action.action_id in action_ids:
                raise ValueError("action IDs must be unique")
            if action.slot_id in slot_ids:
                raise ValueError("slot IDs must be unique")
            action_ids.add(action.action_id)
            slot_ids.add(action.slot_id)
        for expected_slot, (action, result, outcome) in enumerate(
            zip(self.actions, self.results, self.outcomes)
        ):
            if action.slot_id != expected_slot:
                raise ValueError("actions must be in contiguous slot order")
            if result.action != action:
                raise ValueError("result action must agree with action")
            if outcome.action_id != action.action_id:
                raise ValueError("action_id must agree with outcome")
            if outcome.starter_id != action.starter_id:
                raise ValueError("starter_id must agree with outcome")
            if result.status != outcome.status:
                raise ValueError("result status must agree with outcome")
            if result.failure_reason != outcome.failure_reason:
                raise ValueError("result failure_reason must agree with outcome")
            if result.landing_energy != outcome.landing_energy:
                raise ValueError("result landing_energy must agree with outcome")
            if result.force_evaluations != outcome.force_evaluations:
                raise ValueError("result force_evaluations must agree with outcome")
            if result.evaluation_counts != outcome.evaluation_counts:
                raise ValueError("result evaluation_counts must agree with outcome")
            if result.cost_is_exact != outcome.cost_is_exact:
                raise ValueError("result cost_is_exact must agree with outcome")
            if action.force_budget is not None and outcome.force_evaluations > action.force_budget:
                raise ValueError("force_evaluations cannot exceed action force_budget")
            if outcome.posterior_observed != should_observe_posterior(
                outcome.status, outcome.evaluation_counts
            ):
                raise ValueError("posterior_observed must match terminal observation predicate")

    @property
    def batch_id(self) -> int:
        """The shared batch identity, derived from the first ordered action."""
        return self.actions[0].batch_id


__all__ = ["CommittedExplorationBatch"]
