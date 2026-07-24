from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

import pamssw
from pamssw.exploration.actions import (
    AttemptResult,
    AttemptStatus,
    CreditedOutcome,
    PolicySnapshot,
    StarterAction,
)
from pamssw.state import State


def test_package_root_exports_only_the_public_posterior_exploration_types():
    from pamssw.exploration.controller import ExplorationController
    from pamssw.exploration.posterior import StarterProductivityPosterior

    assert pamssw.ExplorationController is ExplorationController
    assert pamssw.StarterProductivityPosterior is StarterProductivityPosterior
    assert set(pamssw.__all__) == {
        "ExplorationController",
        "LSSSWConfig",
        "RelaxConfig",
        "RelaxOutcomeClass",
        "RelaxResult",
        "SSWConfig",
        "SearchMode",
        "SearchResult",
        "StarterProductivityPosterior",
        "State",
        "read_state",
        "relax_minimum",
        "run_ls_ssw",
        "run_ssw",
        "state_from_atoms",
        "state_to_atoms",
        "write_state",
    }

    internal_exports = {
        "AttemptResult",
        "AttemptStatus",
        "BatchLog",
        "CreditedOutcome",
        "ExplorationEventLog",
        "PolicySnapshot",
        "SCHEMA_VERSION",
        "SUPPORTED_POLICIES",
        "StarterAction",
        "Worker",
        "build_policy_snapshot",
        "derive_action_seed",
        "plan_batch",
    }
    assert not (internal_exports & set(pamssw.__all__))
    assert all(not hasattr(pamssw, name) for name in internal_exports)


def _state(x: float = 1.0) -> State:
    return State(numbers=np.array([1]), positions=np.array([[x, 0.0, 0.0]]))


def _action(*, force_budget: int | None = 8) -> StarterAction:
    return StarterAction(
        action_id="batch-00000001-slot-0000",
        batch_id=1,
        slot_id=0,
        policy_name="uniform",
        policy_version=2,
        archive_version=3,
        starter_id=4,
        selection_probability=0.25,
        random_seed=12,
        force_budget=force_budget,
    )


def test_attempt_status_has_only_the_contract_statuses():
    assert set(AttemptStatus) == {
        AttemptStatus.COMPLETED,
        AttemptStatus.INVALID,
        AttemptStatus.FRAGMENTED,
        AttemptStatus.BUDGET_EXHAUSTED,
        AttemptStatus.WORKER_ERROR,
    }


def test_policy_snapshot_exposes_a_normalized_probability_lookup():
    snapshot = PolicySnapshot(
        version=2,
        archive_version=3,
        policy_name="posterior_proportional",
        eligible_starter_ids=(4, 9),
        probabilities=(0.25, 0.75),
        support_complete=True,
    )

    assert snapshot.probability_for(9) == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("eligible_starter_ids", "probabilities", "support_complete"),
    [
        ((), (), True),
        ((0, 0), (0.5, 0.5), True),
        ((0, 1), (0.2,), True),
        ((0, 1), (0.2, 0.2), True),
        ((0, 1), (-0.1, 1.1), True),
        ((0, 1), (float("nan"), 1.0), True),
        ((0, 1), (1.0, 0.0), True),
    ],
)
def test_policy_snapshot_rejects_invalid_probability_vectors(
    eligible_starter_ids, probabilities, support_complete
):
    with pytest.raises(ValueError):
        PolicySnapshot(
            version=0,
            archive_version=0,
            policy_name="uniform",
            eligible_starter_ids=eligible_starter_ids,
            probabilities=probabilities,
            support_complete=support_complete,
        )


def test_policy_snapshot_rejects_unknown_starter_id():
    snapshot = PolicySnapshot(0, 0, "uniform", (0,), (1.0,), True)

    with pytest.raises(KeyError, match="unknown starter_id"):
        snapshot.probability_for(5)


def test_policy_snapshot_is_frozen_and_copies_mutable_sequences():
    starter_ids = [0]
    probabilities = [1.0]
    snapshot = PolicySnapshot(0, 0, "uniform", starter_ids, probabilities, True)
    starter_ids[0] = 3
    probabilities[0] = 0.5

    assert snapshot.eligible_starter_ids == (0,)
    assert snapshot.probabilities == (1.0,)
    with pytest.raises(FrozenInstanceError):
        snapshot.policy_name = "other"


@pytest.mark.parametrize(
    "field,value",
    [
        ("action_id", ""),
        ("policy_name", ""),
        ("batch_id", -1),
        ("slot_id", -1),
        ("policy_version", -1),
        ("archive_version", -1),
        ("starter_id", -1),
        ("selection_probability", 0.0),
        ("selection_probability", float("inf")),
        ("random_seed", -1),
        ("force_budget", 0),
    ],
)
def test_starter_action_rejects_invalid_dispatch_metadata(field, value):
    values = {
        "action_id": "batch-00000001-slot-0000",
        "batch_id": 1,
        "slot_id": 0,
        "policy_name": "uniform",
        "policy_version": 2,
        "archive_version": 3,
        "starter_id": 4,
        "selection_probability": 0.25,
        "random_seed": 12,
        "force_budget": 8,
    }
    values[field] = value

    with pytest.raises(ValueError):
        StarterAction(**values)


def test_starter_action_is_frozen_and_allows_an_unbounded_force_budget():
    action = _action(force_budget=None)

    assert action.force_budget is None
    with pytest.raises(FrozenInstanceError):
        action.slot_id = 1


def test_completed_attempt_requires_finite_landing_data_within_budget():
    result = AttemptResult(
        action=_action(),
        landing_state=_state(),
        landing_energy=-1.0,
        force_evaluations=8,
        status=AttemptStatus.COMPLETED,
        failure_reason=None,
    )

    assert result.status is AttemptStatus.COMPLETED


def test_attempt_result_captures_an_independent_landing_state_snapshot():
    landing_state = State(
        numbers=np.array([1]),
        positions=np.array([[1.0, 0.0, 0.0]]),
        metadata={"labels": ["landing"]},
    )
    result = AttemptResult(
        action=_action(),
        landing_state=landing_state,
        landing_energy=-1.0,
        force_evaluations=1,
        status=AttemptStatus.COMPLETED,
        failure_reason=None,
    )

    landing_state.positions[0, 0] = 9.0
    landing_state.metadata["labels"].append("mutated")

    assert result.landing_state is not None
    assert result.landing_state.positions[0, 0] == pytest.approx(1.0)
    assert result.landing_state.metadata["labels"] == ["landing"]

    result.landing_state.positions[0, 0] = 3.0
    result.landing_state.metadata["labels"].append("record mutation")

    assert result.landing_state.positions[0, 0] == pytest.approx(3.0)
    assert result.landing_state.metadata["labels"] == ["landing", "record mutation"]


def test_attempt_result_deepcopies_cyclic_landing_metadata():
    cycle: dict[str, object] = {}
    cycle["self"] = cycle
    result = AttemptResult(
        action=_action(),
        landing_state=State(
            numbers=np.array([1]),
            positions=np.array([[1.0, 0.0, 0.0]]),
            metadata={"cycle": cycle},
        ),
        landing_energy=-1.0,
        force_evaluations=1,
        status=AttemptStatus.COMPLETED,
        failure_reason=None,
    )

    assert result.landing_state is not None
    assert result.landing_state.metadata["cycle"]["self"] is result.landing_state.metadata["cycle"]


def test_attempt_result_keeps_normal_dataclass_fields_and_replace_behavior():
    result = AttemptResult(
        action=_action(),
        landing_state=State(
            numbers=np.array([1]),
            positions=np.array([[1.0, 0.0, 0.0]]),
            metadata={"labels": ["landing"]},
        ),
        landing_energy=-1.0,
        force_evaluations=1,
        status=AttemptStatus.COMPLETED,
        failure_reason=None,
    )

    assert [item.name for item in fields(result)] == [
        "action",
        "landing_state",
        "landing_energy",
        "force_evaluations",
        "status",
        "failure_reason",
    ]
    replaced = replace(result, force_evaluations=2)
    assert replaced.force_evaluations == 2
    assert replaced.landing_state is not result.landing_state

    assert result.landing_state is not None
    result.landing_state.positions[0, 0] = 4.0
    result.landing_state.metadata["labels"].append("result mutation")

    assert replaced.landing_state is not None
    assert replaced.landing_state.positions[0, 0] == pytest.approx(1.0)
    assert replaced.landing_state.metadata["labels"] == ["landing"]


def test_attempt_result_rejects_non_deepcopyable_landing_metadata():
    class NonDeepcopyable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy")

    with pytest.raises(ValueError, match="landing_state must be deepcopyable") as error:
        AttemptResult(
            action=_action(),
            landing_state=State(
                numbers=np.array([1]),
                positions=np.array([[1.0, 0.0, 0.0]]),
                metadata={"uncopyable": NonDeepcopyable()},
            ),
            landing_energy=-1.0,
            force_evaluations=1,
            status=AttemptStatus.COMPLETED,
            failure_reason=None,
        )

    assert isinstance(error.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("landing_state", "landing_energy", "force_evaluations", "failure_reason"),
    [
        (None, -1.0, 1, None),
        (_state(), float("nan"), 1, None),
        (_state(), -1.0, 1, "unexpected"),
        (_state(), -1.0, 9, None),
        (_state(), -1.0, -1, None),
    ],
)
def test_completed_attempt_rejects_inconsistent_result_data(
    landing_state, landing_energy, force_evaluations, failure_reason
):
    with pytest.raises(ValueError):
        AttemptResult(
            action=_action(),
            landing_state=landing_state,
            landing_energy=landing_energy,
            force_evaluations=force_evaluations,
            status=AttemptStatus.COMPLETED,
            failure_reason=failure_reason,
        )


@pytest.mark.parametrize(
    ("landing_state", "landing_energy", "failure_reason"),
    [
        (None, None, None),
        (_state(), None, "worker failed"),
        (None, -1.0, "worker failed"),
        (None, None, ""),
    ],
)
def test_failed_attempt_requires_only_a_nonempty_failure_reason(
    landing_state, landing_energy, failure_reason
):
    with pytest.raises(ValueError):
        AttemptResult(
            action=_action(),
            landing_state=landing_state,
            landing_energy=landing_energy,
            force_evaluations=1,
            status=AttemptStatus.WORKER_ERROR,
            failure_reason=failure_reason,
        )


def test_credited_outcome_preserves_dispatch_and_commit_facts():
    outcome = CreditedOutcome(
        action_id="a",
        starter_id=2,
        discovered_against_snapshot=True,
        inserted_into_archive=False,
        within_batch_collision=True,
        force_evaluations=7,
        status=AttemptStatus.COMPLETED,
        landing_entry_id=5,
        landing_energy=-3.0,
        failure_reason=None,
    )

    assert outcome.discovered_against_snapshot
    assert outcome.within_batch_collision


@pytest.mark.parametrize(
    "values",
    [
        {"landing_entry_id": None},
        {"landing_energy": None},
        {"failure_reason": "unexpected"},
        {"discovered_against_snapshot": False, "inserted_into_archive": True},
        {
            "discovered_against_snapshot": True,
            "inserted_into_archive": True,
            "within_batch_collision": True,
        },
        {
            "discovered_against_snapshot": True,
            "inserted_into_archive": False,
            "within_batch_collision": False,
        },
    ],
)
def test_credited_outcome_rejects_inconsistent_completed_facts(values):
    outcome_values = {
        "action_id": "a",
        "starter_id": 2,
        "discovered_against_snapshot": True,
        "inserted_into_archive": True,
        "within_batch_collision": False,
        "force_evaluations": 7,
        "status": AttemptStatus.COMPLETED,
        "landing_entry_id": 5,
        "landing_energy": -3.0,
        "failure_reason": None,
    }
    outcome_values.update(values)

    with pytest.raises(ValueError):
        CreditedOutcome(**outcome_values)


def test_credited_outcome_rejects_landing_and_credit_facts_for_failed_attempts():
    with pytest.raises(ValueError):
        CreditedOutcome(
            action_id="a",
            starter_id=2,
            discovered_against_snapshot=True,
            inserted_into_archive=False,
            within_batch_collision=False,
            force_evaluations=0,
            status=AttemptStatus.INVALID,
            landing_entry_id=5,
            landing_energy=-3.0,
            failure_reason="bad geometry",
        )


def test_attempt_result_and_credited_outcome_are_frozen():
    result = AttemptResult(
        action=_action(),
        landing_state=_state(),
        landing_energy=-1.0,
        force_evaluations=1,
        status=AttemptStatus.COMPLETED,
        failure_reason=None,
    )
    outcome = CreditedOutcome(
        action_id="a",
        starter_id=2,
        discovered_against_snapshot=False,
        inserted_into_archive=False,
        within_batch_collision=False,
        force_evaluations=1,
        status=AttemptStatus.COMPLETED,
        landing_entry_id=2,
        landing_energy=-1.0,
        failure_reason=None,
    )

    with pytest.raises(FrozenInstanceError):
        result.force_evaluations = 2
    with pytest.raises(FrozenInstanceError):
        outcome.action_id = "other"
