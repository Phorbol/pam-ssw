import json
import os
import stat
from dataclasses import FrozenInstanceError, replace

import pytest
import numpy as np

from pamssw.accounting import EvaluationCounts, EvaluationPurpose
import pamssw.exploration.event_log as event_log
from pamssw.exploration.actions import (
    AttemptResult,
    AttemptStatus,
    CreditedOutcome,
    PolicySnapshot,
    StarterAction,
)
from pamssw.exploration.committed import CommittedExplorationBatch
from pamssw.exploration.event_log import ExplorationEventLog, SCHEMA_VERSION
from pamssw.state import State


def _snapshot(*, version: int = 7, archive_version: int = 11) -> PolicySnapshot:
    return PolicySnapshot(
        version=version,
        archive_version=archive_version,
        policy_name="posterior_proportional",
        eligible_starter_ids=(3, 8),
        probabilities=(0.2, 0.8),
        support_complete=True,
    )


def _action(
    *,
    batch_id: int = 4,
    slot_id: int = 0,
    starter_id: int = 8,
    selection_probability: float = 0.8,
) -> StarterAction:
    return StarterAction(
        action_id=f"batch-{batch_id:08d}-slot-{slot_id:04d}",
        batch_id=batch_id,
        slot_id=slot_id,
        policy_name="posterior_proportional",
        policy_version=7,
        archive_version=11,
        starter_id=starter_id,
        selection_probability=selection_probability,
        random_seed=100 + slot_id,
        force_budget=12,
    )


def _outcome(
    action: StarterAction,
    *,
    discovered: bool = True,
    status: AttemptStatus = AttemptStatus.COMPLETED,
    failure_reason: str | None = None,
) -> CreditedOutcome:
    if status is AttemptStatus.COMPLETED:
        return CreditedOutcome(
            action_id=action.action_id,
            starter_id=action.starter_id,
            discovered_against_snapshot=discovered,
            inserted_into_archive=discovered,
            within_batch_collision=False,
            force_evaluations=5,
            status=status,
            landing_entry_id=43,
            landing_energy=-1.25,
            failure_reason=None,
        )
    return CreditedOutcome(
        action_id=action.action_id,
        starter_id=action.starter_id,
        discovered_against_snapshot=False,
        inserted_into_archive=False,
        within_batch_collision=False,
        force_evaluations=5,
        status=status,
        landing_entry_id=None,
        landing_energy=None,
        failure_reason=failure_reason or "worker reported exact failure: OOM",
        posterior_observed=False,
    )


def _result(action: StarterAction, outcome: CreditedOutcome) -> AttemptResult:
    return AttemptResult(
        action=action,
        landing_state=(
            State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
            if outcome.status is AttemptStatus.COMPLETED
            else None
        ),
        landing_energy=outcome.landing_energy,
        force_evaluations=outcome.force_evaluations,
        status=outcome.status,
        failure_reason=outcome.failure_reason,
        evaluation_counts=outcome.evaluation_counts,
        cost_is_exact=outcome.cost_is_exact,
    )


def _batch(
    snapshot: PolicySnapshot,
    actions: tuple[StarterAction, ...],
    outcomes: tuple[CreditedOutcome, ...],
) -> CommittedExplorationBatch:
    return CommittedExplorationBatch(
        snapshot=snapshot,
        actions=actions,
        results=tuple(_result(action, outcome) for action, outcome in zip(actions, outcomes)),
        outcomes=outcomes,
    )


def test_committed_batch_requires_complete_slot_aligned_facts() -> None:
    action = _action()
    outcome = _outcome(action, status=AttemptStatus.WORKER_ERROR)

    batch = CommittedExplorationBatch(
        snapshot=_snapshot(),
        actions=(action,),
        results=(_result(action, outcome),),
        outcomes=(outcome,),
    )

    assert batch.batch_id == action.batch_id
    assert batch.actions == (action,)


def _failure_parts() -> tuple[
    PolicySnapshot,
    tuple[StarterAction, ...],
    tuple[AttemptResult, ...],
    tuple[CreditedOutcome, ...],
]:
    action = _action()
    outcome = _outcome(action, status=AttemptStatus.WORKER_ERROR)
    return _snapshot(), (action,), (_result(action, outcome),), (outcome,)


def _two_failure_parts() -> tuple[
    PolicySnapshot,
    tuple[StarterAction, ...],
    tuple[AttemptResult, ...],
    tuple[CreditedOutcome, ...],
]:
    first = _action(slot_id=0, starter_id=8, selection_probability=0.8)
    second = _action(slot_id=1, starter_id=3, selection_probability=0.2)
    first_outcome = _outcome(first, status=AttemptStatus.WORKER_ERROR)
    second_outcome = _outcome(second, status=AttemptStatus.WORKER_ERROR)
    return (
        _snapshot(),
        (first, second),
        (_result(first, first_outcome), _result(second, second_outcome)),
        (first_outcome, second_outcome),
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda snapshot, actions, results, outcomes: (object(), actions, results, outcomes), "snapshot"),
        (lambda snapshot, actions, results, outcomes: (snapshot, list(actions), results, outcomes), "actions"),
        (lambda snapshot, actions, results, outcomes: (snapshot, actions, list(results), outcomes), "results"),
        (lambda snapshot, actions, results, outcomes: (snapshot, actions, results, list(outcomes)), "outcomes"),
        (lambda snapshot, actions, results, outcomes: (snapshot, (), (), ()), "nonempty"),
        (lambda snapshot, actions, results, outcomes: (snapshot, actions, results, ()), "equal lengths"),
        (lambda snapshot, actions, results, outcomes: (snapshot, (object(),), results, outcomes), "StarterAction"),
        (lambda snapshot, actions, results, outcomes: (snapshot, actions, (object(),), outcomes), "AttemptResult"),
        (lambda snapshot, actions, results, outcomes: (snapshot, actions, results, (object(),)), "CreditedOutcome"),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                actions,
                (_result(replace(actions[0], action_id="wrong-result"), outcomes[0]),),
                outcomes,
            ),
            "result action",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                actions,
                results,
                (replace(outcomes[0], action_id="wrong-outcome"),),
            ),
            "action_id",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                actions,
                results,
                (replace(outcomes[0], starter_id=3),),
            ),
            "starter_id",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                actions,
                (replace(results[0], status=AttemptStatus.INVALID),),
                outcomes,
            ),
            "status",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                actions,
                (replace(results[0], cost_is_exact=False),),
                outcomes,
            ),
            "cost_is_exact",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                actions,
                results,
                (
                    replace(
                        outcomes[0],
                        force_evaluations=4,
                        evaluation_counts=EvaluationCounts.unattributed(4),
                    ),
                ),
            ),
            "force_evaluations",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                actions,
                results,
                (
                    replace(
                        outcomes[0],
                        evaluation_counts=EvaluationCounts.from_mapping(
                            {EvaluationPurpose.DIRECTION_ORACLE: outcomes[0].force_evaluations}
                        ),
                    ),
                ),
            ),
            "evaluation_counts",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                actions,
                results,
                (replace(outcomes[0], posterior_observed=True),),
            ),
            "posterior_observed",
        ),
    ],
)
def test_committed_batch_rejects_invalid_slot_facts(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CommittedExplorationBatch(*mutate(*_failure_parts()))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                (actions[0], replace(actions[0], slot_id=1)),
                (results[0], _result(replace(actions[0], slot_id=1), outcomes[0])),
                (outcomes[0], outcomes[0]),
            ),
            "unique",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                (actions[0], replace(actions[1], slot_id=2)),
                (results[0], _result(replace(actions[1], slot_id=2), outcomes[1])),
                outcomes,
            ),
            "contiguous slot order",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                tuple(reversed(actions)),
                tuple(reversed(results)),
                tuple(reversed(outcomes)),
            ),
            "contiguous slot order",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                actions,
                tuple(reversed(results)),
                outcomes,
            ),
            "result action",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                (replace(actions[0], policy_version=snapshot.version + 1), actions[1]),
                (
                    _result(
                        replace(actions[0], policy_version=snapshot.version + 1), outcomes[0]
                    ),
                    results[1],
                ),
                outcomes,
            ),
            "policy_version",
        ),
        (
            lambda snapshot, actions, results, outcomes: (
                snapshot,
                (replace(actions[0], selection_probability=0.2), actions[1]),
                (
                    _result(
                        replace(actions[0], selection_probability=0.2), outcomes[0]
                    ),
                    results[1],
                ),
                outcomes,
            ),
            "selection_probability",
        ),
    ],
)
def test_committed_batch_rejects_invalid_batch_order_or_snapshot_alignment(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        CommittedExplorationBatch(*mutate(*_two_failure_parts()))


def test_committed_batch_fields_are_frozen() -> None:
    batch = CommittedExplorationBatch(*_failure_parts())

    with pytest.raises(FrozenInstanceError):
        batch.actions = ()  # type: ignore[misc]


def test_append_accepts_only_one_complete_committed_batch(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    batch = CommittedExplorationBatch(*_failure_parts())

    ExplorationEventLog(path).append_batch(batch)
    with pytest.raises(ValueError, match="CommittedExplorationBatch"):
        ExplorationEventLog(path).append_batch(object())

    assert ExplorationEventLog(path).reconstruct_posterior().completed_attempts == 0


def test_idempotency_rejects_one_changed_serialized_scalar_fact(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    snapshot, actions, results, outcomes = _failure_parts()
    batch = CommittedExplorationBatch(snapshot, actions, results, outcomes)
    changed_outcome = replace(outcomes[0], failure_reason="a different durable failure")
    changed_batch = CommittedExplorationBatch(
        snapshot,
        actions,
        (_result(actions[0], changed_outcome),),
        (changed_outcome,),
    )

    log = ExplorationEventLog(path)
    log.append_batch(batch)
    with pytest.raises(ValueError, match="batch_id"):
        log.append_batch(changed_batch)


def test_reconstruction_rejects_noncontiguous_action_slots(tmp_path) -> None:
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)
    rows = _read_rows(path)
    rows[2]["slot_id"] = 3
    path.write_text(
        "\n".join(event_log._canonical_json_line(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contiguous slot order"):
        ExplorationEventLog(path).reconstruct_posterior()


def _read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_valid_batch(path, *, batch_id: int = 4):
    snapshot = _snapshot()
    actions = (
        _action(batch_id=batch_id, slot_id=0, starter_id=8, selection_probability=0.8),
        _action(batch_id=batch_id, slot_id=1, starter_id=3, selection_probability=0.2),
    )
    outcomes = (
        _outcome(actions[0]),
        _outcome(actions[1], status=AttemptStatus.WORKER_ERROR, failure_reason="exact remote OOM"),
    )
    ExplorationEventLog(path).append_batch(_batch(snapshot, actions, outcomes))
    return actions, outcomes


def test_append_writes_one_sorted_jsonl_payload_in_policy_attempt_commit_order(tmp_path):
    path = tmp_path / "events.jsonl"
    snapshot = _snapshot()
    actions = (
        _action(slot_id=0, starter_id=8, selection_probability=0.8),
        _action(slot_id=1, starter_id=3, selection_probability=0.2),
    )
    outcomes = (
        _outcome(actions[0]),
        _outcome(actions[1], status=AttemptStatus.WORKER_ERROR, failure_reason="exact remote OOM"),
    )

    ExplorationEventLog(path).append_batch(_batch(snapshot, actions, outcomes))

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in raw_lines]
    assert [row["record_type"] for row in rows] == [
        "policy_snapshot",
        "attempt",
        "attempt",
        "batch_commit",
    ]
    assert rows[0] == {
        "archive_version": 11,
        "batch_id": 4,
        "eligible_starter_ids": [3, 8],
        "policy_name": "posterior_proportional",
        "policy_version": 7,
        "probabilities": [0.2, 0.8],
        "record_type": "policy_snapshot",
        "schema_version": SCHEMA_VERSION,
        "support_complete": True,
    }
    assert rows[1]["selection_probability"] == 0.8
    assert rows[2]["selection_probability"] == 0.2
    assert rows[2]["failure_reason"] == "exact remote OOM"
    assert rows[3]["action_ids"] == [action.action_id for action in actions]
    assert all(line == event_log._canonical_json_line(row) for line, row in zip(raw_lines, rows))


def test_append_requires_an_existing_parent_directory_without_creating_it(tmp_path):
    path = tmp_path / "not-created" / "events.jsonl"
    action = _action()

    with pytest.raises(FileNotFoundError, match="parent directory"):
        ExplorationEventLog(path).append_batch(_batch(_snapshot(), (action,), (_outcome(action),)))

    assert not path.parent.exists()


def test_append_rejects_a_non_directory_parent_without_mutating_it(tmp_path):
    parent = tmp_path / "not-a-directory"
    parent.write_text("keep this file", encoding="utf-8")
    action = _action()

    with pytest.raises(ValueError, match="parent path must be a directory"):
        ExplorationEventLog(parent / "events.jsonl").append_batch(
            _batch(_snapshot(), (action,), (_outcome(action),))
        )

    assert parent.read_text(encoding="utf-8") == "keep this file"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support is unavailable")
def test_existing_final_symlink_is_rejected_without_mutating_its_target(tmp_path):
    target = tmp_path / "target.jsonl"
    target.write_text("target stays unchanged\n", encoding="utf-8")
    path = tmp_path / "events.jsonl"
    path.symlink_to(target)
    action = _action()

    with pytest.raises(ValueError, match="final event-log path.*symlink"):
        ExplorationEventLog(path).append_batch(_batch(_snapshot(), (action,), (_outcome(action),)))
    with pytest.raises(ValueError, match="final event-log path.*symlink"):
        ExplorationEventLog(path).reconstruct_posterior()

    assert path.is_symlink()
    assert target.read_text(encoding="utf-8") == "target stays unchanged\n"


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink support is unavailable")
def test_dangling_final_symlink_is_rejected_without_creating_its_target(tmp_path):
    target = tmp_path / "missing-target.jsonl"
    path = tmp_path / "events.jsonl"
    path.symlink_to(target)
    action = _action()

    with pytest.raises(ValueError, match="final event-log path.*symlink"):
        ExplorationEventLog(path).append_batch(_batch(_snapshot(), (action,), (_outcome(action),)))
    with pytest.raises(ValueError, match="final event-log path.*symlink"):
        ExplorationEventLog(path).reconstruct_posterior()

    assert path.is_symlink()
    assert not target.exists()


def test_ordinary_final_file_remains_appendable_and_replayable(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)

    assert not path.is_symlink()
    assert ExplorationEventLog(path).reconstruct_posterior().completed_attempts == 1


def test_reconstructs_committed_attempt_facts_once_per_action(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)

    posterior = ExplorationEventLog(path).reconstruct_posterior()

    assert posterior.counts(8) == (1, 0)
    assert posterior.counts(3) == (0, 0)
    assert posterior.completed_attempts == 1


def test_reconstructs_multiple_committed_batches_in_append_order(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path, batch_id=4)
    _write_valid_batch(path, batch_id=5)

    posterior = ExplorationEventLog(path).reconstruct_posterior()

    assert posterior.counts(8) == (2, 0)
    assert posterior.counts(3) == (0, 0)
    assert posterior.completed_attempts == 2


def test_missing_event_log_reconstructs_an_empty_posterior(tmp_path):
    posterior = ExplorationEventLog(tmp_path / "missing.jsonl").reconstruct_posterior()

    assert posterior.completed_attempts == 0
    assert posterior.counts(8) == (0, 0)


@pytest.mark.parametrize(
    ("snapshot", "actions", "outcomes", "message"),
    [
        (object(), (_action(),), (_outcome(_action()),), "snapshot"),
        (_snapshot(), [_action()], (_outcome(_action()),), "actions"),
        (_snapshot(), (_action(),), [_outcome(_action())], "outcomes"),
        (_snapshot(), (), (), "nonempty"),
        (_snapshot(), (_action(),), (), "equal"),
        (_snapshot(), (_action(),), (_outcome(_action(starter_id=3, selection_probability=0.2)),), "starter_id"),
        (_snapshot(), (replace(_action(), policy_version=9),), (_outcome(replace(_action(), policy_version=9)),), "policy_version"),
    ],
)
def test_append_rejects_type_pairing_and_snapshot_mismatches_before_writing(
    tmp_path, snapshot, actions, outcomes, message
):
    path = tmp_path / "events.jsonl"

    with pytest.raises(ValueError, match=message):
        ExplorationEventLog(path).append_batch(_batch(snapshot, actions, outcomes))

    assert not path.exists()


def test_append_rejects_duplicate_action_ids_or_slots_before_writing(tmp_path):
    path = tmp_path / "events.jsonl"
    first = _action(slot_id=0)
    duplicate_id = replace(first, slot_id=1)
    duplicate_slot = replace(
        _action(slot_id=1, starter_id=3, selection_probability=0.2), slot_id=0
    )

    for actions in ((first, duplicate_id), (first, duplicate_slot)):
        outcomes = tuple(_outcome(action) for action in actions)
        with pytest.raises(ValueError, match="unique"):
            ExplorationEventLog(path).append_batch(_batch(_snapshot(), actions, outcomes))
        assert not path.exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows.__setitem__(0, "not JSON"),
        lambda rows: rows.__setitem__(0, []),
        lambda rows: rows[0].__setitem__("schema_version", 99),
        lambda rows: rows[0].__setitem__("record_type", 9),
        lambda rows: rows[1].__setitem__("discovered_against_snapshot", 1),
        lambda rows: rows[0].__setitem__("record_type", "unknown"),
        lambda rows: rows[1].pop("random_seed"),
        lambda rows: rows[1].__setitem__("unexpected", True),
    ],
)
def test_reconstruction_fails_closed_on_invalid_json_schema_type_boolean_or_record_shape(tmp_path, mutate):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)
    rows = _read_rows(path)
    mutate(rows)
    path.write_text(
        "\n".join(
            row if isinstance(row, str) else event_log._canonical_json_line(row)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ExplorationEventLog(path).reconstruct_posterior()


def test_reconstruction_rejects_attempt_outside_its_active_snapshot(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)
    rows = _read_rows(path)
    path.write_text(json.dumps(rows[1], sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="attempt"):
        ExplorationEventLog(path).reconstruct_posterior()


def test_reconstruction_rejects_commit_action_ids_that_are_not_exact_attempt_order(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)
    rows = _read_rows(path)
    rows[-1]["action_ids"] = list(reversed(rows[-1]["action_ids"]))
    path.write_text(
        "\n".join(event_log._canonical_json_line(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action_ids"):
        ExplorationEventLog(path).reconstruct_posterior()


def test_reconstruction_fails_closed_for_an_incomplete_final_batch_without_leaking_attempts(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)
    rows = _read_rows(path)
    path.write_text(
        "\n".join(event_log._canonical_json_line(row) for row in rows[:-1]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete"):
        ExplorationEventLog(path).reconstruct_posterior()


def test_reconstruction_rejects_invalid_status_or_failure_data(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)
    rows = _read_rows(path)
    rows[1]["status"] = "made_up"
    rows[2]["failure_reason"] = None
    path.write_text(
        "\n".join(event_log._canonical_json_line(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ExplorationEventLog(path).reconstruct_posterior()


def test_append_rejects_a_duplicate_batch_id_with_different_actions_without_changing_file(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path, batch_id=4)
    replacement_action = replace(
        _action(batch_id=4, slot_id=0, starter_id=8, selection_probability=0.8),
        action_id="replacement-action-for-existing-batch",
    )
    before = path.read_bytes()

    with pytest.raises(ValueError, match="batch_id"):
        ExplorationEventLog(path).append_batch(
            _batch(_snapshot(), (replacement_action,), (_outcome(replacement_action),))
        )

    assert path.read_bytes() == before


def test_append_rejects_an_existing_action_id_under_a_different_batch_without_changing_file(tmp_path):
    path = tmp_path / "events.jsonl"
    existing_actions, _ = _write_valid_batch(path, batch_id=4)
    duplicate_action = replace(
        _action(batch_id=5, slot_id=0, starter_id=8, selection_probability=0.8),
        action_id=existing_actions[0].action_id,
    )
    before = path.read_bytes()

    with pytest.raises(ValueError, match="action_id"):
        ExplorationEventLog(path).append_batch(
            _batch(_snapshot(), (duplicate_action,), (_outcome(duplicate_action),))
        )

    assert path.read_bytes() == before


def test_append_treats_reappending_an_identical_batch_as_a_durable_idempotent_retry(tmp_path):
    path = tmp_path / "events.jsonl"
    actions, outcomes = _write_valid_batch(path, batch_id=4)
    before = path.read_bytes()
    before_line_count = len(before.splitlines())

    ExplorationEventLog(path).append_batch(_batch(_snapshot(), actions, outcomes))

    assert path.read_bytes() == before
    assert len(path.read_bytes().splitlines()) == before_line_count
    assert ExplorationEventLog(path).reconstruct_posterior().completed_attempts == 1


def test_reconstruction_rejects_a_handcrafted_duplicate_batch_id_with_different_actions(tmp_path):
    path = tmp_path / "events.jsonl"
    other_path = tmp_path / "other.jsonl"
    _write_valid_batch(path, batch_id=4)
    _write_valid_batch(other_path, batch_id=5)
    duplicate_batch_rows = _read_rows(other_path)
    for row in duplicate_batch_rows:
        row["batch_id"] = 4
    path.write_text(
        path.read_text(encoding="utf-8")
        + "".join(event_log._canonical_json_line(row) + "\n" for row in duplicate_batch_rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate batch_id"):
        ExplorationEventLog(path).reconstruct_posterior()


def test_reconstruction_rejects_a_handcrafted_duplicate_action_id_across_batches(tmp_path):
    path = tmp_path / "events.jsonl"
    other_path = tmp_path / "other.jsonl"
    _write_valid_batch(path, batch_id=4)
    _write_valid_batch(other_path, batch_id=5)
    first_rows = _read_rows(path)
    duplicate_action_rows = _read_rows(other_path)
    duplicate_action_rows[1]["action_id"] = first_rows[1]["action_id"]
    duplicate_action_rows[-1]["action_ids"][0] = first_rows[1]["action_id"]
    path.write_text(
        path.read_text(encoding="utf-8")
        + "".join(event_log._canonical_json_line(row) + "\n" for row in duplicate_action_rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate action_id"):
        ExplorationEventLog(path).reconstruct_posterior()


@pytest.mark.skipif(event_log.os.name != "posix", reason="parent-directory fsync is POSIX-only")
def test_append_and_idempotent_retry_preflight_and_refsync_a_held_parent_directory_fd(
    tmp_path, monkeypatch
):
    path = tmp_path / "events.jsonl"
    fsync_kinds = []

    def record_fsync_kind(fd):
        fsync_kinds.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")

    monkeypatch.setattr(event_log.os, "fsync", record_fsync_kind)

    action = _action()
    log = ExplorationEventLog(path)
    log.append_batch(_batch(_snapshot(), (action,), (_outcome(action),)))
    log.append_batch(_batch(_snapshot(), (action,), (_outcome(action),)))

    assert fsync_kinds == ["directory", "file", "directory", "directory", "file", "directory"]


@pytest.mark.skipif(event_log.os.name != "posix", reason="parent-directory fsync is POSIX-only")
def test_exact_retry_after_a_file_fsync_error_does_not_append_or_double_count(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    action = _action()
    log = ExplorationEventLog(path)
    fsync_kinds = []

    def fail_file_fsync_after_directory_preflight(fd):
        fsync_kinds.append("directory" if stat.S_ISDIR(os.fstat(fd).st_mode) else "file")
        if fsync_kinds == ["directory", "file"]:
            raise OSError("injected file fsync failure")

    monkeypatch.setattr(event_log.os, "fsync", fail_file_fsync_after_directory_preflight)
    with pytest.raises(OSError, match="injected file fsync failure"):
        log.append_batch(_batch(_snapshot(), (action,), (_outcome(action),)))

    bytes_after_failed_fsync = path.read_bytes()
    assert len(bytes_after_failed_fsync.splitlines()) == 3
    assert fsync_kinds == ["directory", "file"]

    monkeypatch.setattr(event_log.os, "fsync", lambda fd: None)
    log.append_batch(_batch(_snapshot(), (action,), (_outcome(action),)))

    assert path.read_bytes() == bytes_after_failed_fsync
    posterior = log.reconstruct_posterior()
    assert posterior.counts(action.starter_id) == (1, 0)
    assert posterior.completed_attempts == 1


@pytest.mark.skipif(event_log.os.name != "posix", reason="parent-directory fsync is POSIX-only")
@pytest.mark.parametrize("already_exists", [False, True])
def test_parent_directory_preflight_fsync_failure_never_mutates_the_log(
    tmp_path, monkeypatch, already_exists
):
    path = tmp_path / "events.jsonl"
    if already_exists:
        _write_valid_batch(path, batch_id=4)
        action = _action(batch_id=5)
        before = path.read_bytes()
    else:
        action = _action()
        before = None

    def fail_preflight_directory_fsync(fd):
        raise OSError("injected directory fsync failure")

    monkeypatch.setattr(event_log.os, "fsync", fail_preflight_directory_fsync)
    with pytest.raises(OSError, match="parent directory preflight fsync"):
        ExplorationEventLog(path).append_batch(_batch(_snapshot(), (action,), (_outcome(action),)))

    if before is None:
        assert not path.exists()
    else:
        assert path.read_bytes() == before


@pytest.mark.skipif(
    event_log.os.name != "posix" or not hasattr(event_log.os, "O_NOFOLLOW"),
    reason="race-safe final-component opening requires POSIX O_NOFOLLOW",
)
def test_append_opens_only_the_final_component_relative_to_the_held_parent_fd(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    action = _action()
    real_open = event_log.os.open
    open_calls = []

    def record_open(name, flags, mode=0o777, *, dir_fd=None):
        open_calls.append((name, flags, mode, dir_fd))
        return real_open(name, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(event_log.os, "open", record_open)
    ExplorationEventLog(path).append_batch(_batch(_snapshot(), (action,), (_outcome(action),)))

    final_component_calls = [call for call in open_calls if call[3] is not None]
    assert len(final_component_calls) == 1
    name, flags, mode, parent_fd = final_component_calls[0]
    assert name == path.name
    assert flags & event_log.os.O_WRONLY
    assert flags & event_log.os.O_APPEND
    assert flags & event_log.os.O_CREAT
    assert flags & event_log.os.O_NOFOLLOW
    assert mode == 0o666
    assert isinstance(parent_fd, int)


def test_reconstruction_rejects_duplicate_json_keys_at_any_object_depth(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    snapshot = json.loads(raw_lines[0])
    raw_lines[0] = (
        json.dumps(snapshot, sort_keys=True, separators=(",", ":"))[:-1]
        + ',"ignored":{"key":1,"key":2}}'
    )
    path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        ExplorationEventLog(path).reconstruct_posterior()


def test_schema_v2_round_trips_exact_terminal_accounting_and_flags_in_canonical_order(tmp_path):
    path = tmp_path / "events.jsonl"
    action = _action()
    exact_counts = EvaluationCounts.from_mapping(
        {
            EvaluationPurpose.STARTER_TRUE_QUENCH: 1,
            EvaluationPurpose.DIRECTION_ORACLE: 2,
            EvaluationPurpose.LANDING_TRUE_QUENCH: 1,
        }
    )
    outcome = replace(
        _outcome(action),
        force_evaluations=exact_counts.total,
        evaluation_counts=exact_counts,
        cost_is_exact=True,
        posterior_observed=True,
    )

    ExplorationEventLog(path).append_batch(_batch(_snapshot(), (action,), (outcome,)))

    rows = _read_rows(path)
    assert SCHEMA_VERSION == 2
    assert rows[1]["evaluation_counts"] == {
        purpose.value: exact_counts.count(purpose) for purpose in EvaluationPurpose
    }
    assert list(rows[1]["evaluation_counts"]) == [purpose.value for purpose in EvaluationPurpose]
    assert rows[1]["cost_is_exact"] is True
    assert rows[1]["posterior_observed"] is True
    parsed_outcome = event_log._parse_committed_log(path).outcomes[0]
    assert parsed_outcome.evaluation_counts == exact_counts
    assert parsed_outcome.cost_is_exact is True
    assert parsed_outcome.posterior_observed is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda rows: rows[0].__setitem__("schema_version", 1),
        lambda rows: rows[1]["evaluation_counts"].pop("direction_oracle"),
        lambda rows: rows[1]["evaluation_counts"].__setitem__("unknown_purpose", 0),
        lambda rows: rows[1]["evaluation_counts"].__setitem__("direction_oracle", 99),
        lambda rows: rows[1].__setitem__("cost_is_exact", 1),
        lambda rows: rows[1].__setitem__("posterior_observed", 1),
    ],
)
def test_schema_v2_decoder_rejects_lossy_or_invalid_terminal_accounting(tmp_path, mutate):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)
    rows = _read_rows(path)
    mutate(rows)
    path.write_text(
        "\n".join(event_log._canonical_json_line(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        ExplorationEventLog(path).reconstruct_posterior()


def test_schema_v2_decoder_rejects_noncanonical_evaluation_count_order(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)
    rows = _read_rows(path)
    canonical_counts = rows[1]["evaluation_counts"]
    reordered_counts = dict(reversed(tuple(canonical_counts.items())))
    assert reordered_counts == canonical_counts
    assert tuple(reordered_counts) != tuple(canonical_counts)
    rows[1]["evaluation_counts"] = reordered_counts
    path.write_text(
        "\n".join(event_log._canonical_json_line(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical order"):
        ExplorationEventLog(path).reconstruct_posterior()


def test_schema_v2_reconstruction_recomputes_observation_and_idempotent_retry_preserves_fields(tmp_path):
    path = tmp_path / "events.jsonl"
    first = _action(slot_id=0, starter_id=8, selection_probability=0.8)
    second = _action(slot_id=1, starter_id=3, selection_probability=0.2)
    first_counts = EvaluationCounts.from_mapping({EvaluationPurpose.DIRECTION_ORACLE: 2})
    outcomes = (
        replace(
            _outcome(first),
            force_evaluations=first_counts.total,
            evaluation_counts=first_counts,
            posterior_observed=True,
        ),
        CreditedOutcome(
            action_id=second.action_id,
            starter_id=second.starter_id,
            discovered_against_snapshot=False,
            inserted_into_archive=False,
            within_batch_collision=False,
            force_evaluations=0,
            status=AttemptStatus.INVALID,
            landing_entry_id=None,
            landing_energy=None,
            failure_reason="zero physical evaluations",
            evaluation_counts=EvaluationCounts.zero(),
            cost_is_exact=True,
            posterior_observed=False,
        ),
    )
    log = ExplorationEventLog(path)

    log.append_batch(_batch(_snapshot(), (first, second), outcomes))
    before = path.read_bytes()
    log.append_batch(_batch(_snapshot(), (first, second), outcomes))

    parsed_outcomes = event_log._parse_committed_log(path).outcomes
    assert path.read_bytes() == before
    assert [(item.evaluation_counts, item.cost_is_exact, item.posterior_observed) for item in parsed_outcomes] == [
        (outcome.evaluation_counts, outcome.cost_is_exact, outcome.posterior_observed)
        for outcome in outcomes
    ]
    posterior = log.reconstruct_posterior()
    assert posterior.counts(first.starter_id) == (1, 0)
    assert posterior.counts(second.starter_id) == (0, 0)
    assert posterior.completed_attempts == 1
