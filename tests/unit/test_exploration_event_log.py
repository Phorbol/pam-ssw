import json
from dataclasses import replace

import pytest

import pamssw.exploration.event_log as event_log
from pamssw.exploration.actions import (
    AttemptStatus,
    CreditedOutcome,
    PolicySnapshot,
    StarterAction,
)
from pamssw.exploration.event_log import ExplorationEventLog, SCHEMA_VERSION


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
    )


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
    ExplorationEventLog(path).append_batch(snapshot, actions, outcomes)
    return actions, outcomes


def test_append_writes_one_sorted_jsonl_payload_in_policy_attempt_commit_order(tmp_path):
    path = tmp_path / "nested" / "events.jsonl"
    snapshot = _snapshot()
    actions = (
        _action(slot_id=0, starter_id=8, selection_probability=0.8),
        _action(slot_id=1, starter_id=3, selection_probability=0.2),
    )
    outcomes = (
        _outcome(actions[0]),
        _outcome(actions[1], status=AttemptStatus.WORKER_ERROR, failure_reason="exact remote OOM"),
    )

    ExplorationEventLog(path).append_batch(snapshot, actions, outcomes)

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
    assert all(line == json.dumps(row, sort_keys=True, separators=(",", ":")) for line, row in zip(raw_lines, rows))


def test_reconstructs_committed_attempt_facts_once_per_action(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)

    posterior = ExplorationEventLog(path).reconstruct_posterior()

    assert posterior.counts(8) == (1, 0)
    assert posterior.counts(3) == (0, 1)
    assert posterior.completed_attempts == 2


def test_reconstructs_multiple_committed_batches_in_append_order(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path, batch_id=4)
    _write_valid_batch(path, batch_id=5)

    posterior = ExplorationEventLog(path).reconstruct_posterior()

    assert posterior.counts(8) == (2, 0)
    assert posterior.counts(3) == (0, 2)
    assert posterior.completed_attempts == 4


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
        ExplorationEventLog(path).append_batch(snapshot, actions, outcomes)

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
            ExplorationEventLog(path).append_batch(_snapshot(), actions, outcomes)
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
        "\n".join(row if isinstance(row, str) else json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows)
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
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="action_ids"):
        ExplorationEventLog(path).reconstruct_posterior()


def test_reconstruction_fails_closed_for_an_incomplete_final_batch_without_leaking_attempts(tmp_path):
    path = tmp_path / "events.jsonl"
    _write_valid_batch(path)
    rows = _read_rows(path)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows[:-1]) + "\n",
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
        "\n".join(json.dumps(row, sort_keys=True, separators=(",", ":")) for row in rows) + "\n",
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
            _snapshot(), (replacement_action,), (_outcome(replacement_action),)
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
            _snapshot(), (duplicate_action,), (_outcome(duplicate_action),)
        )

    assert path.read_bytes() == before


def test_append_treats_reappending_an_identical_batch_as_a_durable_idempotent_retry(tmp_path):
    path = tmp_path / "events.jsonl"
    actions, outcomes = _write_valid_batch(path, batch_id=4)
    before = path.read_bytes()
    before_line_count = len(before.splitlines())

    ExplorationEventLog(path).append_batch(_snapshot(), actions, outcomes)

    assert path.read_bytes() == before
    assert len(path.read_bytes().splitlines()) == before_line_count
    assert ExplorationEventLog(path).reconstruct_posterior().completed_attempts == 2


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
        + "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in duplicate_batch_rows),
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
        + "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in duplicate_action_rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate action_id"):
        ExplorationEventLog(path).reconstruct_posterior()


@pytest.mark.skipif(event_log.os.name != "posix", reason="parent-directory fsync is POSIX-only")
def test_append_fsyncs_file_before_the_parent_directory_with_a_closed_directory_fd(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    directory_fd = 991
    fsync_fds = []
    directory_opens = []
    closed_fds = []

    def open_directory(directory, flags):
        directory_opens.append((directory, flags))
        return directory_fd

    monkeypatch.setattr(event_log.os, "open", open_directory)
    monkeypatch.setattr(event_log.os, "close", closed_fds.append)
    monkeypatch.setattr(event_log.os, "fsync", fsync_fds.append)

    action = _action()
    ExplorationEventLog(path).append_batch(_snapshot(), (action,), (_outcome(action),))

    expected_flags = event_log.os.O_RDONLY | getattr(event_log.os, "O_DIRECTORY", 0)
    assert len(fsync_fds) == 2
    assert fsync_fds[0] != directory_fd
    assert fsync_fds[1] == directory_fd
    assert directory_opens == [(str(path.parent), expected_flags)]
    assert closed_fds == [directory_fd]


@pytest.mark.skipif(event_log.os.name != "posix", reason="parent-directory fsync is POSIX-only")
def test_exact_retry_after_a_file_fsync_error_does_not_append_or_double_count(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    action = _action()
    log = ExplorationEventLog(path)
    directory_fd = 992
    fsync_fds = []

    monkeypatch.setattr(event_log.os, "open", lambda directory, flags: directory_fd)
    monkeypatch.setattr(event_log.os, "close", lambda fd: None)

    def fail_first_fsync(fd):
        fsync_fds.append(fd)
        if len(fsync_fds) == 1:
            raise OSError("injected file fsync failure")

    monkeypatch.setattr(event_log.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError, match="injected file fsync failure"):
        log.append_batch(_snapshot(), (action,), (_outcome(action),))

    bytes_after_failed_fsync = path.read_bytes()
    assert len(bytes_after_failed_fsync.splitlines()) == 3

    monkeypatch.setattr(event_log.os, "fsync", fsync_fds.append)
    log.append_batch(_snapshot(), (action,), (_outcome(action),))

    assert path.read_bytes() == bytes_after_failed_fsync
    posterior = log.reconstruct_posterior()
    assert posterior.counts(action.starter_id) == (1, 0)
    assert posterior.completed_attempts == 1


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
