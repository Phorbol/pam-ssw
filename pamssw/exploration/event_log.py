"""Durable, append-only facts for committed exploration batches.

The log deliberately records policy and attempt metadata, rather than archive
geometry.  A committed log can rebuild starter productivity counts; it cannot
rebuild an archive or a trajectory.
"""

from __future__ import annotations

import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from ..accounting import EvaluationCounts, EvaluationPurpose
from .actions import (
    AttemptStatus,
    CreditedOutcome,
    PolicySnapshot,
    StarterAction,
    should_observe_posterior,
)
from .posterior import StarterProductivityPosterior


SCHEMA_VERSION = 2

_POLICY_SNAPSHOT_FIELDS = frozenset(
    {
        "archive_version",
        "batch_id",
        "eligible_starter_ids",
        "policy_name",
        "policy_version",
        "probabilities",
        "record_type",
        "schema_version",
        "support_complete",
    }
)
_ATTEMPT_FIELDS = frozenset(
    {
        "action_id",
        "archive_version",
        "batch_id",
        "discovered_against_snapshot",
        "evaluation_counts",
        "failure_reason",
        "force_budget",
        "force_evaluations",
        "inserted_into_archive",
        "landing_energy",
        "landing_entry_id",
        "policy_name",
        "policy_version",
        "random_seed",
        "record_type",
        "schema_version",
        "selection_probability",
        "slot_id",
        "starter_id",
        "status",
        "within_batch_collision",
        "cost_is_exact",
        "posterior_observed",
    }
)
_BATCH_COMMIT_FIELDS = frozenset(
    {"action_ids", "batch_id", "record_type", "schema_version"}
)


class _EventLogError(ValueError):
    """Raised when an event log cannot be trusted for posterior replay."""


def _nonempty_string(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _EventLogError(f"{name} must be a nonempty string")
    return value


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _EventLogError(f"{name} must be a non-negative integer")
    return value


def _positive_int(name: str, value: object) -> int:
    value = _nonnegative_int(name, value)
    if value == 0:
        raise _EventLogError(f"{name} must be a positive integer")
    return value


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _EventLogError(f"{name} must be a finite JSON number")
    number = float(value)
    if not math.isfinite(number):
        raise _EventLogError(f"{name} must be a finite JSON number")
    return number


def _boolean(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise _EventLogError(f"{name} must be a boolean")
    return value


def _optional_positive_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    return _positive_int(name, value)


def _optional_nonnegative_int(name: str, value: object) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(name, value)


def _optional_finite_number(name: str, value: object) -> float | None:
    if value is None:
        return None
    return _finite_number(name, value)


def _optional_failure_reason(value: object) -> str | None:
    if value is None:
        return None
    return _nonempty_string("failure_reason", value)


def _evaluation_counts_row(evaluation_counts: EvaluationCounts) -> dict[str, int]:
    if not isinstance(evaluation_counts, EvaluationCounts):
        raise _EventLogError("evaluation_counts must be an EvaluationCounts")
    return {
        purpose.value: evaluation_counts.count(purpose) for purpose in EvaluationPurpose
    }


def _parse_evaluation_counts(value: object) -> EvaluationCounts:
    if not isinstance(value, dict):
        raise _EventLogError("evaluation_counts must be a JSON object")
    expected_purposes = tuple(purpose.value for purpose in EvaluationPurpose)
    if set(value) != set(expected_purposes):
        missing = sorted(set(expected_purposes) - set(value))
        unknown = sorted(set(value) - set(expected_purposes))
        raise _EventLogError(
            f"evaluation_counts must include every purpose exactly once; missing={missing!r}, unknown={unknown!r}"
        )
    if tuple(value) != expected_purposes:
        raise _EventLogError("evaluation_counts keys must use canonical order")
    return EvaluationCounts(
        tuple(_nonnegative_int(f"evaluation_counts.{purpose}", value[purpose]) for purpose in expected_purposes)
    )


def _require_fields(row: dict[str, object], expected: frozenset[str], line_number: int) -> None:
    if set(row) != expected:
        missing = sorted(expected - set(row))
        unknown = sorted(set(row) - expected)
        raise _EventLogError(
            f"line {line_number}: invalid record fields; missing={missing!r}, unknown={unknown!r}"
        )


def _require_schema(row: dict[str, object], line_number: int) -> None:
    if _nonnegative_int("schema_version", row["schema_version"]) != SCHEMA_VERSION:
        raise _EventLogError(f"line {line_number}: unsupported schema_version")


def _snapshot_row(snapshot: PolicySnapshot, batch_id: int) -> dict[str, object]:
    return {
        "record_type": "policy_snapshot",
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "policy_name": snapshot.policy_name,
        "policy_version": snapshot.version,
        "archive_version": snapshot.archive_version,
        "eligible_starter_ids": list(snapshot.eligible_starter_ids),
        "probabilities": list(snapshot.probabilities),
        "support_complete": snapshot.support_complete,
    }


def _attempt_row(action: StarterAction, outcome: CreditedOutcome) -> dict[str, object]:
    return {
        "record_type": "attempt",
        "schema_version": SCHEMA_VERSION,
        "batch_id": action.batch_id,
        "action_id": action.action_id,
        "slot_id": action.slot_id,
        "policy_name": action.policy_name,
        "policy_version": action.policy_version,
        "archive_version": action.archive_version,
        "starter_id": action.starter_id,
        "selection_probability": action.selection_probability,
        "random_seed": action.random_seed,
        "force_budget": action.force_budget,
        "status": outcome.status.value,
        "failure_reason": outcome.failure_reason,
        "force_evaluations": outcome.force_evaluations,
        "evaluation_counts": _evaluation_counts_row(outcome.evaluation_counts),
        "cost_is_exact": outcome.cost_is_exact,
        "posterior_observed": outcome.posterior_observed,
        "discovered_against_snapshot": outcome.discovered_against_snapshot,
        "inserted_into_archive": outcome.inserted_into_archive,
        "within_batch_collision": outcome.within_batch_collision,
        "landing_entry_id": outcome.landing_entry_id,
        "landing_energy": outcome.landing_energy,
    }


def _commit_row(batch_id: int, action_ids: tuple[str, ...]) -> dict[str, object]:
    return {
        "record_type": "batch_commit",
        "schema_version": SCHEMA_VERSION,
        "batch_id": batch_id,
        "action_ids": list(action_ids),
    }


def _validate_action_against_snapshot(action: StarterAction, snapshot: PolicySnapshot, batch_id: int) -> None:
    if action.batch_id != batch_id:
        raise _EventLogError("all actions must share one batch_id")
    if action.policy_name != snapshot.policy_name:
        raise _EventLogError("action policy_name must match snapshot")
    if action.policy_version != snapshot.version:
        raise _EventLogError("action policy_version must match snapshot")
    if action.archive_version != snapshot.archive_version:
        raise _EventLogError("action archive_version must match snapshot")
    if action.starter_id not in snapshot.eligible_starter_ids:
        raise _EventLogError("action starter_id must be eligible in snapshot")
    if action.selection_probability != snapshot.probability_for(action.starter_id):
        raise _EventLogError("action selection_probability must match snapshot")


def _validate_append_inputs(
    snapshot: object,
    actions: object,
    outcomes: object,
) -> tuple[PolicySnapshot, tuple[StarterAction, ...], tuple[CreditedOutcome, ...]]:
    if not isinstance(snapshot, PolicySnapshot):
        raise _EventLogError("snapshot must be a PolicySnapshot")
    if not isinstance(actions, tuple):
        raise _EventLogError("actions must be a tuple")
    if not isinstance(outcomes, tuple):
        raise _EventLogError("outcomes must be a tuple")
    if not actions:
        raise _EventLogError("actions and outcomes must be nonempty")
    if len(actions) != len(outcomes):
        raise _EventLogError("actions and outcomes must have equal lengths")
    if not all(isinstance(action, StarterAction) for action in actions):
        raise _EventLogError("actions must contain StarterAction values")
    if not all(isinstance(outcome, CreditedOutcome) for outcome in outcomes):
        raise _EventLogError("outcomes must contain CreditedOutcome values")

    typed_actions = actions
    typed_outcomes = outcomes
    batch_id = typed_actions[0].batch_id
    action_ids: set[str] = set()
    slot_ids: set[int] = set()
    for action, outcome in zip(typed_actions, typed_outcomes):
        _validate_action_against_snapshot(action, snapshot, batch_id)
        if action.action_id != outcome.action_id:
            raise _EventLogError("action_id must agree with outcome")
        if action.starter_id != outcome.starter_id:
            raise _EventLogError("starter_id must agree with outcome")
        if action.force_budget is not None and outcome.force_evaluations > action.force_budget:
            raise _EventLogError("force_evaluations cannot exceed action force_budget")
        if outcome.posterior_observed != should_observe_posterior(
            outcome.status, outcome.evaluation_counts
        ):
            raise _EventLogError("posterior_observed must match terminal observation predicate")
        if action.action_id in action_ids:
            raise _EventLogError("action IDs must be unique")
        if action.slot_id in slot_ids:
            raise _EventLogError("slot IDs must be unique")
        action_ids.add(action.action_id)
        slot_ids.add(action.slot_id)
    return snapshot, typed_actions, typed_outcomes


def _parse_snapshot(row: dict[str, object], line_number: int) -> tuple[PolicySnapshot, int]:
    _require_fields(row, _POLICY_SNAPSHOT_FIELDS, line_number)
    _require_schema(row, line_number)
    if row["record_type"] != "policy_snapshot":
        raise _EventLogError(f"line {line_number}: expected policy_snapshot")
    batch_id = _nonnegative_int("batch_id", row["batch_id"])
    eligible_starter_ids = row["eligible_starter_ids"]
    probabilities = row["probabilities"]
    if not isinstance(eligible_starter_ids, list):
        raise _EventLogError("eligible_starter_ids must be a JSON array")
    if not isinstance(probabilities, list):
        raise _EventLogError("probabilities must be a JSON array")
    parsed_starter_ids = tuple(
        _nonnegative_int("eligible_starter_ids", starter_id) for starter_id in eligible_starter_ids
    )
    parsed_probabilities = tuple(
        _finite_number("probabilities", probability) for probability in probabilities
    )
    snapshot = PolicySnapshot(
        version=_nonnegative_int("policy_version", row["policy_version"]),
        archive_version=_nonnegative_int("archive_version", row["archive_version"]),
        policy_name=_nonempty_string("policy_name", row["policy_name"]),
        eligible_starter_ids=parsed_starter_ids,
        probabilities=parsed_probabilities,
        support_complete=_boolean("support_complete", row["support_complete"]),
    )
    return snapshot, batch_id


def _parse_attempt(
    row: dict[str, object],
    line_number: int,
    snapshot: PolicySnapshot,
    batch_id: int,
) -> tuple[StarterAction, CreditedOutcome]:
    _require_fields(row, _ATTEMPT_FIELDS, line_number)
    _require_schema(row, line_number)
    if row["record_type"] != "attempt":
        raise _EventLogError(f"line {line_number}: expected attempt")
    if _nonnegative_int("batch_id", row["batch_id"]) != batch_id:
        raise _EventLogError(f"line {line_number}: attempt batch_id does not match active snapshot")
    action = StarterAction(
        action_id=_nonempty_string("action_id", row["action_id"]),
        batch_id=batch_id,
        slot_id=_nonnegative_int("slot_id", row["slot_id"]),
        policy_name=_nonempty_string("policy_name", row["policy_name"]),
        policy_version=_nonnegative_int("policy_version", row["policy_version"]),
        archive_version=_nonnegative_int("archive_version", row["archive_version"]),
        starter_id=_nonnegative_int("starter_id", row["starter_id"]),
        selection_probability=_finite_number("selection_probability", row["selection_probability"]),
        random_seed=_nonnegative_int("random_seed", row["random_seed"]),
        force_budget=_optional_positive_int("force_budget", row["force_budget"]),
    )
    _validate_action_against_snapshot(action, snapshot, batch_id)
    status_value = row["status"]
    if not isinstance(status_value, str):
        raise _EventLogError("status must be an AttemptStatus value")
    try:
        status = AttemptStatus(status_value)
    except ValueError as exc:
        raise _EventLogError("status must be an AttemptStatus value") from exc
    outcome = CreditedOutcome(
        action_id=action.action_id,
        starter_id=_nonnegative_int("starter_id", row["starter_id"]),
        discovered_against_snapshot=_boolean(
            "discovered_against_snapshot", row["discovered_against_snapshot"]
        ),
        inserted_into_archive=_boolean("inserted_into_archive", row["inserted_into_archive"]),
        within_batch_collision=_boolean("within_batch_collision", row["within_batch_collision"]),
        force_evaluations=_nonnegative_int("force_evaluations", row["force_evaluations"]),
        evaluation_counts=_parse_evaluation_counts(row["evaluation_counts"]),
        cost_is_exact=_boolean("cost_is_exact", row["cost_is_exact"]),
        posterior_observed=_boolean("posterior_observed", row["posterior_observed"]),
        status=status,
        landing_entry_id=_optional_nonnegative_int("landing_entry_id", row["landing_entry_id"]),
        landing_energy=_optional_finite_number("landing_energy", row["landing_energy"]),
        failure_reason=_optional_failure_reason(row["failure_reason"]),
    )
    if action.force_budget is not None and outcome.force_evaluations > action.force_budget:
        raise _EventLogError("force_evaluations cannot exceed action force_budget")
    if outcome.posterior_observed != should_observe_posterior(
        outcome.status, outcome.evaluation_counts
    ):
        raise _EventLogError("posterior_observed must match terminal observation predicate")
    return action, outcome


def _parse_commit(row: dict[str, object], line_number: int, batch_id: int) -> tuple[str, ...]:
    _require_fields(row, _BATCH_COMMIT_FIELDS, line_number)
    _require_schema(row, line_number)
    if row["record_type"] != "batch_commit":
        raise _EventLogError(f"line {line_number}: expected batch_commit")
    if _nonnegative_int("batch_id", row["batch_id"]) != batch_id:
        raise _EventLogError(f"line {line_number}: batch_commit batch_id does not match active snapshot")
    raw_action_ids = row["action_ids"]
    if not isinstance(raw_action_ids, list):
        raise _EventLogError("action_ids must be a JSON array")
    action_ids = tuple(_nonempty_string("action_ids", action_id) for action_id in raw_action_ids)
    if len(set(action_ids)) != len(action_ids):
        raise _EventLogError("batch_commit action_ids must be unique")
    return action_ids


def _reject_json_constant(token: str) -> object:
    raise _EventLogError(f"invalid JSON constant: {token}")


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    row: dict[str, object] = {}
    for key, value in pairs:
        if key in row:
            raise _EventLogError(f"duplicate JSON object key: {key!r}")
        row[key] = value
    return row


def _canonical_json_line(row: dict[str, object]) -> str:
    """Keep top-level rows sorted while preserving enum order inside count objects."""
    sorted_row = {key: row[key] for key in sorted(row)}
    return json.dumps(sorted_row, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class _CommittedBatch:
    """Canonical policy, action, and outcome facts for one committed batch."""

    snapshot: PolicySnapshot
    actions: tuple[StarterAction, ...]
    outcomes: tuple[CreditedOutcome, ...]


@dataclass(frozen=True)
class _ParsedEventLog:
    """Committed attempt facts and identities recovered from one valid file."""

    outcomes: tuple[CreditedOutcome, ...]
    batch_ids: frozenset[int]
    action_ids: frozenset[str]
    batches: dict[int, _CommittedBatch]


def _fsync_held_parent_directory(directory_fd: int, phase: str) -> None:
    """Fsync an already opened parent directory, preserving a clear phase error."""
    try:
        os.fsync(directory_fd)
    except OSError as exc:
        raise OSError(f"parent directory {phase} fsync is unsupported or failed") from exc


def _open_preflight_fsynced_parent_directory(path: Path) -> int:
    """Open and capability-check an existing POSIX parent directory without mutation."""
    parent = path.parent
    if not parent.exists():
        raise FileNotFoundError(f"event-log parent directory does not exist: {parent}")
    if not parent.is_dir():
        raise ValueError(f"event-log parent path must be a directory: {parent}")
    if os.name != "posix":
        raise OSError("parent directory fsync is supported only on POSIX platforms")
    try:
        directory_fd = os.open(
            os.fspath(parent), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
    except OSError as exc:
        raise OSError("could not open parent directory for fsync") from exc
    try:
        _fsync_held_parent_directory(directory_fd, "preflight")
    except BaseException:
        try:
            os.close(directory_fd)
        except OSError:
            pass
        raise
    return directory_fd


def _fsync_existing_file(path: Path) -> None:
    """Fsync a previously written idempotent retry without reopening its directory."""
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _reject_final_path_symlink(path: Path) -> None:
    """Reject a symlink at the event log's final path component only."""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return
    except NotADirectoryError:
        return
    except OSError as exc:
        raise OSError("could not inspect final event-log path") from exc
    if stat.S_ISLNK(mode):
        raise ValueError("final event-log path must not be a symlink")


def _require_no_follow_append_support() -> int:
    """Return the POSIX no-follow flag needed for race-safe final-path writes."""
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW"):
        raise OSError("race-safe event-log append requires POSIX os.O_NOFOLLOW support")
    return os.O_NOFOLLOW


def _open_append_text_file(path: Path, parent_fd: int):
    """Open only ``path.name`` beneath ``parent_fd`` without following a final symlink."""
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | _require_no_follow_append_support()
    try:
        file_fd = os.open(path.name, flags, 0o666, dir_fd=parent_fd)
    except OSError as exc:
        raise OSError("could not open final event-log path without following symlinks") from exc
    try:
        return os.fdopen(file_fd, "a", encoding="utf-8")
    except BaseException:
        os.close(file_fd)
        raise


def _parse_committed_log(path: Path) -> _ParsedEventLog:
    """Strictly validate a complete log before exposing any committed facts."""
    _reject_final_path_symlink(path)
    if not path.exists():
        return _ParsedEventLog((), frozenset(), frozenset(), {})

    active_snapshot: PolicySnapshot | None = None
    active_batch_id: int | None = None
    active_action_ids: list[str] = []
    active_actions: list[StarterAction] = []
    active_slot_ids: set[int] = set()
    active_outcomes: list[CreditedOutcome] = []
    seen_batch_ids: set[int] = set()
    seen_action_ids: set[str] = set()
    committed_outcomes: list[CreditedOutcome] = []
    committed_batches: dict[int, _CommittedBatch] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(
                    line,
                    parse_constant=_reject_json_constant,
                    object_pairs_hook=_reject_duplicate_json_keys,
                )
            except json.JSONDecodeError as exc:
                raise _EventLogError(f"line {line_number}: invalid JSON") from exc
            except _EventLogError as exc:
                raise _EventLogError(f"line {line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise _EventLogError(f"line {line_number}: event-log row must be an object")
            record_type = row.get("record_type")
            if record_type == "policy_snapshot":
                if active_snapshot is not None:
                    raise _EventLogError(f"line {line_number}: nested policy_snapshot before batch_commit")
                active_snapshot, active_batch_id = _parse_snapshot(row, line_number)
                if active_batch_id in seen_batch_ids:
                    raise _EventLogError(f"line {line_number}: duplicate batch_id")
                seen_batch_ids.add(active_batch_id)
                active_action_ids = []
                active_actions = []
                active_slot_ids = set()
                active_outcomes = []
                continue
            if record_type == "attempt":
                if active_snapshot is None or active_batch_id is None:
                    raise _EventLogError(f"line {line_number}: attempt outside policy_snapshot")
                action, outcome = _parse_attempt(row, line_number, active_snapshot, active_batch_id)
                if action.action_id in seen_action_ids:
                    raise _EventLogError(f"line {line_number}: duplicate action_id")
                if action.slot_id in active_slot_ids:
                    raise _EventLogError(f"line {line_number}: duplicate slot_id")
                seen_action_ids.add(action.action_id)
                active_action_ids.append(action.action_id)
                active_actions.append(action)
                active_slot_ids.add(action.slot_id)
                active_outcomes.append(outcome)
                continue
            if record_type == "batch_commit":
                if active_snapshot is None or active_batch_id is None:
                    raise _EventLogError(f"line {line_number}: batch_commit outside policy_snapshot")
                committed_action_ids = _parse_commit(row, line_number, active_batch_id)
                if tuple(active_action_ids) != committed_action_ids:
                    raise _EventLogError(
                        f"line {line_number}: batch_commit action_ids do not match attempt order"
                    )
                if not active_outcomes:
                    raise _EventLogError(f"line {line_number}: batch_commit requires attempts")
                committed_outcomes.extend(active_outcomes)
                committed_batches[active_batch_id] = _CommittedBatch(
                    snapshot=active_snapshot,
                    actions=tuple(active_actions),
                    outcomes=tuple(active_outcomes),
                )
                active_snapshot = None
                active_batch_id = None
                active_action_ids = []
                active_actions = []
                active_slot_ids = set()
                active_outcomes = []
                continue
            if not isinstance(record_type, str):
                raise _EventLogError(f"line {line_number}: record_type must be a string")
            raise _EventLogError(f"line {line_number}: unknown record_type {record_type!r}")

    if active_snapshot is not None:
        raise _EventLogError("incomplete final batch without batch_commit")
    return _ParsedEventLog(
        outcomes=tuple(committed_outcomes),
        batch_ids=frozenset(seen_batch_ids),
        action_ids=frozenset(seen_action_ids),
        batches=committed_batches,
    )


class ExplorationEventLog:
    """Append and replay committed exploration facts from a JSONL file.

    The parent directory must already exist and the final path component cannot
    be a symlink. On POSIX, each append holds a parent-directory FD after a
    capability fsync preflight; a preflight failure occurs before file mutation.
    It opens the final component with ``O_NOFOLLOW`` and fsyncs the held parent
    FD after the file. Appends preflight the entire log for sequential
    idempotency, so they are O(total log size), a deliberate phase-1 scaling
    boundary. This append-only log has no locking or checksum and does not
    reconstruct archive geometry.
    """

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def append_batch(
        self,
        snapshot: PolicySnapshot,
        actions: tuple[StarterAction, ...],
        outcomes: tuple[CreditedOutcome, ...],
    ) -> None:
        """Durably append one snapshot, its attempts, and its commit marker."""
        snapshot, actions, outcomes = _validate_append_inputs(snapshot, actions, outcomes)
        batch_id = actions[0].batch_id
        action_ids = tuple(action.action_id for action in actions)
        candidate = _CommittedBatch(snapshot, actions, outcomes)
        _reject_final_path_symlink(self.path)
        _require_no_follow_append_support()
        directory_fd = _open_preflight_fsynced_parent_directory(self.path)
        try:
            existing = _parse_committed_log(self.path)
            existing_batch = existing.batches.get(batch_id)
            if existing_batch is not None:
                if existing_batch == candidate:
                    _fsync_existing_file(self.path)
                    _fsync_held_parent_directory(directory_fd, "post-retry")
                    return
                raise _EventLogError("batch_id already exists in event log")
            if set(action_ids) & existing.action_ids:
                raise _EventLogError("action_id already exists in event log")
            rows = [_snapshot_row(snapshot, batch_id)]
            rows.extend(_attempt_row(action, outcome) for action, outcome in zip(actions, outcomes))
            rows.append(_commit_row(batch_id, action_ids))
            payload = "".join(
                _canonical_json_line(row) + "\n"
                for row in rows
            )

            with _open_append_text_file(self.path, directory_fd) as handle:
                written = handle.write(payload)
                if written != len(payload):
                    raise OSError("short event-log write")
                handle.flush()
                os.fsync(handle.fileno())
            _fsync_held_parent_directory(directory_fd, "post-write")
        finally:
            os.close(directory_fd)

    def reconstruct_posterior(self) -> StarterProductivityPosterior:
        """Return a posterior rebuilt only from fully committed attempt facts.

        A malformed, interleaved, or incomplete log is rejected rather than
        returning counts from any valid prefix.
        """
        parsed = _parse_committed_log(self.path)
        posterior = StarterProductivityPosterior()
        for outcome in parsed.outcomes:
            posterior_observed = should_observe_posterior(
                outcome.status, outcome.evaluation_counts
            )
            if outcome.posterior_observed != posterior_observed:
                raise _EventLogError("posterior_observed must match terminal observation predicate")
            if posterior_observed:
                posterior.update(outcome.starter_id, outcome.discovered_against_snapshot)
        return posterior


__all__ = ["ExplorationEventLog", "SCHEMA_VERSION"]
