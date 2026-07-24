"""Synchronous, deterministic batch dispatch for generic exploration workers.

The controller owns archive and posterior mutation.  Workers receive isolated
starter-state copies and return terminal :class:`AttemptResult` values; their
completion order cannot affect the committed batch order or credit.
"""

from __future__ import annotations

from concurrent.futures import Executor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
import threading
from typing import Callable, Protocol

from ..archive import MinimaArchive, MinimaEntry
from ..state import State
from .actions import AttemptResult, AttemptStatus, CreditedOutcome, PolicySnapshot, StarterAction
from .batch import plan_batch
from .policies import SUPPORTED_POLICIES, build_policy_snapshot
from .posterior import StarterProductivityPosterior


class BatchLog(Protocol):
    """Append one fully finalized batch before the controller commits it."""

    def append_batch(
        self,
        snapshot: PolicySnapshot,
        actions: tuple[StarterAction, ...],
        outcomes: tuple[CreditedOutcome, ...],
    ) -> None: ...


Worker = Callable[[StarterAction, State], AttemptResult]


@dataclass(frozen=True)
class _PendingCommit:
    """One finalized batch whose audit write may have completed indeterminately."""

    snapshot: PolicySnapshot
    actions: tuple[StarterAction, ...]
    outcomes: tuple[CreditedOutcome, ...]
    shadow_archive: MinimaArchive
    shadow_posterior: StarterProductivityPosterior
    next_policy_version: int
    next_archive_version: int
    next_batch_id: int


def _nonnegative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _policy_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("policy_name must be a nonempty string")
    if value == "legacy_ucb":
        raise ValueError("legacy UCB is an external comparator and is not a supported policy")
    if value not in SUPPORTED_POLICIES:
        raise ValueError(f"unsupported policy: {value}")
    return value


class ExplorationController:
    """Commit deterministic synchronous batches against immutable dispatch state.

    Phase 1 always starts from the supplied archive clone and an empty
    productivity posterior.  It intentionally neither resumes an existing log
    nor changes the archive object passed to the constructor.

    Calls to :meth:`run_batch` on one controller are serialized. A shared event
    log therefore requires a single controller/writer in this phase.
    """

    def __init__(
        self,
        archive: MinimaArchive,
        policy_name: str,
        master_seed: int,
        event_log: BatchLog,
    ) -> None:
        if not isinstance(archive, MinimaArchive):
            raise ValueError("archive must be a MinimaArchive")
        self.policy_name = _policy_name(policy_name)
        self.master_seed = _nonnegative_int("master_seed", master_seed)
        if not callable(getattr(event_log, "append_batch", None)):
            raise ValueError("event_log must provide a callable append_batch method")

        self.archive = archive.clone()
        self.event_log = event_log
        self.posterior = StarterProductivityPosterior()
        self.policy_version = 0
        self.archive_version = 0
        self.batch_id = 0
        self._run_lock = threading.Lock()
        self._pending_commit: _PendingCommit | None = None

    @property
    def has_pending_commit(self) -> bool:
        """Whether a finalized batch still needs an exact append retry."""
        with self._run_lock:
            return self._pending_commit is not None

    def run_batch(
        self,
        executor: Executor,
        worker: Worker,
        batch_size: int,
        force_budget: int | None,
    ) -> tuple[CreditedOutcome, ...]:
        """Execute one batch and commit it atomically after the log append.

        A worker exception becomes a zero-cost ``WORKER_ERROR`` result.  A
        worker that knows it has spent a nonzero force budget before failing
        must catch its own exception and return that explicit ``AttemptResult``
        so the exact cost is recorded.
        """
        with self._run_lock:
            if self._pending_commit is not None:
                return self._reconcile_pending_commit()
            if not isinstance(executor, Executor):
                raise ValueError("executor must be a concurrent.futures.Executor")
            if not callable(worker):
                raise ValueError("worker must be callable")

            planning_posterior = self.posterior.clone()
            snapshot = build_policy_snapshot(
                self.policy_name,
                tuple(entry.entry_id for entry in self.archive.entries),
                planning_posterior,
                self.policy_version,
                self.archive_version,
            )
            actions = plan_batch(snapshot, self.batch_id, batch_size, self.master_seed, force_budget)
            dispatch_archive = self.archive.clone()
            dispatch_entries = _entries_by_id(dispatch_archive)

            futures = {}
            result_by_slot: dict[int, AttemptResult] = {}
            for index, action in enumerate(actions):
                try:
                    future = executor.submit(
                        worker,
                        action,
                        deepcopy(dispatch_entries[action.starter_id].state),
                    )
                except Exception as exc:
                    for undispatched_action in actions[index:]:
                        result_by_slot[undispatched_action.slot_id] = _worker_error_result(
                            undispatched_action,
                            exc,
                        )
                    break
                futures[future] = action

            for future in as_completed(futures):
                action = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = _worker_error_result(action, exc)
                if not isinstance(result, AttemptResult):
                    raise ValueError("worker must return an AttemptResult")
                if result.action != action:
                    raise ValueError("worker returned a result for the wrong action")
                result_by_slot[action.slot_id] = result

            ordered_results = tuple(result_by_slot[action.slot_id] for action in actions)
            shadow_archive = self.archive.clone()
            shadow_posterior = self.posterior.clone()
            outcomes = tuple(
                _credit_result(result, dispatch_archive, shadow_archive, shadow_posterior)
                for result in ordered_results
            )
            self._pending_commit = _PendingCommit(
                snapshot=snapshot,
                actions=actions,
                outcomes=outcomes,
                shadow_archive=shadow_archive,
                shadow_posterior=shadow_posterior,
                next_policy_version=self.policy_version + 1,
                next_archive_version=self.archive_version + 1,
                next_batch_id=self.batch_id + 1,
            )
            return self._reconcile_pending_commit()

    def _reconcile_pending_commit(self) -> tuple[CreditedOutcome, ...]:
        """Append and install the exact finalized batch currently pending."""
        pending = self._pending_commit
        if pending is None:
            raise RuntimeError("no pending exploration batch to reconcile")
        self.event_log.append_batch(pending.snapshot, pending.actions, pending.outcomes)
        self.archive = pending.shadow_archive
        self.posterior = pending.shadow_posterior
        self.policy_version = pending.next_policy_version
        self.archive_version = pending.next_archive_version
        self.batch_id = pending.next_batch_id
        self._pending_commit = None
        return pending.outcomes


def _entries_by_id(archive: MinimaArchive) -> dict[int, MinimaEntry]:
    entries = {entry.entry_id: entry for entry in archive.entries}
    if len(entries) != len(archive.entries):
        raise ValueError("archive entry IDs must be unique")
    return entries


def _worker_error_result(action: StarterAction, exc: Exception) -> AttemptResult:
    return AttemptResult(
        action=action,
        landing_state=None,
        landing_energy=None,
        force_evaluations=0,
        status=AttemptStatus.WORKER_ERROR,
        failure_reason=f"{type(exc).__name__}: {exc}",
    )


def _credit_result(
    result: AttemptResult,
    dispatch_archive: MinimaArchive,
    shadow_archive: MinimaArchive,
    shadow_posterior: StarterProductivityPosterior,
) -> CreditedOutcome:
    discovered = False
    inserted = False
    collision = False
    landing_entry_id: int | None = None
    landing_energy: float | None = None

    if result.status is AttemptStatus.COMPLETED:
        assert result.landing_state is not None
        assert result.landing_energy is not None
        discovered = dispatch_archive.find_match(result.landing_state, result.landing_energy) is None
        before = len(shadow_archive.entries)
        landing = shadow_archive.add(
            deepcopy(result.landing_state),
            result.landing_energy,
            parent_id=result.action.starter_id,
        )
        inserted = len(shadow_archive.entries) > before
        collision = discovered and not inserted
        landing_entry_id = landing.entry_id
        landing_energy = result.landing_energy

    shadow_posterior.update(result.action.starter_id, discovered=discovered)
    return CreditedOutcome(
        action_id=result.action.action_id,
        starter_id=result.action.starter_id,
        discovered_against_snapshot=discovered,
        inserted_into_archive=inserted,
        within_batch_collision=collision,
        force_evaluations=result.force_evaluations,
        status=result.status,
        landing_entry_id=landing_entry_id,
        landing_energy=landing_energy,
        failure_reason=result.failure_reason,
    )


__all__ = ["BatchLog", "ExplorationController", "Worker"]
