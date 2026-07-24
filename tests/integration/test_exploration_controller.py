from __future__ import annotations

import json
import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from pamssw.archive import MinimaArchive
from pamssw.exploration.actions import AttemptResult, AttemptStatus, StarterAction
from pamssw.exploration.controller import ExplorationController
from pamssw.exploration.event_log import ExplorationEventLog
from pamssw.state import State


def _state(x: float, *, label: str = "") -> State:
    return State(
        numbers=np.array([1]),
        positions=np.array([[x, 0.0, 0.0]]),
        metadata={"label": label},
    )


def _archive() -> MinimaArchive:
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.05)
    archive.add(_state(-1.0, label="left"), -1.0, parent_id=None)
    archive.add(_state(1.0, label="right"), -0.9, parent_id=None)
    return archive


def _completed(action: StarterAction, x: float, energy: float) -> AttemptResult:
    return AttemptResult(
        action=action,
        landing_state=_state(x),
        landing_energy=energy,
        force_evaluations=3,
        status=AttemptStatus.COMPLETED,
        failure_reason=None,
    )


def _archive_fingerprint(archive: MinimaArchive) -> tuple[object, ...]:
    def state_fingerprint(state: State) -> tuple[object, ...]:
        return (
            tuple(state.numbers.tolist()),
            tuple(tuple(row) for row in state.positions.tolist()),
            None if state.cell is None else tuple(tuple(row) for row in state.cell.tolist()),
            state.pbc,
            tuple(state.fixed_mask.tolist()),
            tuple(sorted(state.metadata.items())),
        )

    entries = tuple(
        (
            entry.entry_id,
            state_fingerprint(entry.state),
            entry.energy,
            entry.parent_id,
            entry.visits,
            None if entry.descriptor is None else tuple(entry.descriptor.tolist()),
            entry.node_trials,
            entry.node_successes,
            entry.frontier_value,
            entry.duplicate_hits,
            entry.node_duplicate_failures,
            entry.frontier_score,
            entry.is_frontier,
            entry.is_dead,
        )
        for entry in archive.entries
    )
    prototypes = tuple(
        (
            tuple(prototype.descriptor.tolist()),
            prototype.representative_entry_id,
            prototype.weight,
        )
        for prototype in archive.prototypes
    )
    return archive.energy_tol, archive.rmsd_tol, archive.max_prototypes, entries, prototypes


def _controller_fingerprint(controller: ExplorationController) -> tuple[object, ...]:
    return (
        _archive_fingerprint(controller.archive),
        tuple(
            (starter_id, controller.posterior.counts(starter_id))
            for starter_id in range(len(controller.archive.entries))
        ),
        controller.posterior.completed_attempts,
        controller.policy_version,
        controller.archive_version,
        controller.batch_id,
    )


def test_parallel_controller_commits_and_logs_in_slot_order_despite_completion_order(tmp_path: Path) -> None:
    finished_slots: list[int] = []
    finish_lock = threading.Lock()
    slot_one_finished = threading.Event()
    slot_two_finished = threading.Event()

    def worker(action: StarterAction, starter_state: State) -> AttemptResult:
        if action.slot_id == 0:
            assert slot_one_finished.wait(timeout=2)
            assert slot_two_finished.wait(timeout=2)
        elif action.slot_id == 1:
            assert slot_two_finished.wait(timeout=2)
            slot_one_finished.set()
        else:
            slot_two_finished.set()
        with finish_lock:
            finished_slots.append(action.slot_id)
        return _completed(action, 3.0 + action.slot_id, -2.0 - action.slot_id)

    event_path = tmp_path / "events.jsonl"
    controller = ExplorationController(_archive(), "uniform", 4, ExplorationEventLog(event_path))

    with ThreadPoolExecutor(max_workers=3) as executor:
        outcomes = controller.run_batch(executor, worker, batch_size=3, force_budget=10)

    assert finished_slots == [2, 1, 0]
    assert [outcome.action_id for outcome in outcomes] == [
        "batch-00000000-slot-0000",
        "batch-00000000-slot-0001",
        "batch-00000000-slot-0002",
    ]
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert [row["record_type"] for row in rows] == [
        "policy_snapshot",
        "attempt",
        "attempt",
        "attempt",
        "batch_commit",
    ]
    assert [row["slot_id"] for row in rows[1:-1]] == [0, 1, 2]
    assert rows[-1]["action_ids"] == [outcome.action_id for outcome in outcomes]
    assert controller.posterior.completed_attempts == 3
    assert (controller.policy_version, controller.archive_version, controller.batch_id) == (1, 1, 1)


def test_workers_receive_isolated_starter_state_copies_and_cannot_mutate_archives(tmp_path: Path) -> None:
    input_archive = _archive()
    input_before = _archive_fingerprint(input_archive)
    controller = ExplorationController(input_archive, "uniform", 9, ExplorationEventLog(tmp_path / "events.jsonl"))
    seen: list[tuple[int, State, float, str]] = []
    seen_lock = threading.Lock()

    def worker(action: StarterAction, starter_state: State) -> AttemptResult:
        with seen_lock:
            seen.append(
                (
                    action.slot_id,
                    starter_state,
                    float(starter_state.positions[0, 0]),
                    str(starter_state.metadata["label"]),
                )
            )
        starter_state.positions[0, 0] = 99.0 + action.slot_id
        starter_state.metadata["label"] = "worker-mutated"
        return _completed(action, 3.0 + action.slot_id, -2.0 - action.slot_id)

    with ThreadPoolExecutor(max_workers=3) as executor:
        controller.run_batch(executor, worker, batch_size=3, force_budget=None)

    assert [item[2:] for item in sorted(seen)] == [(-1.0, "left")] * 3
    assert len({id(item[1]) for item in seen}) == 3
    assert controller.archive.entries[0].state.positions[0, 0] == pytest.approx(-1.0)
    assert controller.archive.entries[0].state.metadata["label"] == "left"
    assert _archive_fingerprint(input_archive) == input_before


def test_same_novel_landing_is_credited_against_dispatch_snapshot_and_inserted_once(tmp_path: Path) -> None:
    def worker(action: StarterAction, starter_state: State) -> AttemptResult:
        return _completed(action, 4.0, -3.0)

    controller = ExplorationController(_archive(), "uniform", 8, ExplorationEventLog(tmp_path / "events.jsonl"))
    before = len(controller.archive.entries)

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = controller.run_batch(executor, worker, batch_size=2, force_budget=None)

    assert [outcome.discovered_against_snapshot for outcome in outcomes] == [True, True]
    assert [outcome.inserted_into_archive for outcome in outcomes] == [True, False]
    assert [outcome.within_batch_collision for outcome in outcomes] == [False, True]
    assert len(controller.archive.entries) == before + 1
    assert controller.posterior.completed_attempts == 2
    assert sum(controller.posterior.counts(entry.entry_id)[0] for entry in controller.archive.entries) == 2


def test_existing_landing_is_a_false_discovery_without_collision(tmp_path: Path) -> None:
    def worker(action: StarterAction, starter_state: State) -> AttemptResult:
        return _completed(action, -1.0, -1.0)

    controller = ExplorationController(_archive(), "uniform", 5, ExplorationEventLog(tmp_path / "events.jsonl"))
    before = len(controller.archive.entries)

    with ThreadPoolExecutor(max_workers=1) as executor:
        (outcome,) = controller.run_batch(executor, worker, batch_size=1, force_budget=None)

    assert outcome.discovered_against_snapshot is False
    assert outcome.inserted_into_archive is False
    assert outcome.within_batch_collision is False
    assert len(controller.archive.entries) == before
    assert controller.posterior.counts(outcome.starter_id) == (0, 1)


def test_worker_reported_and_unexpected_failures_are_each_credited_once(tmp_path: Path) -> None:
    def worker(action: StarterAction, starter_state: State) -> AttemptResult:
        if action.slot_id == 0:
            return AttemptResult(
                action=action,
                landing_state=None,
                landing_energy=None,
                force_evaluations=2,
                status=AttemptStatus.INVALID,
                failure_reason="worker rejected geometry",
            )
        raise RuntimeError("boom")

    controller = ExplorationController(
        _archive(), "posterior_proportional", 2, ExplorationEventLog(tmp_path / "events.jsonl")
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = controller.run_batch(executor, worker, batch_size=2, force_budget=5)

    assert [outcome.status for outcome in outcomes] == [AttemptStatus.INVALID, AttemptStatus.WORKER_ERROR]
    assert outcomes[0].force_evaluations == 2
    assert outcomes[0].failure_reason == "worker rejected geometry"
    assert outcomes[1].force_evaluations == 0
    assert outcomes[1].failure_reason == "RuntimeError: boom"
    assert all(not outcome.discovered_against_snapshot for outcome in outcomes)
    assert controller.posterior.completed_attempts == 2
    assert sum(controller.posterior.counts(entry.entry_id)[1] for entry in controller.archive.entries) == 2


@pytest.mark.parametrize(
    ("worker_factory", "message"),
    [
        (
            lambda: lambda action, starter_state: _completed(
                replace(action, action_id="wrong-action"), 3.0, -2.0
            ),
            "wrong action",
        ),
        (lambda: lambda action, starter_state: object(), "AttemptResult"),
    ],
)
def test_invalid_worker_returns_abort_without_state_or_log_mutation(
    tmp_path: Path,
    worker_factory: Callable[[], Callable[[StarterAction, State], object]],
    message: str,
) -> None:
    event_path = tmp_path / "events.jsonl"
    controller = ExplorationController(_archive(), "uniform", 1, ExplorationEventLog(event_path))
    before = _controller_fingerprint(controller)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ValueError, match=message):
            controller.run_batch(executor, worker_factory(), batch_size=1, force_budget=None)

    assert _controller_fingerprint(controller) == before
    assert not event_path.exists()


def test_log_failure_rolls_back_entries_nested_archive_data_posterior_and_versions() -> None:
    class FailingLog:
        def __init__(self) -> None:
            self.calls = 0

        def append_batch(self, snapshot, actions, outcomes) -> None:
            self.calls += 1
            raise OSError("disk full")

    def worker(action: StarterAction, starter_state: State) -> AttemptResult:
        if action.slot_id == 0:
            return _completed(action, -1.0, -1.0)
        return _completed(action, 5.0, -4.0)

    event_log = FailingLog()
    controller = ExplorationController(_archive(), "uniform", 1, event_log)
    before = _controller_fingerprint(controller)

    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(OSError, match="disk full"):
            controller.run_batch(executor, worker, batch_size=2, force_budget=None)

    assert event_log.calls == 1
    assert _controller_fingerprint(controller) == before


def test_postwrite_log_error_retries_exact_pending_batch_without_reexecuting_worker(tmp_path: Path) -> None:
    class PostWriteFsyncErrorLog:
        def __init__(self, path: Path) -> None:
            self._log = ExplorationEventLog(path)
            self.calls = 0
            self.raise_once = True

        def append_batch(self, snapshot, actions, outcomes) -> None:
            self.calls += 1
            self._log.append_batch(snapshot, actions, outcomes)
            if self.raise_once:
                self.raise_once = False
                raise OSError("post-write fsync acknowledgement failed")

    event_path = tmp_path / "events.jsonl"
    event_log = PostWriteFsyncErrorLog(event_path)
    controller = ExplorationController(_archive(), "uniform", 23, event_log)
    before = _controller_fingerprint(controller)
    worker_calls: list[int] = []

    def nondeterministic_worker(action: StarterAction, starter_state: State) -> AttemptResult:
        worker_calls.append(action.slot_id)
        return _completed(action, 10.0 + len(worker_calls), -10.0 - len(worker_calls))

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(OSError, match="post-write fsync"):
            controller.run_batch(executor, nondeterministic_worker, batch_size=1, force_budget=None)

        assert worker_calls == [0]
        assert _controller_fingerprint(controller) == before
        rows_before_retry = event_path.read_text(encoding="utf-8")

        def must_not_run(action: StarterAction, starter_state: State) -> AttemptResult:
            pytest.fail("pending commit reconciliation must not dispatch a worker")

        outcomes = controller.run_batch(executor, must_not_run, batch_size=9, force_budget=1)

    assert worker_calls == [0]
    assert event_log.calls == 2
    assert event_path.read_text(encoding="utf-8") == rows_before_retry
    assert outcomes[0].landing_energy == pytest.approx(-11.0)
    assert controller.posterior.completed_attempts == 1
    assert ExplorationEventLog(event_path).reconstruct_posterior().counts(outcomes[0].starter_id) == (
        1,
        0,
    )


def test_controller_exposes_no_locking_pending_status_property(tmp_path: Path) -> None:
    controller = ExplorationController(
        _archive(), "uniform", 23, ExplorationEventLog(tmp_path / "events.jsonl")
    )

    assert not hasattr(controller, "has_pending_commit")


def test_worker_and_batch_log_callback_can_inspect_public_counters_during_batch(tmp_path: Path) -> None:
    observed_by_worker: list[tuple[int, int, int]] = []
    observed_by_log: list[tuple[int, int, int]] = []

    class CounterInspectingLog:
        def __init__(self, path: Path) -> None:
            self._log = ExplorationEventLog(path)

        def append_batch(self, snapshot, actions, outcomes) -> None:
            observed_by_log.append(
                (controller.policy_version, controller.archive_version, controller.batch_id)
            )
            self._log.append_batch(snapshot, actions, outcomes)

    event_log = CounterInspectingLog(tmp_path / "events.jsonl")
    controller = ExplorationController(_archive(), "uniform", 28, event_log)

    def worker(action: StarterAction, starter_state: State) -> AttemptResult:
        observed_by_worker.append(
            (controller.policy_version, controller.archive_version, controller.batch_id)
        )
        return _completed(action, 9.0, -9.0)

    with ThreadPoolExecutor(max_workers=1) as executor:
        controller.run_batch(executor, worker, batch_size=1, force_budget=None)

    assert observed_by_worker == [(0, 0, 0)]
    assert observed_by_log == [(0, 0, 0)]
    assert (controller.policy_version, controller.archive_version, controller.batch_id) == (1, 1, 1)


def test_completed_landing_state_is_owned_by_archive_after_worker_returns(tmp_path: Path) -> None:
    retained: dict[str, object] = {}

    def worker(action: StarterAction, starter_state: State) -> AttemptResult:
        original_landing = _state(6.0, label="worker-original")
        result = AttemptResult(
            action=action,
            landing_state=original_landing,
            landing_energy=-6.0,
            force_evaluations=2,
            status=AttemptStatus.COMPLETED,
            failure_reason=None,
        )
        retained["original_landing"] = original_landing
        retained["result"] = result
        return result

    controller = ExplorationController(_archive(), "uniform", 3, ExplorationEventLog(tmp_path / "events.jsonl"))
    with ThreadPoolExecutor(max_workers=1) as executor:
        (outcome,) = controller.run_batch(executor, worker, batch_size=1, force_budget=None)

    landing = controller.archive.entries[outcome.landing_entry_id]
    assert landing.descriptor is not None
    descriptor_before = landing.descriptor.copy()
    original_landing = retained["original_landing"]
    result = retained["result"]
    assert isinstance(original_landing, State)
    assert isinstance(result, AttemptResult)
    original_landing.positions[0, 0] = 60.0
    result.landing_state.positions[0, 0] = 600.0

    assert landing.state.positions[0, 0] == pytest.approx(6.0)
    np.testing.assert_allclose(landing.descriptor, descriptor_before)


def test_partial_executor_submission_failure_finalizes_every_planned_slot(tmp_path: Path) -> None:
    class FailsOnSecondSubmit(Executor):
        def __init__(self) -> None:
            self._inner = ThreadPoolExecutor(max_workers=1)
            self.submit_calls = 0

        def submit(self, fn, /, *args, **kwargs):
            self.submit_calls += 1
            if self.submit_calls == 2:
                raise RuntimeError("submit transport failed")
            return self._inner.submit(fn, *args, **kwargs)

        def shutdown(self, wait: bool = True, *, cancel_futures: bool = False) -> None:
            self._inner.shutdown(wait=wait, cancel_futures=cancel_futures)

    event_path = tmp_path / "events.jsonl"
    controller = ExplorationController(_archive(), "uniform", 12, ExplorationEventLog(event_path))
    worker_slots: list[int] = []

    def worker(action: StarterAction, starter_state: State) -> AttemptResult:
        worker_slots.append(action.slot_id)
        return _completed(action, 8.0, -8.0)

    executor = FailsOnSecondSubmit()
    try:
        outcomes = controller.run_batch(executor, worker, batch_size=3, force_budget=5)
    finally:
        executor.shutdown()

    assert executor.submit_calls == 2
    assert worker_slots == [0]
    assert [outcome.status for outcome in outcomes] == [
        AttemptStatus.COMPLETED,
        AttemptStatus.WORKER_ERROR,
        AttemptStatus.WORKER_ERROR,
    ]
    assert [outcome.force_evaluations for outcome in outcomes] == [3, 0, 0]
    assert [outcome.failure_reason for outcome in outcomes[1:]] == [
        "RuntimeError: submit transport failed",
        "RuntimeError: submit transport failed",
    ]
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert [row["slot_id"] for row in rows[1:-1]] == [0, 1, 2]
    assert [row["status"] for row in rows[1:-1]] == ["completed", "worker_error", "worker_error"]
    assert controller.posterior.completed_attempts == 3
    assert ExplorationEventLog(event_path).reconstruct_posterior().completed_attempts == 3


def test_simultaneous_controller_calls_are_serialized_into_sequential_batches(tmp_path: Path) -> None:
    class BlockingFirstAppendLog:
        def __init__(self, path: Path) -> None:
            self._log = ExplorationEventLog(path)
            self._lock = threading.Lock()
            self._calls = 0
            self.first_append_entered = threading.Event()
            self.second_append_entered = threading.Event()
            self.release_first_append = threading.Event()

        def append_batch(self, snapshot, actions, outcomes) -> None:
            with self._lock:
                self._calls += 1
                call_number = self._calls
            if call_number == 1:
                self.first_append_entered.set()
                assert self.release_first_append.wait(timeout=2)
            else:
                self.second_append_entered.set()
            self._log.append_batch(snapshot, actions, outcomes)

    event_path = tmp_path / "events.jsonl"
    event_log = BlockingFirstAppendLog(event_path)
    controller = ExplorationController(_archive(), "uniform", 31, event_log)
    callers_ready = threading.Barrier(3)

    def worker(action: StarterAction, starter_state: State) -> AttemptResult:
        return _completed(action, 20.0 + action.batch_id, -20.0 - action.batch_id)

    def caller():
        callers_ready.wait(timeout=2)
        return controller.run_batch(action_executor, worker, batch_size=1, force_budget=None)

    with ThreadPoolExecutor(max_workers=2) as action_executor:
        with ThreadPoolExecutor(max_workers=2) as caller_executor:
            first = caller_executor.submit(caller)
            second = caller_executor.submit(caller)
            callers_ready.wait(timeout=2)
            assert event_log.first_append_entered.wait(timeout=2)
            assert not event_log.second_append_entered.wait(timeout=0.25)
            event_log.release_first_append.set()
            outcomes = (first.result(timeout=5), second.result(timeout=5))

    action_ids = sorted(outcome[0].action_id for outcome in outcomes)
    assert action_ids == ["batch-00000000-slot-0000", "batch-00000001-slot-0000"]
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    assert [row["batch_id"] for row in rows if row["record_type"] == "policy_snapshot"] == [0, 1]
    assert sum(row["record_type"] == "attempt" for row in rows) == 2
    assert controller.posterior.completed_attempts == 2
    assert ExplorationEventLog(event_path).reconstruct_posterior().completed_attempts == 2


@pytest.mark.parametrize("policy_name", ["uniform", "posterior_proportional", "minimal_ucb"])
def test_all_phase_one_policies_run_and_actions_are_deterministic_for_same_initial_state(
    tmp_path: Path, policy_name: str
) -> None:
    captured_first: list[StarterAction] = []
    captured_second: list[StarterAction] = []

    def worker_for(captured: list[StarterAction]):
        def worker(action: StarterAction, starter_state: State) -> AttemptResult:
            captured.append(action)
            return AttemptResult(
                action=action,
                landing_state=None,
                landing_energy=None,
                force_evaluations=1,
                status=AttemptStatus.BUDGET_EXHAUSTED,
                failure_reason="synthetic budget exhaustion",
            )

        return worker

    first = ExplorationController(
        _archive(), policy_name, 17, ExplorationEventLog(tmp_path / f"{policy_name}-one.jsonl")
    )
    second = ExplorationController(
        _archive(), policy_name, 17, ExplorationEventLog(tmp_path / f"{policy_name}-two.jsonl")
    )

    with ThreadPoolExecutor(max_workers=3) as executor:
        first.run_batch(executor, worker_for(captured_first), batch_size=3, force_budget=3)
    with ThreadPoolExecutor(max_workers=3) as executor:
        second.run_batch(executor, worker_for(captured_second), batch_size=3, force_budget=3)

    assert sorted(captured_first, key=lambda action: action.slot_id) == sorted(
        captured_second, key=lambda action: action.slot_id
    )
    assert first.posterior.completed_attempts == second.posterior.completed_attempts == 3


def test_empty_archive_fails_before_dispatch_or_controller_mutation(tmp_path: Path) -> None:
    empty_archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.05)
    event_path = tmp_path / "events.jsonl"
    controller = ExplorationController(empty_archive, "uniform", 1, ExplorationEventLog(event_path))
    before = _controller_fingerprint(controller)

    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ValueError, match="eligible_starter_ids cannot be empty"):
            controller.run_batch(
                executor,
                lambda action, starter_state: _completed(action, 3.0, -2.0),
                batch_size=1,
                force_budget=None,
            )

    assert _controller_fingerprint(controller) == before
    assert not event_path.exists()


@pytest.mark.parametrize(
    ("archive", "policy_name", "master_seed", "event_log", "message"),
    [
        (object(), "uniform", 0, object(), "archive"),
        (_archive(), "", 0, object(), "policy_name"),
        (_archive(), "not-a-policy", 0, object(), "unsupported policy"),
        (_archive(), "uniform", -1, object(), "master_seed"),
        (_archive(), "uniform", True, object(), "master_seed"),
        (_archive(), "uniform", 0, object(), "event_log"),
    ],
)
def test_constructor_rejects_invalid_phase_one_inputs(
    archive: object, policy_name: object, master_seed: object, event_log: object, message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        ExplorationController(archive, policy_name, master_seed, event_log)
