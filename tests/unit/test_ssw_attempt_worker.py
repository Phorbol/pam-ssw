from types import SimpleNamespace

import numpy as np
import pytest

import pamssw
from pamssw.accounting import BudgetExceeded, EvaluationCounts, EvaluationPurpose
import pamssw.exploration as exploration
import pamssw.exploration.ssw_worker as worker_module
from pamssw.config import LSSSWConfig, SSWConfig
from pamssw.exploration import SSWAttemptWorker
from pamssw.exploration.actions import AttemptStatus, StarterAction
from pamssw.result import SearchResult, WalkRecord
from pamssw.state import State


def _state(x: float = 1.0) -> State:
    return State(numbers=np.array([1]), positions=np.array([[x, 0.0, 0.0]]))


def _action(*, force_budget: int | None = 8, random_seed: int = 12) -> StarterAction:
    return StarterAction(
        action_id="batch-00000001-slot-0000",
        batch_id=1,
        slot_id=0,
        policy_name="uniform",
        policy_version=2,
        archive_version=3,
        starter_id=4,
        selection_probability=0.25,
        random_seed=random_seed,
        force_budget=force_budget,
    )


class _Calculator:
    def evaluate(self, state: State) -> tuple[float, np.ndarray]:
        return 0.0, np.zeros_like(state.positions)

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        return 0.0, np.zeros_like(flat_positions)


class _Counter:
    def __init__(
        self,
        force_evaluations: int,
        exhausted: bool,
        evaluation_counts: EvaluationCounts | None = None,
    ) -> None:
        self.force_evaluations = force_evaluations
        self._exhausted = exhausted
        self._evaluation_counts = evaluation_counts or EvaluationCounts.unattributed(force_evaluations)

    def exhausted(self) -> bool:
        return self._exhausted

    def snapshot(self) -> EvaluationCounts:
        return self._evaluation_counts


def _archive_entry(entry_id: int, *, x: float = 1.0, energy: float = 0.5) -> SimpleNamespace:
    return SimpleNamespace(entry_id=entry_id, state=_state(x), energy=energy)


def _search_result(
    *,
    history: list[WalkRecord] | None = None,
    stats: dict[str, object] | None = None,
    entries: list[SimpleNamespace] | None = None,
    best_energy: float = -10.0,
) -> SearchResult:
    return SearchResult(
        best_state=_state(-3.0),
        best_energy=best_energy,
        archive=SimpleNamespace(entries=[] if entries is None else entries),
        walk_history=[] if history is None else history,
        stats=(
            {
                "force_evaluations": 3,
                "budget_exhausted": 0,
                "fragment_rejections": 0,
            }
            if stats is None
            else stats
        ),
    )


def _walk_record(discovered_entry_id: int = 7) -> WalkRecord:
    return WalkRecord(
        seed_entry_id=4,
        discovered_entry_id=discovered_entry_id,
        energy=99.0,
        accepted_new_basin=True,
    )


def _install_fake_walker(
    monkeypatch: pytest.MonkeyPatch,
    outcome: SearchResult | BaseException,
    *,
    force_evaluations: int = 3,
    exhausted: bool = False,
    evaluation_counts: EvaluationCounts | None = None,
    constructor_error: Exception | None = None,
) -> list[SimpleNamespace]:
    instances: list[SimpleNamespace] = []

    class FakeSurfaceWalker:
        def __init__(self, calculator, config, softening_enabled) -> None:
            if constructor_error is not None:
                raise constructor_error
            instance = SimpleNamespace(
                calculator=_Counter(force_evaluations, exhausted, evaluation_counts),
                config=config,
                softening_enabled=softening_enabled,
                calculator_from_factory=calculator,
                run_starters=[],
            )
            self.__dict__.update(instance.__dict__)
            instances.append(self)

        def run(self, starter: State) -> SearchResult:
            self.run_starters.append(starter)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(worker_module, "SurfaceWalker", FakeSurfaceWalker)
    return instances


def _make_worker(
    monkeypatch: pytest.MonkeyPatch,
    outcome: SearchResult | BaseException,
    *,
    force_evaluations: int = 3,
    exhausted: bool = False,
    evaluation_counts: EvaluationCounts | None = None,
    config: SSWConfig | None = None,
) -> tuple[SSWAttemptWorker, list[SimpleNamespace], list[_Calculator]]:
    instances = _install_fake_walker(
        monkeypatch,
        outcome,
        force_evaluations=force_evaluations,
        exhausted=exhausted,
        evaluation_counts=evaluation_counts,
    )
    calculators: list[_Calculator] = []

    def factory() -> _Calculator:
        calculator = _Calculator()
        calculators.append(calculator)
        return calculator

    return SSWAttemptWorker(factory, config or SSWConfig()), instances, calculators


def test_exploration_exports_the_isolated_ssw_attempt_worker_only_locally():
    assert exploration.SSWAttemptWorker is SSWAttemptWorker
    assert "SSWAttemptWorker" in exploration.__all__
    assert "SSWAttemptWorker" not in pamssw.__all__
    assert not hasattr(pamssw, "SSWAttemptWorker")


@pytest.mark.parametrize(
    "factory,config,softening_enabled,error",
    [
        (None, SSWConfig(), False, "calculator_factory must be callable"),
        (lambda: _Calculator(), object(), False, "config must be an SSWConfig"),
        (lambda: _Calculator(), SSWConfig(), 1, "softening_enabled must be a boolean"),
        (
            lambda: _Calculator(),
            SSWConfig(),
            True,
            "softening_enabled requires an LSSSWConfig",
        ),
    ],
)
def test_constructor_validates_factory_config_and_softening(factory, config, softening_enabled, error):
    with pytest.raises(ValueError, match=error):
        SSWAttemptWorker(factory, config, softening_enabled=softening_enabled)


def test_softening_worker_accepts_an_ls_ssw_config_and_passes_the_flag(monkeypatch):
    instances = _install_fake_walker(monkeypatch, _search_result())
    soft_worker = SSWAttemptWorker(
        lambda: _Calculator(), LSSSWConfig(), softening_enabled=True
    )

    result = soft_worker(_action(), _state())

    assert result.status is AttemptStatus.INVALID
    assert instances[0].softening_enabled is True


@pytest.mark.parametrize(
    "config",
    [
        SSWConfig(proposal_pool_size=2),
        SSWConfig(proposal_duplicate_rescue_optimizer="ase-fire"),
    ],
)
def test_constructor_rejects_hidden_internal_proposal_competition(config):
    with pytest.raises(ValueError, match="internal proposal competition"):
        SSWAttemptWorker(lambda: _Calculator(), config)


@pytest.mark.parametrize(
    "config",
    [
        SSWConfig(accepted_structures_log="accepted.jsonl"),
        SSWConfig(accepted_structures_dir="accepted"),
        SSWConfig(write_proposal_minima=True, proposal_minima_dir="proposals"),
        SSWConfig(write_relaxation_trajectories=True, relaxation_trajectory_dir="trajectories"),
        SSWConfig(direction_diagnostics_enabled=True, direction_diagnostics_path="directions.jsonl"),
        SSWConfig(direction_archive_enabled=True, direction_archive_path="directions.jsonl"),
    ],
)
def test_constructor_rejects_shared_filesystem_output_modes(config):
    with pytest.raises(ValueError, match="shared filesystem output"):
        SSWAttemptWorker(lambda: _Calculator(), config)


def test_action_overrides_config_and_factory_is_called_once_per_action(monkeypatch):
    result = _search_result(
        history=[_walk_record()],
        entries=[_archive_entry(7, energy=3.5)],
    )
    worker, instances, calculators = _make_worker(
        monkeypatch,
        result,
        config=SSWConfig(max_trials=9, rng_seed=2, max_force_evals=99),
    )
    action = _action(force_budget=5, random_seed=31)
    starter = _state(2.0)

    mapped = worker(action, starter)

    assert len(calculators) == 1
    assert len(instances) == 1
    assert instances[0].config.max_trials == 1
    assert instances[0].config.rng_seed == 31
    assert instances[0].config.max_force_evals == 5
    assert instances[0].calculator_from_factory is calculators[0]
    assert mapped.status is AttemptStatus.COMPLETED
    assert mapped.force_evaluations == 3


def test_each_action_uses_a_fresh_walker_and_deepcopied_starter(monkeypatch):
    result = _search_result()
    worker, instances, calculators = _make_worker(monkeypatch, result)
    starter = _state(2.0)

    first = worker(_action(random_seed=1), starter)
    instances[0].run_starters[0].positions[0, 0] = 99.0
    second = worker(_action(random_seed=2), starter)

    assert first.status is AttemptStatus.INVALID
    assert second.status is AttemptStatus.INVALID
    assert len(calculators) == 2
    assert len(instances) == 2
    assert starter.positions[0, 0] == 2.0


def test_invalid_starter_returns_zero_cost_invalid_without_calling_factory():
    calls = 0

    def factory() -> _Calculator:
        nonlocal calls
        calls += 1
        return _Calculator()

    invalid = State(numbers=np.array([1]), positions=np.array([[np.nan, 0.0, 0.0]]))
    worker = SSWAttemptWorker(factory, SSWConfig())
    result = worker(_action(), invalid)

    assert result.status is AttemptStatus.INVALID
    assert result.force_evaluations == 0
    assert result.failure_reason == "invalid_starter_geometry"
    assert calls == 0
    diagnostics = worker.diagnostics_snapshot()
    assert len(diagnostics) == 1
    assert dict(diagnostics[0].stats)["diagnostic_stage"] == "invalid_starter"
    assert dict(diagnostics[0].stats)["force_evaluations"] == 0


@pytest.mark.parametrize("action,starter", [(object(), _state()), (_action(), object())])
def test_call_rejects_non_action_or_non_state_inputs(action, starter):
    worker = SSWAttemptWorker(lambda: _Calculator(), SSWConfig())

    with pytest.raises(ValueError):
        worker(action, starter)


def test_factory_failure_is_a_zero_cost_worker_error():
    def factory() -> _Calculator:
        raise RuntimeError("factory boom")

    worker = SSWAttemptWorker(factory, SSWConfig())
    result = worker(_action(), _state())

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == 0
    assert result.failure_reason == "factory_error: RuntimeError: factory boom"
    assert dict(worker.diagnostics_snapshot()[0].stats)["diagnostic_stage"] == "factory_error"


def test_malformed_factory_calculator_is_a_zero_cost_worker_error():
    worker = SSWAttemptWorker(lambda: object(), SSWConfig())
    result = worker(_action(), _state())

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == 0
    assert result.failure_reason == "calculator_error: missing callable evaluate and evaluate_flat"
    assert dict(worker.diagnostics_snapshot()[0].stats)["diagnostic_stage"] == "calculator_error"


def test_walker_construction_failure_is_a_zero_cost_worker_error(monkeypatch):
    worker, _, _ = _make_worker(monkeypatch, _search_result())
    _install_fake_walker(monkeypatch, _search_result(), constructor_error=RuntimeError("construct boom"))

    result = worker(_action(), _state())

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == 0
    assert result.failure_reason == "constructor_error: RuntimeError: construct boom"
    assert dict(worker.diagnostics_snapshot()[0].stats)["diagnostic_stage"] == "constructor_error"


def test_diagnostic_failure_cannot_change_the_terminal_action_result(monkeypatch):
    worker, _, _ = _make_worker(monkeypatch, _search_result())

    def fail_diagnostics(*args, **kwargs):
        raise RuntimeError("diagnostic boom")

    monkeypatch.setattr(worker, "_record_diagnostics", fail_diagnostics)

    result = worker(_action(), _state())

    assert result.status is AttemptStatus.INVALID
    assert result.force_evaluations == 3
    diagnostics = worker.diagnostics_snapshot()
    assert len(diagnostics) == 1
    stats = dict(diagnostics[0].stats)
    assert stats["diagnostics_error"] == "RuntimeError"
    assert stats["force_evaluations"] == result.force_evaluations


def test_landing_comes_from_the_walk_record_not_the_search_best_state(monkeypatch):
    landing = _archive_entry(7, x=5.0, energy=9.0)
    worker, _, _ = _make_worker(
        monkeypatch,
        _search_result(history=[_walk_record(7)], entries=[landing], best_energy=-100.0),
    )

    result = worker(_action(), _state())

    assert result.status is AttemptStatus.COMPLETED
    assert result.landing_energy == 9.0
    assert result.landing_state is not landing.state
    assert result.landing_state.positions[0, 0] == 5.0


def test_valid_landing_takes_precedence_over_an_exactly_exhausted_counter(monkeypatch):
    worker, _, _ = _make_worker(
        monkeypatch,
        _search_result(
            history=[_walk_record(7)],
            entries=[_archive_entry(7, energy=1.5)],
            stats={"force_evaluations": 5, "budget_exhausted": 1, "fragment_rejections": 0},
        ),
        force_evaluations=5,
        exhausted=True,
    )

    result = worker(_action(force_budget=5), _state())

    assert result.status is AttemptStatus.COMPLETED
    assert result.force_evaluations == 5


@pytest.mark.parametrize(
    ("stats", "expected_status", "expected_reason"),
    [
        (
            {"force_evaluations": 3, "budget_exhausted": 1, "fragment_rejections": 1},
            AttemptStatus.BUDGET_EXHAUSTED,
            "budget_exhausted_without_landing",
        ),
        (
            {"force_evaluations": 3, "budget_exhausted": 0, "fragment_rejections": 1},
            AttemptStatus.FRAGMENTED,
            "fragment_rejections_without_landing",
        ),
        (
            {"force_evaluations": 3, "budget_exhausted": 0, "fragment_rejections": 0},
            AttemptStatus.INVALID,
            "no_landing_minimum",
        ),
    ],
)
def test_no_landing_maps_to_the_three_terminal_statuses(
    monkeypatch, stats, expected_status, expected_reason
):
    worker, _, _ = _make_worker(monkeypatch, _search_result(stats=stats))

    result = worker(_action(), _state())

    assert result.status is expected_status
    assert result.force_evaluations == 3
    assert result.failure_reason == expected_reason


def test_budget_exception_uses_counter_exhaustion_without_parsing_the_message(monkeypatch):
    from pamssw.accounting import BudgetExceeded

    worker, _, _ = _make_worker(
        monkeypatch,
        BudgetExceeded("this text must not affect the status"),
        force_evaluations=5,
        exhausted=True,
    )

    result = worker(_action(force_budget=5), _state())

    assert result.status is AttemptStatus.BUDGET_EXHAUSTED
    assert result.force_evaluations == 5
    assert result.failure_reason == "budget_exhausted"


def test_budget_exception_without_exhausted_counter_is_invalid(monkeypatch):
    from pamssw.accounting import BudgetExceeded

    worker, _, _ = _make_worker(
        monkeypatch,
        BudgetExceeded("budget-looking message"),
        force_evaluations=3,
        exhausted=False,
    )

    result = worker(_action(force_budget=5), _state())

    assert result.status is AttemptStatus.INVALID
    assert result.force_evaluations == 3
    assert result.failure_reason == "budget_exception_without_exhaustion"


def test_more_than_one_walk_record_is_a_mapping_worker_error_with_exact_count(monkeypatch):
    worker, _, _ = _make_worker(
        monkeypatch,
        _search_result(
            history=[_walk_record(7), _walk_record(8)],
            entries=[_archive_entry(7), _archive_entry(8)],
            stats={"force_evaluations": 4, "budget_exhausted": 0, "fragment_rejections": 0},
        ),
        force_evaluations=4,
    )

    result = worker(_action(), _state())

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == 4
    assert result.failure_reason == "result_mapping_error: ValueError: expected at most one walk record"


@pytest.mark.parametrize(
    "entries",
    [
        [],
        [_archive_entry(7), _archive_entry(7, x=2.0)],
    ],
)
def test_absent_or_duplicate_discovered_id_is_a_mapping_worker_error(monkeypatch, entries):
    worker, _, _ = _make_worker(
        monkeypatch,
        _search_result(
            history=[_walk_record(7)],
            entries=entries,
            stats={"force_evaluations": 4, "budget_exhausted": 0, "fragment_rejections": 0},
        ),
        force_evaluations=4,
    )

    result = worker(_action(), _state())

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == 4
    assert result.failure_reason == "result_mapping_error: ValueError: discovered entry id must resolve exactly once"


@pytest.mark.parametrize(
    "stats",
    [
        {},
        {"force_evaluations": -1, "budget_exhausted": 0, "fragment_rejections": 0},
        {"force_evaluations": 3.0, "budget_exhausted": 0, "fragment_rejections": 0},
        {"force_evaluations": 3, "budget_exhausted": True, "fragment_rejections": 0},
        {"force_evaluations": 3, "budget_exhausted": 0, "fragment_rejections": -1},
    ],
)
def test_missing_or_invalid_required_stats_are_mapping_worker_errors(monkeypatch, stats):
    worker, _, _ = _make_worker(monkeypatch, _search_result(stats=stats), force_evaluations=4)

    result = worker(_action(), _state())

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == 4
    assert result.failure_reason.startswith("result_mapping_error: ValueError:")


def test_run_errors_become_worker_errors_with_the_exact_counter(monkeypatch):
    worker, _, _ = _make_worker(monkeypatch, RuntimeError("walk boom"), force_evaluations=4)

    result = worker(_action(), _state())

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == 4
    assert result.failure_reason == "run_error: RuntimeError: walk boom"


def test_base_exceptions_propagate_from_the_walker(monkeypatch):
    worker, _, _ = _make_worker(monkeypatch, KeyboardInterrupt(), force_evaluations=4)

    with pytest.raises(KeyboardInterrupt):
        worker(_action(), _state())


def test_completed_worker_result_uses_the_exact_purpose_snapshot_not_legacy_stats(monkeypatch):
    exact_counts = EvaluationCounts.from_mapping(
        {
            EvaluationPurpose.STARTER_TRUE_QUENCH: 1,
            EvaluationPurpose.DIRECTION_ORACLE: 2,
            EvaluationPurpose.LANDING_TRUE_QUENCH: 1,
        }
    )
    worker, _, _ = _make_worker(
        monkeypatch,
        _search_result(
            history=[_walk_record(7)],
            entries=[_archive_entry(7, energy=1.5)],
            stats={"force_evaluations": 4, "budget_exhausted": 0, "fragment_rejections": 0},
        ),
        force_evaluations=exact_counts.total,
        evaluation_counts=exact_counts,
    )

    result = worker(_action(), _state())

    assert result.status is AttemptStatus.COMPLETED
    assert result.force_evaluations == exact_counts.total
    assert result.evaluation_counts == exact_counts
    assert result.cost_is_exact is True


@pytest.mark.parametrize(
    "mode",
    ["completed", "budget", "run_error", "fragmented", "no_landing", "mapping_error"],
)
def test_post_construction_terminal_paths_preserve_the_exact_calculator_snapshot(monkeypatch, mode):
    exact_counts = EvaluationCounts.from_mapping(
        {
            EvaluationPurpose.DIRECTION_ORACLE: 2,
            EvaluationPurpose.BIASED_PROPOSAL_RELAX: 1,
            EvaluationPurpose.LANDING_TRUE_QUENCH: 1,
        }
    )
    if mode == "completed":
        outcome = _search_result(
            history=[_walk_record(7)],
            entries=[_archive_entry(7)],
            stats={"force_evaluations": 4, "budget_exhausted": 0, "fragment_rejections": 0},
        )
        exhausted = False
    elif mode == "budget":
        outcome = BudgetExceeded("exact budget exhaustion")
        exhausted = True
    elif mode == "run_error":
        outcome = RuntimeError("run failure")
        exhausted = False
    elif mode == "fragmented":
        outcome = _search_result(
            stats={"force_evaluations": 4, "budget_exhausted": 0, "fragment_rejections": 1}
        )
        exhausted = False
    elif mode == "no_landing":
        outcome = _search_result(
            stats={"force_evaluations": 4, "budget_exhausted": 0, "fragment_rejections": 0}
        )
        exhausted = False
    else:
        outcome = _search_result(
            history=[_walk_record(7), _walk_record(8)],
            entries=[_archive_entry(7), _archive_entry(8)],
            stats={"force_evaluations": 4, "budget_exhausted": 0, "fragment_rejections": 0},
        )
        exhausted = False
    worker, _, _ = _make_worker(
        monkeypatch,
        outcome,
        force_evaluations=exact_counts.total,
        exhausted=exhausted,
        evaluation_counts=exact_counts,
    )

    result = worker(_action(), _state())

    assert result.evaluation_counts == exact_counts
    assert result.evaluation_counts.total == result.force_evaluations == exact_counts.total
    assert result.cost_is_exact is True


def test_result_mapping_mismatch_returns_worker_error_with_the_same_exact_snapshot(monkeypatch):
    exact_counts = EvaluationCounts.from_mapping({EvaluationPurpose.DIRECTION_ORACLE: 4})
    worker, _, _ = _make_worker(
        monkeypatch,
        _search_result(
            stats={"force_evaluations": 3, "budget_exhausted": 0, "fragment_rejections": 0}
        ),
        force_evaluations=exact_counts.total,
        evaluation_counts=exact_counts,
    )

    result = worker(_action(), _state())

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.failure_reason == (
        "result_mapping_error: ValueError: search result force_evaluations must equal calculator snapshot"
    )
    assert result.force_evaluations == exact_counts.total
    assert result.evaluation_counts == exact_counts
    assert result.cost_is_exact is True


@pytest.mark.parametrize("failure_kind", ["invalid_starter", "factory", "calculator", "constructor"])
def test_pre_calculator_terminal_paths_are_exact_zero_snapshots(monkeypatch, failure_kind):
    if failure_kind == "invalid_starter":
        worker = SSWAttemptWorker(lambda: _Calculator(), SSWConfig())
        starter = State(numbers=np.array([1]), positions=np.array([[np.nan, 0.0, 0.0]]))
    elif failure_kind == "factory":
        def failing_factory():
            raise RuntimeError("factory failure")

        worker = SSWAttemptWorker(failing_factory, SSWConfig())
        starter = _state()
    elif failure_kind == "calculator":
        worker = SSWAttemptWorker(lambda: object(), SSWConfig())
        starter = _state()
    else:
        _install_fake_walker(
            monkeypatch,
            _search_result(),
            constructor_error=RuntimeError("constructor failure"),
        )
        worker = SSWAttemptWorker(lambda: _Calculator(), SSWConfig())
        starter = _state()

    result = worker(_action(), starter)

    assert result.force_evaluations == 0
    assert result.evaluation_counts == EvaluationCounts.zero()
    assert result.cost_is_exact is True
    diagnostics = worker.diagnostics_snapshot()
    assert len(diagnostics) == 1
    assert dict(diagnostics[0].stats)["force_evaluations"] == 0
