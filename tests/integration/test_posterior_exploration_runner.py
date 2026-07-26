import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pamssw.accounting import BudgetExceeded, EvaluationPurpose
from pamssw.calculators import AnalyticCalculator
from pamssw.config import LSSSWConfig, SSWConfig
from pamssw.exploration import PosteriorExplorationConfig, PosteriorExplorationResult
from pamssw.exploration.actions import AttemptStatus
from pamssw.exploration.campaign import CampaignStopReason
from pamssw.exploration.runner import (
    _bootstrap_minimum,
    run_posterior_ls_ssw,
    run_posterior_ssw,
)
from pamssw.exploration.event_log import ExplorationEventLog
from pamssw import run_ls_ssw, run_ssw
from pamssw.potentials import DoubleWell2D
from pamssw.result import RelaxResult
from pamssw.state import State


class CountingAnalyticCalculator:
    def __init__(self) -> None:
        self.calculator = AnalyticCalculator(DoubleWell2D())
        self.calls = 0

    def evaluate(self, state: State):
        self.calls += 1
        return self.calculator.evaluate(state)

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        self.calls += 1
        return self.calculator.evaluate_flat(flat_positions, template)


class CollapsingCalculator:
    def evaluate(self, state: State):
        flat = state.flatten_positions()
        return float(np.dot(flat, flat)), 2.0 * flat.reshape(state.n_atoms, 3)

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        flat = np.asarray(flat_positions, dtype=float)
        return float(np.dot(flat, flat)), 2.0 * flat


def _runner_state() -> State:
    return State(
        numbers=np.ones(4, dtype=int),
        positions=np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    )


def _runner_ssw_config() -> SSWConfig:
    return SSWConfig(
        max_steps_per_walk=1,
        oracle_candidates=2,
        proposal_pool_size=1,
        rng_seed=19,
    )


def _runner_exploration_config(
    run_directory: Path,
    *,
    policy_name: str = "uniform",
    batch_size: int = 3,
    max_workers: int = 2,
    action_force_budget: int = 30,
    total_force_budget: int = 130,
    master_seed: int = 37,
) -> PosteriorExplorationConfig:
    return PosteriorExplorationConfig(
        policy_name=policy_name,
        batch_size=batch_size,
        max_workers=max_workers,
        action_force_budget=action_force_budget,
        total_force_budget=total_force_budget,
        master_seed=master_seed,
        run_directory=run_directory,
    )


def _result_fingerprint(result) -> tuple[object, ...]:
    entries = tuple(
        (
            entry.entry_id,
            entry.parent_id,
            entry.energy,
            tuple(tuple(row) for row in entry.state.positions.tolist()),
            entry.visits,
        )
        for entry in result.archive.entries
    )
    posterior = tuple(
        result.posterior.counts(entry.entry_id) for entry in result.archive.entries
    )
    return (
        entries,
        posterior,
        result.completed_batches,
        result.completed_attempts,
        result.failed_attempts,
        result.posterior_observed_attempts,
        result.bootstrap_evaluations,
        result.action_evaluations,
        result.total_evaluations,
        result.purpose_counts,
        result.unused_force_budget,
        result.stop_reason,
        result.benchmark_eligible,
        result.benchmark_ineligibility_reasons,
    )


def _committed_batches(event_path: Path) -> list[list[dict[str, object]]]:
    rows = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines()]
    batches: list[list[dict[str, object]]] = []
    active: list[dict[str, object]] | None = None
    for row in rows:
        if row["record_type"] == "policy_snapshot":
            assert active is None
            active = []
        elif row["record_type"] == "attempt":
            assert active is not None
            active.append(row)
        elif row["record_type"] == "batch_commit":
            assert active is not None
            batches.append(active)
            active = None
        else:  # pragma: no cover - strict event log has no other record classes.
            raise AssertionError(f"unexpected event row: {row!r}")
    assert active is None
    return batches


def test_bootstrap_minimum_relaxes_with_exact_purpose_accounting_and_isolated_state():
    initial = State(numbers=np.array([1]), positions=np.array([[-0.8, 0.0, 0.0]]))
    caller_positions = initial.positions.copy()
    calculator = CountingAnalyticCalculator()

    relaxed, energy, counts = _bootstrap_minimum(
        initial,
        lambda: calculator,
        SSWConfig(),
        total_force_budget=80,
    )

    assert np.isfinite(energy)
    assert counts.count(EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH) > 0
    assert counts.count(EvaluationPurpose.POST_RELAX_VALIDATION) == 1
    assert counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
    assert counts.total == calculator.calls
    np.testing.assert_allclose(initial.positions, caller_positions)
    assert relaxed is not initial
    assert not np.shares_memory(relaxed.positions, initial.positions)
    relaxed.positions[0, 0] = 99.0
    np.testing.assert_allclose(initial.positions, caller_positions)


def test_bootstrap_minimum_rejects_invalid_raw_geometry_before_factory():
    initial = State(numbers=np.array([1]), positions=np.array([[np.nan, 0.0, 0.0]]))
    factory_calls = 0

    def factory() -> CountingAnalyticCalculator:
        nonlocal factory_calls
        factory_calls += 1
        return CountingAnalyticCalculator()

    with pytest.raises(ValueError, match="invalid initial geometry"):
        _bootstrap_minimum(initial, factory, SSWConfig(), total_force_budget=80)

    assert factory_calls == 0


def test_bootstrap_minimum_propagates_force_budget_exhaustion():
    with pytest.raises(BudgetExceeded, match="force-evaluation budget exhausted"):
        _bootstrap_minimum(
            State(numbers=np.array([1]), positions=np.array([[-0.8, 0.0, 0.0]])),
            lambda: CountingAnalyticCalculator(),
            SSWConfig(),
            total_force_budget=1,
        )


def test_bootstrap_minimum_rejects_an_uncertified_relaxation(monkeypatch):
    initial = State(numbers=np.array([1]), positions=np.array([[-0.8, 0.0, 0.0]]))

    def unconverged_relax(self, state, fmax, maxiter, **kwargs):
        return RelaxResult(
            state=state,
            energy=0.0,
            gradient_norm=10.0 * fmax,
            n_iter=maxiter,
        )

    monkeypatch.setattr("pamssw.exploration.runner.Relaxer.relax", unconverged_relax)

    with pytest.raises(ValueError, match="bootstrap.*converge"):
        _bootstrap_minimum(
            initial,
            lambda: CountingAnalyticCalculator(),
            SSWConfig(quench_maxiter=1),
            total_force_budget=100,
        )


@pytest.mark.parametrize(
    ("initial_state", "calculator_factory", "ssw_config", "total_force_budget", "error_type"),
    [
        (object(), lambda: CountingAnalyticCalculator(), SSWConfig(), 80, TypeError),
        (
            State(numbers=np.array([1]), positions=np.array([[-0.8, 0.0, 0.0]])),
            object(),
            SSWConfig(),
            80,
            TypeError,
        ),
        (
            State(numbers=np.array([1]), positions=np.array([[-0.8, 0.0, 0.0]])),
            lambda: object(),
            SSWConfig(),
            80,
            TypeError,
        ),
        (
            State(numbers=np.array([1]), positions=np.array([[-0.8, 0.0, 0.0]])),
            lambda: CountingAnalyticCalculator(),
            object(),
            80,
            TypeError,
        ),
        (
            State(numbers=np.array([1]), positions=np.array([[-0.8, 0.0, 0.0]])),
            lambda: CountingAnalyticCalculator(),
            SSWConfig(),
            0,
            ValueError,
        ),
        (
            State(numbers=np.array([1]), positions=np.array([[-0.8, 0.0, 0.0]])),
            lambda: CountingAnalyticCalculator(),
            SSWConfig(),
            -1,
            ValueError,
        ),
        (
            State(numbers=np.array([1]), positions=np.array([[-0.8, 0.0, 0.0]])),
            lambda: CountingAnalyticCalculator(),
            SSWConfig(),
            True,
            TypeError,
        ),
    ],
)
def test_bootstrap_minimum_rejects_invalid_arguments(
    initial_state,
    calculator_factory,
    ssw_config,
    total_force_budget,
    error_type,
):
    with pytest.raises(error_type):
        _bootstrap_minimum(
            initial_state,
            calculator_factory,
            ssw_config,
            total_force_budget=total_force_budget,
        )


def test_bootstrap_minimum_rejects_invalid_final_geometry():
    initial = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )

    with pytest.raises(ValueError, match="invalid final relaxed state"):
        _bootstrap_minimum(
            initial,
            CollapsingCalculator,
            SSWConfig(quench_maxiter=30),
            total_force_budget=80,
        )


def test_posterior_runner_uses_real_threaded_ssw_with_exact_budget_and_event_ledger(tmp_path: Path):
    run_directory = tmp_path / "real"
    exploration_config = _runner_exploration_config(run_directory)

    result = run_posterior_ssw(
        _runner_state(),
        lambda: AnalyticCalculator(DoubleWell2D()),
        _runner_ssw_config(),
        exploration_config,
    )

    event_path = run_directory / "events.jsonl"
    assert result.completed_batches >= 1
    assert result.total_evaluations == result.bootstrap_evaluations + result.action_evaluations
    assert result.total_evaluations == result.purpose_counts.total
    assert result.total_evaluations + result.unused_force_budget == result.total_force_budget
    assert result.total_evaluations <= result.total_force_budget
    assert result.purpose_counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
    assert result.unused_force_budget < exploration_config.action_force_budget
    assert event_path.is_file()
    assert ExplorationEventLog(event_path).reconstruct_posterior().completed_attempts == (
        result.posterior_observed_attempts
    )

    remaining = exploration_config.total_force_budget - result.bootstrap_evaluations
    batches = _committed_batches(event_path)
    assert len(batches) == result.completed_batches
    for attempts in batches:
        assert len(attempts) == min(exploration_config.batch_size, remaining // exploration_config.action_force_budget)
        assert attempts
        assert all(attempt["force_budget"] == exploration_config.action_force_budget for attempt in attempts)
        assert all(
            sum(attempt["evaluation_counts"].values()) == attempt["force_evaluations"]
            for attempt in attempts
        )
        remaining -= sum(
            sum(attempt["evaluation_counts"].values()) for attempt in attempts
        )


@pytest.mark.parametrize("invalid_run_directory", ("existing", "missing-parent"))
def test_posterior_runner_rejects_run_directory_before_any_calculator_factory_call(
    tmp_path: Path, invalid_run_directory: str
):
    factory_calls = 0
    existing_directory = tmp_path / "already-exists"
    if invalid_run_directory == "existing":
        existing_directory.mkdir()
    else:
        existing_directory = tmp_path / "missing" / "run"

    def factory() -> CountingAnalyticCalculator:
        nonlocal factory_calls
        factory_calls += 1
        return CountingAnalyticCalculator()

    with pytest.raises((FileExistsError, FileNotFoundError)):
        run_posterior_ssw(
            _runner_state(),
            factory,
            _runner_ssw_config(),
            _runner_exploration_config(existing_directory),
        )

    assert factory_calls == 0


def test_posterior_runner_preflights_static_worker_config_before_bootstrap_or_directory_creation(
    tmp_path: Path,
):
    factory_calls = 0
    run_directory = tmp_path / "worker-config"

    def factory() -> CountingAnalyticCalculator:
        nonlocal factory_calls
        factory_calls += 1
        return CountingAnalyticCalculator()

    with pytest.raises(ValueError, match="proposal_pool_size"):
        run_posterior_ssw(
            _runner_state(),
            factory,
            SSWConfig(proposal_pool_size=2),
            _runner_exploration_config(run_directory),
        )

    assert factory_calls == 0
    assert not run_directory.exists()


def test_posterior_ls_runner_executes_real_campaign_with_exact_accounting(tmp_path: Path):
    exploration_config = _runner_exploration_config(
        tmp_path / "ls-real",
        batch_size=1,
        max_workers=1,
        action_force_budget=100,
        total_force_budget=180,
    )
    result = run_posterior_ls_ssw(
        _runner_state(),
        lambda: AnalyticCalculator(DoubleWell2D()),
        LSSSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=2,
            local_softening_mode="manual",
            local_softening_pairs=[(0, 1)],
            rng_seed=19,
        ),
        exploration_config,
    )

    assert isinstance(result, PosteriorExplorationResult)
    assert result.completed_batches >= 1
    assert result.total_evaluations == result.bootstrap_evaluations + result.action_evaluations
    assert result.total_evaluations == result.purpose_counts.total
    assert result.total_evaluations + result.unused_force_budget == result.total_force_budget
    assert result.total_evaluations <= result.total_force_budget
    assert result.purpose_counts.count(EvaluationPurpose.UNATTRIBUTED) == 0


def test_posterior_runner_stops_after_bootstrap_when_only_a_budget_tail_remains(tmp_path: Path):
    run_directory = tmp_path / "bootstrap-tail"
    bootstrap_state = State(numbers=np.array([1]), positions=np.array([[-0.8, 0.0, 0.0]]))
    bootstrap_counts = _bootstrap_minimum(
        bootstrap_state,
        lambda: AnalyticCalculator(DoubleWell2D()),
        _runner_ssw_config(),
        total_force_budget=80,
    )[2]
    exploration_config = _runner_exploration_config(
        run_directory,
        batch_size=1,
        max_workers=1,
        action_force_budget=20,
        total_force_budget=bootstrap_counts.total + 19,
    )

    result = run_posterior_ssw(
        bootstrap_state,
        lambda: AnalyticCalculator(DoubleWell2D()),
        _runner_ssw_config(),
        exploration_config,
    )

    assert result.completed_batches == 0
    assert result.completed_attempts == 0
    assert result.failed_attempts == 0
    assert result.stop_reason is CampaignStopReason.BUDGET_TAIL
    assert result.run_directory == run_directory
    assert run_directory.is_dir()
    assert not (run_directory / "events.jsonl").exists()


def test_posterior_runner_records_zero_cost_factory_failure_as_ineligible_stall(tmp_path: Path):
    calls = 0

    def bootstrap_then_fail_factory() -> CountingAnalyticCalculator:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("actions must not receive a calculator")
        return CountingAnalyticCalculator()

    exploration_config = _runner_exploration_config(
        tmp_path / "zero-cost-stall",
        total_force_budget=150,
    )
    result = run_posterior_ssw(
        _runner_state(),
        bootstrap_then_fail_factory,
        _runner_ssw_config(),
        exploration_config,
    )

    assert result.completed_batches == 1
    assert result.completed_attempts == 0
    assert result.failed_attempts == exploration_config.batch_size
    assert result.posterior_observed_attempts == 0
    assert result.action_evaluations == 0
    assert result.stop_reason is CampaignStopReason.ZERO_COST_STALL
    assert result.benchmark_eligible is False
    assert result.benchmark_ineligibility_reasons == (
        "non_posterior_observed_attempt",
        "zero_cost_stall",
    )
    attempt_rows = _committed_batches(exploration_config.run_directory / "events.jsonl")[0]
    assert len(attempt_rows) == exploration_config.batch_size
    assert all(row["status"] == AttemptStatus.WORKER_ERROR.value for row in attempt_rows)
    assert all(row["force_evaluations"] == 0 for row in attempt_rows)


def test_posterior_runner_uses_no_residual_action_and_replays_event_batch_widths(tmp_path: Path):
    exploration_config = _runner_exploration_config(
        tmp_path / "widths",
        action_force_budget=30,
        total_force_budget=130,
    )
    result = run_posterior_ssw(
        _runner_state(),
        lambda: AnalyticCalculator(DoubleWell2D()),
        _runner_ssw_config(),
        exploration_config,
    )

    remaining = exploration_config.total_force_budget - result.bootstrap_evaluations
    for attempts in _committed_batches(exploration_config.run_directory / "events.jsonl"):
        expected_width = min(exploration_config.batch_size, remaining // exploration_config.action_force_budget)
        assert len(attempts) == expected_width
        assert expected_width > 0
        assert all(row["force_budget"] == exploration_config.action_force_budget for row in attempts)
        remaining -= sum(sum(row["evaluation_counts"].values()) for row in attempts)
    assert remaining == result.unused_force_budget
    assert remaining < exploration_config.action_force_budget


def test_posterior_runner_public_config_pairing_is_explicit(tmp_path: Path):
    exploration_config = _runner_exploration_config(tmp_path / "config", total_force_budget=80)
    with pytest.raises(TypeError):
        run_posterior_ssw(
            _runner_state(),
            lambda: AnalyticCalculator(DoubleWell2D()),
            LSSSWConfig(),
            exploration_config,
        )
    with pytest.raises(TypeError):
        run_posterior_ls_ssw(
            _runner_state(),
            lambda: AnalyticCalculator(DoubleWell2D()),
            SSWConfig(),
            exploration_config,
        )
    with pytest.raises(TypeError):
        run_posterior_ssw(
            _runner_state(),
            lambda: AnalyticCalculator(DoubleWell2D()),
            _runner_ssw_config(),
            object(),
        )


@pytest.mark.parametrize("policy_name", ("uniform", "posterior_proportional", "minimal_ucb"))
def test_posterior_runner_policies_share_path_and_are_reproducible(
    tmp_path: Path, policy_name: str
):
    config_one = _runner_exploration_config(
        tmp_path / f"{policy_name}-one",
        policy_name=policy_name,
        batch_size=2,
        max_workers=2,
        total_force_budget=100,
    )
    config_two = replace(config_one, run_directory=tmp_path / f"{policy_name}-two")
    arguments = (_runner_state(), lambda: AnalyticCalculator(DoubleWell2D()), _runner_ssw_config())

    first = run_posterior_ssw(*arguments, config_one)
    second = run_posterior_ssw(*arguments, config_two)

    assert first.policy_name == second.policy_name == policy_name
    assert _result_fingerprint(first) == _result_fingerprint(second)


def test_posterior_runner_exports_do_not_replace_legacy_runner_exports():
    import pamssw
    import pamssw.exploration
    import pamssw.runner

    assert callable(pamssw.run_posterior_ssw)
    assert callable(pamssw.run_posterior_ls_ssw)
    assert callable(pamssw.exploration.run_posterior_ssw)
    assert callable(pamssw.exploration.run_posterior_ls_ssw)
    assert pamssw.run_ssw is run_ssw is pamssw.runner.run_ssw
    assert pamssw.run_ls_ssw is run_ls_ssw is pamssw.runner.run_ls_ssw
