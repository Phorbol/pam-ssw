from __future__ import annotations

import numpy as np

from pamssw.calculators import AnalyticCalculator
from pamssw.config import SSWConfig
from pamssw.exploration import SSWAttemptWorker
from pamssw.exploration.actions import AttemptStatus, StarterAction
from pamssw.potentials import DoubleWell2D
from pamssw.result import RelaxResult
from pamssw.state import State


class RecordingAnalyticCalculator:
    """Real analytic calculator with raw calls recorded before delegation."""

    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self.calculator = AnalyticCalculator(DoubleWell2D())
        self.fail_on_call = fail_on_call
        self.calls = 0
        self.evaluate_calls = 0
        self.evaluate_flat_calls = 0

    def _record_call(self) -> None:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("synthetic calculator failure")

    def evaluate(self, state: State):
        self.evaluate_calls += 1
        self._record_call()
        return self.calculator.evaluate(state)

    def evaluate_flat(self, flat_positions: np.ndarray, template: State):
        self.evaluate_flat_calls += 1
        self._record_call()
        return self.calculator.evaluate_flat(flat_positions, template)


def _state() -> State:
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


def _action(*, force_budget: int, random_seed: int = 31) -> StarterAction:
    return StarterAction(
        action_id="batch-00000001-slot-0000",
        batch_id=1,
        slot_id=0,
        policy_name="uniform",
        policy_version=1,
        archive_version=2,
        starter_id=0,
        selection_probability=1.0,
        random_seed=random_seed,
        force_budget=force_budget,
    )


def _base_config() -> SSWConfig:
    return SSWConfig(
        max_trials=7,
        max_steps_per_walk=1,
        oracle_candidates=2,
        direction_probe_enabled=True,
        direction_probe_top_k=1,
        max_force_evals=999,
        rng_seed=19,
    )


def _worker_with_fresh_calculators(config: SSWConfig, *, fail_on_call: int | None = None):
    calculators: list[RecordingAnalyticCalculator] = []

    def factory() -> RecordingAnalyticCalculator:
        calculator = RecordingAnalyticCalculator(fail_on_call=fail_on_call)
        calculators.append(calculator)
        return calculator

    return SSWAttemptWorker(factory, config), calculators


def test_real_worker_uses_one_analytic_calculator_and_reports_exact_cost_within_budget():
    config = _base_config()
    worker, calculators = _worker_with_fresh_calculators(config)
    action = _action(force_budget=400)

    result = worker(action, _state())

    assert len(calculators) == 1
    assert result.force_evaluations == calculators[0].calls <= action.force_budget
    assert calculators[0].calls == calculators[0].evaluate_calls + calculators[0].evaluate_flat_calls
    assert result.status is AttemptStatus.COMPLETED
    assert config.max_trials == 7
    assert config.rng_seed == 19
    assert config.max_force_evals == 999


def test_real_worker_repeats_status_cost_and_landing_for_the_same_action_seed():
    action = _action(force_budget=400)
    first_worker, first_calculators = _worker_with_fresh_calculators(_base_config())
    second_worker, second_calculators = _worker_with_fresh_calculators(_base_config())

    first = first_worker(action, _state())
    second = second_worker(action, _state())

    assert first.status is second.status
    assert first.force_evaluations == second.force_evaluations
    assert first.force_evaluations == first_calculators[0].calls
    assert second.force_evaluations == second_calculators[0].calls
    assert first_calculators[0].calls == (
        first_calculators[0].evaluate_calls + first_calculators[0].evaluate_flat_calls
    )
    assert second_calculators[0].calls == (
        second_calculators[0].evaluate_calls + second_calculators[0].evaluate_flat_calls
    )
    assert first.landing_energy == second.landing_energy
    if first.landing_state is None or second.landing_state is None:
        assert first.landing_state is second.landing_state is None
    else:
        np.testing.assert_array_equal(first.landing_state.positions, second.landing_state.positions)


def test_real_worker_stops_at_a_small_force_budget_without_overrunning_the_calculator():
    worker, calculators = _worker_with_fresh_calculators(_base_config())

    result = worker(_action(force_budget=5), _state())

    assert result.status is AttemptStatus.BUDGET_EXHAUSTED
    assert result.force_evaluations == calculators[0].calls == 5
    assert calculators[0].calls == calculators[0].evaluate_calls + calculators[0].evaluate_flat_calls


def test_real_worker_preserves_the_cost_of_a_failing_analytic_calculator_call():
    worker, calculators = _worker_with_fresh_calculators(_base_config(), fail_on_call=3)

    result = worker(_action(force_budget=400), _state())

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == calculators[0].calls == 3
    assert calculators[0].calls == calculators[0].evaluate_calls + calculators[0].evaluate_flat_calls
    assert "synthetic calculator failure" in result.failure_reason


def test_post_relax_validation_calculator_failure_is_a_worker_error_with_exact_cost(monkeypatch):
    relaxed_states: list[State] = []

    class ImmediateRelaxer:
        def __init__(self, evaluator, optimizer) -> None:
            self.evaluator = evaluator
            self.optimizer = optimizer

        def relax(self, state, fmax, maxiter, trajectory_callback=None, trajectory_stride=1):
            relaxed_states.append(state)
            return RelaxResult(state=state, energy=0.0, gradient_norm=0.0, n_iter=0)

    monkeypatch.setattr("pamssw.walker.Relaxer", ImmediateRelaxer)
    worker, calculators = _worker_with_fresh_calculators(_base_config(), fail_on_call=1)

    result = worker(_action(force_budget=400), _state())

    assert len(relaxed_states) == 1
    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == calculators[0].calls == 1
    assert calculators[0].evaluate_calls == 0
    assert calculators[0].evaluate_flat_calls == 1
    assert result.failure_reason == "run_error: RuntimeError: synthetic calculator failure"
