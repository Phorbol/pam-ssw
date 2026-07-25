import numpy as np
import pytest

from pamssw.accounting import BudgetExceeded, EvalCounter
from pamssw.calculators import EnergyResult
from pamssw.state import State


class RecordingCalculator:
    def __init__(self, fail_on_call: int | None = None) -> None:
        self.fail_on_call = fail_on_call
        self.calls = 0

    def evaluate(self, state: State) -> EnergyResult:
        self._record_call()
        return EnergyResult(energy=1.0, gradient=np.zeros_like(state.positions))

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        self._record_call()
        return 1.0, np.zeros_like(flat_positions)

    def _record_call(self) -> None:
        self.calls += 1
        if self.calls == self.fail_on_call:
            raise RuntimeError("calculator failed")


@pytest.fixture
def state() -> State:
    return State(numbers=np.array([1]), positions=np.zeros((1, 3)))


def _evaluate(counter: EvalCounter, method: str, state: State) -> None:
    if method == "evaluate":
        counter.evaluate(state)
    else:
        counter.evaluate_flat(state.flatten_positions(), state)


@pytest.mark.parametrize("method", ["evaluate", "evaluate_flat"])
def test_started_failing_calculator_call_is_counted(method: str, state: State):
    calculator = RecordingCalculator(fail_on_call=1)
    counter = EvalCounter(calculator)

    with pytest.raises(RuntimeError, match="calculator failed"):
        _evaluate(counter, method, state)

    assert calculator.calls == 1
    assert counter.force_evaluations == 1
    assert counter.energy_evaluations == 1


def test_evaluate_and_evaluate_flat_share_one_force_evaluation_budget(state: State):
    calculator = RecordingCalculator()
    counter = EvalCounter(calculator, max_force_evals=2)

    counter.evaluate(state)
    counter.evaluate_flat(state.flatten_positions(), state)

    assert calculator.calls == 2
    assert counter.force_evaluations == 2
    assert counter.energy_evaluations == 2
    with pytest.raises(BudgetExceeded):
        counter.evaluate(state)


@pytest.mark.parametrize("method", ["evaluate", "evaluate_flat"])
def test_budget_rejection_does_not_delegate_or_increment(method: str, state: State):
    calculator = RecordingCalculator()
    counter = EvalCounter(calculator, max_force_evals=1)
    counter.evaluate_flat(state.flatten_positions(), state)

    with pytest.raises(BudgetExceeded):
        _evaluate(counter, method, state)

    assert calculator.calls == 1
    assert counter.force_evaluations == 1
    assert counter.energy_evaluations == 1
