import numpy as np
import pytest
from dataclasses import FrozenInstanceError

from pamssw.accounting import BudgetExceeded, EvalCounter, EvaluationCounts, EvaluationPurpose
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


def test_evaluation_counts_from_mapping_uses_canonical_purpose_order():
    counts = EvaluationCounts.from_mapping(
        {
            EvaluationPurpose.DIRECTION_ORACLE: 2,
            EvaluationPurpose.LANDING_TRUE_QUENCH: 3,
        }
    )

    assert counts.total == 5
    assert counts.count(EvaluationPurpose.DIRECTION_ORACLE) == 2
    assert counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
    assert counts.as_dict()["landing_true_quench"] == 3


def test_evaluation_counts_unattributed_assigns_only_unattributed_total():
    counts = EvaluationCounts.unattributed(4)

    assert counts.total == 4
    assert counts.count(EvaluationPurpose.UNATTRIBUTED) == 4
    assert counts.as_dict() == {
        purpose.value: 4 if purpose is EvaluationPurpose.UNATTRIBUTED else 0
        for purpose in EvaluationPurpose
    }


def test_evaluation_counts_zero_is_immutable_and_canonical():
    counts = EvaluationCounts.zero()

    assert counts.values == (0,) * len(EvaluationPurpose)
    assert counts.total == 0
    assert all(counts.count(purpose) == 0 for purpose in EvaluationPurpose)
    with pytest.raises(FrozenInstanceError):
        counts.values = (1,) * len(EvaluationPurpose)


@pytest.mark.parametrize(
    ("values", "error"),
    [
        ((0,) * 7, ValueError),
        ((0,) * 7 + (True,), TypeError),
        ((0,) * 7 + (-1,), ValueError),
    ],
)
def test_evaluation_counts_reject_invalid_values(values: tuple[int, ...], error: type[Exception]):
    with pytest.raises(error):
        EvaluationCounts(values)


@pytest.mark.parametrize(
    "mapping",
    [
        {"unknown": 1},
        {EvaluationPurpose.DIRECTION_ORACLE: True},
        {EvaluationPurpose.DIRECTION_ORACLE: -1},
        {
            EvaluationPurpose.DIRECTION_ORACLE: 1,
            "direction_oracle": 2,
        },
    ],
)
def test_evaluation_counts_from_mapping_rejects_invalid_entries(mapping):
    with pytest.raises((TypeError, ValueError)):
        EvaluationCounts.from_mapping(mapping)


def test_evaluation_counts_from_mapping_does_not_alias_input_mapping():
    source = {"direction_oracle": 2}

    counts = EvaluationCounts.from_mapping(source)
    source["direction_oracle"] = 99

    assert counts.count(EvaluationPurpose.DIRECTION_ORACLE) == 2


def test_purpose_counts_started_delegated_failure(state: State):
    counter = EvalCounter(RecordingCalculator(fail_on_call=1))

    with counter.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        with pytest.raises(RuntimeError, match="calculator failed"):
            counter.evaluate(state)

    counts = counter.snapshot()
    assert counts.total == 1
    assert counts.count(EvaluationPurpose.DIRECTION_ORACLE) == 1


def test_purpose_budget_rejection_does_not_increment_counts(state: State):
    calculator = RecordingCalculator()
    counter = EvalCounter(calculator, max_force_evals=0)

    with counter.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        with pytest.raises(BudgetExceeded):
            counter.evaluate(state)

    counts = counter.snapshot()
    assert calculator.calls == 0
    assert counts.total == 0
    assert counts.count(EvaluationPurpose.DIRECTION_ORACLE) == 0


def test_nested_purpose_restores_outer_purpose_after_inner_exception(state: State):
    counter = EvalCounter(RecordingCalculator(fail_on_call=1))

    with counter.purpose(EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH):
        with pytest.raises(RuntimeError, match="calculator failed"):
            with counter.purpose(EvaluationPurpose.DIRECTION_ORACLE):
                counter.evaluate(state)
        assert counter.snapshot().count(EvaluationPurpose.DIRECTION_ORACLE) == 1
        counter.evaluate_flat(state.flatten_positions(), state)

    counts = counter.snapshot()
    assert counts.count(EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH) == 1
    assert counts.count(EvaluationPurpose.DIRECTION_ORACLE) == 1


def test_purpose_counts_successful_evaluate_and_evaluate_flat(state: State):
    counter = EvalCounter(RecordingCalculator())

    with counter.purpose(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK):
        counter.evaluate(state)
    with counter.purpose(EvaluationPurpose.LANDING_TRUE_QUENCH):
        counter.evaluate_flat(state.flatten_positions(), state)

    counts = counter.snapshot()
    assert counts.total == 2
    assert counts.count(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK) == 1
    assert counts.count(EvaluationPurpose.LANDING_TRUE_QUENCH) == 1


def test_unscoped_evaluations_use_unattributed_purpose(state: State):
    counter = EvalCounter(RecordingCalculator())

    counter.evaluate_flat(state.flatten_positions(), state)

    assert counter.snapshot().count(EvaluationPurpose.UNATTRIBUTED) == 1


def test_purpose_rejects_non_enum_value(state: State):
    counter = EvalCounter(RecordingCalculator())

    with pytest.raises(TypeError):
        with counter.purpose("direction_oracle"):
            counter.evaluate(state)


def test_snapshot_is_fresh_and_immutable(state: State):
    counter = EvalCounter(RecordingCalculator())
    before = counter.snapshot()

    counter.evaluate(state)

    assert before.total == 0
    with pytest.raises(FrozenInstanceError):
        before.values = (1,) * len(EvaluationPurpose)
    with pytest.raises(TypeError):
        before.values[0] = 1


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
