import numpy as np
import pytest

from pamssw.accounting import BudgetExceeded, EvaluationPurpose
from pamssw.calculators import AnalyticCalculator
from pamssw.config import SSWConfig
from pamssw.exploration.runner import _bootstrap_minimum
from pamssw.potentials import DoubleWell2D
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
