import numpy as np
import pytest

from pamssw import SSWConfig, State, run_ssw
from pamssw.accounting import BudgetExceeded, EvaluationPurpose
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import DoubleWell2D
from pamssw.walker import SurfaceWalker


class CountingAnalyticCalculator:
    def __init__(self, calculator: AnalyticCalculator, fail_on_evaluate_call: int | None = None) -> None:
        self.calculator = calculator
        self.fail_on_evaluate_call = fail_on_evaluate_call
        self.evaluate_calls = 0
        self.evaluate_flat_calls = 0
        self.failed_evaluate_calls = 0

    @property
    def total_calls(self) -> int:
        return self.evaluate_calls + self.evaluate_flat_calls

    def evaluate(self, state):
        self.evaluate_calls += 1
        if self.evaluate_calls == self.fail_on_evaluate_call:
            self.failed_evaluate_calls += 1
            raise RuntimeError("synthetic probe evaluation failure")
        return self.calculator.evaluate(state)

    def evaluate_flat(self, flat_positions, template):
        self.evaluate_flat_calls += 1
        return self.calculator.evaluate_flat(flat_positions, template)


def test_fresh_walker_bootstrap_true_quench_has_closed_purpose_ledger():
    calculator = CountingAnalyticCalculator(AnalyticCalculator(DoubleWell2D()))
    walker = SurfaceWalker(
        calculator=calculator,
        config=SSWConfig(max_trials=1, max_steps_per_walk=1, oracle_candidates=1, rng_seed=4),
        softening_enabled=False,
    )

    result = walker.relax_true_minimum(
        State(numbers=np.array([1]), positions=np.array([[-1.0, 0.0, 0.0]])),
        quench_purpose=EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH,
    )

    counts = walker.calculator.snapshot()
    assert result.energy == pytest.approx(0.0, abs=1e-12)
    assert counts.count(EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH) > 0
    assert counts.count(EvaluationPurpose.POST_RELAX_VALIDATION) == 1
    assert counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
    assert counts.total == calculator.total_calls == walker.calculator.force_evaluations


def test_surface_walker_run_labels_initial_raw_state_quench_as_bootstrap():
    calculator = CountingAnalyticCalculator(AnalyticCalculator(DoubleWell2D()))
    walker = SurfaceWalker(
        calculator=calculator,
        config=SSWConfig(
            max_trials=1,
            max_steps_per_walk=1,
            oracle_candidates=1,
            rng_seed=4,
        ),
        softening_enabled=False,
    )

    walker.run(
        State(
            numbers=np.array([1]),
            positions=np.array([[-1.0, 0.0, 0.0]]),
        )
    )

    counts = walker.calculator.snapshot()
    assert counts.count(EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH) > 0
    assert counts.count(EvaluationPurpose.STARTER_TRUE_QUENCH) == 0
    assert counts.count(EvaluationPurpose.LANDING_TRUE_QUENCH) > 0
    assert counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
    assert counts.total == calculator.total_calls


def test_ssw_local_relaxation_accounting_is_exact():
    result = run_ssw(
        initial_state=State(
            numbers=np.ones(4, dtype=int),
            positions=np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(
            max_trials=3,
            max_steps_per_walk=2,
            rng_seed=3,
        ),
    )

    assert result.stats["n_trials"] == 3
    assert result.stats["local_relaxations"] == 1 + 3
    assert len(result.walk_history) == 3
    assert result.stats["coordinate_system"] == "cartesian_fixed_cell"
    assert result.stats["variable_cell_supported"] == 0
    assert result.stats["force_evaluations"] > 0
    assert result.stats["energy_evaluations"] == result.stats["force_evaluations"]


def test_force_evaluation_accounting_matches_wrapped_calculator_calls():
    calculator = AnalyticCalculator(DoubleWell2D())
    result = run_ssw(
        initial_state=State(numbers=np.array([1]), positions=np.array([[-1.0, 0.0, 0.0]])),
        calculator=calculator,
        config=SSWConfig(max_trials=2, max_steps_per_walk=2, oracle_candidates=2, rng_seed=4),
    )

    counter = result.stats["force_evaluations"]

    assert isinstance(counter, int)
    assert counter > result.stats["local_relaxations"]


def test_surface_walker_accounts_direction_probe_evaluations():
    calculator = CountingAnalyticCalculator(AnalyticCalculator(DoubleWell2D()))
    config = SSWConfig(
        max_trials=1,
        max_steps_per_walk=1,
        oracle_candidates=3,
        direction_probe_enabled=True,
        direction_probe_top_k=2,
        max_force_evals=80,
        rng_seed=4,
    )
    walker = SurfaceWalker(calculator=calculator, config=config, softening_enabled=False)
    probe_call_counts = []
    probe_refine = walker.oracle._probe_refine

    def record_probe_calls(*args, **kwargs):
        calls_before = calculator.total_calls
        result = probe_refine(*args, **kwargs)
        probe_call_counts.append(calculator.total_calls - calls_before)
        return result

    walker.oracle._probe_refine = record_probe_calls
    result = walker.run(State(numbers=np.array([1]), positions=np.array([[-1.0, 0.0, 0.0]])))

    assert probe_call_counts == [config.direction_probe_top_k + 1]
    assert result.stats["force_evaluations"] == calculator.total_calls
    assert walker.oracle.calculator is walker.calculator
    assert result.stats["force_evaluations"] <= config.max_force_evals


def test_surface_walker_propagates_direction_probe_budget_exhaustion():
    calculator = CountingAnalyticCalculator(AnalyticCalculator(DoubleWell2D()))
    config = SSWConfig(
        max_steps_per_walk=1,
        oracle_candidates=1,
        direction_probe_enabled=True,
        direction_probe_top_k=1,
        max_force_evals=1,
        rng_seed=4,
    )
    walker = SurfaceWalker(calculator=calculator, config=config, softening_enabled=False)
    walker._reset_direction_stats()
    recorded_choices = []
    record_direction_choice = walker._record_direction_choice

    def record_choice(choice):
        recorded_choices.append(choice)
        record_direction_choice(choice)

    def fail_if_choice_continues(*args, **kwargs):
        raise AssertionError("direction choice was recorded after probe budget exhaustion")

    walker.oracle._directional_hvp = lambda state, proposal, direction: np.zeros_like(direction)
    walker._record_direction_choice = record_choice
    walker._true_directional_curvature = fail_if_choice_continues

    with pytest.raises(BudgetExceeded, match="force-evaluation budget exhausted"):
        walker._walk_candidate_from_seed(State(numbers=np.array([1]), positions=np.array([[-1.0, 0.0, 0.0]])))

    assert recorded_choices == []
    assert walker._direction_choices == 0
    assert walker.calculator.force_evaluations == config.max_force_evals
    assert calculator.total_calls == walker.calculator.force_evaluations


def test_surface_walker_skips_ordinary_direction_probe_failures():
    calculator = CountingAnalyticCalculator(AnalyticCalculator(DoubleWell2D()), fail_on_evaluate_call=2)
    config = SSWConfig(
        max_trials=1,
        max_steps_per_walk=1,
        oracle_candidates=2,
        direction_probe_enabled=True,
        direction_probe_top_k=2,
        max_force_evals=80,
        rng_seed=4,
    )
    walker = SurfaceWalker(calculator=calculator, config=config, softening_enabled=False)
    probe_evaluate_calls = []
    probe_refine = walker.oracle._probe_refine

    def record_probe_evaluations(*args, **kwargs):
        calls_before = calculator.evaluate_calls
        result = probe_refine(*args, **kwargs)
        probe_evaluate_calls.append(calculator.evaluate_calls - calls_before)
        return result

    walker.oracle._probe_refine = record_probe_evaluations
    result = walker.run(State(numbers=np.array([1]), positions=np.array([[-1.0, 0.0, 0.0]])))

    assert calculator.failed_evaluate_calls == 1
    assert probe_evaluate_calls == [config.direction_probe_top_k + 1]
    assert result.stats["direction_choices"] == 1
    assert result.stats["force_evaluations"] == calculator.total_calls
    assert result.stats["force_evaluations"] <= config.max_force_evals


def test_force_evaluation_budget_limits_started_trials():
    result = run_ssw(
        initial_state=State(numbers=np.array([1]), positions=np.array([[-1.0, 0.0, 0.0]])),
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=20, max_steps_per_walk=2, oracle_candidates=2, max_force_evals=80, rng_seed=4),
    )

    assert result.stats["force_evaluations"] <= 80
    assert result.stats["n_trials"] < 20


def test_default_proposal_pool_uses_only_one_ssw_walk_for_cluster():
    state = State(
        numbers=np.full(38, 18),
        positions=np.column_stack(
            [
                np.arange(38, dtype=float) * 0.8,
                np.zeros(38),
                np.zeros(38),
            ]
        ),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, rng_seed=5),
        softening_enabled=False,
    )
    result = run_ssw(state, AnalyticCalculator(DoubleWell2D()), SSWConfig(max_trials=1, rng_seed=5))

    labels = [proposal.label for proposal in walker._proposal_pool(result.archive.entries[0].state, result.archive, 0)]

    assert labels == ["ssw_walk"]


def test_configured_proposal_pool_uses_multiple_ssw_walks_only():
    state = State(
        numbers=np.full(8, 18),
        positions=np.column_stack(
            [
                np.arange(8, dtype=float) * 0.8,
                np.zeros(8),
                np.zeros(8),
            ]
        ),
    )
    result = run_ssw(state, AnalyticCalculator(DoubleWell2D()), SSWConfig(max_trials=1, rng_seed=8))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, proposal_pool_size=3, rng_seed=8),
        softening_enabled=False,
    )

    labels = [proposal.label for proposal in walker._proposal_pool(result.archive.entries[0].state, result.archive, 0)]

    assert labels == ["ssw_walk", "ssw_walk", "ssw_walk"]


def test_periodic_state_uses_only_ssw_walk_proposal_by_default():
    state = State(
        numbers=np.full(4, 18),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        cell=np.eye(3) * 6.0,
        pbc=(True, True, True),
    )
    result = run_ssw(state, AnalyticCalculator(DoubleWell2D()), SSWConfig(max_trials=1, rng_seed=7))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, rng_seed=7),
        softening_enabled=False,
    )

    proposals = walker._proposal_pool(result.archive.entries[0].state, result.archive, 0)

    assert [proposal.label for proposal in proposals] == ["ssw_walk"]
