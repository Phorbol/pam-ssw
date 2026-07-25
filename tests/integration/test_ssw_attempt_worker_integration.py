from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pamssw.accounting import EvaluationPurpose
from pamssw.calculators import AnalyticCalculator
from pamssw.config import LSSSWConfig, SSWConfig
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


def _ls_base_config() -> LSSSWConfig:
    return LSSSWConfig(
        max_trials=7,
        max_steps_per_walk=1,
        oracle_candidates=2,
        direction_probe_enabled=True,
        direction_probe_top_k=1,
        max_force_evals=999,
        rng_seed=19,
        local_softening_mode="manual",
        local_softening_pairs=[(0, 1)],
    )


def _worker_with_fresh_calculators(
    config: SSWConfig,
    *,
    fail_on_call: int | None = None,
    softening_enabled: bool = False,
):
    calculators: list[RecordingAnalyticCalculator] = []

    def factory() -> RecordingAnalyticCalculator:
        calculator = RecordingAnalyticCalculator(fail_on_call=fail_on_call)
        calculators.append(calculator)
        return calculator

    return SSWAttemptWorker(factory, config, softening_enabled=softening_enabled), calculators


_SSW_COMPLETED_LANDING = np.array(
    [
        [-1.0, -1.4554479271314136e-17, -2.5197624815554368e-18],
        [5.6914139176231935e-18, -0.21213203435596426, 1.1279032098637582e-19],
        [-1.444168895032776e-17, 1.2121320343559643, 2.2941818395826863e-18],
        [3.058861115081377e-18, 3.3972320780405076e-18, 1.0],
    ]
)
_LS_SSW_COMPLETED_LANDING = np.array(
    [
        [-1.000022482285951, 6.456599136324924e-06, -4.048361096694965e-19],
        [0.8817682755669265, -0.4141983845138378, 2.4491929225442876e-18],
        [-1.444168895032776e-17, 1.2121320343559643, 2.2941818395826863e-18],
        [3.058861115081377e-18, 3.3972320780405076e-18, 1.0],
    ]
)


def _assert_terminal_baseline(
    result,
    raw_calculator: RecordingAnalyticCalculator,
    *,
    status: AttemptStatus,
    force_evaluations: int,
    landing_energy: float | None,
    landing_positions: np.ndarray | None,
) -> None:
    assert result.status is status
    assert result.force_evaluations == raw_calculator.calls == force_evaluations
    assert result.evaluation_counts.total == raw_calculator.calls
    assert raw_calculator.calls == raw_calculator.evaluate_calls + raw_calculator.evaluate_flat_calls
    if landing_energy is None:
        assert result.landing_energy is None
        assert result.landing_state is None
    else:
        assert result.landing_energy == pytest.approx(landing_energy, abs=1e-12)
        assert result.landing_state is not None
        np.testing.assert_allclose(result.landing_state.positions, landing_positions, rtol=0.0, atol=1e-12)


def _assert_closed_physical_ledger(result) -> None:
    assert result.evaluation_counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
    assert result.evaluation_counts.total == result.force_evaluations


@pytest.mark.parametrize(
    ("config_factory", "softening_enabled", "force_evaluations", "landing_energy", "landing_positions"),
    [
        (
            _base_config,
            False,
            21,
            1.0750373417042004e-34,
            _SSW_COMPLETED_LANDING,
        ),
        (
            _ls_base_config,
            True,
            69,
            2.0427020177514484e-09,
            _LS_SSW_COMPLETED_LANDING,
        ),
    ],
    ids=("ssw", "ls_ssw"),
)
def test_real_worker_purpose_ledger_preserves_completed_analytic_baselines(
    config_factory,
    softening_enabled,
    force_evaluations,
    landing_energy,
    landing_positions,
):
    config = config_factory()
    worker, calculators = _worker_with_fresh_calculators(config, softening_enabled=softening_enabled)
    action = _action(force_budget=400)

    result = worker(action, _state())

    assert len(calculators) == 1
    _assert_terminal_baseline(
        result,
        calculators[0],
        status=AttemptStatus.COMPLETED,
        force_evaluations=force_evaluations,
        landing_energy=landing_energy,
        landing_positions=landing_positions,
    )
    _assert_closed_physical_ledger(result)
    for purpose in (
        EvaluationPurpose.STARTER_TRUE_QUENCH,
        EvaluationPurpose.DIRECTION_ORACLE,
        EvaluationPurpose.ESCAPE_TRUE_PES_CHECK,
        EvaluationPurpose.BIASED_PROPOSAL_RELAX,
        EvaluationPurpose.LANDING_TRUE_QUENCH,
        EvaluationPurpose.POST_RELAX_VALIDATION,
    ):
        assert result.evaluation_counts.count(purpose) > 0
    assert config.max_trials == 7
    assert config.rng_seed == 19
    assert config.max_force_evals == 999


def test_real_worker_purpose_ledger_preserves_duplicate_candidate_baseline():
    worker, calculators = _worker_with_fresh_calculators(replace(_base_config(), dedup_rmsd_tol=10.0))

    result = worker(_action(force_budget=400), _state())

    _assert_terminal_baseline(
        result,
        calculators[0],
        status=AttemptStatus.COMPLETED,
        force_evaluations=21,
        landing_energy=0.0,
        landing_positions=_state().positions,
    )
    _assert_closed_physical_ledger(result)


def test_real_worker_purpose_ledger_preserves_fragmented_baseline():
    worker, calculators = _worker_with_fresh_calculators(
        replace(_base_config(), fragment_guard_factor=1.01)
    )

    result = worker(_action(force_budget=400), _state())

    _assert_terminal_baseline(
        result,
        calculators[0],
        status=AttemptStatus.FRAGMENTED,
        force_evaluations=21,
        landing_energy=None,
        landing_positions=None,
    )
    _assert_closed_physical_ledger(result)


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


def test_real_worker_purpose_ledger_preserves_exact_budget_exhaustion_baseline():
    worker, calculators = _worker_with_fresh_calculators(_base_config())

    result = worker(_action(force_budget=5), _state())

    assert result.status is AttemptStatus.BUDGET_EXHAUSTED
    assert result.force_evaluations == calculators[0].calls == 5
    assert calculators[0].calls == calculators[0].evaluate_calls + calculators[0].evaluate_flat_calls
    _assert_closed_physical_ledger(result)


def test_real_worker_purpose_ledger_preserves_exact_worker_error_baseline():
    worker, calculators = _worker_with_fresh_calculators(_base_config(), fail_on_call=3)

    result = worker(_action(force_budget=400), _state())

    assert result.status is AttemptStatus.WORKER_ERROR
    assert result.force_evaluations == calculators[0].calls == 3
    assert calculators[0].calls == calculators[0].evaluate_calls + calculators[0].evaluate_flat_calls
    assert "synthetic calculator failure" in result.failure_reason
    _assert_closed_physical_ledger(result)


def test_post_relax_validation_purpose_ledger_preserves_worker_error_cost(monkeypatch):
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
    _assert_closed_physical_ledger(result)


def test_pre_calculator_invalid_starter_has_a_zero_purpose_ledger():
    worker, calculators = _worker_with_fresh_calculators(_base_config())
    invalid_state = _state()
    invalid_state.positions[1] = invalid_state.positions[0]

    result = worker(_action(force_budget=400), invalid_state)

    assert calculators == []
    assert result.status is AttemptStatus.INVALID
    assert result.force_evaluations == 0
    assert result.evaluation_counts.total == 0
    assert all(result.evaluation_counts.count(purpose) == 0 for purpose in EvaluationPurpose)
