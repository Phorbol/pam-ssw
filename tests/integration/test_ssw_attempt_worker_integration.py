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


def _ls_known_basin_config() -> LSSSWConfig:
    return replace(_ls_base_config(), local_softening_strength=0.01)


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
        [-0.9999728922495554, 2.1110387920091675e-18, 3.735014415926559e-17],
        [1.4620162511082853, -8.759543837714597e-18, -4.806260488024458e-17],
        [-3.197519170442773e-18, 1.0, -2.485148728762242e-17],
        [1.0260346832093245e-16, -2.4851487287622408e-17, 1.0],
    ]
)
_LS_SSW_DUPLICATE_LANDING = np.array(
    [
        [-0.9999923871678221, 5.325406716043512e-18, 4.41347816547278e-17],
        [1.3405858236332273, -7.62222911491232e-18, -4.4565188455938064e-17],
        [-3.317117156888655e-18, 1.0, -2.5781016613751053e-17],
        [1.064411836118929e-16, -2.5781016613751053e-17, 1.0],
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
    (
        "config_factory",
        "softening_enabled",
        "force_evaluations",
        "direction_oracle_evaluations",
        "escape_true_pes_evaluations",
        "landing_energy",
        "landing_positions",
    ),
    [
        (
            _base_config,
            False,
            13,
            6,
            2,
            1.0750373417042004e-34,
            _SSW_COMPLETED_LANDING,
        ),
        (
            _ls_base_config,
            True,
            56,
            6,
            2,
            2.93924085883713e-09,
            _LS_SSW_COMPLETED_LANDING,
        ),
    ],
    ids=("ssw", "ls_ssw"),
)
def test_real_worker_purpose_ledger_preserves_completed_analytic_baselines(
    config_factory,
    softening_enabled,
    force_evaluations,
    direction_oracle_evaluations,
    escape_true_pes_evaluations,
    landing_energy,
    landing_positions,
):
    config = config_factory()
    worker, calculators = _worker_with_fresh_calculators(config, softening_enabled=softening_enabled)
    action = _action(force_budget=400)

    result = worker(action, _state())

    assert len(calculators) == 1
    diagnostics = worker.diagnostics_snapshot()
    assert len(diagnostics) == 1
    assert diagnostics[0].action_id == action.action_id
    diagnostic_stats = dict(diagnostics[0].stats)
    assert diagnostic_stats["proposal_optimizer"] == config.proposal_optimizer
    assert diagnostic_stats["proposal_relax_count"] >= 1
    assert diagnostic_stats["force_evaluations"] == result.force_evaluations
    assert result.evaluation_counts.count(EvaluationPurpose.DIRECTION_ORACLE) == direction_oracle_evaluations
    assert result.evaluation_counts.count(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK) == escape_true_pes_evaluations
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


@pytest.mark.parametrize(
    "proposal_optimizer",
    ["ase-fire2", "safe-lbfgs-total", "bias-separated-lbfgs"],
)
def test_new_proposal_optimizers_preserve_exact_analytic_worker_ledger(proposal_optimizer):
    config = replace(
        _base_config(),
        proposal_optimizer=proposal_optimizer,
        proposal_trust_radius=None,
    )
    worker, calculators = _worker_with_fresh_calculators(config)

    result = worker(_action(force_budget=400), _state())

    _assert_terminal_baseline(
        result,
        calculators[0],
        status=AttemptStatus.COMPLETED,
        force_evaluations=13,
        landing_energy=1.0750373417042004e-34,
        landing_positions=_SSW_COMPLETED_LANDING,
    )
    _assert_closed_physical_ledger(result)
    assert result.evaluation_counts.count(EvaluationPurpose.BIASED_PROPOSAL_RELAX) == 1


@pytest.mark.parametrize(
    ("config_factory", "softening_enabled", "force_evaluations", "landing_energy", "landing_positions"),
    [
        (_base_config, False, 13, 0.0, _state().positions),
        (_ls_known_basin_config, True, 45, 2.3181909026442303e-10, _LS_SSW_DUPLICATE_LANDING),
    ],
    ids=("ssw", "ls_ssw"),
)
def test_real_worker_purpose_ledger_preserves_duplicate_candidate_baseline(
    config_factory,
    softening_enabled,
    force_evaluations,
    landing_energy,
    landing_positions,
):
    worker, calculators = _worker_with_fresh_calculators(
        replace(config_factory(), dedup_rmsd_tol=10.0),
        softening_enabled=softening_enabled,
    )

    result = worker(_action(force_budget=400), _state())

    _assert_terminal_baseline(
        result,
        calculators[0],
        status=AttemptStatus.COMPLETED,
        force_evaluations=force_evaluations,
        landing_energy=landing_energy,
        landing_positions=landing_positions,
    )
    _assert_closed_physical_ledger(result)


@pytest.mark.parametrize(
    ("config_factory", "softening_enabled", "force_evaluations"),
    [
        (_base_config, False, 13),
        (_ls_base_config, True, 56),
    ],
    ids=("ssw", "ls_ssw"),
)
def test_real_worker_purpose_ledger_preserves_fragmented_baseline(
    config_factory,
    softening_enabled,
    force_evaluations,
):
    worker, calculators = _worker_with_fresh_calculators(
        replace(config_factory(), fragment_guard_factor=1.01),
        softening_enabled=softening_enabled,
    )

    result = worker(_action(force_budget=400), _state())

    _assert_terminal_baseline(
        result,
        calculators[0],
        status=AttemptStatus.FRAGMENTED,
        force_evaluations=force_evaluations,
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


@pytest.mark.parametrize(
    ("config_factory", "softening_enabled"),
    [
        (_base_config, False),
        (_ls_base_config, True),
    ],
    ids=("ssw", "ls_ssw"),
)
def test_real_worker_purpose_ledger_preserves_exact_budget_exhaustion_baseline(
    config_factory,
    softening_enabled,
):
    worker, calculators = _worker_with_fresh_calculators(
        config_factory(),
        softening_enabled=softening_enabled,
    )

    result = worker(_action(force_budget=5), _state())

    _assert_terminal_baseline(
        result,
        calculators[0],
        status=AttemptStatus.BUDGET_EXHAUSTED,
        force_evaluations=5,
        landing_energy=None,
        landing_positions=None,
    )
    _assert_closed_physical_ledger(result)
    diagnostics = worker.diagnostics_snapshot()
    assert len(diagnostics) == 1
    assert dict(diagnostics[0].stats)["force_evaluations"] == 5


@pytest.mark.parametrize(
    ("config_factory", "softening_enabled"),
    [
        (_base_config, False),
        (_ls_base_config, True),
    ],
    ids=("ssw", "ls_ssw"),
)
def test_real_worker_purpose_ledger_preserves_exact_worker_error_baseline(
    config_factory,
    softening_enabled,
):
    worker, calculators = _worker_with_fresh_calculators(
        config_factory(),
        fail_on_call=3,
        softening_enabled=softening_enabled,
    )

    result = worker(_action(force_budget=400), _state())

    _assert_terminal_baseline(
        result,
        calculators[0],
        status=AttemptStatus.WORKER_ERROR,
        force_evaluations=3,
        landing_energy=None,
        landing_positions=None,
    )
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


@pytest.mark.parametrize(
    ("config_factory", "softening_enabled"),
    [
        (_base_config, False),
        (_ls_base_config, True),
    ],
    ids=("ssw", "ls_ssw"),
)
def test_pre_calculator_invalid_starter_has_a_zero_purpose_ledger(
    config_factory,
    softening_enabled,
):
    worker, calculators = _worker_with_fresh_calculators(
        config_factory(),
        softening_enabled=softening_enabled,
    )
    invalid_state = _state()
    invalid_state.positions[1] = invalid_state.positions[0]

    result = worker(_action(force_budget=400), invalid_state)

    assert calculators == []
    assert result.status is AttemptStatus.INVALID
    assert result.force_evaluations == 0
    assert result.evaluation_counts.total == 0
    assert all(result.evaluation_counts.count(purpose) == 0 for purpose in EvaluationPurpose)
