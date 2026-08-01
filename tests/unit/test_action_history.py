from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from pamssw import SSWConfig
from pamssw.accounting import BudgetExceeded, EvaluationPurpose
from pamssw.calculators import AnalyticCalculator
from pamssw.result import (
    ActionRecord,
    RelaxResult,
    UphillStepRecord,
    UphillWalkTrace,
)
from pamssw.state import State
from pamssw.walker import (
    CandidateProposal,
    DirectionCandidateKind,
    DirectionChoice,
    SurfaceWalker,
)


def _step(
    step_index: int,
    energy_before: float,
    energy_after: float,
    *,
    target_eV: float = 0.8,
    direction_oracle_force_evaluations: int = 2,
) -> UphillStepRecord:
    return UphillStepRecord(
        step_index=step_index,
        direction_kind="random",
        target_eV=target_eV,
        true_energy_before_eV=energy_before,
        true_energy_after_eV=energy_after,
        requested_sigma=0.2,
        executed_sigma=0.2,
        base_bias_weight=0.4,
        final_bias_weight=0.4,
        true_curvature=-0.1,
        inner_curvature=-0.2,
        proposal_relax_iterations=3,
        proposal_relax_outcome="useful_progress",
        proposal_relax_termination="maxiter",
        direction_oracle_force_evaluations=direction_oracle_force_evaluations,
        biased_relax_force_evaluations=4,
        true_pes_check_force_evaluations=2,
        displacement_clipped=False,
        step_termination_reason="continued",
    )


def _walk() -> UphillWalkTrace:
    return UphillWalkTrace(
        target_eV=0.8,
        termination_reason="reached_step_cap",
        steps=(
            _step(0, -10.0, -9.6),
            _step(1, -9.6, -9.2),
        ),
    )


def _action(**overrides) -> ActionRecord:
    values = {
        "trial_index": 0,
        "proposal_index": 0,
        "seed_entry_id": 0,
        "seed_energy_eV": -10.0,
        "walk": _walk(),
        "escape_energy_eV": -9.2,
        "landing_energy_eV": -10.2,
        "landing_gradient_norm": 0.04,
        "landing_iterations": 8,
        "landing_converged": True,
        "landing_force_evaluations": 10,
        "accepted_new_basin": True,
        "is_duplicate": False,
        "global_improved": True,
        "status": "accepted",
    }
    values.update(overrides)
    return ActionRecord(**values)


def test_walk_trace_derives_observed_true_pes_heights() -> None:
    trace = _walk()

    assert trace.observed_max_height_eV == pytest.approx(0.8)
    assert trace.observed_terminal_height_eV == pytest.approx(0.8)
    assert trace.target_delivery_ratio == pytest.approx(1.0)


def test_walk_trace_is_immutable() -> None:
    trace = _walk()

    with pytest.raises(FrozenInstanceError):
        trace.target_eV = 1.0  # type: ignore[misc]


@pytest.mark.parametrize("target", [0.0, -0.1])
def test_walk_trace_rejects_non_positive_target(target: float) -> None:
    with pytest.raises(ValueError, match="target_eV must be positive"):
        UphillWalkTrace(target, "reached_step_cap", ())


def test_walk_trace_rejects_unordered_step_indices() -> None:
    with pytest.raises(ValueError, match="ordered from zero"):
        UphillWalkTrace(
            0.8,
            "reached_step_cap",
            (_step(1, -10.0, -9.6), _step(0, -9.6, -9.2)),
        )


def test_step_rejects_negative_purpose_count() -> None:
    with pytest.raises(ValueError, match="force-evaluation counts must be non-negative"):
        _step(0, -10.0, -9.6, direction_oracle_force_evaluations=-1)


def test_action_rejects_escape_energy_inconsistent_with_walk_endpoint() -> None:
    with pytest.raises(ValueError, match="escape_energy_eV must match"):
        _action(escape_energy_eV=-9.0)


def test_action_rejects_simultaneous_new_and_duplicate_outcome() -> None:
    with pytest.raises(ValueError, match="cannot be both new and duplicate"):
        _action(accepted_new_basin=True, is_duplicate=True)


def test_empty_walk_has_no_derived_height() -> None:
    trace = UphillWalkTrace(0.8, "budget_exhausted", ())

    assert trace.observed_max_height_eV is None
    assert trace.observed_terminal_height_eV is None
    assert trace.target_delivery_ratio is None


class _Quadratic:
    def energy_gradient(self, flat_positions, state):
        gradient = np.asarray(flat_positions, dtype=float).copy()
        return 0.5 * float(gradient @ gradient), gradient


def test_walk_trace_reuses_existing_true_pes_checks_without_new_evaluations(
    monkeypatch,
) -> None:
    class DeterministicWalker(SurfaceWalker):
        def _relax_proposal_task(self, task, *, optimizer, trajectory_callback):
            state = task.initial_state
            return RelaxResult(
                state=state,
                energy=0.5 * float(state.flatten_positions() @ state.flatten_positions()),
                gradient_norm=float(np.linalg.norm(state.flatten_positions())),
                n_iter=0,
            )

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    walker = DeterministicWalker(
        calculator=AnalyticCalculator(_Quadratic()),
        config=SSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            direction_curvature_source="inner",
            proposal_relax_steps=1,
            rng_seed=7,
        ),
        softening_enabled=False,
    )
    direction = np.array([1.0, 0.0, 0.0])
    monkeypatch.setattr(
        walker.oracle.generator,
        "generate_initial_direction",
        lambda *args, **kwargs: direction,
    )
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=direction,
            curvature=-0.5,
            true_curvature=-0.5,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(walker, "_build_softening", lambda *args, **kwargs: None)
    before = walker.calculator.snapshot()
    trace_sink = []

    result = walker._walk_candidate_from_seed(
        state,
        step_target=0.8,
        trace_sink=trace_sink,
    )

    after = walker.calculator.snapshot()
    assert len(trace_sink) == 1
    trace = trace_sink[0]
    assert len(trace.steps) == 1
    step = trace.steps[0]
    assert step.step_index == 0
    assert step.true_energy_before_eV == pytest.approx(0.5)
    assert step.true_energy_after_eV == pytest.approx(
        0.5 * float(result.flatten_positions() @ result.flatten_positions())
    )
    assert step.direction_oracle_force_evaluations == 0
    assert step.biased_relax_force_evaluations == 0
    assert step.true_pes_check_force_evaluations == 2
    assert after.total - before.total == 2
    assert after.count(EvaluationPurpose.ESCAPE_TRUE_PES_CHECK) == 2
    assert after.count(EvaluationPurpose.UNATTRIBUTED) == 0


def _search_states() -> tuple[State, State]:
    initial = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    proposal = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
    return initial, proposal


def _run_one_traced_action(monkeypatch, *, outcome: str):
    initial, proposal_state = _search_states()
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(_Quadratic()),
        config=SSWConfig(
            max_trials=1,
            max_steps_per_walk=1,
            proposal_pool_size=1,
            quench_fmax=0.05,
        ),
        softening_enabled=False,
    )
    trace = _walk()
    monkeypatch.setattr(
        walker,
        "_proposal_pool",
        lambda *args, **kwargs: [
            CandidateProposal("test", proposal_state, walk_trace=trace)
        ],
    )
    if outcome == "fragment":
        monkeypatch.setattr(walker, "_is_fragmented_cluster", lambda *args: True)

    def fake_landing(state, trajectory_name=None, *, quench_purpose=None):
        with walker.calculator.purpose(EvaluationPurpose.LANDING_TRUE_QUENCH):
            walker.calculator.evaluate(state)
        if outcome == "budget_exhausted":
            raise BudgetExceeded("synthetic landing stop")
        if outcome == "duplicate":
            return RelaxResult(initial, energy=-10.0, gradient_norm=0.08, n_iter=3)
        return RelaxResult(proposal_state, energy=-10.2, gradient_norm=0.04, n_iter=4)

    monkeypatch.setattr(walker, "relax_true_minimum", fake_landing)
    prequenched = RelaxResult(initial, energy=-10.0, gradient_norm=0.0, n_iter=2)
    return walker.run(initial, prequenched_initial=prequenched)


def test_search_action_history_joins_new_global_minimum_and_landing_cost(
    monkeypatch,
) -> None:
    result = _run_one_traced_action(monkeypatch, outcome="accepted")

    assert len(result.action_history) == 1
    action = result.action_history[0]
    assert (action.trial_index, action.proposal_index, action.seed_entry_id) == (0, 0, 0)
    assert action.walk is _walk() or action.walk == _walk()
    assert action.escape_energy_eV == pytest.approx(-9.2)
    assert action.landing_energy_eV == pytest.approx(-10.2)
    assert action.landing_gradient_norm == pytest.approx(0.04)
    assert action.landing_iterations == 4
    assert action.landing_converged is True
    assert action.landing_force_evaluations == 1
    assert action.accepted_new_basin is True
    assert action.is_duplicate is False
    assert action.global_improved is True
    assert action.status == "accepted"


def test_search_action_history_records_duplicate_landing(monkeypatch) -> None:
    result = _run_one_traced_action(monkeypatch, outcome="duplicate")

    action = result.action_history[0]
    assert action.accepted_new_basin is False
    assert action.is_duplicate is True
    assert action.global_improved is False
    assert action.status == "duplicate"
    assert action.landing_force_evaluations == 1


def test_search_action_history_records_fragment_rejection(monkeypatch) -> None:
    result = _run_one_traced_action(monkeypatch, outcome="fragment")

    action = result.action_history[0]
    assert action.accepted_new_basin is False
    assert action.is_duplicate is None
    assert action.global_improved is False
    assert action.status == "fragment_rejected"
    assert action.landing_force_evaluations == 1


def test_search_action_history_records_landing_budget_exhaustion(monkeypatch) -> None:
    result = _run_one_traced_action(monkeypatch, outcome="budget_exhausted")

    action = result.action_history[0]
    assert action.landing_energy_eV is None
    assert action.landing_iterations is None
    assert action.landing_converged is None
    assert action.landing_force_evaluations == 1
    assert action.accepted_new_basin is None
    assert action.is_duplicate is None
    assert action.global_improved is None
    assert action.status == "landing_budget_exhausted"
    assert result.stats["budget_exhausted"] == 1
