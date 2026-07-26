import numpy as np

from pamssw.proposal_replay import (
    capture_proposal_task,
    proposal_task_from_payload,
    proposal_task_to_payload,
    replay_proposal_task,
)
from pamssw.accounting import EvaluationPurpose
from pamssw.calculators import AnalyticCalculator
from pamssw.config import SSWConfig
from pamssw.state import State


class Quadratic:
    def energy_gradient(self, flat_positions, state):
        flat = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(flat @ flat), flat.copy()


def test_capture_first_task_stops_before_biased_proposal_evaluation():
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    config = SSWConfig(
        max_steps_per_walk=1,
        oracle_candidates=1,
        proposal_relax_steps=4,
        proposal_fmax=0.05,
        rng_seed=7,
    )

    captured = capture_proposal_task(
        state,
        AnalyticCalculator(Quadratic()),
        config,
        target_bias_count=1,
    )

    assert len(captured.task.biases) == 1
    assert captured.task.maxiter == 4
    assert (
        captured.evaluation_counts.count(EvaluationPurpose.BIASED_PROPOSAL_RELAX)
        == 0
    )


def test_task_json_payload_round_trip_preserves_periodic_fixed_state_and_biases():
    state = State(
        numbers=np.array([1, 8]),
        positions=np.array([[0.1, 0.2, 0.3], [1.1, 0.2, 0.3]]),
        cell=np.diag([4.0, 5.0, 6.0]),
        pbc=(True, True, False),
        fixed_mask=np.array([True, False]),
    )
    config = SSWConfig(
        max_steps_per_walk=1,
        oracle_candidates=1,
        proposal_relax_steps=4,
        proposal_fmax=0.05,
        rng_seed=3,
    )
    task = capture_proposal_task(
        state,
        AnalyticCalculator(Quadratic()),
        config,
        target_bias_count=1,
    ).task

    restored = proposal_task_from_payload(proposal_task_to_payload(task))

    np.testing.assert_array_equal(restored.initial_state.numbers, task.initial_state.numbers)
    np.testing.assert_allclose(restored.initial_state.positions, task.initial_state.positions)
    np.testing.assert_allclose(restored.initial_state.cell, task.initial_state.cell)
    np.testing.assert_array_equal(restored.initial_state.fixed_mask, task.initial_state.fixed_mask)
    assert restored.initial_state.pbc == task.initial_state.pbc
    assert restored.fmax == task.fmax
    assert restored.maxiter == task.maxiter
    assert len(restored.biases) == len(task.biases)
    np.testing.assert_allclose(restored.biases[0].center, task.biases[0].center)
    np.testing.assert_allclose(restored.biases[0].direction, task.biases[0].direction)


def test_replay_returns_exact_force_call_ledger_and_certificate():
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    config = SSWConfig(
        max_steps_per_walk=1,
        oracle_candidates=1,
        proposal_relax_steps=100,
        proposal_fmax=0.05,
        rng_seed=11,
    )
    task = capture_proposal_task(
        state,
        AnalyticCalculator(Quadratic()),
        config,
        target_bias_count=1,
    ).task

    replay = replay_proposal_task(
        task,
        AnalyticCalculator(Quadratic()),
        optimizer="ase-fire",
    )

    assert replay.evaluation_counts.total == replay.result.telemetry.evaluator_calls
    assert replay.result.gradient_norm <= task.fmax
    assert replay.certificate_satisfied
    assert replay.wall_time_s >= 0.0
