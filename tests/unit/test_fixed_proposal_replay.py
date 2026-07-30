import numpy as np
import pytest

from pamssw.proposal_replay import (
    ProposalTaskNotCaptured,
    capture_proposal_task,
    proposal_task_from_payload,
    proposal_task_to_payload,
    retarget_last_gaussian,
    replay_proposal_task,
    replay_proposal_task_observed,
)
from pamssw.accounting import EvaluationPurpose
from pamssw.bias import GaussianBiasTerm, QuadraticBiasTerm
from pamssw.calculators import AnalyticCalculator
from pamssw.config import SSWConfig
from pamssw.state import State
from pamssw.walker import ProposalRelaxationTask


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


def test_uncaptured_task_failure_preserves_spent_evaluation_counts():
    state = State(
        numbers=np.array([1]),
        positions=np.array([[1.0, 0.0, 0.0]]),
    )
    config = SSWConfig(
        max_steps_per_walk=2,
        oracle_candidates=1,
        proposal_relax_steps=4,
        proposal_fmax=0.05,
        walk_trust_radius=1.0e-8,
        rng_seed=7,
    )

    with pytest.raises(ProposalTaskNotCaptured) as captured:
        capture_proposal_task(
            state,
            AnalyticCalculator(Quadratic()),
            config,
            target_bias_count=2,
        )

    assert captured.value.evaluation_counts.total > 0
    assert (
        captured.value.evaluation_counts.count(
            EvaluationPurpose.UNATTRIBUTED
        )
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


def test_retarget_last_gaussian_changes_only_last_bias_and_explicit_displacement():
    state = State(
        numbers=np.array([1]),
        positions=np.array([[1.0, 0.0, 0.0]]),
    )
    task = capture_proposal_task(
        state,
        AnalyticCalculator(Quadratic()),
        SSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            proposal_relax_steps=4,
            proposal_fmax=0.05,
            rng_seed=19,
        ),
        target_bias_count=1,
    ).task
    prefix = GaussianBiasTerm(
        center=task.biases[-1].center - 0.1 * task.biases[-1].direction,
        direction=task.biases[-1].direction,
        sigma=0.1,
        weight=0.2,
    )
    task = ProposalRelaxationTask(
        initial_state=task.initial_state,
        biases=(prefix, *task.biases),
        softening=task.softening,
        fmax=task.fmax,
        maxiter=task.maxiter,
        coordinate_trust_radius=task.coordinate_trust_radius,
    )
    source_sigma = task.biases[-1].sigma
    source_weight = task.biases[-1].weight
    source_prefix_center = task.biases[0].center.copy()
    source_positions = task.initial_state.positions.copy()

    retargeted = retarget_last_gaussian(
        task,
        sigma=0.25,
        weight=0.4,
    )

    assert task.biases[-1].sigma == source_sigma
    assert task.biases[-1].weight == source_weight
    np.testing.assert_allclose(task.initial_state.positions, source_positions)
    assert retargeted.biases[0].sigma == task.biases[0].sigma
    assert retargeted.biases[0].weight == task.biases[0].weight
    np.testing.assert_allclose(retargeted.biases[0].center, source_prefix_center)
    assert retargeted.biases[-1].sigma == 0.25
    assert retargeted.biases[-1].weight == 0.4
    delta = (
        retargeted.initial_state.flatten_positions()
        - retargeted.biases[-1].center
    )
    progress = float(np.dot(delta, retargeted.biases[-1].direction))
    assert progress == pytest.approx(0.25)
    np.testing.assert_allclose(
        delta - progress * retargeted.biases[-1].direction,
        0.0,
        atol=1.0e-12,
    )


def test_observed_replay_reuses_backend_calls_for_true_and_bias_components():
    state = State(
        numbers=np.array([1]),
        positions=np.array([[1.0, 0.0, 0.0]]),
    )
    task = capture_proposal_task(
        state,
        AnalyticCalculator(Quadratic()),
        SSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            proposal_relax_steps=100,
            proposal_fmax=0.05,
            rng_seed=23,
        ),
        target_bias_count=1,
    ).task
    initial_flat = task.initial_state.flatten_positions()
    expected_true_initial = 0.5 * float(initial_flat @ initial_flat)
    expected_bias_initial = sum(
        bias.evaluate(
            initial_flat,
            cell=task.initial_state.cell,
            pbc=task.initial_state.pbc,
        )[0]
        for bias in task.biases
    )

    observed = replay_proposal_task_observed(
        task,
        AnalyticCalculator(Quadratic()),
        optimizer="ase-fire",
    )

    assert (
        observed.evaluation_counts.total
        == observed.result.telemetry.evaluator_calls
    )
    assert observed.observer_only_force_evaluations == 0
    assert observed.initial.true_energy == pytest.approx(expected_true_initial)
    assert observed.initial.bias_energy == pytest.approx(expected_bias_initial)
    assert observed.initial.total_energy == pytest.approx(
        expected_true_initial + expected_bias_initial
    )
    assert observed.final.total_energy == pytest.approx(observed.result.energy)
    endpoint_delta = (
        observed.result.state.flatten_positions()
        - task.biases[-1].center
    )
    expected_progress = float(
        np.dot(endpoint_delta, task.biases[-1].direction)
    )
    assert observed.direction_progress == pytest.approx(expected_progress)
    expected_orthogonal = (
        endpoint_delta
        - expected_progress * task.biases[-1].direction
    )
    assert observed.orthogonal_displacement_norm == pytest.approx(
        np.linalg.norm(expected_orthogonal)
    )


def test_observed_replay_can_override_bias_shape_without_mutating_task():
    state = State(
        numbers=np.array([1]),
        positions=np.array([[1.0, 0.0, 0.0]]),
    )
    task = capture_proposal_task(
        state,
        AnalyticCalculator(Quadratic()),
        SSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            proposal_relax_steps=20,
            proposal_fmax=0.05,
            rng_seed=29,
        ),
        target_bias_count=1,
    ).task
    gaussian = task.biases[-1]
    quadratic = QuadraticBiasTerm(
        center=gaussian.center,
        direction=gaussian.direction,
        sigma=gaussian.sigma,
        weight=gaussian.weight,
    )

    observed = replay_proposal_task_observed(
        task,
        AnalyticCalculator(Quadratic()),
        optimizer="ase-fire",
        biases_override=(quadratic,),
    )

    expected_bias_energy = quadratic.evaluate(
        task.initial_state.flatten_positions(),
        cell=task.initial_state.cell,
        pbc=task.initial_state.pbc,
    )[0]
    assert observed.initial.bias_energy == pytest.approx(expected_bias_energy)
    assert isinstance(task.biases[-1], GaussianBiasTerm)
    assert observed.evaluation_counts.total == (
        observed.result.telemetry.evaluator_calls
    )
