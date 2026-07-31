import numpy as np
import pytest

from pamssw.accounting import BudgetExceeded, EvalCounter, EvaluationPurpose
from pamssw.calculators import AnalyticCalculator
from pamssw.config import SSWConfig
from pamssw.krylov import IntentBlock
from pamssw.rigid import project_out_rigid_body_modes
from pamssw.state import State
from pamssw.walker import (
    CandidateDirectionGenerator,
    DirectionCandidateKind,
    ProposalPotential,
    SoftModeOracle,
    SurfaceWalker,
)


class Quadratic:
    def energy_gradient(self, flat_positions, state):
        gradient = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(gradient @ gradient), gradient


class BatchQuadraticCalculator(AnalyticCalculator):
    def __init__(self, potential) -> None:
        super().__init__(potential)
        self.batch_sizes: list[int] = []

    def evaluate_flat_many(self, flat_positions, templates):
        self.batch_sizes.append(len(flat_positions))
        return tuple(
            self.evaluate_flat(positions, template)
            for positions, template in zip(flat_positions, templates)
        )


def test_first_step_fills_the_oracle_candidate_budget_with_random_directions():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=2, n_bond_pairs=0)

    candidates = generator.generate(state, previous_direction=None)

    assert len(candidates) == 2
    assert [candidate.kind for candidate in candidates] == [
        DirectionCandidateKind.RANDOM,
        DirectionCandidateKind.RANDOM,
    ]


def test_valid_momentum_occupies_one_slot_of_the_oracle_candidate_budget():
    state = State(
        numbers=np.array([1, 1, 1, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 1.0, 0.0],
            ]
        ),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(2),
        n_random=4,
        bond_pairs=[(0, 1)],
        n_bond_pairs=1,
        bond_distance_threshold=2.0,
    )

    candidates = generator.generate(state, previous_direction=np.ones(state.positions.size))

    assert len(candidates) == 4
    assert [candidate.kind for candidate in candidates] == [
        DirectionCandidateKind.MOMENTUM,
        DirectionCandidateKind.BOND,
        DirectionCandidateKind.BOND,
        DirectionCandidateKind.RANDOM,
    ]
    assert generator.last_random_bond_pairs_requested == 1
    assert generator.last_random_bond_pairs_generated == 1


def test_budget_one_keeps_the_momentum_priority_over_bond_and_random_candidates():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(0),
        n_random=1,
        bond_pairs=[(0, 1)],
        n_bond_pairs=1,
        bond_distance_threshold=1.0,
    )

    candidates = generator.generate(state, previous_direction=np.ones(state.positions.size))

    assert len(candidates) == 1
    assert candidates[0].kind is DirectionCandidateKind.MOMENTUM
    assert generator.last_random_bond_pairs_requested == 1
    assert generator.last_random_bond_pairs_generated == 0


def test_invalid_zero_momentum_does_not_consume_an_oracle_candidate_slot():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=1, n_bond_pairs=0)

    candidates = generator.generate(state, previous_direction=np.zeros(state.positions.size))

    assert len(candidates) == 1
    assert candidates[0].kind is DirectionCandidateKind.RANDOM
    assert np.linalg.norm(candidates[0].direction) == pytest.approx(1.0)


def test_subsequent_oracle_step_keeps_two_central_hvps_for_a_budget_of_two():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    calculator = EvalCounter(AnalyticCalculator(Quadratic()))
    oracle = SoftModeOracle(calculator, np.random.default_rng(0), candidates=2, n_bond_pairs=0)

    choice = oracle.choose_direction(
        state,
        ProposalPotential(calculator),
        previous_direction=np.ones(state.positions.size),
    )

    assert choice.candidate_count == 2
    assert calculator.force_evaluations == 4


def test_discrete_choice_records_exact_evaluated_candidate_source_counts():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    calculator = EvalCounter(AnalyticCalculator(Quadratic()))
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(7),
        candidates=4,
        bond_pairs=[(0, 1)],
        n_bond_pairs=0,
        enable_momentum_candidate=True,
    )

    choice = oracle.choose_direction(
        state,
        ProposalPotential(calculator),
        previous_direction=np.ones(state.positions.size),
    )

    assert choice.candidate_count == 4
    assert choice.diagnostics["evaluated_candidate_kind_counts"] == {
        "bond": 1,
        "momentum": 1,
        "random": 2,
    }
    assert sum(
        choice.diagnostics["evaluated_candidate_kind_counts"].values()
    ) == choice.candidate_count
    assert calculator.force_evaluations == 2 * choice.candidate_count


def test_discrete_choice_batches_all_central_hessian_stencils_without_changing_cost():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    backend = BatchQuadraticCalculator(Quadratic())
    calculator = EvalCounter(backend)
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(7),
        candidates=4,
        bond_pairs=[(0, 1)],
        n_bond_pairs=0,
        enable_momentum_candidate=True,
    )

    with calculator.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        choice = oracle.choose_direction(
            state,
            ProposalPotential(calculator),
            previous_direction=np.ones(state.positions.size),
        )

    assert backend.batch_sizes == [8]
    assert choice.candidate_count == 4
    assert choice.curvature == pytest.approx(1.0)
    assert calculator.force_evaluations == 8
    assert calculator.snapshot().count(EvaluationPurpose.DIRECTION_ORACLE) == 8
    assert calculator.snapshot().count(EvaluationPurpose.UNATTRIBUTED) == 0


def test_batched_and_serial_candidate_scoring_select_the_same_physical_direction():
    class CoupledQuadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.diag(np.arange(1.0, 7.0))
            hessian[0, 3] = hessian[3, 0] = 0.4
            gradient = hessian @ flat_positions
            return 0.5 * float(flat_positions @ gradient), gradient

    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    serial_counter = EvalCounter(AnalyticCalculator(CoupledQuadratic()))
    batch_counter = EvalCounter(BatchQuadraticCalculator(CoupledQuadratic()))
    serial_oracle = SoftModeOracle(
        serial_counter,
        np.random.default_rng(19),
        candidates=4,
        n_bond_pairs=0,
    )
    batch_oracle = SoftModeOracle(
        batch_counter,
        np.random.default_rng(19),
        candidates=4,
        n_bond_pairs=0,
    )

    serial = serial_oracle.choose_direction(
        state,
        ProposalPotential(serial_counter),
        previous_direction=None,
    )
    batch = batch_oracle.choose_direction(
        state,
        ProposalPotential(batch_counter),
        previous_direction=None,
    )

    assert batch.kind is serial.kind
    assert batch.curvature == pytest.approx(serial.curvature, rel=0.0, abs=1e-12)
    assert batch.score == pytest.approx(serial.score, rel=0.0, abs=1e-12)
    np.testing.assert_allclose(batch.direction, serial.direction, rtol=0.0, atol=0.0)
    assert batch_counter.force_evaluations == serial_counter.force_evaluations == 8


def test_k4_batch_hessian_stencil_fails_before_partial_evaluation_when_budget_is_seven():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    backend = BatchQuadraticCalculator(Quadratic())
    calculator = EvalCounter(backend, max_force_evals=7)
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(19),
        candidates=4,
        n_bond_pairs=0,
    )

    with calculator.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        with pytest.raises(BudgetExceeded):
            oracle.choose_direction(
                state,
                ProposalPotential(calculator),
                previous_direction=None,
            )

    assert backend.batch_sizes == []
    assert calculator.snapshot().total == 0


def test_block_krylov_reuses_solver_hvps_for_selection_and_true_curvature():
    class CoupledQuadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.array(
                [
                    [2.0, 1.0, 0.0],
                    [1.0, 3.0, 1.0],
                    [0.0, 1.0, 4.0],
                ]
            )
            gradient = hessian @ flat_positions
            return 0.5 * float(flat_positions @ gradient), gradient

    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    calculator = EvalCounter(AnalyticCalculator(CoupledQuadratic()))
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="block_krylov",
        block_krylov_depth=3,
    )

    choice = oracle.choose_direction(
        state,
        ProposalPotential(calculator),
        previous_direction=None,
        krylov_intents=(
            IntentBlock(np.array([[1.0], [0.0], [0.0]])),
            IntentBlock(np.array([[0.0], [1.0], [0.0]])),
        ),
    )

    assert choice.candidate_count == 0
    assert choice.diagnostics["krylov_blocks"] == 2
    assert choice.true_curvature is not None
    assert choice.diagnostics["krylov_hvp_count"] == 6
    assert choice.diagnostics["krylov_dimensions"] == [3, 3]
    assert calculator.force_evaluations == 12


def test_exact_anchor_mode_uses_one_curvature_hvp():
    state = State(
        numbers=np.array([1]),
        positions=np.array([[0.0, 0.0, 0.0]]),
    )
    calculator = EvalCounter(AnalyticCalculator(Quadratic()))
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="exact_anchor",
    )
    anchor = np.array([0.0, -1.0, 0.0])

    choice = oracle.choose_direction(
        state,
        ProposalPotential(calculator),
        previous_direction=None,
        anchor_direction=anchor,
    )

    assert choice.kind is DirectionCandidateKind.ANCHOR
    np.testing.assert_array_equal(choice.direction, anchor)
    assert choice.curvature == pytest.approx(1.0)
    assert choice.true_curvature == pytest.approx(1.0)
    assert choice.candidate_count == 1
    assert choice.diagnostics == {"direction_hvp_count": 1}
    assert calculator.force_evaluations == 2


def test_anchor_krylov_reuses_exact_anchor_block_for_full_hvp_budget():
    class CoupledQuadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.array(
                [
                    [2.0, 1.0, 0.0],
                    [1.0, 3.0, 1.0],
                    [0.0, 1.0, 4.0],
                ]
            )
            gradient = hessian @ flat_positions
            return 0.5 * float(flat_positions @ gradient), gradient

    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    calculator = EvalCounter(AnalyticCalculator(CoupledQuadratic()))
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="anchor_krylov",
        block_krylov_depth=3,
    )
    anchor = np.array([1.0, 0.0, 0.0])

    choice = oracle.choose_direction(
        state,
        ProposalPotential(calculator),
        previous_direction=None,
        anchor_direction=anchor,
        krylov_intents=(IntentBlock(anchor[:, None]),),
    )

    assert choice.kind is DirectionCandidateKind.BLOCK_RITZ
    assert choice.diagnostics["krylov_initial_basis_columns"] == [1]
    assert choice.diagnostics["krylov_hvp_consumed"] == 3
    assert calculator.force_evaluations == 6


def test_energy_bounded_anchor_reuses_krylov_hvps_and_enforces_energy_limit():
    class DiagonalQuadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.diag([1.0, 4.0, 9.0])
            gradient = hessian @ flat_positions
            return 0.5 * float(flat_positions @ gradient), gradient

    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    calculator = EvalCounter(AnalyticCalculator(DiagonalQuadratic()))
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="energy_bounded_anchor",
        block_krylov_depth=3,
    )
    anchor = np.ones(3) / np.sqrt(3.0)

    choice = oracle.choose_direction(
        state,
        ProposalPotential(calculator),
        previous_direction=None,
        anchor_direction=anchor,
        krylov_intents=(IntentBlock(anchor[:, None]),),
        energy_bound_step_scale=0.5,
        energy_bound_target=0.375,
    )

    assert choice.kind is DirectionCandidateKind.ENERGY_BOUNDED_ANCHOR
    assert choice.diagnostics["krylov_hvp_consumed"] == 3
    assert choice.diagnostics["energy_bounded_anchor_feasible"] is True
    assert choice.diagnostics["energy_bounded_anchor_active"] is True
    assert choice.diagnostics["energy_bounded_anchor_step_scale"] == pytest.approx(0.5)
    assert choice.diagnostics["energy_bounded_anchor_energy_target"] == pytest.approx(0.375)
    assert choice.diagnostics["energy_bounded_anchor_curvature_limit"] == pytest.approx(3.0)
    assert choice.diagnostics["energy_bounded_anchor_true_curvature"] == pytest.approx(3.0)
    assert choice.diagnostics["energy_bounded_anchor_quadratic_energy"] == pytest.approx(0.375)
    assert 0.0 < choice.diagnostics["energy_bounded_anchor_overlap"] < 1.0
    assert calculator.force_evaluations == 6


def test_direction_modes_generate_the_same_anchor_before_arm_specific_intents():
    state = State(
        numbers=np.array([6, 6, 6, 6]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 1.4, 0.0],
                [0.0, 0.0, 1.4],
            ]
        ),
    )

    def context(mode):
        walker = SurfaceWalker(
            calculator=AnalyticCalculator(Quadratic()),
            config=SSWConfig(
                rng_seed=17,
                direction_selection_mode=mode,
                block_krylov_blocks=1,
                block_krylov_depth=3,
            ),
            softening_enabled=False,
        )
        return walker._initialize_walk_direction_context(
            state,
            trial_index=0,
        )

    detached_anchor, detached_intents = context("block_krylov")
    exact_anchor, exact_intents = context("exact_anchor")
    lanczos_anchor, lanczos_intents = context("anchor_krylov")
    bounded_anchor, bounded_intents = context("energy_bounded_anchor")

    np.testing.assert_array_equal(detached_anchor, exact_anchor)
    np.testing.assert_array_equal(exact_anchor, lanczos_anchor)
    np.testing.assert_array_equal(lanczos_anchor, bounded_anchor)
    assert detached_intents is not None
    assert exact_intents is None
    assert lanczos_intents is not None
    assert bounded_intents is not None
    assert len(lanczos_intents) == 1
    assert lanczos_intents[0].basis.shape == (
        state.positions.size,
        1,
    )
    np.testing.assert_array_equal(
        lanczos_intents[0].basis[:, 0],
        lanczos_anchor,
    )
    np.testing.assert_array_equal(
        bounded_intents[0].basis[:, 0],
        bounded_anchor,
    )


def test_block_krylov_projects_hvps_back_into_fixed_and_internal_subspace():
    class FixedCoupledQuadratic:
        def __init__(self):
            self.hessian = np.diag(np.arange(1.0, 13.0))
            self.hessian[0, 3] = 4.0
            self.hessian[3, 0] = 4.0

        def energy_gradient(self, flat_positions, state):
            gradient = self.hessian @ flat_positions
            return 0.5 * float(flat_positions @ gradient), gradient

    state = State(
        numbers=np.ones(4, dtype=int),
        positions=np.array(
            [
                [3.0, 3.0, 3.0],
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        fixed_mask=np.array([True, False, False, False]),
    )
    initial = np.zeros(state.positions.size)
    initial[3] = 1.0
    initial = project_out_rigid_body_modes(state, initial)
    initial[state.fixed_mask.repeat(3)] = 0.0
    initial /= np.linalg.norm(initial)
    calculator = EvalCounter(AnalyticCalculator(FixedCoupledQuadratic()))
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="block_krylov",
        block_krylov_depth=2,
    )

    choice = oracle.choose_direction(
        state,
        ProposalPotential(calculator),
        previous_direction=None,
        krylov_intents=(IntentBlock(initial[:, None]),),
    )

    direction_by_atom = choice.direction.reshape(state.n_atoms, 3)
    np.testing.assert_allclose(direction_by_atom[state.fixed_mask], 0.0, rtol=0.0, atol=1e-12)
    movable_squared_amplitudes = np.sum(
        np.square(direction_by_atom[state.movable_mask]),
        axis=1,
    )
    expected_participation = 1.0 / float(
        np.dot(movable_squared_amplitudes, movable_squared_amplitudes)
    )
    assert choice.diagnostics["direction_participation_ratio"] == pytest.approx(
        expected_participation,
        rel=1e-12,
    )
    assert choice.diagnostics["krylov_hvp_count"] == 2
    assert choice.diagnostics["krylov_dimensions"] == [2]
    assert calculator.force_evaluations == 4
