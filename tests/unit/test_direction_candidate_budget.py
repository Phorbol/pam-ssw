import numpy as np
import pytest

from pamssw.accounting import EvalCounter
from pamssw.calculators import AnalyticCalculator
from pamssw.state import State
from pamssw.walker import (
    CandidateDirectionGenerator,
    DirectionCandidateKind,
    ProposalPotential,
    SoftModeOracle,
)


class Quadratic:
    def energy_gradient(self, flat_positions, state):
        gradient = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(gradient @ gradient), gradient


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
