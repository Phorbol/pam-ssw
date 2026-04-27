import numpy as np

from pamssw import SSWConfig
from pamssw.archive import MinimaArchive
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import DoubleWell2D
from pamssw.state import State
from pamssw.walker import (
    CandidateDirectionGenerator,
    DirectionCandidateKind,
    DirectionCandidate,
    DirectionScorer,
    ProposalPotential,
    SoftModeOracle,
    SurfaceWalker,
)


def test_bias_weight_matches_curvature_inversion_rule():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(target_negative_curvature=0.2),
        softening_enabled=False,
    )

    assert walker._bias_weight(curvature=0.3, sigma=2.0) == 2.0
    assert walker._bias_weight(curvature=-0.5, sigma=2.0) == 0.0


def test_soft_mode_oracle_returns_best_candidate_without_random_mixing():
    class Quadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.diag([1.0, 4.0, 9.0])
            gradient = hessian @ flat_positions
            energy = 0.5 * float(flat_positions @ gradient)
            return energy, gradient

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    rng = np.random.default_rng(0)
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), rng, candidates=1)
    direction = np.array([1.0, 0.0, 0.0])

    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=direction,
    )

    np.testing.assert_allclose(choice.direction, direction)


def test_direction_generator_exposes_documented_enabled_and_guarded_kinds():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=2)

    candidates = generator.generate(state, previous_direction=np.array([1.0, 0.0, 0.0]))

    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.SOFT) == 1
    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.RANDOM) == 2
    assert DirectionCandidateKind.BOND not in [candidate.kind for candidate in candidates]
    assert DirectionCandidateKind.CELL not in [candidate.kind for candidate in candidates]


def test_direction_generator_adds_bond_candidate_when_pairs_are_provided():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=0, bond_pairs=[(0, 1)])

    candidates = generator.generate(state, previous_direction=None)

    assert [candidate.kind for candidate in candidates] == [DirectionCandidateKind.BOND]
    np.testing.assert_allclose(candidates[0].direction.reshape(2, 3)[0], np.array([-1.0, 0.0, 0.0]) / np.sqrt(2.0))
    np.testing.assert_allclose(candidates[0].direction.reshape(2, 3)[1], np.array([1.0, 0.0, 0.0]) / np.sqrt(2.0))


def test_direction_scorer_penalizes_damage_risk_and_discontinuity():
    scorer = DirectionScorer(damage_weight=10.0, continuity_weight=1.0)
    previous = np.array([1.0, 0.0, 0.0])

    smooth = scorer.score(curvature=1.0, sigma=0.5, direction=previous, previous_direction=previous, damage_risk=0.0)
    damaging = scorer.score(curvature=1.0, sigma=0.5, direction=-previous, previous_direction=previous, damage_risk=1.0)

    assert smooth > damaging


def test_direction_scorer_rewards_score_only_novelty_gain_from_archive():
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=1e-6)
    state = State(numbers=np.array([1, 1]), positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    archive.add(state, 0.0, parent_id=None)
    scorer = DirectionScorer(novelty_weight=10.0)

    near = DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0]) / np.sqrt(2.0))
    far = DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([-1.0, 0.0, 0.0, 1.0, 0.0, 0.0]) / np.sqrt(2.0))
    near_score = scorer.score_candidate(
        state=state,
        candidate=near,
        curvature=0.0,
        sigma=0.2,
        previous_direction=None,
        archive=archive,
    )
    far_score = scorer.score_candidate(
        state=state,
        candidate=far,
        curvature=0.0,
        sigma=2.0,
        previous_direction=None,
        archive=archive,
    )

    assert far_score > near_score
