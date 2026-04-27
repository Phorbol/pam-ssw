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
    geometry_damage_risk,
    ProposalPotential,
    SoftModeOracle,
    StepTargetController,
    SurfaceWalker,
    TrustRegionBiasController,
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


def test_geometry_damage_risk_penalizes_overlap_without_compactness_reward():
    state = State(numbers=np.array([1, 1]), positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    toward_overlap = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0)
    apart = -toward_overlap

    risky = geometry_damage_risk(state, toward_overlap, sigma=0.6)
    safe = geometry_damage_risk(state, apart, sigma=0.6)

    assert risky > 0.0
    assert safe == 0.0


def test_geometry_damage_risk_uses_relative_structure_scale():
    small = State(numbers=np.array([1, 1]), positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    large = State(numbers=np.array([1, 1]), positions=np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]))
    direction = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0)

    small_risk = geometry_damage_risk(small, direction, sigma=0.6)
    large_risk = geometry_damage_risk(large, direction, sigma=6.0)

    assert small_risk > 0.0
    np.testing.assert_allclose(large_risk, small_risk)


def test_trust_region_controller_shrinks_after_bad_local_model():
    controller = TrustRegionBiasController()

    update = controller.update(
        curvature=2.0,
        sigma=0.5,
        true_delta=2.0,
        sigma_scale=1.0,
        weight_scale=1.0,
    )

    assert update.action == "shrink"
    assert update.sigma_scale < 1.0
    assert update.weight_scale < 1.0
    assert update.model_error > controller.error_tolerance


def test_trust_region_controller_expands_after_acceptable_local_model():
    controller = TrustRegionBiasController()

    update = controller.update(
        curvature=2.0,
        sigma=0.5,
        true_delta=0.26,
        sigma_scale=1.0,
        weight_scale=1.0,
    )

    assert update.action == "expand"
    assert update.sigma_scale > 1.0
    assert update.weight_scale > 1.0


def test_surface_walker_reports_trust_region_diagnostics():
    initial = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, max_steps_per_walk=2, oracle_candidates=2, rng_seed=0),
        softening_enabled=False,
    )

    result = walker.run(initial)

    assert result.stats["trust_region_steps"] == 2
    assert "trust_model_error_mean" in result.stats
    assert "trust_shrink_steps" in result.stats
    assert "trust_expand_steps" in result.stats
    assert "trust_damage_events" in result.stats


def test_surface_walker_reports_bounded_archive_prototype_diagnostics():
    initial = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=2, max_steps_per_walk=1, oracle_candidates=2, max_prototypes=2, rng_seed=0),
        softening_enabled=False,
    )

    result = walker.run(initial)

    assert result.stats["archive_prototypes"] <= 2
    assert result.stats["archive_max_prototypes"] == 2


def test_step_target_controller_uses_archive_energy_scale():
    archive = MinimaArchive(energy_tol=1e-8, rmsd_tol=1e-8)
    archive.add(State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]])), -10.0, parent_id=None)
    archive.add(State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]])), -8.0, parent_id=None)
    controller = StepTargetController(fallback_target=0.6)

    target = controller.target(archive)

    assert target != 0.6
    assert target > controller.min_target


def test_step_target_controller_feedback_increases_on_low_escape_and_decreases_on_damage():
    archive = MinimaArchive(energy_tol=1e-8, rmsd_tol=1e-8)
    archive.add(State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]])), -10.0, parent_id=None)
    archive.add(State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]])), -8.0, parent_id=None)
    controller = StepTargetController(fallback_target=0.6)

    base = controller.target(archive)
    controller.record_trial(escaped=False, damaged=False)
    increased = controller.target(archive)
    for _ in range(controller.feedback_warmup_trials):
        controller.record_trial(escaped=False, damaged=True)
    decreased = controller.target(archive)

    assert increased > base
    assert decreased < increased


def test_surface_walker_reports_adaptive_step_target_diagnostics():
    initial = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, max_steps_per_walk=1, oracle_candidates=2, rng_seed=0),
        softening_enabled=False,
    )

    result = walker.run(initial)

    assert "adaptive_step_target" in result.stats
    assert "adaptive_step_multiplier" in result.stats


def test_surface_walker_reports_observable_frontier_diagnostics():
    initial = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, max_steps_per_walk=1, oracle_candidates=2, rng_seed=0),
        softening_enabled=False,
    )

    result = walker.run(initial)

    assert "frontier_nodes" in result.stats
    assert "dead_nodes" in result.stats
    assert "mean_frontier_score" in result.stats
