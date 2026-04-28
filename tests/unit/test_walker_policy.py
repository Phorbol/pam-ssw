import numpy as np
import pytest

from pamssw import LSSSWConfig, SSWConfig
from pamssw.archive import MinimaArchive
from pamssw.bias import GaussianBiasTerm
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import DoubleWell2D
from pamssw.state import State
from pamssw.walker import (
    CandidateDirectionGenerator,
    DirectionCandidateKind,
    DirectionCandidate,
    DirectionScorer,
    ProposalPotential,
    GeometryValidator,
    SoftModeOracle,
    StepTargetController,
    SurfaceWalker,
    TrustRegionBiasController,
)


class Quadratic:
    def energy_gradient(self, flat_positions, state):
        gradient = np.asarray(flat_positions, dtype=float).copy()
        energy = 0.5 * float(gradient @ gradient)
        return energy, gradient


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


def test_soft_mode_oracle_scores_with_actual_walk_sigma():
    class Quadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.diag([1.0, 4.0, 9.0])
            gradient = hessian @ flat_positions
            energy = 0.5 * float(flat_positions @ gradient)
            return energy, gradient

    class CapturingScorer(DirectionScorer):
        def __init__(self):
            super().__init__()
            self.sigmas = []

        def score_candidate(self, **kwargs):
            self.sigmas.append(kwargs["sigma"])
            return super().score_candidate(**kwargs)

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), np.random.default_rng(0), candidates=1)
    oracle.scorer = CapturingScorer()

    oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=np.array([1.0, 0.0, 0.0]),
        step_scale_fn=lambda curvature: 0.123,
    )

    assert oracle.scorer.sigmas
    assert all(sigma == 0.123 for sigma in oracle.scorer.sigmas)


def test_soft_mode_oracle_uses_configured_hvp_epsilon():
    class Quadratic:
        def energy_gradient(self, flat_positions, state):
            gradient = flat_positions.copy()
            return 0.5 * float(flat_positions @ flat_positions), gradient

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), np.random.default_rng(0), candidates=1, hvp_epsilon=2e-4)

    curvature = oracle._directional_curvature(
        state,
        ProposalPotential(AnalyticCalculator(Quadratic())),
        np.array([1.0, 0.0, 0.0]),
    )

    assert curvature == pytest.approx(1.0)
    assert oracle.hvp_epsilon == 2e-4


def test_direction_generator_exposes_documented_enabled_and_guarded_kinds():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=2)

    candidates = generator.generate(state, previous_direction=np.array([1.0, 0.0, 0.0]))

    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.SOFT) == 1
    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.RANDOM) == 2
    assert DirectionCandidateKind.BOND not in [candidate.kind for candidate in candidates]


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


def test_direction_generator_adds_random_non_neighbor_bond_candidates():
    state = State(
        numbers=np.full(4, 18),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [4.0, 0.0, 0.0],
                [4.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(2),
        n_random=0,
        n_bond_pairs=3,
        bond_distance_threshold=2.0,
    )

    candidates = generator.generate(state, previous_direction=None)

    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.BOND) > 0
    assert generator.last_random_bond_pairs_requested == 3
    assert generator.last_random_bond_pairs_generated > 0
    assert generator.last_random_bond_candidates_valid > 0


def test_initial_direction_mixes_random_and_bond_components():
    state = State(
        numbers=np.full(3, 18),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=float),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(3),
        n_random=0,
        n_bond_pairs=2,
        bond_distance_threshold=1.5,
    )

    directions = [
        generator.generate_initial_direction(
            state,
            step_index=i,
            max_steps=6,
            lambda_start=0.1,
            lambda_end=1.0,
            n_bond_pairs=2,
            bond_distance_threshold=1.5,
        )
        for i in range(10)
    ]
    cosines = []
    for i, left in enumerate(directions):
        for right in directions[i + 1 :]:
            cosines.append(abs(float(np.dot(left, right))))

    assert float(np.mean(cosines)) < 0.5
    assert generator.last_initial_bond_pair is not None


def test_direction_generator_projects_random_candidates_out_of_rigid_modes():
    state = State(
        numbers=np.full(4, 18),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        ),
    )
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=4)

    candidates = generator.generate(state, previous_direction=None)

    assert len(candidates) == 4
    assert max(candidate.rigid_body_overlap for candidate in candidates) > 0.0
    for candidate in candidates:
        assert candidate.post_projection_rigid_body_overlap < 1e-10


def test_direction_scorer_penalizes_damage_risk_and_discontinuity():
    scorer = DirectionScorer(damage_weight=10.0, continuity_weight=1.0)
    previous = np.array([1.0, 0.0, 0.0])

    smooth = scorer.score(
        curvature=1.0,
        sigma=0.5,
        direction=previous,
        previous_direction=previous,
        anchor_direction=None,
        damage_risk=0.0,
    )
    damaging = scorer.score(
        curvature=1.0,
        sigma=0.5,
        direction=-previous,
        previous_direction=previous,
        anchor_direction=None,
        damage_risk=1.0,
    )

    assert smooth > damaging


def test_direction_scorer_penalizes_deviation_from_anchor_direction():
    scorer = DirectionScorer(anchor_weight=2.0, continuity_weight=0.0, damage_weight=0.0)
    anchor = np.array([1.0, 0.0, 0.0])

    aligned = scorer.score(
        curvature=0.0,
        sigma=1.0,
        direction=anchor,
        previous_direction=None,
        anchor_direction=anchor,
        damage_risk=0.0,
    )
    opposite = scorer.score(
        curvature=0.0,
        sigma=1.0,
        direction=-anchor,
        previous_direction=None,
        anchor_direction=anchor,
        damage_risk=0.0,
    )

    assert aligned > opposite


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
        anchor_direction=None,
        archive=archive,
    )
    far_score = scorer.score_candidate(
        state=state,
        candidate=far,
        curvature=0.0,
        sigma=2.0,
        previous_direction=None,
        anchor_direction=None,
        archive=archive,
    )

    assert far_score > near_score


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


def test_trust_region_prediction_includes_true_gradient_linear_term():
    controller = TrustRegionBiasController()

    predicted = controller.predicted_delta(curvature=2.0, sigma=0.5, g_parallel=-1.0)

    assert predicted == pytest.approx(-0.25)


def test_trust_region_accepts_model_when_linear_term_explains_delta():
    controller = TrustRegionBiasController()

    update = controller.update(
        curvature=2.0,
        sigma=0.5,
        true_delta=-0.25,
        g_parallel=-1.0,
        sigma_scale=1.0,
        weight_scale=1.0,
    )

    assert update.action == "expand"
    assert update.model_error < controller.error_tolerance


def test_bias_weight_is_clipped_by_configured_maximum():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(target_negative_curvature=0.2, bias_weight_max=1.5),
        softening_enabled=False,
    )

    assert walker._bias_weight(curvature=100.0, sigma=2.0) == 1.5


def test_bias_weight_uses_configured_minimum_floor():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(target_negative_curvature=0.2, bias_weight_min=0.05, bias_weight_max=1.5),
        softening_enabled=False,
    )

    assert walker._bias_weight(curvature=-0.5, sigma=2.0) == 0.05


def test_true_curvature_excludes_accumulated_gaussian_bias():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))

    class Quadratic:
        def energy_gradient(self, flat_positions, state):
            gradient = flat_positions.copy()
            return 0.5 * float(flat_positions @ flat_positions), gradient

    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_steps_per_walk=1),
        softening_enabled=False,
    )
    direction = np.array([1.0, 0.0, 0.0])

    biased = ProposalPotential(
        walker.calculator,
        biases=[GaussianBiasTerm(center=state.flatten_positions(), direction=direction, sigma=0.5, weight=10.0)],
    )

    biased_curvature = walker.oracle._directional_curvature(state, biased, direction)
    true_curvature = walker._true_directional_curvature(state, direction)

    assert biased_curvature < 0.0
    assert true_curvature == pytest.approx(1.0, rel=1e-3)


def test_geometry_validator_rejects_nan_positions_and_nonfinite_energy():
    valid = GeometryValidator(min_distance=0.5)
    bad_positions = State(numbers=np.array([1]), positions=np.array([[np.nan, 0.0, 0.0]]))

    assert not valid.is_valid_state(bad_positions)

    class NonFinite:
        def evaluate_flat(self, flat_positions, state):
            return float("nan"), np.zeros_like(flat_positions)

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))

    assert not valid.is_valid_evaluation(state, NonFinite())


def test_geometry_validator_rejects_nonperiodic_atom_overlap():
    validator = GeometryValidator(min_distance=0.5)
    overlapped = State(numbers=np.array([1, 1]), positions=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    periodic = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        cell=np.eye(3),
        pbc=(True, True, True),
    )

    assert not validator.is_valid_state(overlapped)
    assert validator.is_valid_state(periodic)


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


def test_surface_walker_reports_relaxation_convergence_diagnostics():
    initial = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, max_steps_per_walk=1, oracle_candidates=2, rng_seed=0),
        softening_enabled=False,
    )

    result = walker.run(initial)

    assert result.stats["true_quench_count"] >= 1
    assert result.stats["true_quench_max_gradient"] <= 1.1 * walker.config.quench_fmax
    assert result.stats["true_quench_unconverged"] >= 0
    assert result.stats["proposal_relax_count"] == 1
    assert result.stats["proposal_relax_max_gradient"] >= 0.0
    assert "proposal_relax_unconverged" in result.stats
    assert "proposal_relax_median_iterations" in result.stats
    assert "proposal_relax_p90_iterations" in result.stats
    assert result.stats["proposal_relax_max_iterations"] >= result.stats["proposal_relax_min_iterations"]
    assert "proposal_relax_active_bound_fraction_mean" in result.stats
    assert "proposal_relax_displacement_max" in result.stats
    assert "bias_zero_weight_fraction" in result.stats


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
    assert "mean_node_duplicate_failure_rate" in result.stats
    assert "max_node_duplicate_failure_rate" in result.stats


def test_surface_walker_reports_direction_acquisition_diagnostics():
    initial = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, max_steps_per_walk=2, oracle_candidates=2, rng_seed=0),
        softening_enabled=False,
    )

    result = walker.run(initial)

    assert result.stats["direction_choices"] == 2
    assert result.stats["direction_candidate_evaluations"] >= result.stats["direction_choices"]
    assert "direction_rigid_body_overlap_mean" in result.stats
    assert "direction_post_projection_rigid_body_overlap_mean" in result.stats
    assert result.stats["direction_selected_random"] >= 1
    assert result.stats["direction_selected_soft"] >= 0
    assert result.stats["direction_selected_bond"] >= 0
    assert "walk_displacement_clips" in result.stats
    assert "fragment_rejections" in result.stats
    assert "direction_bond_pairs_requested" in result.stats
    assert "direction_bond_pairs_generated" in result.stats
    assert "direction_bond_candidates_valid" in result.stats


def test_standard_surface_walker_generates_bond_candidates():
    initial = State(
        numbers=np.full(4, 18),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [3.2, 0.0, 0.0],
                [3.2, 1.0, 0.0],
            ],
            dtype=float,
        ),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, max_steps_per_walk=1, oracle_candidates=1, n_bond_pairs=2, rng_seed=0),
        softening_enabled=False,
    )

    result = walker.run(initial)

    assert result.stats["direction_candidate_evaluations"] > result.stats["direction_choices"]


def test_walk_displacement_clip_limits_per_atom_motion():
    reference = State(numbers=np.array([1, 1]), positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    candidate = State(numbers=np.array([1, 1]), positions=np.array([[10.0, 0.0, 0.0], [1.0, 0.5, 0.0]]))

    clipped, did_clip = SurfaceWalker._clip_walk_displacement(reference, candidate, max_displacement=2.0)

    assert did_clip
    np.testing.assert_allclose(clipped.positions[0], np.array([2.0, 0.0, 0.0]))
    np.testing.assert_allclose(clipped.positions[1], candidate.positions[1])


def test_walk_displacement_clip_uses_mic_for_periodic_axes():
    cell = np.diag([5.0, 5.0, 5.0])
    reference = State(
        numbers=np.array([1]),
        positions=np.array([[4.8, 0.0, 0.0]]),
        cell=cell,
        pbc=(True, True, True),
    )
    candidate = State(
        numbers=np.array([1]),
        positions=np.array([[0.2, 0.0, 0.0]]),
        cell=cell,
        pbc=(True, True, True),
    )

    clipped, did_clip = SurfaceWalker._clip_walk_displacement(reference, candidate, max_displacement=1.0)

    assert not did_clip
    np.testing.assert_allclose(clipped.positions, candidate.positions)


def test_direction_generator_samples_slab_non_neighbor_pairs_with_mic():
    state = State(
        numbers=np.full(3, 18),
        positions=np.array([[0.2, 0.0, 0.0], [4.8, 0.0, 0.0], [2.5, 0.0, 3.0]]),
        cell=np.diag([5.0, 5.0, 10.0]),
        pbc=(True, True, False),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(4),
        n_random=0,
        n_bond_pairs=2,
        bond_distance_threshold=1.0,
    )

    pairs = generator._random_non_neighbor_pairs(state, n_pairs=2, distance_threshold=1.0)

    assert pairs
    assert (0, 1) not in pairs


def test_fragment_guard_rejects_disconnected_nonperiodic_cluster():
    reference = State(
        numbers=np.array([1, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]]),
    )
    fragmented = State(
        numbers=np.array([1, 1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [12.0, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(fragment_guard_factor=3.0),
        softening_enabled=False,
    )

    assert walker._is_fragmented_cluster(reference, fragmented)


def test_ls_ssw_builds_auto_neighbor_softening_without_manual_pairs():
    state = State(
        numbers=np.array([6, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(local_softening_mode="neighbor_auto"),
        softening_enabled=True,
    )

    softening = walker._build_softening(state)

    assert softening is not None
    assert len(softening.terms) == 1


def test_ls_ssw_manual_mode_with_empty_pairs_still_disables_softening():
    state = State(
        numbers=np.array([6, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(local_softening_mode="manual", local_softening_pairs=[]),
        softening_enabled=True,
    )

    assert walker._build_softening(state) is None


def test_ls_ssw_auto_neighbor_softening_build_stats_increment_predictably():
    state = State(
        numbers=np.array([6, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(local_softening_mode="neighbor_auto"),
        softening_enabled=True,
    )

    first = walker._build_softening(state)
    second = walker._build_softening(state)

    assert first is not None
    assert second is not None
    assert walker._local_softening_builds == 2
    assert walker._local_softening_terms_last == 1
    assert walker._local_softening_terms_built_total == 2
    assert walker._local_softening_terms_total == 2


def test_ls_ssw_reset_local_softening_stats_resets_all_counters():
    state = State(
        numbers=np.array([6, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.09, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(local_softening_mode="neighbor_auto"),
        softening_enabled=True,
    )
    assert walker._build_softening(state) is not None

    walker._reset_local_softening_stats()

    assert walker._local_softening_terms_last == 0
    assert walker._local_softening_terms_total == 0
    assert walker._local_softening_builds == 0
    assert walker._local_softening_terms_built_total == 0


def test_ls_ssw_zero_neighbor_auto_softening_does_not_increment_build_stats():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(local_softening_mode="neighbor_auto"),
        softening_enabled=True,
    )

    assert walker._build_softening(state) is None
    assert walker._local_softening_builds == 0
    assert walker._local_softening_terms_last == 0
    assert walker._local_softening_terms_built_total == 0
    assert walker._local_softening_terms_total == 0


def test_active_neighbors_select_atoms_from_anchor_direction_displacement():
    state = State(
        numbers=np.array([6, 1, 6, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.09, 0.0, 0.0],
            ]
        ),
    )
    direction = np.zeros(12)
    direction[6] = 10.0
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(local_softening_mode="active_neighbors", local_softening_active_count=1),
        softening_enabled=True,
    )

    assert walker._softening_active_indices(state, direction).tolist() == [2]


def test_active_neighbors_build_softening_from_anchor_direction_displacement():
    state = State(
        numbers=np.array([6, 1, 6, 1]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [5.0, 0.0, 0.0],
                [6.09, 0.0, 0.0],
            ]
        ),
    )
    direction = np.zeros(12)
    direction[6] = 10.0
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(local_softening_mode="active_neighbors", local_softening_active_count=1),
        softening_enabled=True,
    )

    softening = walker._build_softening(state, direction)

    assert softening is not None
    assert [(term.atom_i, term.atom_j) for term in softening.terms] == [(2, 3)]
