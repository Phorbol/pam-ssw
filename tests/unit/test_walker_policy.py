import json

import numpy as np
import pytest

from pamssw import LSSSWConfig, SSWConfig
from pamssw.accounting import BudgetExceeded, EvalCounter, EvaluationPurpose
from pamssw.archive import MinimaArchive
from pamssw.bias import GaussianBiasTerm
from pamssw.calculators import AnalyticCalculator
from pamssw.krylov import IntentBlock, KrylovResult
from pamssw.potentials import DoubleWell2D
from pamssw.result import RelaxOutcomeClass, RelaxResult
from pamssw.state import State
from pamssw.softening import LocalSofteningModel, PairSofteningTerm
from pamssw.walker import (
    CandidateDirectionGenerator,
    ContinuationDirectionDegenerate,
    DirectionCandidateKind,
    DirectionCandidate,
    DirectionScorer,
    DirectionRecord,
    DirectionTypeMemory,
    ProposalRelaxationTask,
    ProposalPotential,
    GeometryValidator,
    BiasStrengthController,
    CandidateProposal,
    SoftModeOracle,
    StepLengthController,
    StepTargetController,
    DirectionChoice,
    SurfaceWalker,
    TrustRegionBiasController,
)


class Quadratic:
    def energy_gradient(self, flat_positions, state):
        gradient = np.asarray(flat_positions, dtype=float).copy()
        energy = 0.5 * float(gradient @ gradient)
        return energy, gradient


def test_walk_exposes_optimizer_neutral_proposal_relaxation_task(monkeypatch):
    class TaskCaptured(RuntimeError):
        pass

    captured = []

    class CapturingWalker(SurfaceWalker):
        def _relax_proposal_task(self, task, *, optimizer, trajectory_callback):
            captured.append((task, optimizer, trajectory_callback))
            raise TaskCaptured

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    walker = CapturingWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            proposal_relax_steps=7,
            proposal_fmax=0.05,
            proposal_optimizer="ase-fire2",
            proposal_trust_radius=0.4,
        ),
        softening_enabled=False,
    )
    direction = np.array([1.0, 0.0, 0.0])
    walker.oracle.generator.generate_initial_direction = lambda *args, **kwargs: direction
    walker.oracle.choose_direction = lambda *args, **kwargs: DirectionChoice(
        direction=direction,
        curvature=-0.5,
        kind=DirectionCandidateKind.RANDOM,
        candidate_count=1,
    )
    monkeypatch.setattr(walker, "_build_softening", lambda *args, **kwargs: None)
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda *args, **kwargs: -0.5)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", lambda *args, **kwargs: -0.5)

    with pytest.raises(TaskCaptured):
        walker._walk_candidate_from_seed(state, trial_index=2, proposal_index=3)

    assert len(captured) == 1
    task, optimizer, trajectory_callback = captured[0]
    assert isinstance(task, ProposalRelaxationTask)
    assert optimizer == "ase-fire2"
    assert trajectory_callback is None
    assert task.fmax == pytest.approx(0.05)
    assert task.maxiter == 7
    assert task.coordinate_trust_radius == pytest.approx(0.4)
    assert task.softening is None
    assert len(task.biases) == 1
    assert not hasattr(task, "optimizer")
    assert task.initial_state.numbers.tolist() == [1]
    assert task.initial_state.positions[0, 0] > state.positions[0, 0]


@pytest.mark.parametrize(
    ("scope", "oracle_softened", "proposal_softened"),
    [
        ("none", False, False),
        ("oracle", True, False),
        ("proposal", False, True),
        ("both", True, True),
    ],
)
def test_walk_routes_local_softening_to_documented_scope(
    monkeypatch,
    scope,
    oracle_softened,
    proposal_softened,
):
    class TaskCaptured(RuntimeError):
        pass

    captured = {}

    class CapturingWalker(SurfaceWalker):
        def _relax_proposal_task(self, task, *, optimizer, trajectory_callback):
            captured["task"] = task
            raise TaskCaptured

    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    walker = CapturingWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            n_bond_pairs=0,
            proposal_relax_steps=1,
            local_softening_mode="manual",
            local_softening_pairs=[(0, 1)],
            local_softening_scope=scope,
        ),
        softening_enabled=True,
    )
    direction = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    direction /= np.linalg.norm(direction)
    monkeypatch.setattr(
        walker.oracle.generator,
        "generate_initial_direction",
        lambda *args, **kwargs: direction,
    )

    def choose_direction(*args, **kwargs):
        captured["oracle_proposal"] = kwargs.get("proposal", args[1] if len(args) > 1 else None)
        return DirectionChoice(
            direction=direction,
            curvature=-0.5,
            true_curvature=-0.25,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        )

    monkeypatch.setattr(walker.oracle, "choose_direction", choose_direction)

    with pytest.raises(TaskCaptured):
        walker._walk_candidate_from_seed(state)

    assert (captured["oracle_proposal"].softening is not None) is oracle_softened
    assert (captured["task"].softening is not None) is proposal_softened


def test_paper_ordered_softening_prerelaxes_once_and_attributes_evaluations():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(
            proposal_optimizer="safe-lbfgs-total",
            proposal_fmax=0.01,
            proposal_relax_steps=80,
            local_softening_protocol="paper_ordered",
            local_softening_mode="manual",
            local_softening_pairs=[(0, 1)],
            local_softening_strength=0.2,
            local_softening_xi=0.2,
            local_softening_cutoff=None,
        ),
        softening_enabled=True,
    )

    prepared, frozen = walker._prepare_frozen_local_softening(state)
    counts = walker.calculator.snapshot()

    assert frozen is not None
    assert frozen.reference_scaled_xi
    assert frozen.terms[0].reference_distance == pytest.approx(1.0)
    assert np.linalg.norm(prepared.positions[1] - prepared.positions[0]) > 1.0
    assert counts.count(EvaluationPurpose.LOCAL_SOFTENING_PRE_RELAX) > 0
    assert counts.count(EvaluationPurpose.UNATTRIBUTED) == 0
    diagnostics = walker.local_softening_diagnostics()
    assert diagnostics["protocol"] == "paper_ordered"
    assert diagnostics["pre_relaxations"] == 1
    assert diagnostics["pre_relax_force_evaluations"] == counts.count(
        EvaluationPurpose.LOCAL_SOFTENING_PRE_RELAX
    )
    assert diagnostics["pre_relax_pls_eV_per_atom"] > 0.0
    assert diagnostics["pre_relax_converged"] == 1
    assert diagnostics["pre_relax_gradient_norm"] <= 0.01


def test_paper_ordered_walk_reuses_frozen_softening_after_prerelax(monkeypatch):
    class TaskCaptured(RuntimeError):
        pass

    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    prepared = State(
        numbers=state.numbers.copy(),
        positions=np.array([[-0.6, 0.0, 0.0], [0.6, 0.0, 0.0]]),
    )
    frozen = LocalSofteningModel.from_state(
        state,
        pairs=[(0, 1)],
        strength=0.2,
        mode="manual",
        penalty="buckingham_repulsive",
        xi=0.2,
        reference_scaled_xi=True,
        cutoff=None,
    )
    captured = {}

    class CapturingWalker(SurfaceWalker):
        def _prepare_frozen_local_softening(self, seed_state):
            return prepared, frozen

        def _build_softening(self, *args, **kwargs):
            raise AssertionError("paper-ordered walk rebuilt its frozen penalty")

        def _relax_proposal_task(self, task, *, optimizer, trajectory_callback):
            captured["task"] = task
            raise TaskCaptured

    walker = CapturingWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            n_bond_pairs=0,
            proposal_relax_steps=1,
            local_softening_protocol="paper_ordered",
            local_softening_mode="manual",
            local_softening_pairs=[(0, 1)],
            local_softening_scope="both",
        ),
        softening_enabled=True,
    )
    direction = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    direction /= np.linalg.norm(direction)
    monkeypatch.setattr(
        walker.oracle.generator,
        "generate_initial_direction",
        lambda *args, **kwargs: direction,
    )

    def choose_direction(chosen_state, proposal, *args, **kwargs):
        captured["oracle_state"] = chosen_state
        captured["oracle_softening"] = proposal.softening
        return DirectionChoice(
            direction=direction,
            curvature=-0.5,
            true_curvature=-0.25,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        )

    monkeypatch.setattr(walker.oracle, "choose_direction", choose_direction)

    with pytest.raises(TaskCaptured):
        walker._walk_candidate_from_seed(state)

    assert captured["oracle_state"] is prepared
    assert captured["oracle_softening"] is frozen
    assert captured["task"].softening is not frozen
    assert captured["task"].softening.terms == frozen.terms


def test_proposal_relaxation_task_snapshots_state_and_bias_arrays():
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    center = np.array([1.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])
    task = ProposalRelaxationTask(
        initial_state=state,
        biases=(
            GaussianBiasTerm(
                center=center,
                direction=direction,
                sigma=0.2,
                weight=0.4,
            ),
        ),
        softening=None,
        fmax=0.05,
        maxiter=10,
        coordinate_trust_radius=None,
    )

    state.positions[0, 0] = 9.0
    center[0] = 8.0
    direction[0] = -1.0

    np.testing.assert_allclose(task.initial_state.positions, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(task.biases[0].center, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(task.biases[0].direction, [1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="read-only"):
        task.initial_state.positions[0, 0] = 7.0
    with pytest.raises(ValueError, match="read-only"):
        task.biases[0].center[0] = 7.0


def test_proposal_relaxation_task_defensively_copies_softening_model():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    softening = LocalSofteningModel(
        [
            PairSofteningTerm(
                atom_i=0,
                atom_j=1,
                reference_distance=1.0,
                width=0.2,
                strength=0.3,
            )
        ]
    )
    task = ProposalRelaxationTask(
        initial_state=state,
        biases=(),
        softening=softening,
        fmax=0.05,
        maxiter=10,
        coordinate_trust_radius=None,
    )

    softening.terms.clear()

    assert task.softening is not softening
    assert len(task.softening.terms) == 1


def test_bias_weight_matches_curvature_inversion_rule():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(target_negative_curvature=0.2),
        softening_enabled=False,
    )

    assert walker._bias_weight(curvature=0.3, sigma=2.0) == 2.0
    assert walker._bias_weight(curvature=-0.5, sigma=2.0) == 0.0


def test_gaussian_bias_uses_minimum_image_displacement_for_periodic_centers():
    bias = GaussianBiasTerm(
        center=np.array([9.5, 0.0, 0.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        sigma=0.5,
        weight=2.0,
    )
    cell = np.diag([10.0, 10.0, 10.0])

    energy_inside, gradient_inside = bias.evaluate(
        np.array([9.8, 0.0, 0.0]),
        cell=cell,
        pbc=(True, True, True),
    )
    energy_wrapped, gradient_wrapped = bias.evaluate(
        np.array([-0.2, 0.0, 0.0]),
        cell=cell,
        pbc=(True, True, True),
    )

    assert energy_wrapped == pytest.approx(energy_inside)
    np.testing.assert_allclose(gradient_wrapped, gradient_inside)


def test_gaussian_bias_rejects_mismatched_position_shape():
    bias = GaussianBiasTerm(
        center=np.zeros(6),
        direction=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        sigma=0.5,
        weight=2.0,
    )

    with pytest.raises(ValueError, match="same shape"):
        bias.evaluate(np.zeros(3))


def test_proposal_parts_preserve_tuple_api_fixed_atoms_and_stable_mic_branch():
    class CountingConstantCalculator:
        def __init__(self):
            self.calls = 0

        def evaluate_flat(self, flat_positions, template):
            self.calls += 1
            return 2.5, np.full_like(flat_positions, 0.75)

    state = State(
        numbers=np.array([1]),
        positions=np.array([[9.8, 0.0, 0.0]]),
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=(True, True, True),
        fixed_mask=np.array([True]),
    )
    wrapped = state.with_flat_positions(np.array([-0.2, 0.0, 0.0]))
    calculator = CountingConstantCalculator()
    potential = ProposalPotential(
        calculator,
        biases=[
            GaussianBiasTerm(
                center=np.array([9.5, 0.0, 0.0]),
                direction=np.array([1.0, 0.0, 0.0]),
                sigma=0.5,
                weight=2.0,
            )
        ],
    )

    assert hasattr(potential, "evaluate_parts")
    inside_parts = potential.evaluate_parts(state.flatten_positions(), state)
    wrapped_parts = potential.evaluate_parts(wrapped.flatten_positions(), wrapped)
    energy, gradient = potential.evaluate(state.flatten_positions(), state)

    assert calculator.calls == 3
    assert inside_parts.bias_energy == pytest.approx(wrapped_parts.bias_energy)
    np.testing.assert_allclose(inside_parts.bias_gradient, wrapped_parts.bias_gradient)
    assert inside_parts.softening_energy == 0.0
    np.testing.assert_allclose(inside_parts.softening_gradient, np.zeros(3))
    np.testing.assert_allclose(inside_parts.total_gradient, np.full(3, 0.75) + inside_parts.bias_gradient)
    assert energy == pytest.approx(inside_parts.total_energy)
    np.testing.assert_allclose(gradient, inside_parts.total_gradient)
    assert gradient.flags.writeable
    gradient[0] = gradient[0] + 1.0
    assert gradient[0] != inside_parts.total_gradient[0]


def test_surface_walker_uses_configured_step_length_controller_controls():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(step_error_tolerance=2.0, step_gamma_down=0.7, step_gamma_up=1.3),
        softening_enabled=False,
    )

    controller = walker.trust_controller.step_length
    assert controller.error_tolerance == pytest.approx(2.0)
    assert controller.gamma_down == pytest.approx(0.7)
    assert controller.gamma_up == pytest.approx(1.3)


def test_relax_true_minimum_uses_configured_quench_maxiter(monkeypatch):
    recorded: dict[str, int] = {}

    class RecordingRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator
            self.optimizer = optimizer

        def relax(self, state, fmax, maxiter, trajectory_callback=None, trajectory_stride=1):
            recorded["maxiter"] = maxiter
            return RelaxResult(state=state, energy=0.0, gradient_norm=0.0, n_iter=0)

    monkeypatch.setattr("pamssw.walker.Relaxer", RecordingRelaxer)
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(quench_maxiter=123),
        softening_enabled=False,
    )

    walker.relax_true_minimum(state)

    assert recorded["maxiter"] == 123


def test_relax_true_minimum_records_only_fallback_final_and_charges_both_attempts(
    monkeypatch,
):
    starts: dict[str, list[float]] = {"scipy-lbfgsb": [], "ase-fire": []}

    class ScriptedRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator
            self.optimizer = optimizer

        def relax(self, state, fmax, maxiter, trajectory_callback=None, trajectory_stride=1):
            starts[self.optimizer].append(float(state.positions[0, 0]))
            self.evaluator(state.flatten_positions(), state)
            if self.optimizer == "scipy-lbfgsb":
                terminal = State(
                    numbers=state.numbers.copy(),
                    positions=np.array([[1.0, 0.0, 0.0]]),
                )
                return RelaxResult(
                    state=terminal,
                    energy=1.0,
                    gradient_norm=2.0 * fmax,
                    n_iter=20,
                )
            terminal = State(
                numbers=state.numbers.copy(),
                positions=np.array([[2.0, 0.0, 0.0]]),
            )
            return RelaxResult(
                state=terminal,
                energy=0.5,
                gradient_norm=0.5 * fmax,
                n_iter=4,
            )

    monkeypatch.setattr("pamssw.walker.Relaxer", ScriptedRelaxer)
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            quench_fmax=0.01,
            quench_fallback_optimizer="ase-fire",
        ),
        softening_enabled=False,
    )

    result = walker.relax_true_minimum(
        State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    )

    assert result.gradient_norm == pytest.approx(0.005)
    assert starts == {"scipy-lbfgsb": [0.0], "ase-fire": [1.0]}
    counts = walker.calculator.snapshot()
    assert counts.count(EvaluationPurpose.LANDING_TRUE_QUENCH) == 2
    assert counts.count(EvaluationPurpose.POST_RELAX_VALIDATION) == 1
    diagnostics = walker.relaxation_diagnostics()
    assert diagnostics["true_quench_count"] == 1
    assert diagnostics["true_quench_mean_iterations"] == pytest.approx(4.0)
    assert diagnostics["true_quench_unconverged"] == 0
    assert diagnostics["quench_fallback_optimizer"] == "ase-fire"
    assert diagnostics["quench_fallback_attempts"] == 1
    assert diagnostics["quench_fallback_converged"] == 1


def test_relax_true_minimum_counts_uncertified_fallback_as_final_failure(monkeypatch):
    class UnconvergedRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator
            self.optimizer = optimizer

        def relax(self, state, fmax, maxiter, trajectory_callback=None, trajectory_stride=1):
            self.evaluator(state.flatten_positions(), state)
            return RelaxResult(
                state=state,
                energy=0.0,
                gradient_norm=2.0 * fmax if self.optimizer == "scipy-lbfgsb" else float("nan"),
                n_iter=20 if self.optimizer == "scipy-lbfgsb" else 7,
            )

    monkeypatch.setattr("pamssw.walker.Relaxer", UnconvergedRelaxer)
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            quench_fmax=0.01,
            quench_fallback_optimizer="ase-fire",
        ),
        softening_enabled=False,
    )

    walker.relax_true_minimum(
        State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    )

    diagnostics = walker.relaxation_diagnostics()
    assert diagnostics["true_quench_count"] == 1
    assert diagnostics["true_quench_mean_iterations"] == pytest.approx(7.0)
    assert diagnostics["true_quench_unconverged"] == 1
    assert diagnostics["quench_fallback_attempts"] == 1
    assert diagnostics["quench_fallback_converged"] == 0


def test_relax_true_minimum_counts_fallback_started_before_budget_exhaustion(
    monkeypatch,
):
    optimizer_calls: list[str] = []

    class BudgetedRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator
            self.optimizer = optimizer

        def relax(self, state, fmax, maxiter, trajectory_callback=None, trajectory_stride=1):
            optimizer_calls.append(self.optimizer)
            self.evaluator(state.flatten_positions(), state)
            return RelaxResult(
                state=state,
                energy=0.0,
                gradient_norm=2.0 * fmax,
                n_iter=1,
            )

    monkeypatch.setattr("pamssw.walker.Relaxer", BudgetedRelaxer)
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            quench_fmax=0.01,
            quench_fallback_optimizer="ase-fire",
            max_force_evals=1,
        ),
        softening_enabled=False,
    )

    with pytest.raises(BudgetExceeded, match="force-evaluation budget exhausted") as captured:
        walker.relax_true_minimum(
            State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
        )

    assert optimizer_calls == ["scipy-lbfgsb", "ase-fire"]
    diagnostics = walker.relaxation_diagnostics()
    assert diagnostics["quench_fallback_attempts"] == 1
    assert diagnostics["quench_fallback_converged"] == 0
    counts = walker.calculator.snapshot()
    assert counts.count(EvaluationPurpose.LANDING_TRUE_QUENCH) == 1
    assert counts.count(EvaluationPurpose.POST_RELAX_VALIDATION) == 0
    assert captured.value.evaluation_counts == counts


def test_direction_scoring_proposal_can_ignore_inner_bias_curvature():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    calculator = AnalyticCalculator(Quadratic())
    direction = np.array([1.0, 0.0, 0.0])
    inner = ProposalPotential(
        calculator,
        biases=[
            GaussianBiasTerm(
                center=state.flatten_positions(),
                direction=direction,
                sigma=1.0,
                weight=2.0,
            )
        ],
    )
    walker = SurfaceWalker(
        calculator=calculator,
        config=SSWConfig(direction_curvature_source="true"),
        softening_enabled=False,
    )

    true_scoring = walker._direction_scoring_proposal(inner)

    assert walker.oracle._directional_curvature(state, true_scoring, direction) == pytest.approx(1.0)
    assert walker.oracle._directional_curvature(state, inner, direction) == pytest.approx(-1.0)


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


def test_soft_mode_oracle_scores_all_candidates_with_fixed_reference_sigma():
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
            self.curvatures = []

        def score_candidate(self, **kwargs):
            self.sigmas.append(kwargs["sigma"])
            self.curvatures.append(kwargs["curvature"])
            return super().score_candidate(**kwargs)

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), np.random.default_rng(0), candidates=2)
    oracle.scorer = CapturingScorer()

    oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=np.array([1.0, 0.0, 0.0]),
        step_scale_fn=lambda curvature: 10.0 + curvature,
    )

    assert len(oracle.scorer.sigmas) >= 2
    assert len({round(curvature, 6) for curvature in oracle.scorer.curvatures}) >= 2
    assert all(sigma == pytest.approx(11.0) for sigma in oracle.scorer.sigmas)


def test_soft_mode_oracle_allows_explicit_score_sigma_override():
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
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), np.random.default_rng(0), candidates=2)
    oracle.scorer = CapturingScorer()

    oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=np.array([1.0, 0.0, 0.0]),
        step_scale_fn=lambda curvature: 10.0 + curvature,
        score_sigma=3.5,
    )

    assert len(oracle.scorer.sigmas) >= 2
    assert all(sigma == pytest.approx(3.5) for sigma in oracle.scorer.sigmas)


def test_soft_mode_oracle_can_score_candidates_with_adaptive_sigma():
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
            self.curvatures = []

        def score_candidate(self, **kwargs):
            self.sigmas.append(kwargs["sigma"])
            self.curvatures.append(kwargs["curvature"])
            return super().score_candidate(**kwargs)

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), np.random.default_rng(0), candidates=2)
    oracle.scorer = CapturingScorer()

    oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=np.array([1.0, 0.0, 0.0]),
        score_sigma_fn=lambda curvature: 10.0 + curvature,
    )

    assert len(oracle.scorer.sigmas) >= 2
    assert len({round(curvature, 6) for curvature in oracle.scorer.curvatures}) >= 2
    assert len({round(sigma, 6) for sigma in oracle.scorer.sigmas}) >= 2
    for curvature, sigma in zip(oracle.scorer.curvatures, oracle.scorer.sigmas):
        assert sigma == pytest.approx(10.0 + curvature)


def test_soft_mode_oracle_can_select_rayleigh_ritz_subspace_direction():
    class CoupledQuadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.array(
                [
                    [1.0, -0.8, 0.0],
                    [-0.8, 1.0, 0.0],
                    [0.0, 0.0, 5.0],
                ]
            )
            gradient = hessian @ flat_positions
            energy = 0.5 * float(flat_positions @ gradient)
            return energy, gradient

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(CoupledQuadratic()),
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="rayleigh_ritz",
    )

    def fixed_candidates(*args, **kwargs):
        return [
            DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
            DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([0.0, 1.0, 0.0])),
        ]

    oracle.generator.generate = fixed_candidates
    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(CoupledQuadratic())),
        previous_direction=None,
        score_sigma=1.0,
    )

    assert choice.kind == DirectionCandidateKind.RITZ
    assert choice.curvature == pytest.approx(0.2, rel=1e-5)
    assert abs(float(np.dot(choice.direction, np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)))) == pytest.approx(1.0)


def test_rayleigh_ritz_reuses_native_true_hvps_for_projected_true_curvature(monkeypatch):
    """The Ritz vector inherits a native-HVP subspace projection of true curvature."""

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="rayleigh_ritz",
    )
    candidates = [
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([0.0, 1.0, 0.0])),
    ]
    total_hessian = np.array([[1.0, -0.8, 0.0], [-0.8, 1.0, 0.0], [0.0, 0.0, 5.0]])
    true_hessian = np.diag([1.0, 4.0, 9.0])
    monkeypatch.setattr(oracle.generator, "generate", lambda *args, **kwargs: candidates)
    monkeypatch.setattr(
        oracle,
        "_candidate_directional_hvps",
        lambda _state, _proposal, direction: (total_hessian @ direction, true_hessian @ direction),
    )

    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=None,
        score_sigma=1.0,
    )

    assert choice.kind == DirectionCandidateKind.RITZ
    assert choice.curvature == pytest.approx(0.2, rel=1e-12)
    assert choice.true_curvature == pytest.approx(2.5, rel=1e-12)


def test_block_krylov_selects_lowest_block_without_native_scoring(monkeypatch):
    class AnisotropicQuadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.diag([1.0, 3.0, 7.0])
            gradient = hessian @ flat_positions
            return 0.5 * float(flat_positions @ gradient), gradient

    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    calculator = AnalyticCalculator(AnisotropicQuadratic())
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(0),
        candidates=2,
        direction_selection_mode="block_krylov",
        block_krylov_depth=3,
    )
    monkeypatch.setattr(oracle.generator, "generate", lambda *args, **kwargs: pytest.fail("native generation ran"))
    monkeypatch.setattr(
        DirectionScorer,
        "score_candidate",
        lambda *args, **kwargs: pytest.fail("native scoring ran"),
    )

    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(calculator),
        previous_direction=None,
        anchor_direction=np.array([1.0, 1.0, 0.0])
        / np.sqrt(2.0),
        krylov_intents=(
            IntentBlock(np.array([[1.0], [0.0], [0.0]])),
            IntentBlock(np.array([[0.0], [1.0], [0.0]])),
        ),
    )

    assert choice.kind is DirectionCandidateKind.BLOCK_RITZ
    assert choice.score is None
    assert choice.candidate_count == 0
    assert choice.curvature == pytest.approx(1.0)
    assert choice.true_curvature == pytest.approx(1.0)
    assert set(choice.diagnostics) == {
        "krylov_blocks",
        "krylov_depth",
        "krylov_selected_block",
        "krylov_hvp_count",
        "krylov_hvp_requested",
        "krylov_hvp_consumed",
        "krylov_initial_basis_columns",
        "krylov_dimensions",
        "krylov_initial_ranks",
        "krylov_residual_norm",
        "krylov_initial_span_overlap",
        "krylov_antisymmetry",
        "krylov_termination",
        "direction_participation_ratio",
        "krylov_ritz_spectrum",
    }
    assert choice.diagnostics["krylov_blocks"] == 2
    assert choice.diagnostics["krylov_depth"] == 3
    assert choice.diagnostics["krylov_selected_block"] == 0
    assert choice.diagnostics["krylov_hvp_count"] == 2
    assert choice.diagnostics["krylov_hvp_requested"] == 6
    assert choice.diagnostics["krylov_hvp_consumed"] == 2
    assert choice.diagnostics["krylov_initial_basis_columns"] == [1, 1]
    assert choice.diagnostics["krylov_dimensions"] == [1, 1]
    assert choice.diagnostics["krylov_initial_ranks"] == [1, 1]
    assert choice.diagnostics["direction_participation_ratio"] == pytest.approx(1.0)
    spectrum = choice.diagnostics["krylov_ritz_spectrum"]
    assert len(spectrum) == sum(
        choice.diagnostics["krylov_dimensions"]
    )
    assert sum(point["executed"] for point in spectrum) == 1
    assert spectrum[0]["executed"] is True
    assert [
        point["curvature"] for point in spectrum
    ] == pytest.approx([1.0, 3.0])
    assert [
        point["anchor_abs_overlap"] for point in spectrum
    ] == pytest.approx([1.0 / np.sqrt(2.0)] * 2)
    assert all(
        point["participation_ratio"] == pytest.approx(1.0)
        for point in spectrum
    )
    assert choice.diagnostics["krylov_hvp_count"] == 2


def test_transport_direction_projects_aligns_and_spends_one_hvp():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            direction_selection_mode="transported_direction",
            n_bond_pairs=0,
        ),
        softening_enabled=False,
    )
    previous = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    proposal = ProposalPotential(walker.calculator)
    before = walker.calculator.snapshot().count(
        EvaluationPurpose.DIRECTION_ORACLE
    )

    with walker.calculator.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        choice = walker.oracle.choose_transported_direction(
            state,
            proposal,
            -previous,
            previous,
        )

    after = walker.calculator.snapshot().count(
        EvaluationPurpose.DIRECTION_ORACLE
    )
    assert after - before == 2
    assert np.dot(choice.direction, previous) > 0.0
    assert choice.kind is DirectionCandidateKind.TRANSPORTED
    assert choice.diagnostics["direction_hvp_count"] == 1
    assert choice.diagnostics["continuation_source"] == "selected_mode"


def test_transport_direction_reports_existing_hvp_eigen_residual_without_extra_cost():
    class AnisotropicQuadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.diag(np.arange(1.0, 10.0))
            gradient = hessian @ flat_positions
            return 0.5 * float(flat_positions @ gradient), gradient

    state = State(
        numbers=np.array([6, 6, 6]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 1.2, 0.0],
            ]
        ),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(AnisotropicQuadratic()),
        config=SSWConfig(
            direction_selection_mode="transported_direction",
            n_bond_pairs=0,
        ),
        softening_enabled=False,
    )
    direction = np.arange(1.0, 10.0)
    proposal = ProposalPotential(walker.calculator)
    before = walker.calculator.snapshot().count(
        EvaluationPurpose.DIRECTION_ORACLE
    )

    with walker.calculator.purpose(EvaluationPurpose.DIRECTION_ORACLE):
        choice = walker.oracle.choose_transported_direction(
            state,
            proposal,
            direction,
            direction,
        )

    after = walker.calculator.snapshot().count(
        EvaluationPurpose.DIRECTION_ORACLE
    )
    hessian = np.diag(np.arange(1.0, 10.0))
    total_hvp = hessian @ choice.direction
    curvature = float(choice.direction @ total_hvp)
    residual_norm = float(
        np.linalg.norm(total_hvp - curvature * choice.direction)
    )
    relative_residual = residual_norm / float(np.linalg.norm(total_hvp))

    assert after - before == 2
    assert choice.diagnostics["transported_hvp_norm"] == pytest.approx(
        np.linalg.norm(total_hvp)
    )
    assert choice.diagnostics[
        "transported_residual_norm"
    ] == pytest.approx(residual_norm)
    assert choice.diagnostics[
        "transported_relative_residual"
    ] == pytest.approx(relative_residual)
    assert choice.diagnostics[
        "transported_true_residual_norm"
    ] == pytest.approx(residual_norm)
    assert choice.diagnostics[
        "transported_true_relative_residual"
    ] == pytest.approx(relative_residual)


def test_continuation_projection_rejects_direction_removed_by_fixed_mask():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        fixed_mask=np.array([True, True]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            direction_selection_mode="transported_direction",
            n_bond_pairs=0,
        ),
        softening_enabled=False,
    )
    direction = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])

    with pytest.raises(
        ContinuationDirectionDegenerate,
        match="vanished after projection",
    ):
        walker.oracle.choose_transported_direction(
            state,
            ProposalPotential(walker.calculator),
            direction,
            direction,
        )


def test_continuation_walk_has_common_first_step_and_keeps_selected_mode(
    monkeypatch,
    tmp_path,
):
    class ThreeAtomQuadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.diag(np.arange(1.0, 10.0))
            gradient = hessian @ flat_positions
            return 0.5 * float(flat_positions @ gradient), gradient

    state = State(
        numbers=np.array([6, 6, 6]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 1.2, 0.0],
            ]
        ),
    )
    rows_by_arm = {}
    arm_settings = {
        "fixed_intent_ritz": ("block_krylov", 6),
        "transported_direction": ("transported_direction", 6),
        "continuation_lanczos": ("continuation_krylov", 12),
        "continuation_intent_ritz2": (
            "continuation_intent_krylov",
            1,
        ),
    }

    for arm, (mode, depth) in arm_settings.items():
        diagnostic_path = tmp_path / f"{arm}.jsonl"
        walker = SurfaceWalker(
            calculator=AnalyticCalculator(ThreeAtomQuadratic()),
            config=SSWConfig(
                rng_seed=42,
                max_steps_per_walk=2,
                oracle_candidates=1,
                n_bond_pairs=0,
                proposal_relax_steps=0,
                direction_selection_mode=mode,
                block_krylov_blocks=1,
                block_krylov_depth=depth,
                direction_curvature_source="true",
                target_negative_curvature=10.0,
                direction_diagnostics_enabled=True,
                direction_diagnostics_path=str(diagnostic_path),
            ),
            softening_enabled=False,
        )

        def relax_with_transverse_component(task, **kwargs):
            positions = task.initial_state.positions.copy()
            positions[0, 1] += 0.02
            positions[1, 1] -= 0.02
            return RelaxResult(
                task.initial_state.with_flat_positions(positions.reshape(-1)),
                energy=0.0,
                gradient_norm=0.0,
                n_iter=0,
            )

        monkeypatch.setattr(
            walker,
            "_relax_proposal_task",
            relax_with_transverse_component,
        )
        walker._walk_candidate_from_seed(state)
        rows_by_arm[arm] = [
            json.loads(line)
            for line in diagnostic_path.read_text().splitlines()
        ]

    first_hashes = {
        rows[0]["selected_direction_sha256"]
        for rows in rows_by_arm.values()
    }
    assert len(first_hashes) == 1
    assert all(
        rows[0]["selected_kind"] == "block_ritz"
        for rows in rows_by_arm.values()
    )
    assert (
        rows_by_arm["transported_direction"][1][
            "direction_hvp_count"
        ]
        == 1
    )
    assert (
        rows_by_arm["transported_direction"][1][
            "continuation_source"
        ]
        == "selected_mode"
    )
    assert (
        rows_by_arm["transported_direction"][1][
            "direction_participation_ratio"
        ]
        > 0.0
    )
    assert all(
        row["executed_step_scale"] > 0.0
        for rows in rows_by_arm.values()
        for row in rows
    )
    assert (
        rows_by_arm["continuation_lanczos"][1][
            "krylov_initial_basis_columns"
        ]
        == [1]
    )
    assert (
        rows_by_arm["continuation_lanczos"][1][
            "continuation_source"
        ]
        == "selected_mode"
    )
    assert (
        rows_by_arm["continuation_intent_ritz2"][1][
            "krylov_initial_basis_columns"
        ]
        == [2]
    )
    assert (
        rows_by_arm["continuation_intent_ritz2"][1][
            "krylov_hvp_count"
        ]
        == 2
    )
    assert (
        rows_by_arm["continuation_intent_ritz2"][1][
            "oracle_selection_force_evaluations_delta"
        ]
        == 4
    )
    assert (
        rows_by_arm["continuation_intent_ritz2"][1][
            "continuation_source"
        ]
        == "selected_mode_plus_initial_intent"
    )
    selected_cosine = rows_by_arm["transported_direction"][1][
        "selected_to_previous_selected_abs_cosine"
    ]
    relaxed_cosine = rows_by_arm["transported_direction"][1][
        "selected_to_previous_relaxed_abs_cosine"
    ]
    assert selected_cosine > 0.999
    assert selected_cosine > relaxed_cosine


def test_continuation_walk_stops_without_fallback_on_degenerate_projection(
    monkeypatch,
):
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            rng_seed=42,
            max_steps_per_walk=2,
            oracle_candidates=1,
            n_bond_pairs=0,
            proposal_relax_steps=0,
            direction_selection_mode="transported_direction",
            block_krylov_blocks=1,
            block_krylov_depth=6,
            target_negative_curvature=10.0,
        ),
        softening_enabled=False,
    )
    direction = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    monkeypatch.setattr(
        walker.oracle,
        "_choose_block_krylov_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=direction / np.linalg.norm(direction),
            curvature=1.0,
            kind=DirectionCandidateKind.BLOCK_RITZ,
            candidate_count=0,
            true_curvature=1.0,
        ),
    )
    monkeypatch.setattr(
        walker.oracle,
        "choose_transported_direction",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ContinuationDirectionDegenerate(
                "continuation direction vanished after projection"
            )
        ),
    )
    monkeypatch.setattr(
        walker,
        "_relax_proposal_task",
        lambda task, **kwargs: RelaxResult(
            task.initial_state,
            energy=0.0,
            gradient_norm=0.0,
            n_iter=0,
        ),
    )

    result = walker._walk_candidate_from_seed(state)

    assert isinstance(result, State)
    assert (
        walker._direction_stats_summary()[
            "continuation_projection_degenerate"
        ]
        == 1
    )


def test_walk_executes_a_shared_initial_direction_without_recomputing_it(
    monkeypatch,
    tmp_path,
):
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]]),
    )
    diagnostic_path = tmp_path / "shared-initial.jsonl"
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            rng_seed=42,
            max_steps_per_walk=1,
            oracle_candidates=1,
            n_bond_pairs=0,
            proposal_relax_steps=0,
            direction_selection_mode="transported_direction",
            block_krylov_blocks=1,
            block_krylov_depth=6,
            target_negative_curvature=10.0,
            direction_diagnostics_enabled=True,
            direction_diagnostics_path=str(diagnostic_path),
        ),
        softening_enabled=False,
    )
    direction = (
        np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
        / np.sqrt(2.0)
    )
    shared_choice = DirectionChoice(
        direction=direction,
        curvature=1.0,
        kind=DirectionCandidateKind.BLOCK_RITZ,
        candidate_count=0,
        true_curvature=1.0,
        diagnostics={
            "krylov_blocks": 1,
            "krylov_depth": 6,
            "krylov_hvp_count": 12,
            "krylov_hvp_requested": 12,
            "krylov_hvp_consumed": 12,
            "krylov_initial_basis_columns": [2],
        },
    )
    monkeypatch.setattr(
        walker.oracle,
        "_choose_block_krylov_direction",
        lambda *args, **kwargs: pytest.fail(
            "shared initial direction was recomputed"
        ),
    )
    monkeypatch.setattr(
        walker,
        "_relax_proposal_task",
        lambda task, **kwargs: RelaxResult(
            task.initial_state,
            energy=0.0,
            gradient_norm=0.0,
            n_iter=0,
        ),
    )

    walker._walk_candidate_from_seed(
        state,
        initial_direction_choice=shared_choice,
    )

    [row] = [
        json.loads(line)
        for line in diagnostic_path.read_text().splitlines()
    ]
    assert row["shared_initial_direction"] is True
    assert row["oracle_selection_force_evaluations_delta"] == 0
    assert row["selected_direction_sha256"] == (
        SurfaceWalker._continuation_diagnostics(
            direction,
            None,
            None,
            direction,
        )["selected_direction_sha256"]
    )


@pytest.mark.parametrize(
    ("force_softening_rebuild", "extra_direction_evaluations"),
    [(False, 2), (True, 2)],
)
def test_block_krylov_walk_reuses_one_intent_batch_and_records_exact_diagnostics(
    monkeypatch,
    tmp_path,
    force_softening_rebuild,
    extra_direction_evaluations,
):
    class TwoAtomQuadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            gradient = hessian @ flat_positions
            return 0.5 * float(flat_positions @ gradient), gradient

    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]]),
    )
    diagnostic_path = tmp_path / "block_krylov_directions.jsonl"
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(TwoAtomQuadratic()),
        config=SSWConfig(
            max_steps_per_walk=2,
            oracle_candidates=1,
            n_bond_pairs=0,
            proposal_relax_steps=0,
            direction_selection_mode="block_krylov",
            block_krylov_blocks=2,
            block_krylov_depth=2,
            direction_curvature_source="true",
            target_negative_curvature=10.0,
            direction_diagnostics_enabled=True,
            direction_diagnostics_path=str(diagnostic_path),
        ),
        softening_enabled=False,
    )
    generated_intents = []
    original_generate = walker.oracle.generator.generate_krylov_intents
    original_choose = walker.oracle.choose_direction
    original_directional_curvature = walker.oracle._directional_curvature
    original_bias_weight = walker._bias_weight
    chosen_intents = []
    inner_curvatures = []
    bias_weight_inputs = []

    def generate_krylov_intents(current, *, n_blocks):
        intents = original_generate(current, n_blocks=n_blocks)
        generated_intents.append(intents)
        return intents

    def choose_direction(*args, **kwargs):
        chosen_intents.append(kwargs["krylov_intents"])
        return original_choose(*args, **kwargs)

    def record_inner_curvature(current, proposal, direction):
        curvature = original_directional_curvature(current, proposal, direction)
        inner_curvatures.append((len(proposal.biases), curvature))
        return curvature

    def record_bias_weight(curvature, sigma):
        weight = original_bias_weight(curvature, sigma)
        bias_weight_inputs.append((curvature, sigma, weight))
        return weight

    def relax_to_newest_bias_center(task, **kwargs):
        return RelaxResult(
            task.initial_state.with_flat_positions(task.biases[-1].center),
            energy=0.0,
            gradient_norm=0.0,
            n_iter=0,
        )

    monkeypatch.setattr(walker.oracle.generator, "generate_krylov_intents", generate_krylov_intents)
    monkeypatch.setattr(walker.oracle, "choose_direction", choose_direction)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", record_inner_curvature)
    monkeypatch.setattr(walker, "_bias_weight", record_bias_weight)
    if force_softening_rebuild:
        monkeypatch.setattr(
            walker,
            "_should_rebuild_softening_for_choice",
            lambda *args, **kwargs: True,
        )
    monkeypatch.setattr(
        walker,
        "_relax_proposal_task",
        relax_to_newest_bias_center,
    )

    walker._walk_candidate_from_seed(state)

    assert len(generated_intents) == 1
    assert chosen_intents == [generated_intents[0], generated_intents[0]]
    assert all(intents is generated_intents[0] for intents in chosen_intents)
    rows = [json.loads(line) for line in diagnostic_path.read_text().splitlines()]
    assert len(rows) == 2
    required = {
        "selected_kind",
        "krylov_selected_block",
        "selected_curvature",
        "true_curvature",
        "krylov_residual_norm",
        "krylov_initial_span_overlap",
        "direction_participation_ratio",
        "krylov_antisymmetry",
        "krylov_hvp_requested",
        "krylov_hvp_consumed",
        "oracle_selection_force_evaluations_delta",
        "oracle_selection_wall_seconds",
        "oracle_direction_force_evaluations_delta",
        "oracle_wall_seconds",
    }
    assert all(required <= set(row) for row in rows)
    assert all(row["selected_kind"] == "block_ritz" for row in rows)
    assert all(row["candidate_count"] == 0 for row in rows)
    assert all(row["krylov_blocks"] > 0 for row in rows)
    requested_hvps = sum(intent.basis.shape[1] for intent in generated_intents[0]) * 2
    assert all(row["krylov_hvp_requested"] == requested_hvps for row in rows)
    assert all(row["krylov_hvp_consumed"] >= 1 for row in rows)
    assert all(
        row["oracle_selection_force_evaluations_delta"] == 2 * row["krylov_hvp_consumed"]
        for row in rows
    )
    assert all(
        row["oracle_direction_force_evaluations_delta"]
        == row["oracle_selection_force_evaluations_delta"] + extra_direction_evaluations
        for row in rows
    )
    assert all(
        row["oracle_wall_seconds"] >= row["oracle_selection_wall_seconds"] >= 0.0
        for row in rows
    )
    assert [bias_count for bias_count, _ in inner_curvatures] == [0, 1]
    assert [curvature for curvature, _, _ in bias_weight_inputs] == pytest.approx(
        [curvature for _, curvature in inner_curvatures]
    )
    assert inner_curvatures[1][1] < rows[1]["selected_curvature"] - 5.0
    assert bias_weight_inputs[1][2] == pytest.approx(
        original_bias_weight(inner_curvatures[1][1], bias_weight_inputs[1][1])
    )
    assert walker.calculator.snapshot().count(EvaluationPurpose.DIRECTION_ORACLE) == sum(
        row["oracle_direction_force_evaluations_delta"] for row in rows
    )

    stats = walker._direction_stats_summary()
    assert stats["direction_candidate_evaluations"] == 0


def test_discrete_walk_never_generates_or_passes_block_krylov_intents(monkeypatch):
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.75, 0.0, 0.0], [0.75, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            n_bond_pairs=0,
            proposal_relax_steps=0,
            direction_curvature_source="true",
        ),
        softening_enabled=False,
    )
    saw_krylov_keyword = []
    choose_direction = walker.oracle.choose_direction
    directional_curvature = walker.oracle._directional_curvature
    curvature_recomputations = []
    monkeypatch.setattr(
        walker.oracle.generator,
        "generate_krylov_intents",
        lambda *args, **kwargs: pytest.fail("discrete walk generated Krylov intents"),
    )

    def record_choose(*args, **kwargs):
        saw_krylov_keyword.append("krylov_intents" in kwargs)
        return choose_direction(*args, **kwargs)

    def record_curvature(*args, **kwargs):
        curvature_recomputations.append(True)
        return directional_curvature(*args, **kwargs)

    monkeypatch.setattr(walker.oracle, "choose_direction", record_choose)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", record_curvature)
    monkeypatch.setattr(
        walker,
        "_relax_proposal_task",
        lambda task, **kwargs: RelaxResult(task.initial_state, energy=0.0, gradient_norm=0.0, n_iter=0),
    )

    walker._walk_candidate_from_seed(state)

    assert saw_krylov_keyword == [False]
    assert curvature_recomputations == [True]


def test_direction_diagnostics_reject_nonfinite_values(tmp_path):
    path = tmp_path / "direction_trace.jsonl"
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_diagnostics_enabled=True, direction_diagnostics_path=str(path)),
        softening_enabled=False,
    )

    with pytest.raises(ValueError, match="Out of range float values"):
        walker._record_direction_diagnostics(
            trial_index=0,
            proposal_index=0,
            step_index=0,
            choice=DirectionChoice(
                direction=np.array([1.0, 0.0, 0.0]),
                curvature=1.0,
                kind=DirectionCandidateKind.RANDOM,
                candidate_count=1,
                diagnostics={"nonfinite": np.nan},
            ),
            anchor_direction=None,
        )


def test_block_krylov_requires_nonempty_intents():
    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    calculator = AnalyticCalculator(Quadratic())
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="block_krylov",
    )

    for intents in (None, (), (object(),)):
        with pytest.raises(ValueError, match="krylov_intents"):
            oracle.choose_direction(
                state,
                proposal=ProposalPotential(calculator),
                previous_direction=None,
                krylov_intents=intents,
            )


@pytest.mark.parametrize(
    "field",
    ("curvature", "true_curvature", "residual_norm", "initial_span_overlap", "antisymmetry"),
)
def test_block_krylov_rejects_nonfinite_solver_results(monkeypatch, field):
    values = {
        "direction": np.array([1.0, 0.0, 0.0]),
        "curvature": 1.0,
        "true_curvature": 1.0,
        "residual_norm": 0.0,
        "initial_span_overlap": 1.0,
        "antisymmetry": 0.0,
        "dimension": 1,
        "initial_rank": 1,
        "hvp_count": 1,
        "termination_reason": "krylov_breakdown",
    }
    values[field] = np.nan
    result = KrylovResult(**values)
    monkeypatch.setattr(
        "pamssw.walker.solve_krylov_block",
        lambda *args, **kwargs: result,
    )
    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    calculator = AnalyticCalculator(Quadratic())
    oracle = SoftModeOracle(
        calculator,
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="block_krylov",
    )

    with pytest.raises(ValueError, match=field):
        oracle.choose_direction(
            state,
            proposal=ProposalPotential(calculator),
            previous_direction=None,
            krylov_intents=(IntentBlock(np.array([[1.0], [0.0], [0.0]])),),
        )


def test_surface_walker_passes_block_krylov_depth_to_oracle():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(block_krylov_depth=4),
        softening_enabled=False,
    )

    assert walker.oracle.block_krylov_depth == 4


def test_rayleigh_ritz_projected_true_curvature_has_second_order_difference_from_direct_mixed_stencil():
    """On a nonlinear analytic PES, projection/direct-stencil mismatch scales as epsilon squared."""

    class QuarticPotential:
        coefficient = 3.0

        def energy_gradient(self, flat_positions, state):
            x, y, z = np.asarray(flat_positions, dtype=float)
            energy = 0.5 * (x * x + y * y + 5.0 * z * z) + 0.25 * self.coefficient * x**4
            return energy, np.array([x + self.coefficient * x**3, y, 5.0 * z])

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    potential = QuarticPotential()
    oracle = SoftModeOracle(AnalyticCalculator(potential), np.random.default_rng(0), candidates=0)
    candidates = [
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([0.0, 1.0, 0.0])),
    ]
    total_hessian = np.array([[1.0, -0.8, 0.0], [-0.8, 1.0, 0.0], [0.0, 0.0, 5.0]])
    total_hvps = [total_hessian @ candidate.direction for candidate in candidates]

    def true_hvp(direction, epsilon):
        plus = state.flatten_positions() + epsilon * direction
        minus = state.flatten_positions() - epsilon * direction
        _, gradient_plus = potential.energy_gradient(plus, state)
        _, gradient_minus = potential.energy_gradient(minus, state)
        return (gradient_plus - gradient_minus) / (2.0 * epsilon)

    def projected_and_direct(epsilon):
        projected = oracle._rayleigh_ritz_candidate(
            candidates,
            total_hvps,
            [true_hvp(candidate.direction, epsilon) for candidate in candidates],
        )
        assert projected is not None
        direction, _, projected_true_curvature = projected
        assert projected_true_curvature is not None
        direct_true_curvature = float(np.dot(direction, true_hvp(direction, epsilon)))
        return projected_true_curvature, direct_true_curvature

    projected_coarse, direct_coarse = projected_and_direct(4e-2)
    projected_fine, direct_fine = projected_and_direct(2e-2)
    coarse_error = abs(projected_coarse - direct_coarse)
    fine_error = abs(projected_fine - direct_fine)

    assert coarse_error > 0.0
    assert 0.20 * coarse_error < fine_error < 0.30 * coarse_error


def test_walk_ritz_reuses_true_curvature_without_extra_escape_true_pes_check(monkeypatch):
    """Plain Ritz must not add a fallback true-HVP after native candidates were evaluated."""

    class CoupledQuadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.array(
                [[1.0, -0.8, 0.0], [-0.8, 1.0, 0.0], [0.0, 0.0, 5.0]]
            )
            gradient = hessian @ flat_positions
            return 0.5 * float(flat_positions @ gradient), gradient

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(CoupledQuadratic()),
        config=LSSSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=2,
            n_bond_pairs=0,
            rng_seed=0,
            direction_selection_mode="rayleigh_ritz",
            direction_synthesis_mode="none",
            direction_curvature_source="inner",
            choice_aligned_softening_enabled=False,
            anchor_weight=1e-12,
            continuity_weight=0.0,
            history_push_weight=0.0,
        ),
        softening_enabled=False,
    )
    directions = [
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([0.0, 1.0, 0.0])),
    ]
    monkeypatch.setattr(
        walker.oracle.generator,
        "generate_initial_direction",
        lambda *args, **kwargs: np.array([1.0, 0.0, 0.0]),
    )
    monkeypatch.setattr(walker.oracle.generator, "generate", lambda *args, **kwargs: directions)
    selected = []
    original_choose_direction = walker.oracle.choose_direction

    def capture_choice(*args, **kwargs):
        choice = original_choose_direction(*args, **kwargs)
        selected.append(choice)
        return choice

    monkeypatch.setattr(walker.oracle, "choose_direction", capture_choice)
    monkeypatch.setattr(
        walker,
        "_true_directional_curvature",
        lambda *args, **kwargs: pytest.fail("Ritz must reuse native true curvature"),
    )
    monkeypatch.setattr(
        walker,
        "_relax_proposal_task",
        lambda task, **kwargs: RelaxResult(task.initial_state, energy=0.0, gradient_norm=0.0, n_iter=0),
    )

    walker._walk_candidate_from_seed(state)

    counts = walker.calculator.snapshot().as_dict()
    assert selected[0].kind == DirectionCandidateKind.RITZ
    assert counts[EvaluationPurpose.DIRECTION_ORACLE.value] == 4
    assert counts[EvaluationPurpose.ESCAPE_TRUE_PES_CHECK.value] == 2


class KindScoreScorer(DirectionScorer):
    def __init__(self, scores):
        super().__init__()
        self.scores = scores
        self.seen_kinds = []

    def score_candidate(self, **kwargs):
        kind = kwargs["candidate"].kind
        self.seen_kinds.append(kind)
        return self.scores.get(kind, 0.0)


def test_direction_type_bonus_disabled_preserves_winner_and_candidate_count():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), np.random.default_rng(0), candidates=0)
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.MOMENTUM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.BOND, np.array([0.0, 1.0, 0.0])),
    ]
    oracle.scorer = KindScoreScorer(
        {
            DirectionCandidateKind.MOMENTUM: 1.0,
            DirectionCandidateKind.BOND: 0.99,
        }
    )

    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=None,
        score_sigma=1.0,
        direction_type_bonus_fn=None,
    )

    assert choice.kind == DirectionCandidateKind.MOMENTUM
    assert choice.candidate_count == 2


def test_true_curvature_ranker_uses_paid_true_hvp_not_static_composite(
    monkeypatch,
):
    state = State(
        numbers=np.array([1]),
        positions=np.array([[0.0, 0.0, 0.0]]),
    )
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=0,
        direction_ranking_mode="true_curvature",
    )
    candidates = [
        DirectionCandidate(
            DirectionCandidateKind.MOMENTUM,
            np.array([1.0, 0.0, 0.0]),
        ),
        DirectionCandidate(
            DirectionCandidateKind.BOND,
            np.array([0.0, 1.0, 0.0]),
        ),
    ]
    monkeypatch.setattr(
        oracle.generator,
        "generate",
        lambda *args, **kwargs: candidates,
    )
    monkeypatch.setattr(
        oracle,
        "_candidate_directional_hvps",
        lambda state, proposal, direction: (
            direction,
            (5.0 if direction[0] else -1.0) * direction,
        ),
    )
    oracle.scorer = KindScoreScorer(
        {
            DirectionCandidateKind.MOMENTUM: 10.0,
            DirectionCandidateKind.BOND: 0.0,
        }
    )

    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(
            AnalyticCalculator(Quadratic())
        ),
        previous_direction=None,
        score_sigma=1.0,
    )

    assert choice.kind == DirectionCandidateKind.BOND
    assert choice.true_curvature == pytest.approx(-1.0)
    assert choice.score == pytest.approx(0.0)
    assert choice.diagnostics["direction_ranking_mode"] == "true_curvature"
    assert choice.diagnostics["direction_ranking_score"] == pytest.approx(1.0)


@pytest.mark.parametrize("bonus_kind", [DirectionCandidateKind.BOND, DirectionCandidateKind.RANDOM])
def test_direction_type_bonus_can_flip_close_static_scores_from_momentum(bonus_kind):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), np.random.default_rng(0), candidates=0)
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.MOMENTUM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(bonus_kind, np.array([0.0, 1.0, 0.0])),
    ]
    oracle.scorer = KindScoreScorer(
        {
            DirectionCandidateKind.MOMENTUM: 1.0,
            bonus_kind: 0.99,
        }
    )

    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=None,
        score_sigma=1.0,
        direction_type_bonus_fn=lambda kind: 0.02 if kind == bonus_kind else 0.0,
    )

    assert choice.kind == bonus_kind
    assert choice.candidate_count == 2


def test_direction_type_bonus_applies_to_regularized_ritz_kind(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=0,
        direction_synthesis_mode="regularized_ritz",
    )
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.MOMENTUM, np.array([1.0, 0.0, 0.0])),
    ]
    oracle.scorer = KindScoreScorer(
        {
            DirectionCandidateKind.MOMENTUM: 1.0,
            DirectionCandidateKind.RITZ_REG: 0.5,
        }
    )
    monkeypatch.setattr(
        oracle,
        "_regularized_ritz_candidate",
        lambda *args, **kwargs: (np.array([0.0, 1.0, 0.0]), 0.0),
    )

    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=None,
        score_sigma=1.0,
        direction_type_bonus_fn=lambda kind: 0.6 if kind == DirectionCandidateKind.RITZ_REG else 0.0,
    )

    assert choice.kind == DirectionCandidateKind.RITZ_REG
    assert choice.candidate_count == 2
    assert DirectionCandidateKind.RITZ_REG in oracle.scorer.seen_kinds


def test_direction_type_bonus_applies_to_rayleigh_ritz_kind(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=0,
        direction_selection_mode="rayleigh_ritz",
    )
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.MOMENTUM, np.array([1.0, 0.0, 0.0])),
    ]
    oracle.scorer = KindScoreScorer(
        {
            DirectionCandidateKind.MOMENTUM: 1.0,
            DirectionCandidateKind.RITZ: 0.5,
        }
    )
    monkeypatch.setattr(
        oracle,
        "_rayleigh_ritz_candidate",
        lambda *args, **kwargs: (np.array([0.0, 1.0, 0.0]), 0.0),
    )

    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=None,
        score_sigma=1.0,
        direction_type_bonus_fn=lambda kind: 0.6 if kind == DirectionCandidateKind.RITZ else 0.0,
    )

    assert choice.kind == DirectionCandidateKind.RITZ
    assert choice.candidate_count == 2
    assert DirectionCandidateKind.RITZ in oracle.scorer.seen_kinds


@pytest.mark.parametrize("bad_bonus", [np.nan, np.inf])
def test_direction_type_bonus_rejects_nonfinite_bonus(bad_bonus):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), np.random.default_rng(0), candidates=0)
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.MOMENTUM, np.array([1.0, 0.0, 0.0])),
    ]
    oracle.scorer = KindScoreScorer({DirectionCandidateKind.MOMENTUM: 1.0})

    with pytest.raises(ValueError, match="direction_type_bonus_fn"):
        oracle.choose_direction(
            state,
            proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
            previous_direction=None,
            score_sigma=1.0,
            direction_type_bonus_fn=lambda kind: bad_bonus,
        )


def test_direction_type_memory_disabled_bonus_is_zero_even_after_records():
    memory = DirectionTypeMemory(
        enabled=False,
        success_weight=1.0,
        exploration_weight=1.0,
        window=10,
    )

    memory.record_trial([DirectionCandidateKind.RANDOM], productive=True)

    assert memory.selected_counts[DirectionCandidateKind.RANDOM] == 1
    assert memory.productive_counts[DirectionCandidateKind.RANDOM] == 1
    assert memory.bonus(DirectionCandidateKind.RANDOM) == 0.0


def test_direction_record_stores_copy_safe_readonly_direction_without_normalizing():
    direction = np.array([3.0, 4.0])
    record = DirectionRecord(
        record_id=0,
        trial_index=1,
        proposal_index=2,
        step_index=3,
        seed_entry_id=4,
        kind=DirectionCandidateKind.RANDOM,
        direction=direction,
        curvature=-0.5,
        score=1.2,
        anchor_cosine=0.25,
        accepted_new_basin=True,
        global_improved=False,
        productive=True,
        final_energy=-2.0,
    )

    direction[:] = 0.0

    np.testing.assert_allclose(record.direction, [3.0, 4.0])
    assert record.direction.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        record.direction[0] = 9.0
    assert np.linalg.norm(record.direction) == pytest.approx(5.0)


def test_direction_record_uses_identity_equality_and_hash():
    direction = np.array([1.0])
    first = DirectionRecord(
        record_id=0,
        trial_index=None,
        proposal_index=None,
        step_index=0,
        seed_entry_id=None,
        kind=DirectionCandidateKind.RANDOM,
        direction=direction,
        curvature=0.0,
        score=None,
        anchor_cosine=None,
    )
    second = DirectionRecord(
        record_id=0,
        trial_index=None,
        proposal_index=None,
        step_index=0,
        seed_entry_id=None,
        kind=DirectionCandidateKind.RANDOM,
        direction=direction,
        curvature=0.0,
        score=None,
        anchor_cosine=None,
    )

    assert first == first
    assert first != second
    assert isinstance(hash(first), int)


def test_direction_record_rejects_invalid_direction_archive_identity_fields():
    with pytest.raises(ValueError, match="record_id"):
        DirectionRecord(
            record_id=True,
            trial_index=None,
            proposal_index=None,
            step_index=0,
            seed_entry_id=None,
            kind=DirectionCandidateKind.RANDOM,
            direction=np.array([1.0]),
            curvature=0.0,
            score=None,
            anchor_cosine=None,
        )
    with pytest.raises(ValueError, match="step_index"):
        DirectionRecord(
            record_id=0,
            trial_index=None,
            proposal_index=None,
            step_index=-1,
            seed_entry_id=None,
            kind=DirectionCandidateKind.RANDOM,
            direction=np.array([1.0]),
            curvature=0.0,
            score=None,
            anchor_cosine=None,
        )


def test_direction_record_rejects_invalid_direction_archive_direction_and_kind():
    with pytest.raises(ValueError, match="direction"):
        DirectionRecord(
            record_id=0,
            trial_index=None,
            proposal_index=None,
            step_index=0,
            seed_entry_id=None,
            kind=DirectionCandidateKind.RANDOM,
            direction=np.array([[1.0]]),
            curvature=0.0,
            score=None,
            anchor_cosine=None,
        )
    with pytest.raises(ValueError, match="direction"):
        DirectionRecord(
            record_id=0,
            trial_index=None,
            proposal_index=None,
            step_index=0,
            seed_entry_id=None,
            kind=DirectionCandidateKind.RANDOM,
            direction=np.array([np.nan]),
            curvature=0.0,
            score=None,
            anchor_cosine=None,
        )
    with pytest.raises(ValueError, match="kind"):
        DirectionRecord(
            record_id=0,
            trial_index=None,
            proposal_index=None,
            step_index=0,
            seed_entry_id=None,
            kind="random",
            direction=np.array([1.0]),
            curvature=0.0,
            score=None,
            anchor_cosine=None,
        )


@pytest.mark.parametrize("field_name", ["curvature", "score", "anchor_cosine", "final_energy"])
def test_direction_record_rejects_nonfinite_direction_archive_fields(field_name):
    kwargs = {
        "record_id": 0,
        "trial_index": None,
        "proposal_index": None,
        "step_index": 0,
        "seed_entry_id": None,
        "kind": DirectionCandidateKind.RANDOM,
        "direction": np.array([1.0]),
        "curvature": 0.0,
        "score": None,
        "anchor_cosine": None,
        "final_energy": None,
    }
    kwargs[field_name] = np.inf

    with pytest.raises(ValueError, match=field_name):
        DirectionRecord(**kwargs)


@pytest.mark.parametrize("field_name", ["curvature", "score", "anchor_cosine", "final_energy"])
@pytest.mark.parametrize("bad_value", [True, "1.0", object()])
def test_direction_record_rejects_non_real_direction_archive_fields(field_name, bad_value):
    kwargs = {
        "record_id": 0,
        "trial_index": None,
        "proposal_index": None,
        "step_index": 0,
        "seed_entry_id": None,
        "kind": DirectionCandidateKind.RANDOM,
        "direction": np.array([1.0]),
        "curvature": 0.0,
        "score": None,
        "anchor_cosine": None,
        "final_energy": None,
    }
    kwargs[field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        DirectionRecord(**kwargs)


@pytest.mark.parametrize("field_name", ["accepted_new_basin", "global_improved", "productive"])
@pytest.mark.parametrize("bad_value", [1, "yes"])
def test_direction_record_rejects_non_bool_direction_archive_outcomes(field_name, bad_value):
    kwargs = {
        "record_id": 0,
        "trial_index": None,
        "proposal_index": None,
        "step_index": 0,
        "seed_entry_id": None,
        "kind": DirectionCandidateKind.RANDOM,
        "direction": np.array([1.0]),
        "curvature": 0.0,
        "score": None,
        "anchor_cosine": None,
    }
    kwargs[field_name] = bad_value

    with pytest.raises(ValueError, match=field_name):
        DirectionRecord(**kwargs)


def test_disabled_direction_archive_retains_no_direction_records_after_walk(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_steps_per_walk=1, oracle_candidates=1, proposal_relax_steps=0),
        softening_enabled=False,
    )

    walker.oracle.generator.generate_initial_direction = lambda *args, **kwargs: np.array([1.0, 0.0, 0.0])
    walker.oracle.choose_direction = lambda *args, **kwargs: DirectionChoice(
        direction=np.array([1.0, 0.0, 0.0]),
        curvature=2.5,
        kind=DirectionCandidateKind.RANDOM,
        candidate_count=1,
    )

    class NoRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator

        @staticmethod
        def classify_outcome(**kwargs):
            return RelaxOutcomeClass.USEFUL_PROGRESS

        def relax(self, state, **kwargs):
            return RelaxResult(state=state, energy=0.0, gradient_norm=0.0, n_iter=0)

    import pamssw.walker as walker_module

    monkeypatch.setattr(walker_module, "Relaxer", NoRelaxer)
    monkeypatch.setattr(walker, "_build_softening", lambda *args, **kwargs: None)
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", lambda *args, **kwargs: 1.0)

    walker._walk_candidate_from_seed(state, trial_index=3, proposal_index=4, seed_entry_id=5)

    assert walker._direction_archive_pending_records is None


def test_enabled_direction_archive_captures_selected_direction_record(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            proposal_relax_steps=0,
            direction_archive_enabled=True,
        ),
        softening_enabled=False,
    )

    walker.oracle.generator.generate_initial_direction = lambda *args, **kwargs: np.array([1.0, 0.0, 0.0])
    walker.oracle.choose_direction = lambda *args, **kwargs: DirectionChoice(
        direction=np.array([0.0, 1.0, 0.0]),
        curvature=-0.75,
        kind=DirectionCandidateKind.BOND,
        candidate_count=1,
        score=2.25,
    )

    class NoRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator

        @staticmethod
        def classify_outcome(**kwargs):
            return RelaxOutcomeClass.USEFUL_PROGRESS

        def relax(self, state, **kwargs):
            return RelaxResult(state=state, energy=0.0, gradient_norm=0.0, n_iter=0)

    import pamssw.walker as walker_module

    monkeypatch.setattr(walker_module, "Relaxer", NoRelaxer)
    monkeypatch.setattr(walker, "_build_softening", lambda *args, **kwargs: None)
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", lambda *args, **kwargs: 1.0)

    walker._walk_candidate_from_seed(state, trial_index=3, proposal_index=4, seed_entry_id=5)

    assert walker._direction_archive_pending_records is not None
    assert len(walker._direction_archive_pending_records) == 1
    record = walker._direction_archive_pending_records[0]
    assert record.record_id == 0
    assert record.trial_index == 3
    assert record.proposal_index == 4
    assert record.step_index == 0
    assert record.seed_entry_id == 5
    assert record.kind is DirectionCandidateKind.BOND
    np.testing.assert_allclose(record.direction, [0.0, 1.0, 0.0])
    assert record.curvature == pytest.approx(-0.75)
    assert record.score == pytest.approx(2.25)
    assert record.anchor_cosine == pytest.approx(0.0)
    assert record.accepted_new_basin is None
    assert record.global_improved is None
    assert record.productive is None
    assert record.final_energy is None


def test_direction_archive_capture_is_copy_safe():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )
    direction = np.array([1.0, 0.0, 0.0])
    choice = DirectionChoice(
        direction=direction,
        curvature=1.5,
        kind=DirectionCandidateKind.RANDOM,
        candidate_count=1,
    )

    walker._capture_direction_record(
        trial_index=0,
        proposal_index=0,
        step_index=0,
        seed_entry_id=0,
        choice=choice,
        anchor_direction=None,
    )
    direction[:] = 9.0

    assert walker._direction_archive_pending_records is not None
    np.testing.assert_allclose(walker._direction_archive_pending_records[0].direction, [1.0, 0.0, 0.0])
    assert walker._direction_archive_pending_records[0].anchor_cosine is None


def test_direction_archive_capture_handles_invalid_anchor_with_none_cosine():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )
    choice = DirectionChoice(
        direction=np.array([1.0, 0.0, 0.0]),
        curvature=1.5,
        kind=DirectionCandidateKind.RANDOM,
        candidate_count=1,
    )

    walker._capture_direction_record(
        trial_index=0,
        proposal_index=0,
        step_index=0,
        seed_entry_id=0,
        choice=choice,
        anchor_direction=np.array([np.nan, 0.0, 0.0]),
    )

    assert walker._direction_archive_pending_records is not None
    assert walker._direction_archive_pending_records[0].anchor_cosine is None


def _capture_test_direction_archive_record(walker, trial_index=0, proposal_index=0, step_index=0):
    walker._capture_direction_record(
        trial_index=trial_index,
        proposal_index=proposal_index,
        step_index=step_index,
        seed_entry_id=7,
        choice=DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=-0.5,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
            score=1.25,
        ),
        anchor_direction=np.array([1.0, 0.0, 0.0]),
    )


def test_direction_archive_productive_trial_finalizes_pending_record_with_final_energy():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )
    _capture_test_direction_archive_record(walker, trial_index=2)

    walker._finalize_direction_archive_trial(
        2,
        accepted_new_basin=True,
        global_improved=False,
        final_energy=-3.5,
    )

    assert walker._direction_archive_pending_records == []
    assert walker._direction_archive_records is not None
    assert len(walker._direction_archive_records) == 1
    record = walker._direction_archive_records[0]
    assert record.trial_index == 2
    assert record.accepted_new_basin is True
    assert record.global_improved is False
    assert record.productive is True
    assert record.final_energy == pytest.approx(-3.5)


def test_direction_archive_completed_no_discovery_trial_finalizes_nonproductive_with_no_final_energy():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )
    _capture_test_direction_archive_record(walker, trial_index=0)

    # Completed trials with no relaxed discovery keep an explicit non-productive record.
    walker._finalize_direction_archive_trial(
        0,
        accepted_new_basin=False,
        global_improved=False,
        final_energy=None,
    )

    assert walker._direction_archive_records is not None
    record = walker._direction_archive_records[0]
    assert record.accepted_new_basin is False
    assert record.global_improved is False
    assert record.productive is False
    assert record.final_energy is None


def test_direction_archive_success_only_drops_nonproductive_and_keeps_productive_records():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True, direction_archive_success_only=True),
        softening_enabled=False,
    )
    _capture_test_direction_archive_record(walker, trial_index=0)
    _capture_test_direction_archive_record(walker, trial_index=1)

    walker._finalize_direction_archive_trial(
        0,
        accepted_new_basin=False,
        global_improved=False,
        final_energy=None,
    )
    walker._finalize_direction_archive_trial(
        1,
        accepted_new_basin=False,
        global_improved=True,
        final_energy=-4.0,
    )

    assert walker._direction_archive_records is not None
    assert len(walker._direction_archive_records) == 1
    assert walker._direction_archive_records[0].trial_index == 1
    assert walker._direction_archive_records[0].productive is True


def test_direction_archive_max_records_cap_evicts_oldest_finalized_records():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True, direction_archive_max_records=2),
        softening_enabled=False,
    )
    for trial_index in range(3):
        _capture_test_direction_archive_record(walker, trial_index=trial_index)
        walker._finalize_direction_archive_trial(
            trial_index,
            accepted_new_basin=True,
            global_improved=False,
            final_energy=-float(trial_index),
        )

    assert walker._direction_archive_records is not None
    assert [record.trial_index for record in walker._direction_archive_records] == [1, 2]


def test_direction_archive_stats_summary_disabled_reports_zero_counts():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=False),
        softening_enabled=False,
    )

    summary = walker._direction_archive_stats_summary()

    assert summary == {
        "direction_archive_enabled": 0,
        "direction_archive_active": 0,
        "direction_archive_records": 0,
        "direction_archive_productive_records": 0,
    }


def test_direction_archive_stats_summary_counts_finalized_productive_records():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )
    _capture_test_direction_archive_record(walker, trial_index=0)
    _capture_test_direction_archive_record(walker, trial_index=1)

    walker._finalize_direction_archive_trial(
        0,
        accepted_new_basin=False,
        global_improved=False,
        final_energy=None,
    )
    walker._finalize_direction_archive_trial(
        1,
        accepted_new_basin=True,
        global_improved=False,
        final_energy=-2.0,
    )

    assert walker._direction_archive_stats_summary() == {
        "direction_archive_enabled": 1,
        "direction_archive_active": 1,
        "direction_archive_records": 2,
        "direction_archive_productive_records": 1,
    }


def test_plateau_evolution_uses_in_memory_direction_archive_without_output_archive_enabled():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(plateau_evolution_enabled=True, direction_archive_enabled=False),
        softening_enabled=False,
    )
    _capture_test_direction_archive_record(walker, trial_index=0)

    walker._finalize_direction_archive_trial(
        0,
        accepted_new_basin=True,
        global_improved=False,
        final_energy=-3.0,
    )

    assert walker._direction_archive_records is not None
    assert walker.successful_records(seed_entry_id=7, limit=1)
    summary = walker._direction_archive_stats_summary()
    assert summary["direction_archive_enabled"] == 0
    assert summary["direction_archive_active"] == 1
    assert summary["direction_archive_productive_records"] == 1


def test_direction_archive_writes_finalized_records_jsonl_when_enabled(tmp_path):
    output_path = tmp_path / "nested" / "directions.jsonl"
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True, direction_archive_path=str(output_path)),
        softening_enabled=False,
    )
    _capture_test_direction_archive_record(walker, trial_index=0)

    walker._finalize_direction_archive_trial(
        0,
        accepted_new_basin=True,
        global_improved=False,
        final_energy=-3.0,
    )

    [line] = output_path.read_text().splitlines()
    payload = json.loads(line)
    assert payload["record_id"] == 0
    assert payload["trial_index"] == 0
    assert payload["proposal_index"] == 0
    assert payload["step_index"] == 0
    assert payload["seed_entry_id"] == 7
    assert payload["kind"] == "random"
    assert payload["curvature"] == pytest.approx(-0.5)
    assert payload["score"] == pytest.approx(1.25)
    assert payload["anchor_cosine"] == pytest.approx(1.0)
    assert payload["accepted_new_basin"] is True
    assert payload["global_improved"] is False
    assert payload["productive"] is True
    assert payload["final_energy"] == pytest.approx(-3.0)
    assert payload["direction"] == [1.0, 0.0, 0.0]


def test_disabled_direction_archive_with_path_does_not_write_jsonl(tmp_path):
    output_path = tmp_path / "directions.jsonl"
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=False, direction_archive_path=str(output_path)),
        softening_enabled=False,
    )

    _capture_test_direction_archive_record(walker, trial_index=0)
    walker._finalize_direction_archive_trial(
        0,
        accepted_new_basin=True,
        global_improved=True,
        final_energy=-1.0,
    )

    assert not output_path.exists()


def test_direction_archive_jsonl_is_not_duplicated_by_repeated_finalize_or_query(tmp_path):
    output_path = tmp_path / "directions.jsonl"
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True, direction_archive_path=str(output_path)),
        softening_enabled=False,
    )
    _capture_test_direction_archive_record(walker, trial_index=0)

    walker._finalize_direction_archive_trial(
        0,
        accepted_new_basin=True,
        global_improved=True,
        final_energy=-1.0,
    )
    walker._finalize_direction_archive_trial(
        0,
        accepted_new_basin=True,
        global_improved=True,
        final_energy=-1.0,
    )
    walker.recent_records()
    walker.successful_records()

    assert len(output_path.read_text().splitlines()) == 1


def _direction_archive_query_record(
    record_id,
    *,
    seed_entry_id=7,
    kind=DirectionCandidateKind.RANDOM,
    direction=None,
    productive=True,
):
    return DirectionRecord(
        record_id=record_id,
        trial_index=record_id,
        proposal_index=0,
        step_index=0,
        seed_entry_id=seed_entry_id,
        kind=kind,
        direction=np.array([float(record_id + 1), 0.0, 0.0]) if direction is None else direction,
        curvature=-0.5,
        score=1.0,
        anchor_cosine=None,
        accepted_new_basin=productive,
        global_improved=False,
        productive=productive,
        final_energy=-float(record_id),
    )


def test_direction_archive_recent_records_return_newest_first_and_limit():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )
    assert walker._direction_archive_records is not None
    walker._direction_archive_records.extend(
        [
            _direction_archive_query_record(0),
            _direction_archive_query_record(1),
            _direction_archive_query_record(2),
        ]
    )

    records = walker.recent_records(limit=2)

    assert [record.record_id for record in records] == [2, 1]


def test_direction_archive_successful_records_filter_productive_seed_and_kind():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )
    assert walker._direction_archive_records is not None
    walker._direction_archive_records.extend(
        [
            _direction_archive_query_record(
                0,
                seed_entry_id=7,
                kind=DirectionCandidateKind.RANDOM,
                productive=True,
            ),
            _direction_archive_query_record(
                1,
                seed_entry_id=7,
                kind=DirectionCandidateKind.BOND,
                productive=False,
            ),
            _direction_archive_query_record(
                2,
                seed_entry_id=8,
                kind=DirectionCandidateKind.BOND,
                productive=True,
            ),
            _direction_archive_query_record(
                3,
                seed_entry_id=7,
                kind=DirectionCandidateKind.BOND,
                productive=True,
            ),
        ]
    )

    records = walker.successful_records(seed_entry_id=7, kind=DirectionCandidateKind.BOND)

    assert [record.record_id for record in records] == [3]


def test_direction_archive_successful_records_exclude_losing_proposals_in_successful_trial():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )
    _capture_test_direction_archive_record(walker, trial_index=0, proposal_index=0)
    _capture_test_direction_archive_record(walker, trial_index=0, proposal_index=1)

    walker._finalize_direction_archive_trial(
        0,
        proposal_index=0,
        accepted_new_basin=False,
        global_improved=False,
        final_energy=-1.0,
    )
    walker._finalize_direction_archive_trial(
        0,
        proposal_index=1,
        accepted_new_basin=True,
        global_improved=True,
        final_energy=-2.0,
    )

    records = walker.successful_records()

    assert [record.proposal_index for record in records] == [1]


def test_direction_archive_recent_records_filters_kind_but_includes_nonproductive():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )
    assert walker._direction_archive_records is not None
    walker._direction_archive_records.extend(
        [
            _direction_archive_query_record(0, kind=DirectionCandidateKind.RANDOM, productive=True),
            _direction_archive_query_record(1, kind=DirectionCandidateKind.BOND, productive=False),
            _direction_archive_query_record(2, kind=DirectionCandidateKind.BOND, productive=True),
        ]
    )

    records = walker.recent_records(kind=DirectionCandidateKind.BOND)

    assert [record.record_id for record in records] == [2, 1]
    assert [record.productive for record in records] == [True, False]


def test_archive_escape_momentum_enables_in_memory_direction_archive_without_jsonl(tmp_path):
    output_path = tmp_path / "directions.jsonl"
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            archive_escape_momentum_enabled=True,
            direction_archive_enabled=False,
            direction_archive_path=str(output_path),
        ),
        softening_enabled=False,
    )

    assert walker._direction_archive_records == []
    _capture_test_direction_archive_record(walker, trial_index=0)
    walker._finalize_direction_archive_trial(
        0,
        accepted_new_basin=True,
        global_improved=False,
        final_energy=-1.0,
    )

    assert len(walker.successful_records()) == 1
    assert not output_path.exists()


@pytest.mark.parametrize("method_name", ["recent_records", "successful_records"])
@pytest.mark.parametrize("bad_limit", [0, -1, True, 1.5, "1"])
def test_direction_archive_query_rejects_invalid_limit(method_name, bad_limit):
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )

    with pytest.raises(ValueError, match="limit"):
        getattr(walker, method_name)(limit=bad_limit)


@pytest.mark.parametrize("method_name", ["recent_records", "successful_records"])
def test_direction_archive_query_rejects_invalid_kind(method_name):
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )

    with pytest.raises(ValueError, match="kind"):
        getattr(walker, method_name)(kind="random")


def test_direction_archive_query_returns_normalized_copy_safe_records_without_mutating_stored():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )
    assert walker._direction_archive_records is not None
    stored = _direction_archive_query_record(0, direction=np.array([3.0, 4.0]))
    walker._direction_archive_records.append(stored)

    [record] = walker.recent_records()

    assert record is not stored
    np.testing.assert_allclose(record.direction, [0.6, 0.8])
    np.testing.assert_allclose(stored.direction, [3.0, 4.0])
    assert record.direction.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        record.direction[0] = 9.0


def test_direction_archive_query_raises_on_degenerate_stored_direction():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=True),
        softening_enabled=False,
    )
    assert walker._direction_archive_records is not None
    walker._direction_archive_records.append(_direction_archive_query_record(0, direction=np.array([0.0, 0.0])))

    with pytest.raises(ValueError, match="nonzero"):
        walker.recent_records()


def test_disabled_direction_archive_queries_return_empty_lists_without_error():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_archive_enabled=False),
        softening_enabled=False,
    )

    assert walker.recent_records() == []
    assert walker.successful_records() == []


@pytest.mark.parametrize(
    ("candidate_state", "candidate_energy", "expected_productive", "expected_final_energy"),
    [
        (State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]])), -0.5, True, -0.5),
        (State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]])), -1.0, False, -1.0),
    ],
)
def test_direction_archive_run_finalizes_productive_and_nonproductive_trials(
    monkeypatch,
    candidate_state,
    candidate_energy,
    expected_productive,
    expected_final_energy,
):
    initial = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_trials=1, direction_archive_enabled=True),
        softening_enabled=False,
    )
    relax_calls = 0

    def fake_relax_true_minimum(state, trajectory_name=None, *, quench_purpose=None):
        nonlocal relax_calls
        relax_calls += 1
        if relax_calls == 1:
            return RelaxResult(initial, energy=-1.0, gradient_norm=0.0, n_iter=0)
        return RelaxResult(candidate_state, energy=candidate_energy, gradient_norm=0.0, n_iter=0)

    def proposal_pool(*args, **kwargs):
        _capture_test_direction_archive_record(walker, trial_index=0)
        return [CandidateProposal("test", candidate_state)]

    monkeypatch.setattr(walker, "relax_true_minimum", fake_relax_true_minimum)
    monkeypatch.setattr(walker, "_proposal_pool", proposal_pool)

    walker.run(initial)

    assert walker._direction_archive_pending_records == []
    assert walker._direction_archive_records is not None
    assert len(walker._direction_archive_records) == 1
    record = walker._direction_archive_records[0]
    assert record.productive is expected_productive
    assert record.final_energy == pytest.approx(expected_final_energy)


def test_run_can_reuse_one_prequenched_initial_without_repeating_true_quench(
    monkeypatch,
):
    raw = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    minimum = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    prequenched = RelaxResult(
        minimum,
        energy=-1.25,
        gradient_norm=0.0,
        n_iter=7,
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_trials=1),
        softening_enabled=False,
    )
    monkeypatch.setattr(
        walker,
        "relax_true_minimum",
        lambda *args, **kwargs: pytest.fail("shared bootstrap must not be repeated"),
    )
    monkeypatch.setattr(
        walker,
        "_proposal_pool",
        lambda *args, **kwargs: (_ for _ in ()).throw(BudgetExceeded("stop")),
    )

    result = walker.run(
        minimum,
        prequenched_initial=prequenched,
    )

    assert result.best_energy == pytest.approx(-1.25)
    np.testing.assert_array_equal(result.best_state.positions, minimum.positions)
    assert result.stats["force_evaluations"] == 0


def test_direction_archive_run_does_not_mark_duplicate_after_new_best_as_global_improvement(monkeypatch):
    initial = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    improved = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_trials=1, direction_archive_enabled=True),
        softening_enabled=False,
    )
    relax_results = iter(
        [
            RelaxResult(initial, energy=-1.0, gradient_norm=0.0, n_iter=0),
            RelaxResult(improved, energy=-2.0, gradient_norm=0.0, n_iter=0),
            RelaxResult(improved, energy=-2.0, gradient_norm=0.0, n_iter=0),
        ]
    )

    def proposal_pool(*args, **kwargs):
        _capture_test_direction_archive_record(walker, trial_index=0, proposal_index=0)
        _capture_test_direction_archive_record(walker, trial_index=0, proposal_index=1)
        return [CandidateProposal("first", improved), CandidateProposal("duplicate", improved)]

    monkeypatch.setattr(
        walker,
        "relax_true_minimum",
        lambda state, trajectory_name=None, *, quench_purpose=None: next(relax_results),
    )
    monkeypatch.setattr(walker, "_proposal_pool", proposal_pool)

    walker.run(initial)

    assert walker._direction_archive_records is not None
    records_by_proposal = {record.proposal_index: record for record in walker._direction_archive_records}
    assert records_by_proposal[0].accepted_new_basin is True
    assert records_by_proposal[0].global_improved is True
    assert records_by_proposal[0].productive is True
    assert records_by_proposal[1].accepted_new_basin is False
    assert records_by_proposal[1].global_improved is False
    assert records_by_proposal[1].productive is False


def test_plateau_evolution_run_loop_activates_after_patience_without_improvement(monkeypatch):
    initial = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_trials=3, plateau_evolution_enabled=True, plateau_patience_trials=1),
        softening_enabled=False,
    )
    active_flags = []

    def fake_relax_true_minimum(state, trajectory_name=None, *, quench_purpose=None):
        return RelaxResult(state, energy=-1.0, gradient_norm=0.0, n_iter=0)

    def proposal_pool(*args, **kwargs):
        active_flags.append(kwargs["plateau_evolution_active"])
        return [CandidateProposal("duplicate", initial)]

    monkeypatch.setattr(walker, "relax_true_minimum", fake_relax_true_minimum)
    monkeypatch.setattr(walker, "_proposal_pool", proposal_pool)

    walker.run(initial)

    assert active_flags == [False, True, True]
    assert walker._direction_stats_summary()["plateau_evolution_enabled"] == 1


def test_direction_archive_run_reset_clears_previously_retained_records(monkeypatch):
    initial = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_trials=1, direction_archive_enabled=True),
        softening_enabled=False,
    )
    _capture_test_direction_archive_record(walker, trial_index=0)
    walker._finalize_direction_archive_trial(
        0,
        accepted_new_basin=True,
        global_improved=False,
        final_energy=-1.0,
    )
    assert walker._direction_archive_records

    monkeypatch.setattr(
        walker,
        "relax_true_minimum",
        lambda state, trajectory_name=None, *, quench_purpose=None: RelaxResult(state, 0.0, 0.0, 0),
    )
    monkeypatch.setattr(walker.calculator, "exhausted", lambda: True)

    walker.run(initial)

    assert walker._direction_archive_records == []
    assert walker._direction_archive_pending_records == []


def test_direction_archive_budget_exhaustion_discards_pending_without_productive_finalization(monkeypatch):
    initial = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_trials=1, direction_archive_enabled=True),
        softening_enabled=False,
    )

    monkeypatch.setattr(
        walker,
        "relax_true_minimum",
        lambda state, trajectory_name=None, *, quench_purpose=None: RelaxResult(state, 0.0, 0.0, 0),
    )

    def exhausted_proposal_pool(*args, **kwargs):
        _capture_test_direction_archive_record(walker, trial_index=0)
        raise BudgetExceeded("synthetic budget exhaustion before trial outcome")

    monkeypatch.setattr(walker, "_proposal_pool", exhausted_proposal_pool)

    result = walker.run(initial)

    assert result.stats["budget_exhausted"] == 1
    assert result.stats["direction_archive_enabled"] == 1
    assert result.stats["direction_archive_records"] == 0
    assert result.stats["direction_archive_productive_records"] == 0
    assert walker._direction_archive_records == []
    assert walker._direction_archive_pending_records == []


def test_direction_type_memory_enabled_bonus_combines_success_and_exploration():
    memory = DirectionTypeMemory(
        enabled=True,
        success_weight=2.0,
        exploration_weight=0.5,
        window=10,
    )
    memory.record_trial([DirectionCandidateKind.RANDOM], productive=True)
    memory.record_trial([DirectionCandidateKind.RANDOM], productive=False)
    memory.record_trial([DirectionCandidateKind.BOND], productive=False)

    total_selected = 3
    success_rate = 1 / 2
    exploration = np.sqrt(np.log1p(total_selected) / 2)
    expected = 2.0 * success_rate + 0.5 * exploration

    assert memory.bonus(DirectionCandidateKind.RANDOM) == pytest.approx(expected)


def test_direction_type_memory_credits_duplicate_kinds_once_per_trial():
    memory = DirectionTypeMemory(
        enabled=True,
        success_weight=1.0,
        exploration_weight=0.0,
        window=10,
    )

    memory.record_trial(
        [
            DirectionCandidateKind.RANDOM,
            DirectionCandidateKind.RANDOM,
            DirectionCandidateKind.BOND,
        ],
        productive=True,
    )

    assert memory.selected_counts[DirectionCandidateKind.RANDOM] == 1
    assert memory.productive_counts[DirectionCandidateKind.RANDOM] == 1
    assert memory.selected_counts[DirectionCandidateKind.BOND] == 1
    assert memory.productive_counts[DirectionCandidateKind.BOND] == 1
    assert len(memory.recent_events) == 2


def test_direction_type_memory_rolling_window_evicts_counts():
    memory = DirectionTypeMemory(
        enabled=True,
        success_weight=1.0,
        exploration_weight=0.0,
        window=2,
    )

    memory.record_trial([DirectionCandidateKind.RANDOM], productive=True)
    memory.record_trial([DirectionCandidateKind.BOND], productive=False)
    memory.record_trial([DirectionCandidateKind.MOMENTUM], productive=True)

    assert memory.selected_counts[DirectionCandidateKind.RANDOM] == 0
    assert memory.productive_counts[DirectionCandidateKind.RANDOM] == 0
    assert memory.selected_counts[DirectionCandidateKind.BOND] == 1
    assert memory.productive_counts[DirectionCandidateKind.BOND] == 0
    assert memory.selected_counts[DirectionCandidateKind.MOMENTUM] == 1
    assert memory.productive_counts[DirectionCandidateKind.MOMENTUM] == 1
    assert list(memory.recent_events) == [
        (DirectionCandidateKind.BOND, False),
        (DirectionCandidateKind.MOMENTUM, True),
    ]


def test_direction_type_memory_multikind_overflow_uses_enum_value_order():
    memory = DirectionTypeMemory(
        enabled=True,
        success_weight=1.0,
        exploration_weight=0.0,
        window=2,
    )

    memory.record_trial(
        [
            DirectionCandidateKind.RITZ,
            DirectionCandidateKind.BOND,
            DirectionCandidateKind.MOMENTUM,
        ],
        productive=True,
    )

    assert memory.selected_counts[DirectionCandidateKind.BOND] == 0
    assert memory.productive_counts[DirectionCandidateKind.BOND] == 0
    assert memory.selected_counts[DirectionCandidateKind.MOMENTUM] == 1
    assert memory.productive_counts[DirectionCandidateKind.MOMENTUM] == 1
    assert memory.selected_counts[DirectionCandidateKind.RITZ] == 1
    assert memory.productive_counts[DirectionCandidateKind.RITZ] == 1
    assert list(memory.recent_events) == [
        (DirectionCandidateKind.MOMENTUM, True),
        (DirectionCandidateKind.RITZ, True),
    ]


def test_regularized_ritz_synthesis_adds_anchor_aligned_candidate_without_extra_hvp(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=0,
        direction_synthesis_mode="regularized_ritz",
        regularized_ritz_top_k=2,
        anchor_weight=10.0,
        continuity_weight=0.0,
    )
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([0.0, 1.0, 0.0])),
    ]
    hvp_calls = {"count": 0}

    def fake_hvp(state, proposal, direction):
        hvp_calls["count"] += 1
        return np.zeros_like(direction)

    monkeypatch.setattr(
        oracle,
        "_candidate_directional_hvps",
        lambda state, proposal, direction: (fake_hvp(state, proposal, direction), None),
    )
    anchor = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=None,
        anchor_direction=anchor,
        score_sigma=1.0,
    )

    assert choice.kind == DirectionCandidateKind.RITZ_REG
    assert choice.candidate_count == 3
    assert abs(float(np.dot(choice.direction, anchor))) == pytest.approx(1.0)
    assert hvp_calls["count"] == 2


def test_regularized_ritz_synthesis_none_keeps_candidate_count_and_never_selects_ritz_reg(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=0,
        direction_synthesis_mode="none",
        regularized_ritz_top_k=2,
        anchor_weight=10.0,
        continuity_weight=0.0,
    )
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([0.0, 1.0, 0.0])),
    ]
    hvp_calls = {"count": 0}

    def fake_hvp(state, proposal, direction):
        hvp_calls["count"] += 1
        return np.zeros_like(direction)

    monkeypatch.setattr(
        oracle,
        "_candidate_directional_hvps",
        lambda state, proposal, direction: (fake_hvp(state, proposal, direction), None),
    )
    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=None,
        anchor_direction=np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
        score_sigma=1.0,
    )

    assert choice.kind != DirectionCandidateKind.RITZ_REG
    assert choice.candidate_count == 2
    assert hvp_calls["count"] == 2


def test_regularized_ritz_synthesis_can_be_steered_by_previous_direction(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=0,
        direction_synthesis_mode="regularized_ritz",
        regularized_ritz_top_k=2,
        anchor_weight=0.0,
        continuity_weight=10.0,
    )
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([0.0, 1.0, 0.0])),
    ]

    monkeypatch.setattr(
        oracle,
        "_candidate_directional_hvps",
        lambda state, proposal, direction: (np.zeros_like(direction), None),
    )
    previous = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=previous,
        anchor_direction=None,
        score_sigma=1.0,
    )

    assert choice.kind == DirectionCandidateKind.RITZ_REG
    assert abs(float(np.dot(choice.direction, previous))) == pytest.approx(1.0)


def test_regularized_ritz_synthesis_reconstructs_projected_hessian_from_nonorthogonal_candidates():
    class CoupledQuadratic:
        hessian = np.array(
            [
                [2.0, 1.0, 0.0],
                [1.0, 4.0, 0.0],
                [0.0, 0.0, 9.0],
            ]
        )

        def energy_gradient(self, flat_positions, state):
            gradient = self.hessian @ flat_positions
            energy = 0.5 * float(flat_positions @ gradient)
            return energy, gradient

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(CoupledQuadratic()),
        np.random.default_rng(0),
        candidates=0,
        direction_synthesis_mode="regularized_ritz",
        regularized_ritz_top_k=2,
        anchor_weight=0.0,
        continuity_weight=0.0,
    )
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)),
    ]

    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(CoupledQuadratic())),
        previous_direction=None,
        anchor_direction=None,
        score_sigma=1.0,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(CoupledQuadratic.hessian)
    expected_index = int(np.argmin(eigenvalues))
    expected_direction = eigenvectors[:, expected_index]

    assert choice.kind == DirectionCandidateKind.RITZ_REG
    assert choice.candidate_count == 3
    assert choice.curvature == pytest.approx(eigenvalues[expected_index], rel=1e-5)
    assert abs(float(np.dot(choice.direction, expected_direction))) == pytest.approx(1.0)
    assert choice.true_curvature is None


def test_regularized_ritz_synthesis_tied_scores_tolerates_top_k_larger_than_candidate_count(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=0,
        direction_synthesis_mode="regularized_ritz",
        regularized_ritz_top_k=10,
        anchor_weight=10.0,
        continuity_weight=0.0,
    )
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([0.0, 1.0, 0.0])),
    ]

    monkeypatch.setattr(
        oracle,
        "_candidate_directional_hvps",
        lambda state, proposal, direction: (np.zeros_like(direction), None),
    )
    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=None,
        anchor_direction=np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
        score_sigma=1.0,
    )

    assert choice.kind == DirectionCandidateKind.RITZ_REG
    assert choice.candidate_count == 3


def test_plateau_evolution_crosses_current_and_history_directions(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=0,
        anchor_weight=0.0,
        continuity_weight=0.0,
    )
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.BOND, np.array([0.0, 1.0, 0.0])),
    ]
    history = [
        DirectionRecord(
            record_id=0,
            trial_index=0,
            proposal_index=0,
            step_index=0,
            seed_entry_id=7,
            kind=DirectionCandidateKind.RANDOM,
            direction=np.array([0.0, 1.0, 0.0]),
            curvature=1.0,
            score=1.0,
            anchor_cosine=None,
            accepted_new_basin=True,
            global_improved=False,
            productive=True,
            final_energy=-1.0,
        )
    ]
    hvp_calls = {"count": 0}

    def fake_hvp(state, proposal, direction):
        hvp_calls["count"] += 1
        target = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
        curvature = -10.0 if abs(float(np.dot(direction, target))) > 0.99 else 10.0
        return curvature * direction

    monkeypatch.setattr(
        oracle,
        "_candidate_directional_hvps",
        lambda state, proposal, direction: (fake_hvp(state, proposal, direction), None),
    )
    monkeypatch.setattr(oracle, "_directional_hvp", fake_hvp)
    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=None,
        anchor_direction=None,
        score_sigma=1.0,
        plateau_evolution_active=True,
        plateau_history=history,
        plateau_evolution_children=1,
        plateau_evolution_crossover_pairs=1,
        plateau_evolution_mutation_count=0,
    )

    assert choice.kind == DirectionCandidateKind.EVOLVED
    assert choice.candidate_count == 3
    assert choice.curvature == pytest.approx(-10.0)
    expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    assert abs(float(np.dot(choice.direction, expected))) == pytest.approx(1.0)
    assert hvp_calls["count"] == 3
    assert choice.true_curvature is None


def test_plateau_evolution_inactive_keeps_original_candidate_count(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=0,
        anchor_weight=0.0,
        continuity_weight=0.0,
    )
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.BOND, np.array([0.0, 1.0, 0.0])),
    ]
    monkeypatch.setattr(
        oracle,
        "_candidate_directional_hvps",
        lambda state, proposal, direction: (10.0 * direction, None),
    )

    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=None,
        anchor_direction=None,
        score_sigma=1.0,
        plateau_evolution_active=False,
        plateau_history=[],
        plateau_evolution_children=1,
        plateau_evolution_crossover_pairs=1,
        plateau_evolution_mutation_count=0,
    )

    assert choice.kind != DirectionCandidateKind.EVOLVED
    assert choice.candidate_count == 2


def test_archive_escape_momentum_candidate_can_win_direction_choice(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=0,
        anchor_weight=0.0,
        continuity_weight=0.0,
    )
    oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.BOND, np.array([0.0, 1.0, 0.0])),
    ]
    archive_record = DirectionRecord(
        record_id=0,
        trial_index=0,
        proposal_index=0,
        step_index=7,
        seed_entry_id=7,
        kind=DirectionCandidateKind.MOMENTUM,
        direction=np.array([0.0, 0.0, 1.0]),
        curvature=1.0,
        score=1.0,
        anchor_cosine=None,
        accepted_new_basin=True,
        global_improved=False,
        productive=True,
        final_energy=-1.0,
    )
    hvp_calls = {"count": 0}

    def fake_hvp(state, proposal, direction):
        hvp_calls["count"] += 1
        target = np.array([0.0, 0.0, 1.0])
        curvature = -5.0 if abs(float(np.dot(direction, target))) > 0.99 else 10.0
        return curvature * direction

    monkeypatch.setattr(
        oracle,
        "_candidate_directional_hvps",
        lambda state, proposal, direction: (fake_hvp(state, proposal, direction), None),
    )
    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=None,
        anchor_direction=None,
        score_sigma=1.0,
        archive_momentum_history=[archive_record],
        archive_momentum_limit=1,
    )

    assert choice.kind == DirectionCandidateKind.ARCHIVE_MOMENTUM
    assert choice.candidate_count == 3
    assert choice.curvature == pytest.approx(-5.0)
    assert hvp_calls["count"] == 3


def test_direction_stats_summary_counts_selected_regularized_ritz_candidate():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(),
        softening_enabled=False,
    )
    walker._record_direction_choice(
        DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=0.0,
            kind=DirectionCandidateKind.RITZ_REG,
            candidate_count=1,
        )
    )
    walker._record_direction_choice(
        DirectionChoice(
            direction=np.array([0.0, 1.0, 0.0]),
            curvature=0.0,
            kind=DirectionCandidateKind.ANCHOR,
            candidate_count=1,
        )
    )
    walker._record_direction_choice(
        DirectionChoice(
            direction=np.array([0.0, 0.0, 1.0]),
            curvature=0.0,
            kind=DirectionCandidateKind.BOND_FORM,
            candidate_count=1,
        )
    )
    walker._record_direction_choice(
        DirectionChoice(
            direction=np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0),
            curvature=0.0,
            kind=DirectionCandidateKind.BOND_BREAK,
            candidate_count=1,
        )
    )
    walker._record_direction_choice(
        DirectionChoice(
            direction=np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0),
            curvature=0.0,
            kind=DirectionCandidateKind.EVOLVED,
            candidate_count=3,
            evolved_candidate_count=2,
        )
    )

    summary = walker._direction_stats_summary()

    assert summary["direction_selected_ritz_reg"] == 1
    assert summary["direction_selected_anchor"] == 1
    assert summary["direction_selected_bond_form"] == 1
    assert summary["direction_selected_bond_break"] == 1
    assert summary["direction_selected_evolved"] == 1
    assert summary["plateau_evolution_candidate_steps"] == 1
    assert summary["plateau_evolution_candidates_generated"] == 2


def test_surface_walker_plumbs_regularized_ritz_synthesis_config_to_oracle():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(direction_synthesis_mode="regularized_ritz", regularized_ritz_top_k=3),
        softening_enabled=False,
    )

    assert walker.oracle.direction_synthesis_mode == "regularized_ritz"
    assert walker.oracle.regularized_ritz_top_k == 3


def test_surface_walker_can_use_fixed_reference_direction_score_sigma():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(target_uphill_energy=0.8, direction_score_sigma_mode="fixed_reference"),
        softening_enabled=False,
    )
    trust_scaled = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(target_uphill_energy=0.8, direction_score_sigma_mode="trust_scaled"),
        softening_enabled=False,
    )

    assert walker._direction_score_sigma(sigma_scale=0.25, step_target=0.8) == pytest.approx(np.sqrt(1.6))
    assert trust_scaled._direction_score_sigma(sigma_scale=0.25, step_target=0.8) == pytest.approx(np.sqrt(1.6) * 0.25)


def test_surface_walker_defaults_to_adaptive_direction_score_sigma():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(target_uphill_energy=0.8),
        softening_enabled=False,
    )
    score_sigma_fn = walker._direction_score_sigma_fn(sigma_scale=1.0, step_target=0.8)

    assert score_sigma_fn is not None
    assert score_sigma_fn(1.0) == pytest.approx(np.sqrt(1.6))
    assert score_sigma_fn(4.0) == pytest.approx(np.sqrt(0.4))


def test_direction_scorer_rewards_independent_history_push():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    scorer = DirectionScorer(history_push_weight=0.5)
    with_push = DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0]))
    against_push = DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([-1.0, 0.0, 0.0]))

    pushed_score = scorer.score_candidate(
        state=state,
        candidate=with_push,
        curvature=1.0,
        sigma=1.0,
        previous_direction=None,
        anchor_direction=None,
        archive=None,
        history_push=2.0,
    )
    opposed_score = scorer.score_candidate(
        state=state,
        candidate=against_push,
        curvature=1.0,
        sigma=1.0,
        previous_direction=None,
        anchor_direction=None,
        archive=None,
        history_push=-2.0,
    )

    assert pushed_score - opposed_score == pytest.approx(2.0)


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

    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.MOMENTUM) == 1
    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.RANDOM) == 1
    assert DirectionCandidateKind.BOND not in [candidate.kind for candidate in candidates]


def test_direction_generator_can_disable_momentum_candidate():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=2, enable_momentum_candidate=False)

    candidates = generator.generate(state, previous_direction=np.array([1.0, 0.0, 0.0]))

    assert DirectionCandidateKind.MOMENTUM not in [candidate.kind for candidate in candidates]
    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.RANDOM) == 2


def test_anchor_candidate_disabled_preserves_candidate_pool_size():
    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    generator = CandidateDirectionGenerator(
        rng=np.random.default_rng(0),
        n_random=4,
        n_bond_pairs=0,
        enable_anchor_candidate=False,
    )

    candidates = generator.generate(
        state,
        previous_direction=None,
        anchor_direction=np.array([1.0, 0.0, 0.0]),
    )

    assert len(candidates) == 4
    assert all(candidate.kind == DirectionCandidateKind.RANDOM for candidate in candidates)


def test_anchor_candidate_enabled_is_deprecated_noop():
    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    generator = CandidateDirectionGenerator(
        rng=np.random.default_rng(0),
        n_random=4,
        n_bond_pairs=0,
        enable_anchor_candidate=True,
    )

    candidates = generator.generate(
        state,
        previous_direction=None,
        anchor_direction=np.array([1.0, 0.0, 0.0]),
    )

    kinds = [candidate.kind for candidate in candidates]
    assert DirectionCandidateKind.ANCHOR not in kinds
    assert kinds.count(DirectionCandidateKind.RANDOM) == 4


def test_krylov_intent_generator_returns_unique_orthonormal_random_pair_blocks():
    state = State(
        numbers=np.array([6, 6, 6]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 1.2, 0.0],
            ]
        ),
    )
    generator = CandidateDirectionGenerator(np.random.default_rng(27), n_random=0)

    intents = generator.generate_krylov_intents(state, n_blocks=2)

    assert isinstance(intents, tuple)
    assert len(intents) == 2
    assert all(intent.basis.shape == (9, 2) for intent in intents)
    assert all(intent.pair is not None for intent in intents)
    assert len({intent.pair for intent in intents}) == 2
    for intent in intents:
        np.testing.assert_allclose(intent.basis.T @ intent.basis, np.eye(2), rtol=0.0, atol=1e-12)


def test_krylov_intent_generator_returns_random_only_rank_one_block_for_one_movable_atom():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        fixed_mask=np.array([False, True]),
    )
    generator = CandidateDirectionGenerator(np.random.default_rng(27), n_random=0)

    [intent] = generator.generate_krylov_intents(state, n_blocks=1)

    assert intent.pair is None
    assert intent.basis.shape == (6, 1)
    assert np.linalg.norm(intent.basis[:, 0]) == pytest.approx(1.0)


def test_krylov_intent_generator_rejects_periodic_single_atom_translation():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        cell=np.diag([8.0, 8.0, 8.0]),
        pbc=(True, True, True),
        fixed_mask=np.array([False, True]),
    )
    generator = CandidateDirectionGenerator(np.random.default_rng(27), n_random=0)

    with pytest.raises(ValueError, match="no projected random direction"):
        generator.generate_krylov_intents(state, n_blocks=1)


def test_krylov_intent_generator_rejects_all_fixed_state():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
        fixed_mask=np.array([True, True]),
    )
    generator = CandidateDirectionGenerator(np.random.default_rng(27), n_random=0)

    with pytest.raises(ValueError, match="no projected random direction"):
        generator.generate_krylov_intents(state, n_blocks=1)


def test_krylov_intent_generator_degrades_dependent_pair_axis_to_rank_one():
    class PairAlignedRng:
        def normal(self, size):
            assert size == 6
            return np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])

        def permutation(self, population_size):
            assert population_size == 1
            return np.array([0])

    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(PairAlignedRng(), n_random=0)

    [intent] = generator.generate_krylov_intents(state, n_blocks=1)

    assert intent.pair is None
    assert intent.basis.shape == (6, 1)
    assert np.linalg.norm(intent.basis[:, 0]) == pytest.approx(1.0)


def test_krylov_intent_generator_requires_positive_block_count():
    state = State(numbers=np.array([6]), positions=np.array([[0.0, 0.0, 0.0]]))
    generator = CandidateDirectionGenerator(np.random.default_rng(27), n_random=0)

    with pytest.raises(ValueError, match="positive"):
        generator.generate_krylov_intents(state, n_blocks=0)


def test_krylov_intent_generator_does_not_use_non_neighbor_or_closest_pair_sampling(monkeypatch):
    state = State(
        numbers=np.array([6, 6, 6]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 1.2, 0.0],
            ]
        ),
    )
    generator = CandidateDirectionGenerator(np.random.default_rng(27), n_random=0)

    def unexpected_legacy_pair_sampling(*args, **kwargs):
        raise AssertionError("Krylov intent generation must not use legacy pair sampling")

    monkeypatch.setattr(generator, "_random_non_neighbor_pairs", unexpected_legacy_pair_sampling)
    monkeypatch.setattr(generator, "_closest_mic_pairs", unexpected_legacy_pair_sampling)

    intents = generator.generate_krylov_intents(state, n_blocks=2)

    assert len(intents) == 2


class FixedNormalRng:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def normal(self, size):
        assert size == self.values.size
        return self.values.copy()


class ReverseChoiceRng:
    def choice(self, population_size, size, replace=False):
        assert replace is False
        return np.arange(population_size - 1, population_size - size - 1, -1)


def test_mass_weighted_random_is_direction_equivalent_for_same_mass_atoms():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
    raw = np.array([1.0, 2.0, 3.0, -1.0, -2.0, -3.0])
    unit_generator = CandidateDirectionGenerator(
        rng=FixedNormalRng(raw),
        n_random=1,
        random_direction_distribution="unit_gaussian",
    )
    mass_generator = CandidateDirectionGenerator(
        rng=FixedNormalRng(raw),
        n_random=1,
        random_direction_distribution="mass_weighted",
    )

    unit = unit_generator.generate(state, previous_direction=None)[0].direction
    mass_weighted = mass_generator.generate(state, previous_direction=None)[0].direction

    np.testing.assert_allclose(mass_weighted, unit)


def test_mass_weighted_random_gives_larger_components_to_light_atoms():
    state = State(
        numbers=np.array([46, 8]),
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(
        rng=FixedNormalRng(np.ones(6)),
        n_random=1,
        random_direction_distribution="mass_weighted",
    )

    [candidate] = generator.generate(state, previous_direction=None)
    atom_norms = np.linalg.norm(candidate.direction.reshape(2, 3), axis=1)

    assert atom_norms[1] > atom_norms[0]


def test_direction_generator_anchor_mixing_only_changes_momentum_candidate():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    previous = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    anchor = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=2, bond_pairs=[(0, 1)])

    candidates = generator.generate(state, previous_direction=previous, anchor_direction=anchor, anchor_mixing_alpha=0.6)

    momentum = next(candidate for candidate in candidates if candidate.kind == DirectionCandidateKind.MOMENTUM)
    bond = next(candidate for candidate in candidates if candidate.kind == DirectionCandidateKind.BOND)
    anchor_unit = anchor / np.linalg.norm(anchor)

    assert float(np.dot(momentum.direction, anchor_unit)) == pytest.approx(0.6)
    np.testing.assert_allclose(bond.direction.reshape(2, 3)[0], np.array([-1.0, 0.0, 0.0]) / np.sqrt(2.0))


def test_anchor_mixing_replaces_score_layer_anchor_penalty():
    class Quadratic:
        def energy_gradient(self, flat_positions, state):
            gradient = np.asarray(flat_positions, dtype=float).copy()
            return 0.5 * float(gradient @ gradient), gradient

    class CapturingScorer(DirectionScorer):
        def __init__(self):
            super().__init__(anchor_weight=10.0)
            self.anchor_directions = []

        def score_candidate(self, **kwargs):
            self.anchor_directions.append(kwargs["anchor_direction"])
            return super().score_candidate(**kwargs)

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    oracle = SoftModeOracle(
        AnalyticCalculator(Quadratic()),
        np.random.default_rng(0),
        candidates=1,
        anchor_mixing_alpha=0.6,
    )
    oracle.scorer = CapturingScorer()

    oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=np.array([1.0, 0.0, 0.0]),
        anchor_direction=np.array([0.0, 1.0, 0.0]),
    )

    assert oracle.scorer.anchor_directions
    assert all(anchor_direction is None for anchor_direction in oracle.scorer.anchor_directions)


def test_surface_walker_applies_direction_scorer_weights_from_config():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(continuity_weight=0.0, history_push_weight=0.3, enable_momentum_candidate=False),
        softening_enabled=False,
    )

    assert walker.oracle.scorer.continuity_weight == 0.0
    assert walker.oracle.scorer.history_push_weight == 0.3
    assert walker.oracle.generator.enable_momentum_candidate is False


def test_direction_diagnostics_disabled_does_not_create_trace(tmp_path):
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            max_trials=1,
            max_steps_per_walk=1,
            oracle_candidates=1,
            n_bond_pairs=0,
            proposal_relax_steps=0,
            direction_diagnostics_enabled=False,
            direction_diagnostics_path=str(tmp_path / "direction_trace.jsonl"),
        ),
        softening_enabled=False,
    )

    walker._record_direction_diagnostics(
        trial_index=0,
        proposal_index=0,
        step_index=0,
        choice=DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
        anchor_direction=None,
    )

    assert not (tmp_path / "direction_trace.jsonl").exists()


def test_direction_diagnostics_records_selected_kind_and_anchor_cosine(tmp_path):
    path = tmp_path / "direction_trace.jsonl"
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_diagnostics_enabled=True, direction_diagnostics_path=str(path)),
        softening_enabled=False,
    )
    choice = DirectionChoice(
        direction=np.array([1.0, 0.0, 0.0]),
        curvature=2.0,
        kind=DirectionCandidateKind.RANDOM,
        candidate_count=3,
    )

    walker._reset_direction_diagnostics()
    walker._record_direction_diagnostics(
        trial_index=4,
        proposal_index=1,
        step_index=2,
        choice=choice,
        anchor_direction=np.array([1.0, 0.0, 0.0]),
    )

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["anchor_cosine"] == pytest.approx(1.0)
    assert row["candidate_count"] == 3
    assert row["curvature"] == pytest.approx(2.0)
    assert row["proposal"] == 1
    assert row["selected_kind"] == "random"
    assert row["step"] == 2
    assert row["trial"] == 4


def test_choice_aligned_softening_rebuild_gate_is_default_off():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(choice_aligned_softening_enabled=False),
        softening_enabled=True,
    )

    assert walker._should_rebuild_softening_for_choice(
        np.array([1.0, 0.0, 0.0]),
        np.array([-1.0, 0.0, 0.0]),
    ) is False


def test_choice_aligned_softening_rebuild_gate_uses_cosine_threshold():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(
            choice_aligned_softening_enabled=True,
            choice_aligned_softening_cos_threshold=0.3,
        ),
        softening_enabled=True,
    )

    assert walker._should_rebuild_softening_for_choice(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ) is True
    assert walker._should_rebuild_softening_for_choice(
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ) is False


def test_choice_aligned_softening_rebuild_gate_requires_active_softening(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            n_bond_pairs=0,
            proposal_relax_steps=0,
            choice_aligned_softening_enabled=True,
            choice_aligned_softening_cos_threshold=0.3,
        ),
        softening_enabled=False,
    )
    calls = {"curvature": 0, "build": 0}
    captured_bias_curvatures = []

    assert walker._should_rebuild_softening_for_choice(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
    ) is False

    walker.oracle.generator.generate_initial_direction = lambda *args, **kwargs: np.array([1.0, 0.0, 0.0])
    walker.oracle.choose_direction = lambda *args, **kwargs: DirectionChoice(
        direction=np.array([0.0, 1.0, 0.0]),
        curvature=99.0,
        kind=DirectionCandidateKind.RANDOM,
        candidate_count=1,
    )

    def fake_build(current, direction=None):
        calls["build"] += 1
        return None

    def fail_if_recomputed(current, proposal, direction):
        calls["curvature"] += 1
        raise AssertionError("disabled local softening must not trigger curvature recompute")

    def fake_bias_weight(curvature, sigma):
        captured_bias_curvatures.append(curvature)
        return 0.5

    monkeypatch.setattr(walker, "_build_softening", fake_build)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", fail_if_recomputed)
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda current, direction: 1.0)
    monkeypatch.setattr(walker, "_bias_weight", fake_bias_weight)

    class NoRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator

        @staticmethod
        def classify_outcome(**kwargs):
            return RelaxOutcomeClass.USEFUL_PROGRESS

        def relax(
            self,
            state,
            fmax,
            maxiter,
            coordinate_trust_radius=None,
            trajectory_callback=None,
            trajectory_stride=1,
        ):
            energy, gradient = self.evaluator(state.flatten_positions(), state)
            return RelaxResult(
                state=state,
                energy=energy,
                gradient_norm=float(np.linalg.norm(gradient)),
                n_iter=0,
            )

    import pamssw.walker as walker_module

    monkeypatch.setattr(walker_module, "Relaxer", NoRelaxer)
    walker._walk_candidate_from_seed(state)

    assert calls == {"curvature": 0, "build": 1}
    assert captured_bias_curvatures == [pytest.approx(99.0)]


def test_choice_aligned_softening_rebuild_gate_rejects_degenerate_inputs():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(choice_aligned_softening_enabled=True),
        softening_enabled=True,
    )

    assert walker._should_rebuild_softening_for_choice(None, np.array([1.0, 0.0, 0.0])) is False
    assert walker._should_rebuild_softening_for_choice(np.array([1.0, 0.0, 0.0]), None) is False
    assert walker._should_rebuild_softening_for_choice(
        np.array([1.0, 0.0, 0.0]),
        np.array([1.0, 0.0]),
    ) is False
    assert walker._should_rebuild_softening_for_choice(
        np.array([1.0, 0.0, 0.0]),
        np.array([np.nan, 0.0, 0.0]),
    ) is False
    assert walker._should_rebuild_softening_for_choice(
        np.array([np.inf, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ) is False
    assert walker._should_rebuild_softening_for_choice(
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
    ) is False
    assert walker._should_rebuild_softening_for_choice(
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    ) is False


def test_choice_aligned_softening_recomputes_inner_curvature(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            n_bond_pairs=0,
            proposal_relax_steps=0,
            choice_aligned_softening_enabled=True,
            choice_aligned_softening_cos_threshold=0.3,
        ),
        softening_enabled=True,
    )
    calls = {"curvature": 0, "build": 0}

    class SentinelSoftening:
        def evaluate(self, flat_positions):
            return 0.0, np.zeros_like(flat_positions)

    anchor_softening = SentinelSoftening()
    choice_softening = SentinelSoftening()
    captured_bias_curvatures = []

    walker.oracle.generator.generate_initial_direction = lambda *args, **kwargs: np.array([1.0, 0.0, 0.0])
    walker.oracle.choose_direction = lambda *args, **kwargs: DirectionChoice(
        direction=np.array([0.0, 1.0, 0.0]),
        curvature=99.0,
        kind=DirectionCandidateKind.RANDOM,
        candidate_count=1,
    )

    def fake_build(current, direction=None):
        calls["build"] += 1
        if np.allclose(direction, np.array([1.0, 0.0, 0.0])):
            return anchor_softening
        if np.allclose(direction, np.array([0.0, 1.0, 0.0])):
            return choice_softening
        raise AssertionError(f"unexpected softening direction: {direction}")

    def fake_curvature(current, proposal, direction):
        calls["curvature"] += 1
        assert proposal.softening is choice_softening
        return 1.0

    def fake_bias_weight(curvature, sigma):
        captured_bias_curvatures.append(curvature)
        return 0.5

    monkeypatch.setattr(walker, "_build_softening", fake_build)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", fake_curvature)
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda current, direction: 1.0)
    monkeypatch.setattr(walker, "_bias_weight", fake_bias_weight)

    class NoRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator

        @staticmethod
        def classify_outcome(**kwargs):
            return RelaxOutcomeClass.USEFUL_PROGRESS

        def relax(
            self,
            state,
            fmax,
            maxiter,
            coordinate_trust_radius=None,
            trajectory_callback=None,
            trajectory_stride=1,
        ):
            energy, gradient = self.evaluator(state.flatten_positions(), state)
            return RelaxResult(
                state=state,
                energy=energy,
                gradient_norm=float(np.linalg.norm(gradient)),
                n_iter=0,
            )

    import pamssw.walker as walker_module

    monkeypatch.setattr(walker_module, "Relaxer", NoRelaxer)
    walker._walk_candidate_from_seed(state)

    assert calls == {"curvature": 1, "build": 2}
    assert captured_bias_curvatures == [pytest.approx(1.0)]


def test_direction_generator_adds_bond_candidate_when_pairs_are_provided():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=1, bond_pairs=[(0, 1)])

    candidates = generator.generate(state, previous_direction=None)

    assert [candidate.kind for candidate in candidates] == [DirectionCandidateKind.BOND]
    np.testing.assert_allclose(candidates[0].direction.reshape(2, 3)[0], np.array([-1.0, 0.0, 0.0]) / np.sqrt(2.0))
    np.testing.assert_allclose(candidates[0].direction.reshape(2, 3)[1], np.array([1.0, 0.0, 0.0]) / np.sqrt(2.0))


def test_bond_form_direction_decreases_pair_distance():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=0)

    direction = generator._bond_form_direction(state, 0, 1)
    assert direction is not None
    displaced = state.positions + 0.1 * direction.reshape(2, 3)

    assert np.linalg.norm(displaced[1] - displaced[0]) < 3.0


def test_bond_break_direction_increases_pair_distance():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=0)

    direction = generator._bond_break_direction(state, 0, 1)
    assert direction is not None
    displaced = state.positions + 0.1 * direction.reshape(2, 3)

    assert np.linalg.norm(displaced[1] - displaced[0]) > 1.4


def test_bond_form_break_split_rejects_too_far_formation_pairs():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(0),
        n_random=1,
        enable_bond_form_break_split=True,
        n_bond_formation_pairs=1,
        n_bond_breaking_pairs=0,
        bond_distance_threshold=1.0,
        bond_formation_max_distance=4.0,
    )

    candidates = generator.generate(state, previous_direction=None)

    assert all(candidate.kind is not DirectionCandidateKind.BOND_FORM for candidate in candidates)


def test_bond_form_break_split_generates_break_candidates_for_short_pairs():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(0),
        n_random=1,
        enable_bond_form_break_split=True,
        n_bond_formation_pairs=0,
        n_bond_breaking_pairs=1,
        bond_breaking_max_distance=2.0,
    )

    candidates = generator.generate(state, previous_direction=None)

    assert [candidate.kind for candidate in candidates] == [DirectionCandidateKind.BOND_BREAK]


def test_bond_form_break_split_keeps_explicit_bond_pairs_as_legacy_bond_candidates():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(0),
        n_random=1,
        bond_pairs=[(0, 1)],
        enable_bond_form_break_split=True,
        n_bond_formation_pairs=0,
        n_bond_breaking_pairs=0,
    )

    candidates = generator.generate(state, previous_direction=None)

    assert [candidate.kind for candidate in candidates] == [DirectionCandidateKind.BOND]


def test_bond_form_break_split_routes_per_call_pair_budget_by_configured_ratio():
    generator = CandidateDirectionGenerator(
        np.random.default_rng(0),
        n_random=0,
        enable_bond_form_break_split=True,
        n_bond_formation_pairs=2,
        n_bond_breaking_pairs=1,
    )

    assert generator._split_bond_pair_counts(6) == (4, 2)


def test_bond_form_break_pair_window_samples_even_when_eligible_count_fits_budget():
    state = State(
        numbers=np.array([6, 6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(ReverseChoiceRng(), n_random=0)

    pairs = generator._random_pairs_in_distance_window(
        state,
        n_pairs=3,
        min_distance=0.0,
        max_distance=10.0,
    )

    assert pairs == [(1, 2), (0, 2), (0, 1)]


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
        n_random=3,
        n_bond_pairs=3,
        bond_distance_threshold=2.0,
    )

    candidates = generator.generate(state, previous_direction=None)

    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.BOND) > 0
    assert generator.last_random_bond_pairs_requested == 3
    assert generator.last_random_bond_pairs_generated > 0
    assert generator.last_random_bond_candidates_valid > 0


def test_direction_generator_allows_per_call_bond_pair_override():
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
        n_bond_pairs=1,
        bond_distance_threshold=2.0,
    )

    generator.generate(state, previous_direction=None, n_bond_pairs=3)

    assert generator.n_bond_pairs == 1
    assert generator.last_random_bond_pairs_requested == 3


def test_direction_generator_adds_periodic_random_bond_candidates_with_mic_direction():
    state = State(
        numbers=np.full(2, 18),
        positions=np.array([[2.0, 0.0, 0.0], [8.0, 0.0, 0.0]], dtype=float),
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=(True, True, True),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(0),
        n_random=1,
        n_bond_pairs=1,
        bond_distance_threshold=3.0,
    )

    candidates = generator.generate(state, previous_direction=None)

    assert [candidate.kind for candidate in candidates] == [DirectionCandidateKind.BOND]
    assert generator.last_random_bond_pairs_generated == 1
    direction = candidates[0].direction.reshape(2, 3)
    trial_positions = state.positions + 0.5 * direction
    delta = trial_positions[1] - trial_positions[0]
    delta[0] -= round(delta[0] / 10.0) * 10.0
    assert np.linalg.norm(delta) > 4.0


def test_direction_generator_falls_back_to_closest_mic_pairs_when_non_neighbors_are_sparse():
    state = State(
        numbers=np.full(4, 18),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=(True, True, True),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(0),
        n_random=3,
        n_bond_pairs=3,
        bond_distance_threshold=20.0,
    )

    candidates = generator.generate(state, previous_direction=None)

    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.BOND) == 3
    assert generator.last_random_bond_pairs_requested == 3
    assert generator.last_random_bond_pairs_generated == 3
    assert generator.last_fallback_bond_pairs_generated == 3
    assert generator.last_random_bond_candidates_valid == 3


def test_direction_generator_resets_public_fallback_counter_without_generate_wrapper():
    state = State(
        numbers=np.full(4, 18),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=float,
        ),
        cell=np.diag([10.0, 10.0, 10.0]),
        pbc=(True, True, True),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(0),
        n_random=0,
        n_bond_pairs=3,
        bond_distance_threshold=20.0,
    )

    generator._random_non_neighbor_pairs(state, n_pairs=3, distance_threshold=20.0)
    assert generator.last_fallback_bond_pairs_generated == 3
    generator._random_non_neighbor_pairs(state, n_pairs=0, distance_threshold=20.0)
    assert generator.last_fallback_bond_pairs_generated == 0


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


def test_walk_initial_anchor_uses_trial_progress_for_bond_mixing():
    state = State(numbers=np.array([1]), positions=np.array([[0.1, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_trials=5, max_steps_per_walk=1),
        softening_enabled=False,
    )
    observed: dict[str, int] = {}

    def record_initial_direction(
        current,
        step_index,
        max_steps,
        lambda_start,
        lambda_end,
        n_bond_pairs,
        bond_distance_threshold,
    ):
        observed["step_index"] = step_index
        observed["max_steps"] = max_steps
        raise RuntimeError("stop after anchor")

    walker.oracle.generator.generate_initial_direction = record_initial_direction

    with pytest.raises(RuntimeError, match="stop after anchor"):
        walker._walk_candidate_from_seed(state, trial_index=4)

    assert observed == {"step_index": 4, "max_steps": 5}


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


def test_direction_scorer_allows_continuity_weight_override():
    scorer = DirectionScorer(damage_weight=0.0, continuity_weight=1.0)
    previous = np.array([1.0, 0.0, 0.0])

    default_score = scorer.score(
        curvature=0.0,
        sigma=1.0,
        direction=-previous,
        previous_direction=previous,
        anchor_direction=None,
        damage_risk=0.0,
    )
    overridden_score = scorer.score(
        curvature=0.0,
        sigma=1.0,
        direction=-previous,
        previous_direction=previous,
        anchor_direction=None,
        damage_risk=0.0,
        continuity_weight=0.0,
    )

    assert overridden_score > default_score


def test_surface_walker_gates_continuity_from_previous_relax_outcome():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(continuity_weight=0.1),
        softening_enabled=False,
    )

    assert walker._continuity_weight_for_outcome(None) == pytest.approx(0.1)
    assert walker._continuity_weight_for_outcome(RelaxOutcomeClass.USEFUL_PROGRESS) == pytest.approx(0.1)
    assert walker._continuity_weight_for_outcome(RelaxOutcomeClass.CONVERGED_PRODUCTIVE) == pytest.approx(0.1)
    assert walker._continuity_weight_for_outcome(RelaxOutcomeClass.STAGNATED) == pytest.approx(0.0)
    assert walker._continuity_weight_for_outcome(RelaxOutcomeClass.CONVERGED_UNPRODUCTIVE) == pytest.approx(0.0)
    assert walker._continuity_weight_for_outcome(RelaxOutcomeClass.DAMAGED) == pytest.approx(0.05)


def test_surface_walker_can_disable_outcome_gated_continuity():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(continuity_weight=0.1, enable_outcome_gated_continuity=False),
        softening_enabled=False,
    )

    assert walker._continuity_weight_for_outcome(RelaxOutcomeClass.STAGNATED) == pytest.approx(0.1)
    assert walker._continuity_weight_for_outcome(RelaxOutcomeClass.CONVERGED_UNPRODUCTIVE) == pytest.approx(0.1)


def test_surface_walker_boosts_bond_pairs_after_stagnated_outcome():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(n_bond_pairs=2, stagnation_bond_pair_boost=3, max_stagnation_bond_pairs=4),
        softening_enabled=False,
    )

    assert walker._n_bond_pairs_for_outcome(None) == 2
    assert walker._n_bond_pairs_for_outcome(RelaxOutcomeClass.USEFUL_PROGRESS) == 2
    assert walker._n_bond_pairs_for_outcome(RelaxOutcomeClass.STAGNATED) == 4
    assert walker._n_bond_pairs_for_outcome(RelaxOutcomeClass.CONVERGED_UNPRODUCTIVE) == 4


def test_stagnation_bond_pair_cap_only_applies_after_stagnation():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(n_bond_pairs=12, stagnation_bond_pair_boost=3, max_stagnation_bond_pairs=10),
        softening_enabled=False,
    )

    assert walker._n_bond_pairs_for_outcome(None) == 12
    assert walker._n_bond_pairs_for_outcome(RelaxOutcomeClass.USEFUL_PROGRESS) == 12
    assert walker._n_bond_pairs_for_outcome(RelaxOutcomeClass.STAGNATED) == 10


def test_surface_walker_uses_rescue_optimizer_only_after_unproductive_outcome():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(proposal_optimizer="ase-fire", proposal_optimizer_alt="ase-lbfgs"),
        softening_enabled=False,
    )

    assert walker._proposal_optimizer_for_outcome(None) == "ase-fire"
    assert walker._proposal_optimizer_for_outcome(RelaxOutcomeClass.USEFUL_PROGRESS) == "ase-fire"
    assert walker._proposal_optimizer_for_outcome(RelaxOutcomeClass.STAGNATED) == "ase-lbfgs"
    assert walker._proposal_optimizer_for_outcome(RelaxOutcomeClass.CONVERGED_UNPRODUCTIVE) == "ase-lbfgs"


def test_surface_walker_duplicate_rescue_optimizer_override_takes_precedence():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(proposal_optimizer="ase-fire", proposal_optimizer_alt="ase-lbfgs"),
        softening_enabled=False,
    )

    assert walker._proposal_optimizer_for_outcome(None, override="ase-lbfgs") == "ase-lbfgs"
    assert walker._proposal_optimizer_for_outcome(RelaxOutcomeClass.STAGNATED, override="ase-fire") == "ase-fire"


def test_candidate_proposal_disables_recursive_duplicate_rescue():
    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))

    normal = CandidateProposal("ssw_walk", state)
    rescue = CandidateProposal("duplicate_rescue", state, allow_duplicate_rescue=False)

    assert normal.allow_duplicate_rescue
    assert not rescue.allow_duplicate_rescue


def test_surface_walker_disabled_direction_type_ucb_passes_no_bonus(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_steps_per_walk=1, oracle_candidates=1, proposal_relax_steps=0),
        softening_enabled=False,
    )
    seen = {}

    walker.oracle.generator.generate_initial_direction = lambda *args, **kwargs: np.array([1.0, 0.0, 0.0])

    def choose_direction(*args, **kwargs):
        seen["bonus_fn"] = kwargs["direction_type_bonus_fn"]
        return DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.MOMENTUM,
            candidate_count=1,
        )

    class NoRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator

        @staticmethod
        def classify_outcome(**kwargs):
            return RelaxOutcomeClass.USEFUL_PROGRESS

        def relax(self, state, **kwargs):
            return RelaxResult(state=state, energy=0.0, gradient_norm=0.0, n_iter=0)

    import pamssw.walker as walker_module

    monkeypatch.setattr(walker_module, "Relaxer", NoRelaxer)
    monkeypatch.setattr(walker, "_build_softening", lambda *args, **kwargs: None)
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(walker.oracle, "choose_direction", choose_direction)

    walker._walk_candidate_from_seed(state)

    assert seen["bonus_fn"] is None


def test_surface_walker_enabled_direction_type_ucb_biases_choice(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            proposal_relax_steps=0,
            direction_type_ucb_enabled=True,
            direction_type_success_weight=1.0,
            direction_type_exploration_weight=0.0,
        ),
        softening_enabled=False,
    )
    walker.direction_type_memory.record_trial([DirectionCandidateKind.BOND], productive=True)
    selected_direction_kinds: set[DirectionCandidateKind] = set()

    walker.oracle.generator.generate_initial_direction = lambda *args, **kwargs: np.array([1.0, 0.0, 0.0])
    walker.oracle.generator.generate = lambda *args, **kwargs: [
        DirectionCandidate(DirectionCandidateKind.MOMENTUM, np.array([1.0, 0.0, 0.0])),
        DirectionCandidate(DirectionCandidateKind.BOND, np.array([0.0, 1.0, 0.0])),
    ]
    walker.oracle.scorer = KindScoreScorer(
        {
            DirectionCandidateKind.MOMENTUM: 1.0,
            DirectionCandidateKind.BOND: 0.5,
        }
    )

    class NoRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator

        @staticmethod
        def classify_outcome(**kwargs):
            return RelaxOutcomeClass.USEFUL_PROGRESS

        def relax(self, state, **kwargs):
            return RelaxResult(state=state, energy=0.0, gradient_norm=0.0, n_iter=0)

    import pamssw.walker as walker_module

    monkeypatch.setattr(walker_module, "Relaxer", NoRelaxer)
    monkeypatch.setattr(walker, "_build_softening", lambda *args, **kwargs: None)
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", lambda *args, **kwargs: 1.0)

    walker._walk_candidate_from_seed(state, selected_direction_kinds=selected_direction_kinds)

    assert selected_direction_kinds == {DirectionCandidateKind.BOND}


def test_proposal_pool_tracks_repeated_selected_kind_once(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(proposal_pool_size=1),
        softening_enabled=False,
    )

    def fake_walk(seed_state, archive, step_target, **kwargs):
        kwargs["selected_direction_kinds"].add(DirectionCandidateKind.RANDOM)
        kwargs["selected_direction_kinds"].add(DirectionCandidateKind.RANDOM)
        return seed_state

    monkeypatch.setattr(walker, "_walk_candidate_from_seed", fake_walk)

    proposals = walker._proposal_pool(state, archive=None, trial_index=0)

    assert proposals[0].selected_direction_kinds == frozenset({DirectionCandidateKind.RANDOM})


@pytest.mark.parametrize(
    ("is_new", "candidate_energy", "expected_productive"),
    [
        (True, -0.5, True),
        (False, -1.1, True),
        (False, -1.0, False),
    ],
)
def test_run_records_direction_type_trial_productivity(monkeypatch, is_new, candidate_energy, expected_productive):
    initial = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    candidate_state = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    if not is_new:
        candidate_state = initial
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_trials=1, direction_type_ucb_enabled=True),
        softening_enabled=False,
    )
    relax_calls = 0

    def fake_relax_true_minimum(state, trajectory_name=None, *, quench_purpose=None):
        nonlocal relax_calls
        relax_calls += 1
        if relax_calls == 1:
            return RelaxResult(initial, energy=-1.0, gradient_norm=0.0, n_iter=0)
        return RelaxResult(candidate_state, energy=candidate_energy, gradient_norm=0.0, n_iter=0)

    monkeypatch.setattr(walker, "relax_true_minimum", fake_relax_true_minimum)
    monkeypatch.setattr(
        walker,
        "_proposal_pool",
        lambda *args, **kwargs: [
            CandidateProposal("test", candidate_state, selected_direction_kinds=frozenset({DirectionCandidateKind.RANDOM}))
        ],
    )

    result = walker.run(initial)

    assert result.stats["direction_type_selected_random"] == 1
    assert result.stats["direction_type_productive_random"] == int(expected_productive)


def test_run_resets_direction_type_memory_at_start(monkeypatch):
    initial = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_trials=1, direction_type_ucb_enabled=True),
        softening_enabled=False,
    )
    walker.direction_type_memory.record_trial([DirectionCandidateKind.BOND], productive=True)
    monkeypatch.setattr(walker.calculator, "exhausted", lambda: True)

    result = walker.run(initial)

    assert result.stats["direction_type_selected_bond"] == 0
    assert result.stats["direction_type_productive_bond"] == 0


def test_direction_stats_summary_includes_direction_type_fields():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(direction_type_ucb_enabled=True),
        softening_enabled=False,
    )
    walker.direction_type_memory.record_trial([DirectionCandidateKind.RITZ_REG], productive=False)

    summary = walker._direction_stats_summary()

    assert summary["direction_type_ucb_enabled"] == 1
    for kind in DirectionCandidateKind:
        assert f"direction_type_selected_{kind.value}" in summary
        assert f"direction_type_productive_{kind.value}" in summary


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


def test_direction_scorer_uses_best_multiscale_novelty_gain():
    class RecordingArchive:
        def __init__(self):
            self.calls = 0

        def coverage_gain(self, descriptor):
            self.calls += 1
            return {1: 0.1, 2: 0.9}[self.calls]

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    candidate = DirectionCandidate(DirectionCandidateKind.RANDOM, np.array([1.0, 0.0, 0.0]))
    archive = RecordingArchive()
    scorer = DirectionScorer(novelty_weight=2.0, novelty_probe_scales=(0.5, 1.5))

    score = scorer.score_candidate(
        state=state,
        candidate=candidate,
        curvature=0.0,
        sigma=1.0,
        previous_direction=None,
        anchor_direction=None,
        archive=archive,
    )

    assert archive.calls == 2
    assert score == pytest.approx(1.8)


def test_trust_region_controller_shrinks_step_length_after_bad_local_model_without_weakening_bias():
    controller = TrustRegionBiasController()

    update = controller.update(
        curvature=2.0,
        sigma=0.5,
        true_delta=2.0,
        sigma_scale=1.0,
        weight_scale=1.0,
        bias_weight=0.5,
    )

    assert update.action == "shrink"
    assert update.sigma_scale < 1.0
    assert update.weight_scale == 1.0
    assert update.model_error > controller.error_tolerance


def test_bias_strength_controller_drops_on_bias_induced_damage():
    controller = BiasStrengthController()

    update = controller.update(
        curvature=1.0,
        sigma=1.0,
        bias_weight=2.0,
        weight_scale=1.0,
        bias_induced_damage=True,
    )

    assert update.action == "shrink"
    assert update.weight_scale < 1.0


def test_bias_strength_controller_increases_when_bias_does_not_flip_curvature():
    controller = BiasStrengthController()

    update = controller.update(
        curvature=10.0,
        sigma=1.0,
        bias_weight=1.0,
        weight_scale=1.0,
    )

    assert update.action == "expand"
    assert update.weight_scale > 1.0


def test_trust_region_prediction_includes_true_gradient_linear_term():
    controller = StepLengthController()

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


def test_trust_region_uses_error_floor_for_near_zero_prediction():
    controller = TrustRegionBiasController()

    update = controller.update(
        curvature=0.0,
        sigma=0.5,
        true_delta=0.03,
        sigma_scale=1.0,
        weight_scale=1.0,
        error_floor=0.06,
    )

    assert update.action == "expand"
    assert update.model_error == pytest.approx(0.5, rel=1e-6)


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


def test_oracle_candidate_hvp_exposes_true_curvature_from_same_parts_evaluations(monkeypatch):
    """A selected native candidate carries the true-PES curvature already evaluated for its HVP."""

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    calculator = EvalCounter(AnalyticCalculator(Quadratic()))
    oracle = SoftModeOracle(calculator, np.random.default_rng(0), candidates=1)
    direction = np.array([1.0, 0.0, 0.0])
    candidate = DirectionCandidate(DirectionCandidateKind.RANDOM, direction)
    monkeypatch.setattr(oracle.generator, "generate", lambda *args, **kwargs: [candidate])
    proposal = ProposalPotential(
        calculator,
        biases=[GaussianBiasTerm(center=state.flatten_positions(), direction=direction, sigma=1.0, weight=4.0)],
    )

    choice = oracle.choose_direction(state, proposal, previous_direction=None)

    assert calculator.force_evaluations == 2
    assert choice.curvature == pytest.approx(-3.0, rel=1e-3)
    assert choice.true_curvature == pytest.approx(1.0, rel=1e-3)


def test_walk_uses_native_oracle_true_curvature_without_a_second_central_hvp(monkeypatch):
    """The walker must not re-evaluate true curvature when its own oracle selected the candidate."""

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_steps_per_walk=1, oracle_candidates=1, n_bond_pairs=0, rng_seed=0),
        softening_enabled=False,
    )
    direction = np.array([1.0, 0.0, 0.0])
    candidate = DirectionCandidate(DirectionCandidateKind.RANDOM, direction)
    monkeypatch.setattr(walker.oracle.generator, "generate_initial_direction", lambda *args, **kwargs: direction)
    monkeypatch.setattr(walker.oracle.generator, "generate", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(
        walker,
        "_true_directional_curvature",
        lambda *args, **kwargs: pytest.fail("native candidate must reuse its oracle true curvature"),
    )
    monkeypatch.setattr(
        walker,
        "_relax_proposal_task",
        lambda task, **kwargs: RelaxResult(task.initial_state, energy=0.0, gradient_norm=0.0, n_iter=0),
    )

    walker._walk_candidate_from_seed(state)

    assert walker.calculator.force_evaluations == 4


def test_walk_reuses_true_after_only_for_the_identical_next_step_state(monkeypatch):
    """The next step consumes the previous true-after result exactly once, saving one true-PES evaluation."""

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_steps_per_walk=2, oracle_candidates=1, n_bond_pairs=0, rng_seed=0),
        softening_enabled=False,
    )
    direction = np.array([1.0, 0.0, 0.0])
    monkeypatch.setattr(walker.oracle.generator, "generate_initial_direction", lambda *args, **kwargs: direction)
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=direction,
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(
        walker,
        "_relax_proposal_task",
        lambda task, **kwargs: RelaxResult(task.initial_state, energy=0.0, gradient_norm=0.0, n_iter=0),
    )

    walker._walk_candidate_from_seed(state)

    assert walker.calculator.force_evaluations == 3


def test_external_direction_choice_falls_back_to_a_direct_true_curvature_hvp(monkeypatch):
    """A DirectionChoice not produced by the native candidate loop carries no reuse evidence."""

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_steps_per_walk=1, oracle_candidates=1, n_bond_pairs=0, rng_seed=0),
        softening_enabled=False,
    )
    direction = np.array([1.0, 0.0, 0.0])
    true_curvature_calls = []
    monkeypatch.setattr(walker.oracle.generator, "generate_initial_direction", lambda *args, **kwargs: direction)
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=direction,
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(
        walker,
        "_true_directional_curvature",
        lambda *args, **kwargs: true_curvature_calls.append(1) or 1.0,
    )
    monkeypatch.setattr(
        walker,
        "_relax_proposal_task",
        lambda task, **kwargs: RelaxResult(task.initial_state, energy=0.0, gradient_norm=0.0, n_iter=0),
    )

    walker._walk_candidate_from_seed(state)

    assert true_curvature_calls == [1]


def test_geometry_validator_rejects_nan_positions_and_nonfinite_energy():
    valid = GeometryValidator(min_distance=0.5)
    bad_positions = State(numbers=np.array([1]), positions=np.array([[np.nan, 0.0, 0.0]]))

    assert not valid.is_valid_state(bad_positions)

    class NonFinite:
        def evaluate_flat(self, flat_positions, state):
            return float("nan"), np.zeros_like(flat_positions)

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))

    assert not valid.is_valid_evaluation(state, NonFinite())


def test_geometry_validator_rejects_atom_overlap_with_or_without_pbc():
    validator = GeometryValidator(min_distance=0.5)
    overlapped = State(numbers=np.array([1, 1]), positions=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]))
    periodic = State(
        numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]),
        cell=np.eye(3),
        pbc=(True, True, True),
    )

    assert not validator.is_valid_state(overlapped)
    assert not validator.is_valid_state(periodic)


def test_geometry_validator_rejects_covalent_radius_collisions_but_allows_hydrogen_bonds():
    validator = GeometryValidator(min_distance=0.5)
    compressed_carbon = State(numbers=np.array([6, 6]), positions=np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]))
    hydrogen_bond = State(numbers=np.array([1, 1]), positions=np.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]))

    assert not validator.is_valid_state(compressed_carbon)
    assert validator.is_valid_state(hydrogen_bond)


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
    assert decreased >= increased


def test_step_target_controller_ignores_damage_rate_for_multiplier():
    controller = StepTargetController(fallback_target=0.6)
    base_multiplier = controller.multiplier

    for _ in range(controller.feedback_warmup_trials):
        controller.record_trial(escaped=True, damaged=True)

    assert controller.multiplier == base_multiplier
    assert controller.stats()["adaptive_damage_warning"] == 1


def test_step_target_controller_filters_tiny_fake_escapes_when_evidence_is_available():
    controller = StepTargetController(
        fallback_target=0.6,
        min_escape_energy_delta=0.1,
        min_escape_descriptor_delta=0.1,
        min_escape_novelty=0.2,
    )

    controller.record_trial(
        escaped=True,
        damaged=False,
        seed_energy=-10.0,
        new_energy=-9.99,
        descriptor_delta=0.02,
        novelty_gain=0.05,
    )

    assert controller.escapes == 0
    assert controller.raw_escapes == 1
    assert controller.multiplier > 1.0


def test_step_target_controller_counts_uphill_or_novel_escapes_as_meaningful():
    controller = StepTargetController(
        fallback_target=0.6,
        min_escape_energy_delta=0.1,
        min_escape_descriptor_delta=0.1,
        min_escape_novelty=0.2,
    )

    controller.record_trial(
        escaped=True,
        damaged=False,
        seed_energy=-10.0,
        new_energy=-9.85,
        descriptor_delta=0.0,
        novelty_gain=0.0,
    )
    controller.record_trial(
        escaped=True,
        damaged=False,
        seed_energy=-10.0,
        new_energy=-9.99,
        descriptor_delta=0.0,
        novelty_gain=0.3,
    )

    assert controller.escapes == 2
    assert controller.stats()["adaptive_escape_rate"] == pytest.approx(1.0)


def test_step_target_controller_counts_structurally_distant_escape_as_meaningful():
    controller = StepTargetController(
        fallback_target=0.6,
        min_escape_energy_delta=0.1,
        min_escape_descriptor_delta=0.1,
        min_escape_novelty=1.01,
    )

    controller.record_trial(
        escaped=True,
        damaged=False,
        seed_energy=-10.0,
        new_energy=-9.99,
        descriptor_delta=0.2,
        novelty_gain=0.0,
    )

    assert controller.escapes == 1


def test_step_target_controller_default_disables_novelty_only_escape():
    controller = StepTargetController(fallback_target=0.6)

    controller.record_trial(
        escaped=True,
        damaged=False,
        seed_energy=-10.0,
        new_energy=-9.99,
        descriptor_delta=0.0,
        novelty_gain=1.0,
    )

    assert controller.escapes == 0
    assert controller.raw_escapes == 1


def test_step_target_controller_boosts_after_global_progress_stalls():
    controller = StepTargetController(
        fallback_target=0.6,
        progress_patience=2,
        progress_boost_factor=1.5,
        progress_max_boost=2.0,
    )
    base = controller.target()

    controller.record_trial(
        escaped=True,
        damaged=False,
        energy_delta=0.5,
        descriptor_delta=0.2,
        global_improved=False,
    )
    assert controller.target() == pytest.approx(base)

    controller.record_trial(
        escaped=True,
        damaged=False,
        energy_delta=0.5,
        descriptor_delta=0.2,
        global_improved=False,
    )

    assert controller.target() == pytest.approx(base * 1.5)
    assert controller.stats()["adaptive_no_global_progress_trials"] == 2
    assert controller.stats()["adaptive_progress_boost"] == pytest.approx(1.5)


def test_step_target_controller_progress_boost_recovers_on_unsafe_trial():
    controller = StepTargetController(
        fallback_target=0.6,
        progress_patience=1,
        progress_boost_factor=1.5,
        progress_max_boost=2.0,
        progress_duplicate_tolerance=0.75,
    )
    base = controller.target()
    controller.record_trial(escaped=True, damaged=False, energy_delta=0.5, global_improved=False)
    assert controller.target() > base

    controller.record_trial(escaped=True, damaged=False, energy_delta=0.5, global_improved=False, duplicate_rate=1.0)

    assert controller.target() == pytest.approx(base)
    assert controller.stats()["adaptive_no_global_progress_trials"] == 0
    assert controller.stats()["adaptive_progress_boost"] == pytest.approx(1.0)


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
    assert result.stats["proposal_relax_evaluator_calls"] >= 1
    assert result.stats["proposal_relax_backend_evaluations"] >= 1
    assert result.stats["proposal_relax_reporting_cache_hits"] >= 1
    assert result.stats["proposal_relax_reporting_evaluator_calls"] >= 0
    assert result.stats["proposal_relax_finalization_requests"] == result.stats["proposal_relax_count"]
    assert result.stats["proposal_relax_explicit_finalization_calls"] >= 0
    assert result.stats["proposal_relax_accepted_steps"] >= 0
    assert result.stats["proposal_relax_rejected_steps"] >= 0
    assert result.stats["proposal_relax_accepted_secants"] >= 0
    assert result.stats["proposal_relax_rejected_secants"] >= 0
    assert result.stats["proposal_relax_line_search_evaluations"] >= 0
    assert result.stats["proposal_relax_mic_branch_resets"] >= 0
    assert np.isfinite(result.stats["proposal_relax_bias_secant_curvature_sum"])
    assert (
        result.stats["proposal_relax_gradient_measure_raw_active_max_force"]
        + result.stats["proposal_relax_gradient_measure_projected_active_kkt_residual"]
        + result.stats["proposal_relax_gradient_measure_unknown"]
        == result.stats["proposal_relax_count"]
    )
    assert (
        result.stats["proposal_relax_termination_converged"]
        + result.stats["proposal_relax_termination_maxiter"]
        + result.stats["proposal_relax_termination_optimizer_stopped"]
        + result.stats["proposal_relax_termination_unconverged"]
        == result.stats["proposal_relax_count"]
    )
    assert "bias_zero_weight_fraction" in result.stats


def test_surface_walker_writes_accepted_structure_log(tmp_path):
    log_path = tmp_path / "accepted_structures.jsonl"
    accepted_dir = tmp_path / "accepted_minima"
    initial = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(
            max_trials=1,
            max_steps_per_walk=1,
            oracle_candidates=2,
            rng_seed=0,
            accepted_structures_log=str(log_path),
            accepted_structures_dir=str(accepted_dir),
        ),
        softening_enabled=False,
    )

    result = walker.run(initial)

    lines = log_path.read_text().splitlines()
    accepted_records = [record for record in result.walk_history if record.accepted_new_basin]
    assert len(lines) == len(accepted_records)
    assert len(list(accepted_dir.glob("*.xyz"))) == len(accepted_records)
    if accepted_records:
        payload = json.loads(lines[0])
        assert payload["trial_index"] == 1
        assert payload["seed_entry_id"] == accepted_records[0].seed_entry_id
        assert payload["discovered_entry_id"] == accepted_records[0].discovered_entry_id
        assert payload["energy"] == pytest.approx(accepted_records[0].energy)
        assert payload["best_energy"] == pytest.approx(result.best_energy)
        assert len(payload["descriptor"]) == 20


def test_surface_walker_can_write_all_proposal_minima(tmp_path):
    proposal_dir = tmp_path / "proposal_minima"
    initial = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(
            max_trials=1,
            max_steps_per_walk=1,
            oracle_candidates=2,
            rng_seed=0,
            write_proposal_minima=True,
            proposal_minima_dir=str(proposal_dir),
        ),
        softening_enabled=False,
    )

    result = walker.run(initial)

    proposal_files = list(proposal_dir.glob("*.xyz"))
    assert len(proposal_files) == result.stats["local_relaxations"] - 1


def test_surface_walker_can_write_relaxation_trajectories(tmp_path):
    trajectory_dir = tmp_path / "relaxation_trajectories"
    initial = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(
            max_trials=1,
            max_steps_per_walk=1,
            oracle_candidates=2,
            rng_seed=0,
            write_relaxation_trajectories=True,
            relaxation_trajectory_dir=str(trajectory_dir),
        ),
        softening_enabled=False,
    )

    walker.run(initial)

    assert list(trajectory_dir.glob("*proposal_relax.xyz"))
    assert list(trajectory_dir.glob("*true_quench.xyz"))


def test_surface_walker_warns_once_when_trajectory_context_is_missing(tmp_path, capsys):
    trajectory_dir = tmp_path / "relaxation_trajectories"
    initial = State(numbers=np.array([1]), positions=np.array([[0.2, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(
            max_trials=1,
            max_steps_per_walk=1,
            oracle_candidates=2,
            rng_seed=0,
            write_relaxation_trajectories=True,
            relaxation_trajectory_dir=str(trajectory_dir),
        ),
        softening_enabled=False,
    )

    walker._walk_candidate_from_seed(initial)
    walker._walk_candidate_from_seed(initial)

    captured = capsys.readouterr()
    assert captured.err.count("missing trajectory context") == 1


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


def test_surface_walker_reseeds_from_frontier_after_consecutive_seed_limit():
    class AlwaysFirstSelector:
        def select(self, archive, rng):
            return archive.entries[0]

        def score_entry(self, archive, entry, total_trials=None, policy=None):
            return 1.0 / (1.0 + entry.entry_id)

    from pamssw.archive import MinimaArchive

    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.01)
    first = archive.add(State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]])), -2.0, None)
    second = archive.add(State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]])), -1.9, None)
    first.is_frontier = True
    second.is_frontier = True
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, same_seed_max_consecutive=2),
        softening_enabled=False,
    )
    walker.selector = AlwaysFirstSelector()

    assert walker._select_seed_entry(archive).entry_id == first.entry_id
    assert walker._select_seed_entry(archive).entry_id == first.entry_id
    reseeded = walker._select_seed_entry(archive)

    assert reseeded.entry_id == second.entry_id
    assert walker._seed_diversity_reseeds == 1
    assert second.node_trials == 1


def test_metropolis_chain_updates_only_through_acceptance_rule():
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.01)
    current = archive.add(State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]])), -2.0, None)
    lower = archive.add(State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]])), -3.0, current.entry_id)
    higher = archive.add(State(numbers=np.array([1]), positions=np.array([[2.0, 0.0, 0.0]])), -1.0, current.entry_id)
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(seed_selection_mode="metropolis_chain", metropolis_temperature=0.01, rng_seed=0),
        softening_enabled=False,
    )

    assert walker._update_metropolis_chain(current, lower, is_new=True).entry_id == lower.entry_id
    assert walker._update_metropolis_chain(lower, higher, is_new=True).entry_id == lower.entry_id
    assert walker._update_metropolis_chain(lower, lower, is_new=False).entry_id == lower.entry_id
    stats = walker._metropolis_stats_summary(lower)

    assert stats["metropolis_downhill_accepts"] == 1
    assert stats["metropolis_uphill_rejects"] == 1
    assert stats["metropolis_duplicate_rejects"] == 1
    assert stats["metropolis_acceptance_rate"] == pytest.approx(1.0 / 3.0)


def test_metropolis_seed_selection_counts_visits_without_bandit_selector():
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.01)
    entry = archive.add(State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]])), -2.0, None)
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(seed_selection_mode="metropolis_chain"),
        softening_enabled=False,
    )

    selected = walker._select_metropolis_seed_entry(entry)

    assert selected.entry_id == entry.entry_id
    assert entry.visits == 2
    assert entry.node_trials == 1
    assert walker._same_seed_consecutive == 1


def test_uniform_archive_seed_selection_samples_entries_without_bandit_selector():
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.01)
    archive.add(State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]])), -3.0, None)
    selected_entry = archive.add(
        State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]])),
        -2.0,
        None,
    )
    archive.add(State(numbers=np.array([1]), positions=np.array([[2.0, 0.0, 0.0]])), -1.0, None)
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(seed_selection_mode="uniform_archive", rng_seed=0),
        softening_enabled=False,
    )

    class SelectMiddle:
        @staticmethod
        def integers(upper):
            assert upper == 3
            return 1

    walker.selection_rng = SelectMiddle()
    selected = walker._select_uniform_seed_entry(archive)

    assert selected.entry_id == selected_entry.entry_id
    assert selected.visits == 2
    assert selected.node_trials == 1
    assert walker._same_seed_consecutive == 1


def test_starter_selection_does_not_advance_physical_action_random_stream():
    archive = MinimaArchive(energy_tol=1e-6, rmsd_tol=0.01)
    entry = archive.add(
        State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]])),
        -2.0,
        None,
    )
    walkers = {
        mode: SurfaceWalker(
            calculator=AnalyticCalculator(DoubleWell2D()),
            config=SSWConfig(seed_selection_mode=mode, rng_seed=17),
            softening_enabled=False,
        )
        for mode in ("uniform_archive", "archive_ucb", "metropolis_chain")
    }

    walkers["uniform_archive"]._select_uniform_seed_entry(archive.clone())
    walkers["archive_ucb"]._select_seed_entry(archive.clone())
    walkers["metropolis_chain"]._select_metropolis_seed_entry(entry)

    action_draws = {
        mode: walker.rng.normal(size=12)
        for mode, walker in walkers.items()
    }
    assert np.array_equal(action_draws["uniform_archive"], action_draws["archive_ucb"])
    assert np.array_equal(action_draws["uniform_archive"], action_draws["metropolis_chain"])


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
    assert result.stats["direction_selected_momentum"] >= 0
    assert result.stats["direction_selected_bond"] >= 0
    assert result.stats["direction_selected_block_ritz"] == 0
    assert "walk_displacement_clips" in result.stats
    assert "fragment_rejections" in result.stats
    assert "direction_bond_pairs_requested" in result.stats
    assert "direction_bond_pairs_generated" in result.stats
    assert "direction_bond_candidates_valid" in result.stats


def test_uphill_control_telemetry_reports_requested_and_actual_controls():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(bias_weight_max=10.0),
        softening_enabled=False,
    )

    walker._record_uphill_control(
        requested_sigma=0.8,
        executed_sigma=0.6,
        base_weight=10.0,
        final_weight=11.5,
        true_curvature=4.0,
        inner_curvature=2.0,
    )

    stats = walker._direction_stats_summary()
    assert stats["uphill_control_steps"] == 1
    assert stats["uphill_requested_sigma_mean"] == pytest.approx(0.8)
    assert stats["uphill_executed_sigma_mean"] == pytest.approx(0.6)
    assert stats["uphill_sigma_capped_steps"] == 1
    assert stats["uphill_base_weight_mean"] == pytest.approx(10.0)
    assert stats["uphill_final_weight_mean"] == pytest.approx(11.5)
    assert stats["uphill_base_weight_at_config_max_steps"] == 1
    assert stats["uphill_final_weight_above_config_max_steps"] == 1
    assert stats["uphill_true_curvature_mean"] == pytest.approx(4.0)
    assert stats["uphill_inner_curvature_mean"] == pytest.approx(2.0)


def test_walk_termination_telemetry_distinguishes_step_cap_and_radius_clip():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(),
        softening_enabled=False,
    )

    assert (
        walker._direction_stats_summary()[
            "walk_termination_reached_step_cap"
        ]
        == 0
    )
    walker._record_walk_termination("reached_step_cap")
    walker._record_walk_termination("walk_displacement_clipped")

    stats = walker._direction_stats_summary()
    assert stats["walk_terminations"] == 2
    assert stats["walk_termination_reached_step_cap"] == 1
    assert stats["walk_termination_walk_displacement_clipped"] == 1
    assert stats["walk_termination_last_reason"] == "walk_displacement_clipped"


def test_walk_records_actual_uphill_controls_and_step_cap_termination(
    monkeypatch,
):
    state = State(
        numbers=np.array([1]),
        positions=np.array([[1.0, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            max_steps_per_walk=1,
            oracle_candidates=1,
            direction_selection_mode="energy_bounded_anchor",
            step_length_mode="per_atom_rms",
            step_rms_scope="all_atoms",
            target_step_rms=0.1,
            max_step_rms=0.2,
            target_uphill_energy=0.0025,
            proposal_relax_steps=1,
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
            curvature=2.0,
            true_curvature=2.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(
        walker,
        "_relax_proposal_task",
        lambda task, **kwargs: RelaxResult(
            task.initial_state,
            energy=0.0,
            gradient_norm=0.0,
            n_iter=0,
        ),
    )

    walker._walk_candidate_from_seed(state)

    stats = walker._direction_stats_summary()
    assert stats["uphill_control_steps"] == 1
    assert stats["uphill_requested_sigma_mean"] == pytest.approx(0.1)
    assert stats["uphill_executed_sigma_mean"] == pytest.approx(0.05)
    assert stats["uphill_sigma_capped_steps"] == 1
    assert stats["walk_termination_reached_step_cap"] == 1


def test_walk_records_radius_clip_termination(monkeypatch):
    state = State(
        numbers=np.array([1]),
        positions=np.array([[1.0, 0.0, 0.0]]),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            max_steps_per_walk=2,
            oracle_candidates=1,
            proposal_relax_steps=1,
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
            curvature=2.0,
            true_curvature=2.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(
        walker,
        "_relax_proposal_task",
        lambda task, **kwargs: RelaxResult(
            task.initial_state,
            energy=0.0,
            gradient_norm=0.0,
            n_iter=0,
        ),
    )
    monkeypatch.setattr(
        walker,
        "_clip_walk_displacement",
        lambda **kwargs: (kwargs["candidate"], True),
    )

    walker._walk_candidate_from_seed(state)

    stats = walker._direction_stats_summary()
    assert stats["walk_terminations"] == 1
    assert stats["walk_termination_walk_displacement_clipped"] == 1


def test_surface_walker_rejects_unphysical_energy_drop_before_archive(monkeypatch):
    initial = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
    )
    collapsed = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.1, 0.0]], dtype=float),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_trials=1, max_energy_drop_per_atom=5.0),
        softening_enabled=False,
    )
    relax_calls = 0

    def fake_relax_true_minimum(state, trajectory_name=None, *, quench_purpose=None):
        nonlocal relax_calls
        relax_calls += 1
        if relax_calls == 1:
            return RelaxResult(initial, energy=-10.0, gradient_norm=0.0, n_iter=0)
        return RelaxResult(collapsed, energy=-20.1, gradient_norm=0.0, n_iter=0)

    monkeypatch.setattr(walker, "relax_true_minimum", fake_relax_true_minimum)
    monkeypatch.setattr(
        walker,
        "_proposal_pool",
        lambda seed_state, archive, trial_index, step_target, **kwargs: [CandidateProposal("test", collapsed)],
    )

    result = walker.run(initial)

    assert result.best_energy == pytest.approx(-10.0)
    assert result.stats["n_minima"] == 1
    assert result.stats["energy_sanity_rejections"] == 1


def test_standard_surface_walker_respects_one_candidate_budget():
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

    assert result.stats["direction_candidate_evaluations"] == result.stats["direction_choices"]


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


def test_direction_step_metrics_reports_all_and_active_atom_scales():
    state = State(numbers=np.array([1, 1, 1]), positions=np.zeros((3, 3)))
    direction = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]) / np.sqrt(2.0)

    metrics = SurfaceWalker._direction_step_metrics(
        state,
        direction,
        sigma=0.6,
        active_threshold=1e-4,
    )

    assert metrics["direction_full_norm"] == pytest.approx(1.0)
    assert metrics["direction_per_atom_rms_all"] == pytest.approx(np.sqrt(1.0 / 3.0))
    assert metrics["direction_per_atom_rms_active"] == pytest.approx(np.sqrt(1.0 / 2.0))
    assert metrics["predicted_max_atom_displacement"] == pytest.approx(0.6 / np.sqrt(2.0))


def test_step_displacement_stats_summary_reports_recorded_step_scales():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(),
        softening_enabled=False,
    )
    state = State(numbers=np.array([1, 1]), positions=np.zeros((2, 3)))
    direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    walker._record_step_displacement_metrics(state, direction, sigma=0.2)
    walker._record_step_displacement_metrics(state, direction, sigma=0.4)

    summary = walker._direction_stats_summary()
    assert summary["step_displacement_rms_mean"] == pytest.approx(0.3 / np.sqrt(2.0))
    assert summary["step_displacement_rms_max"] == pytest.approx(0.4 / np.sqrt(2.0))
    assert summary["step_displacement_max_atom_mean"] == pytest.approx(0.3)
    assert summary["step_displacement_max_atom_max"] == pytest.approx(0.4)


def test_per_atom_rms_step_mode_gives_sparse_and_delocalized_directions_comparable_rms():
    state = State(numbers=np.array([1, 1, 1, 1]), positions=np.zeros((4, 3)))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(step_length_mode="per_atom_rms", target_step_rms=0.2, max_step_rms=0.35),
        softening_enabled=False,
    )
    sparse = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) / np.sqrt(2.0)
    delocalized = np.ones(12) / np.sqrt(12.0)

    sparse_sigma = walker._execution_step_scale(state, sparse, curvature=100.0, sigma_scale=1.0)
    delocalized_sigma = walker._execution_step_scale(state, delocalized, curvature=0.01, sigma_scale=1.0)

    sparse_rms = SurfaceWalker._direction_step_metrics(state, sparse, sparse_sigma, 1e-4)[
        "step_displacement_rms_all"
    ]
    delocalized_rms = SurfaceWalker._direction_step_metrics(state, delocalized, delocalized_sigma, 1e-4)[
        "step_displacement_rms_all"
    ]
    assert sparse_rms == pytest.approx(0.2)
    assert delocalized_rms == pytest.approx(0.2)


def test_per_atom_rms_step_mode_honors_trust_region_sigma_scale_and_cap():
    state = State(numbers=np.array([1, 1, 1, 1]), positions=np.zeros((4, 3)))
    direction = np.ones(12) / np.sqrt(12.0)
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(step_length_mode="per_atom_rms", target_step_rms=0.2, max_step_rms=0.35),
        softening_enabled=False,
    )

    shrunk_sigma = walker._execution_step_scale(state, direction, curvature=0.01, sigma_scale=0.5)
    expanded_sigma = walker._execution_step_scale(state, direction, curvature=0.01, sigma_scale=2.0)
    expanded_nominal_sigma = walker._nominal_execution_step_scale(
        state,
        direction,
        curvature=0.01,
        sigma_scale=2.0,
    )

    shrunk_rms = SurfaceWalker._direction_step_metrics(state, direction, shrunk_sigma, 1e-4)[
        "step_displacement_rms_all"
    ]
    expanded_rms = SurfaceWalker._direction_step_metrics(state, direction, expanded_sigma, 1e-4)[
        "step_displacement_rms_all"
    ]
    assert shrunk_rms == pytest.approx(0.1)
    assert expanded_rms == pytest.approx(0.35)
    assert expanded_nominal_sigma > expanded_sigma
    assert SurfaceWalker._direction_step_metrics(
        state,
        direction,
        expanded_nominal_sigma,
        1e-4,
    )["step_displacement_rms_all"] == pytest.approx(0.4)


def test_energy_bounded_anchor_uses_the_exact_all_atom_execution_step():
    state = State(
        numbers=np.array([6, 6, 6, 6]),
        positions=np.zeros((4, 3)),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            direction_selection_mode="energy_bounded_anchor",
            step_length_mode="per_atom_rms",
            step_rms_scope="all_atoms",
            target_step_rms=0.08,
            max_step_rms=0.15,
            target_uphill_energy=0.8,
        ),
        softening_enabled=False,
    )

    step_scale, energy_target = walker._energy_bounded_direction_inputs(
        state,
        sigma_scale=2.0,
        step_target=0.6,
    )

    assert step_scale == pytest.approx(0.15 * np.sqrt(4.0))
    assert energy_target == pytest.approx(0.6)


def test_energy_bounded_anchor_caps_infeasible_execution_step_from_energy():
    requested = 0.9424517674661126
    curvature = 2.2436482352245
    target = 0.8

    capped = SurfaceWalker._energy_bounded_execution_step_scale(
        requested_step_scale=requested,
        true_curvature=curvature,
        energy_target=target,
    )

    assert capped == pytest.approx(np.sqrt(2.0 * target / curvature))
    assert capped < requested
    assert 0.5 * capped * capped * curvature == pytest.approx(target)
    assert (
        SurfaceWalker._energy_bounded_execution_step_scale(
            requested_step_scale=0.5,
            true_curvature=curvature,
            energy_target=target,
        )
        == pytest.approx(0.5)
    )


def test_per_atom_rms_active_scope_uses_only_active_movable_atoms():
    state = State(
        numbers=np.array([1, 1, 1, 1]),
        positions=np.zeros((4, 3)),
        fixed_mask=np.array([False, False, False, True]),
    )
    direction = np.array(
        [
            1.0,
            0.0,
            0.0,
            -1.0,
            0.0,
            0.0,
            0.01,
            0.0,
            0.0,
            10.0,
            0.0,
            0.0,
        ],
        dtype=float,
    )
    direction = direction / np.linalg.norm(direction)
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            step_length_mode="per_atom_rms",
            target_step_rms=0.2,
            max_step_rms=0.35,
            step_rms_scope="active_atoms",
            step_active_threshold=0.1,
        ),
        softening_enabled=False,
    )

    sigma = walker._execution_step_scale(state, direction, curvature=0.01, sigma_scale=1.0)
    metrics = SurfaceWalker._direction_step_metrics(state, direction, sigma, 0.1)

    assert metrics["step_displacement_rms_active"] == pytest.approx(0.2)
    assert metrics["predicted_max_atom_displacement"] == pytest.approx(0.2)
    assert metrics["step_displacement_rms_all"] < metrics["step_displacement_rms_active"]


def test_curvature_adaptive_step_mode_preserves_scaled_step_scale():
    state = State(numbers=np.array([1, 1]), positions=np.zeros((2, 3)))
    direction = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0)
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(step_length_mode="curvature_adaptive"),
        softening_enabled=False,
    )

    sigma = walker._execution_step_scale(state, direction, curvature=2.0, sigma_scale=0.5, step_target=0.8)

    assert sigma == pytest.approx(walker._scaled_step_scale(2.0, 0.5, step_target=0.8))


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


def test_walk_rebuilds_local_softening_for_each_micro_step():
    state = State(numbers=np.array([1]), positions=np.array([[0.1, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(
            max_steps_per_walk=3,
            oracle_candidates=1,
            n_bond_pairs=0,
            proposal_relax_steps=1,
            proposal_fmax=100.0,
            proposal_trust_radius=None,
            walk_trust_radius=100.0,
        ),
        softening_enabled=True,
    )
    build_positions = []

    def record_build(current_state, direction=None):
        build_positions.append(current_state.positions.copy())
        return None

    walker._build_softening = record_build

    walker._walk_candidate_from_seed(state)

    assert len(build_positions) == 3
    assert any(not np.allclose(build_positions[0], positions) for positions in build_positions[1:])


def test_walk_early_stop_hook_stops_after_already_paid_true_energy_check():
    observed = []

    class StopAfterFirstWalker(SurfaceWalker):
        def _walk_early_stop_reason(
            self,
            *,
            step_index,
            walk_reference,
            current,
            true_energy,
        ):
            observed.append(
                {
                    "step": int(step_index) + 1,
                    "walk_reference": walk_reference,
                    "current": current,
                    "true_energy": float(true_energy),
                }
            )
            return "unit_test_stop"

    state = State(
        numbers=np.array([1]),
        positions=np.array([[0.1, 0.0, 0.0]]),
    )
    walker = StopAfterFirstWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(
            max_steps_per_walk=3,
            oracle_candidates=1,
            n_bond_pairs=0,
            proposal_relax_steps=1,
            proposal_fmax=100.0,
            proposal_trust_radius=None,
            walk_trust_radius=100.0,
        ),
        softening_enabled=True,
    )

    walker._walk_candidate_from_seed(state)

    assert len(observed) == 1
    assert observed[0]["step"] == 1
    assert walker._walk_termination_last_reason == "unit_test_stop"
    assert walker.calculator.snapshot().count(
        EvaluationPurpose.ESCAPE_TRUE_PES_CHECK
    ) == 2


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
