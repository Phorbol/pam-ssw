import json

import numpy as np
import pytest

from pamssw import LSSSWConfig, SSWConfig
from pamssw.accounting import BudgetExceeded
from pamssw.archive import MinimaArchive
from pamssw.bias import GaussianBiasTerm
from pamssw.calculators import AnalyticCalculator, EnergyResult
from pamssw.potentials import DoubleWell2D
from pamssw.result import RelaxOutcomeClass, RelaxResult
from pamssw.reference_dimer import ReferenceDimerResult, ReferenceDimerRotator, sample_global_mode, sample_mixed_mode
from pamssw.state import State
from pamssw.walker import (
    CandidateDirectionGenerator,
    DirectionCandidateKind,
    DirectionCandidate,
    DirectionScorer,
    DirectionRecord,
    DirectionTypeMemory,
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
    TrustRegionUpdate,
)


class Quadratic:
    def energy_gradient(self, flat_positions, state):
        gradient = np.asarray(flat_positions, dtype=float).copy()
        energy = 0.5 * float(gradient @ gradient)
        return energy, gradient


def test_direction_candidate_kind_includes_reference_dimer():
    assert DirectionCandidateKind.REFERENCE_DIMER.value == "reference_dimer"


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


def test_direct_qp_scalar_step_matches_closed_form_unconstrained():
    gradient = np.array([2.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])

    step = SurfaceWalker._solve_direct_qp_scalar_step(
        gradient=gradient,
        direction=direction,
        sigma=1.0,
        gamma=2.0,
        kappa=8.0,
        trust_radius=10.0,
    )

    np.testing.assert_allclose(step, np.array([0.6, 0.0, 0.0]))


def test_direct_qp_scalar_step_clips_to_trust_radius():
    gradient = np.array([-10.0, 0.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])

    step = SurfaceWalker._solve_direct_qp_scalar_step(
        gradient=gradient,
        direction=direction,
        sigma=1.0,
        gamma=1.0,
        kappa=1.0,
        trust_radius=0.25,
    )

    assert np.linalg.norm(step) == pytest.approx(0.25)
    np.testing.assert_allclose(step, np.array([0.25, 0.0, 0.0]))


def test_direct_qp_scalar_step_normalizes_direction():
    gradient = np.array([0.0, 0.0, 0.0])
    direction = np.array([2.0, 0.0, 0.0])

    step = SurfaceWalker._solve_direct_qp_scalar_step(
        gradient=gradient,
        direction=direction,
        sigma=1.0,
        gamma=1.0,
        kappa=3.0,
        trust_radius=10.0,
    )

    np.testing.assert_allclose(step, np.array([0.75, 0.0, 0.0]))


def test_direct_qp_rank1_step_uses_directional_and_floor_curvatures():
    gradient = np.array([10.0, 4.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])

    step = SurfaceWalker._solve_direct_qp_rank1_step(
        gradient=gradient,
        direction=direction,
        sigma=1.0,
        gamma_floor=2.0,
        directional_curvature=18.0,
        kappa=6.0,
        trust_radius=10.0,
    )

    np.testing.assert_allclose(step, np.array([-1.0 / 6.0, -0.5, 0.0]))


def test_direct_qp_rank1_step_clips_to_trust_radius():
    gradient = np.array([-10.0, -10.0, 0.0])
    direction = np.array([1.0, 0.0, 0.0])

    step = SurfaceWalker._solve_direct_qp_rank1_step(
        gradient=gradient,
        direction=direction,
        sigma=1.0,
        gamma_floor=1.0,
        directional_curvature=1.0,
        kappa=1.0,
        trust_radius=0.25,
    )

    assert np.linalg.norm(step) == pytest.approx(0.25)


def test_direct_qp_stats_summary_records_step_quality():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(proposal_step_mode="direct_qp"),
        softening_enabled=False,
    )

    walker._record_direct_qp_result(
        step_norm=0.5,
        progress=0.4,
        target_error=0.1,
        predicted_delta=0.2,
        true_delta=0.3,
        model_error=0.5,
        gamma=1.0,
        kappa=4.0,
        action="expand",
        rejected=False,
    )

    summary = walker._direct_qp_stats_summary()

    assert summary["direct_qp_steps"] == 1
    assert summary["direct_qp_rejected"] == 0
    assert summary["direct_qp_mean_step_norm"] == pytest.approx(0.5)
    assert summary["direct_qp_mean_progress"] == pytest.approx(0.4)
    assert summary["direct_qp_mean_target_error"] == pytest.approx(0.1)
    assert summary["direct_qp_mean_model_error"] == pytest.approx(0.5)
    assert summary["direct_qp_trust_expand_steps"] == 1
    assert summary["direct_qp_gamma_mean"] == pytest.approx(1.0)
    assert summary["direct_qp_kappa_mean"] == pytest.approx(4.0)


def test_direct_qp_walk_step_moves_without_calling_proposal_relax(monkeypatch):
    class RaisingRelaxer:
        def __init__(self, evaluator, optimizer):
            pass

        def relax(self, *args, **kwargs):
            raise AssertionError("proposal relax should not be called in direct_qp mode")

    monkeypatch.setattr("pamssw.walker.Relaxer", RaisingRelaxer)

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            proposal_step_mode="direct_qp",
            max_steps_per_walk=1,
            proposal_trust_radius=0.5,
            direct_qp_gamma=1.0,
            direct_qp_kappa=4.0,
        ),
        softening_enabled=False,
    )
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )

    candidate = walker._walk_candidate_from_seed(state)

    assert candidate.positions[0, 0] > state.positions[0, 0]
    assert walker._direct_qp_steps == 1
    assert walker._relax_stats["proposal_relax"]["count"] == 0
    assert walker._bias_steps == 0


def test_direct_qp_walk_step_can_run_bare_pes_micro_corrector(monkeypatch):
    calls = []

    class RecordingRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator
            self.optimizer = optimizer

        def relax(self, state, fmax, maxiter, coordinate_trust_radius=None, **kwargs):
            calls.append(
                {
                    "optimizer": self.optimizer,
                    "fmax": fmax,
                    "maxiter": maxiter,
                    "coordinate_trust_radius": coordinate_trust_radius,
                    "energy_before": self.evaluator(state.flatten_positions(), state)[0],
                }
            )
            corrected = State(
                numbers=state.numbers.copy(),
                positions=state.positions * 0.5,
                cell=None if state.cell is None else state.cell.copy(),
                pbc=state.pbc,
                fixed_mask=state.fixed_mask.copy(),
                metadata=state.metadata.copy(),
            )
            return RelaxResult(
                state=corrected,
                energy=0.5,
                gradient_norm=0.1,
                n_iter=maxiter,
                displacement_rms=0.25,
                displacement_max=0.5,
            )

    monkeypatch.setattr("pamssw.walker.Relaxer", RecordingRelaxer)

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            proposal_step_mode="direct_qp",
            max_steps_per_walk=1,
            proposal_trust_radius=10.0,
            direct_qp_gamma=1.0,
            direct_qp_kappa=4.0,
            direct_qp_micro_steps=3,
            direct_qp_micro_optimizer="scipy-lbfgsb",
            direct_qp_micro_fmax=0.2,
            direct_qp_micro_trust_radius=0.3,
            target_step_rms=0.15,
            max_step_rms=0.15,
        ),
        softening_enabled=False,
    )
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )

    candidate = walker._walk_candidate_from_seed(state)
    summary = walker._direct_qp_stats_summary()

    assert len(calls) == 1
    assert calls[0]["optimizer"] == "scipy-lbfgsb"
    assert calls[0]["maxiter"] == 3
    assert calls[0]["fmax"] == pytest.approx(0.2)
    assert calls[0]["coordinate_trust_radius"] == pytest.approx(0.3)
    assert calls[0]["energy_before"] > 0.0
    assert candidate.positions[0, 0] == pytest.approx((4.0 * 0.15 / 5.0) * 0.5)
    assert summary["direct_qp_micro_count"] == 1
    assert summary["direct_qp_micro_mean_iterations"] == pytest.approx(3.0)
    assert summary["direct_qp_micro_displacement_rms_mean"] == pytest.approx(0.25)


def test_direct_qp_micro_steps_can_follow_model_error():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            direct_qp_micro_steps=2,
            direct_qp_micro_max_steps=20,
            direct_qp_micro_mode="adaptive_model_error",
            direct_qp_micro_model_error_threshold=2.0,
            direct_qp_micro_model_error_high=10.0,
        ),
        softening_enabled=False,
    )

    assert walker._direct_qp_micro_steps_for_model_error(0.5) == 2
    assert walker._direct_qp_micro_steps_for_model_error(2.0) == 2
    assert walker._direct_qp_micro_steps_for_model_error(6.0) == 11
    assert walker._direct_qp_micro_steps_for_model_error(12.0) == 20


def test_direct_qp_micro_steps_model_error_gate_can_skip_low_error():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            direct_qp_micro_steps=2,
            direct_qp_micro_max_steps=18,
            direct_qp_micro_mode="model_error",
            direct_qp_micro_model_error_threshold=2.0,
        ),
        softening_enabled=False,
    )

    assert walker._direct_qp_micro_steps_for_model_error(1.9) == 0
    assert walker._direct_qp_micro_steps_for_model_error(2.1) == 18


def test_direct_qp_micro_corrector_uses_adaptive_model_error_steps(monkeypatch):
    calls = []

    class RecordingRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator
            self.optimizer = optimizer

        def relax(self, state, *, fmax, maxiter, coordinate_trust_radius):
            calls.append(maxiter)
            return RelaxResult(
                state=state,
                energy=0.0,
                gradient_norm=0.1,
                n_iter=maxiter,
                displacement_rms=0.0,
                displacement_max=0.0,
            )

    monkeypatch.setattr("pamssw.walker.Relaxer", RecordingRelaxer)
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            direct_qp_micro_steps=2,
            direct_qp_micro_max_steps=20,
            direct_qp_micro_mode="adaptive_model_error",
            direct_qp_micro_model_error_threshold=2.0,
            direct_qp_micro_model_error_high=10.0,
        ),
        softening_enabled=False,
    )

    result = walker._direct_qp_micro_correct(state, model_error=6.0)

    assert result is not None
    assert calls == [11]


def test_direct_qp_walk_step_uses_true_directional_curvature_as_scalar_gamma(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            proposal_step_mode="direct_qp",
            max_steps_per_walk=1,
            proposal_trust_radius=10.0,
            direct_qp_gamma=1.0,
            direct_qp_kappa=4.0,
            target_step_rms=0.15,
            max_step_rms=0.15,
        ),
        softening_enabled=False,
    )
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda state, direction: 12.0)

    candidate = walker._walk_candidate_from_seed(state)

    assert walker._direct_qp_gamma_sum == pytest.approx(12.0)
    assert candidate.positions[0, 0] == pytest.approx((4.0 * 0.15) / (12.0 + 4.0))


def test_direct_qp_walk_step_can_scale_kappa_from_directional_curvature(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            proposal_step_mode="direct_qp",
            max_steps_per_walk=1,
            proposal_trust_radius=10.0,
            direct_qp_gamma=1.0,
            direct_qp_kappa=4.0,
            direct_qp_kappa_mode="adaptive_curvature",
            direct_qp_kappa_curvature_ratio=15.0,
            target_step_rms=0.15,
            max_step_rms=0.15,
        ),
        softening_enabled=False,
    )
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda state, direction: 12.0)

    candidate = walker._walk_candidate_from_seed(state)

    assert walker._direct_qp_gamma_sum == pytest.approx(12.0)
    assert walker._direct_qp_kappa_sum == pytest.approx(180.0)
    assert candidate.positions[0, 0] == pytest.approx((180.0 * 0.15) / (12.0 + 180.0))


def test_direct_qp_walk_step_uses_rank1_directional_hessian(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            proposal_step_mode="direct_qp",
            direct_qp_hessian="rank1",
            max_steps_per_walk=1,
            proposal_trust_radius=10.0,
            direct_qp_gamma=1.0,
            direct_qp_kappa=4.0,
            target_step_rms=0.15,
            max_step_rms=0.15,
        ),
        softening_enabled=False,
    )
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda state, direction: 12.0)

    candidate = walker._walk_candidate_from_seed(state)

    assert walker._direct_qp_gamma_sum == pytest.approx(1.0)
    assert candidate.positions[0, 0] == pytest.approx((4.0 * 0.15) / (12.0 + 4.0))


def test_direct_qp_rank1_floor_can_use_previous_curvature_history(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            proposal_step_mode="direct_qp",
            direct_qp_hessian="rank1",
            direct_qp_gamma=1.0,
            direct_qp_gamma_mode="curvature_history",
            direct_qp_gamma_history_quantile=0.5,
            direct_qp_gamma_history_min_samples=3,
            max_steps_per_walk=1,
            proposal_trust_radius=10.0,
            direct_qp_kappa=4.0,
            target_step_rms=0.15,
            max_step_rms=0.15,
        ),
        softening_enabled=False,
    )
    walker._direct_qp_curvature_history.extend([4.0, 8.0, 12.0])
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda state, direction: 40.0)

    candidate = walker._walk_candidate_from_seed(state)

    assert walker._direct_qp_gamma_sum == pytest.approx(8.0)
    assert candidate.positions[0, 0] == pytest.approx((4.0 * 0.15) / (40.0 + 4.0))
    assert walker._direct_qp_curvature_history[-1] == pytest.approx(40.0)


def test_direct_qp_rank1_floor_uses_history_only_after_model_error_gate():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            proposal_step_mode="direct_qp",
            direct_qp_hessian="rank1",
            direct_qp_gamma=1.0,
            direct_qp_gamma_mode="model_error_gated_history",
            direct_qp_gamma_history_quantile=0.5,
            direct_qp_gamma_history_min_samples=3,
            direct_qp_gamma_model_error_threshold=2.0,
            direct_qp_gamma_model_error_streak=2,
        ),
        softening_enabled=False,
    )
    walker._direct_qp_curvature_history.extend([4.0, 8.0, 12.0])

    assert walker._direct_qp_rank1_gamma_floor() == pytest.approx(1.0)

    walker._record_direct_qp_result(
        step_norm=1.0,
        progress=0.5,
        target_error=0.1,
        predicted_delta=0.0,
        true_delta=0.0,
        model_error=3.0,
        gamma=1.0,
        kappa=4.0,
        action="expand",
        rejected=False,
    )
    assert walker._direct_qp_rank1_gamma_floor() == pytest.approx(1.0)

    walker._record_direct_qp_result(
        step_norm=1.0,
        progress=0.5,
        target_error=0.1,
        predicted_delta=0.0,
        true_delta=0.0,
        model_error=3.0,
        gamma=1.0,
        kappa=4.0,
        action="expand",
        rejected=False,
    )
    assert walker._direct_qp_rank1_gamma_floor() == pytest.approx(8.0)

    walker._record_direct_qp_result(
        step_norm=1.0,
        progress=0.5,
        target_error=0.1,
        predicted_delta=0.0,
        true_delta=0.0,
        model_error=0.5,
        gamma=1.0,
        kappa=4.0,
        action="expand",
        rejected=False,
    )
    assert walker._direct_qp_rank1_gamma_floor() == pytest.approx(1.0)


def test_direct_qp_walk_step_caps_adaptive_kappa(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            proposal_step_mode="direct_qp",
            max_steps_per_walk=1,
            proposal_trust_radius=10.0,
            direct_qp_gamma=1.0,
            direct_qp_kappa=4.0,
            direct_qp_kappa_mode="adaptive_curvature",
            direct_qp_kappa_curvature_ratio=15.0,
            direct_qp_kappa_max=240.0,
            target_step_rms=0.15,
            max_step_rms=0.15,
        ),
        softening_enabled=False,
    )
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda state, direction: 40.0)

    candidate = walker._walk_candidate_from_seed(state)

    assert walker._direct_qp_gamma_sum == pytest.approx(40.0)
    assert walker._direct_qp_kappa_sum == pytest.approx(240.0)
    assert candidate.positions[0, 0] == pytest.approx((240.0 * 0.15) / (40.0 + 240.0))


def test_direct_qp_walk_step_rejects_invalid_geometry(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(proposal_step_mode="direct_qp", max_steps_per_walk=1),
        softening_enabled=False,
    )
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )
    monkeypatch.setattr(GeometryValidator, "is_valid_state", lambda self, candidate: False)

    candidate = walker._walk_candidate_from_seed(state)

    np.testing.assert_allclose(candidate.positions, state.positions)
    assert walker._direct_qp_rejected == 1


def test_bias_relax_default_still_calls_proposal_relax(monkeypatch):
    calls = {"count": 0}

    class RecordingRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator
            self.optimizer = optimizer

        def relax(self, state, fmax, maxiter, coordinate_trust_radius=None, trajectory_callback=None, trajectory_stride=1):
            calls["count"] += 1
            energy, gradient = self.evaluator(state.flatten_positions(), state)
            return RelaxResult(
                state=state,
                energy=energy,
                gradient_norm=float(np.linalg.norm(gradient)),
                n_iter=1,
            )

        @staticmethod
        def classify_outcome(**kwargs):
            return RelaxOutcomeClass.USEFUL_PROGRESS

    monkeypatch.setattr("pamssw.walker.Relaxer", RecordingRelaxer)

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_steps_per_walk=1),
        softening_enabled=False,
    )
    monkeypatch.setattr(
        walker.oracle,
        "choose_direction",
        lambda *args, **kwargs: DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        ),
    )

    walker._walk_candidate_from_seed(state)

    assert calls["count"] == 1
    assert walker._bias_steps == 1


def test_walk_early_exit_default_stops_when_real_energy_drops_below_seed(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(max_steps_per_walk=3, proposal_relax_steps=0, walk_trust_radius=10.0),
        softening_enabled=False,
    )
    choose_calls = {"count": 0}

    walker.oracle.generator.generate_initial_direction = lambda *args, **kwargs: np.array([-1.0, 0.0, 0.0])

    def choose_direction(*args, **kwargs):
        choose_calls["count"] += 1
        return DirectionChoice(
            direction=np.array([-1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        )

    class LoweringRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator

        @staticmethod
        def classify_outcome(**kwargs):
            return RelaxOutcomeClass.USEFUL_PROGRESS

        def relax(self, state, **kwargs):
            lowered = State(numbers=state.numbers, positions=np.array([[0.0, 0.0, 0.0]]))
            return RelaxResult(state=lowered, energy=0.0, gradient_norm=0.0, n_iter=0)

    import pamssw.walker as walker_module

    monkeypatch.setattr(walker_module, "Relaxer", LoweringRelaxer)
    monkeypatch.setattr(walker, "_build_softening", lambda *args, **kwargs: None)
    monkeypatch.setattr(walker, "_execution_step_scale", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(walker, "_bias_weight", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(walker.oracle, "choose_direction", choose_direction)
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", lambda *args, **kwargs: 1.0)

    result = walker._walk_candidate_from_seed(state)

    assert choose_calls["count"] == 1
    assert walker._walk_early_stops == 1
    np.testing.assert_allclose(result.positions, [[0.0, 0.0, 0.0]])


def test_walk_early_exit_can_be_disabled(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(
            max_steps_per_walk=3,
            proposal_relax_steps=0,
            walk_trust_radius=10.0,
            early_exit_enabled=False,
        ),
        softening_enabled=False,
    )
    choose_calls = {"count": 0}

    walker.oracle.generator.generate_initial_direction = lambda *args, **kwargs: np.array([-1.0, 0.0, 0.0])

    def choose_direction(*args, **kwargs):
        choose_calls["count"] += 1
        return DirectionChoice(
            direction=np.array([-1.0, 0.0, 0.0]),
            curvature=1.0,
            kind=DirectionCandidateKind.RANDOM,
            candidate_count=1,
        )

    class LoweringRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator

        @staticmethod
        def classify_outcome(**kwargs):
            return RelaxOutcomeClass.USEFUL_PROGRESS

        def relax(self, state, **kwargs):
            lowered = State(numbers=state.numbers, positions=np.array([[0.0, 0.0, 0.0]]))
            return RelaxResult(state=lowered, energy=0.0, gradient_norm=0.0, n_iter=0)

    import pamssw.walker as walker_module

    monkeypatch.setattr(walker_module, "Relaxer", LoweringRelaxer)
    monkeypatch.setattr(walker, "_build_softening", lambda *args, **kwargs: None)
    monkeypatch.setattr(walker, "_execution_step_scale", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(walker, "_bias_weight", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(walker.oracle, "choose_direction", choose_direction)
    monkeypatch.setattr(walker, "_true_directional_curvature", lambda *args, **kwargs: 1.0)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", lambda *args, **kwargs: 1.0)

    walker._walk_candidate_from_seed(state)

    assert choose_calls["count"] == 3
    assert walker._walk_early_stops == 0


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


def test_reference_dimer_sample_mixed_mode_returns_normalized_direction():
    rng = np.random.default_rng(7)
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=float,
    )

    direction, info = sample_mixed_mode(
        positions,
        rng=rng,
        lam=0.5,
        min_distance=3.0,
    )

    assert direction.shape == positions.shape
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    assert info["lambda"] == pytest.approx(0.5)
    assert info["pair"] in {(0, 1), (0, 2), (1, 2)}


def test_reference_dimer_sample_mixed_mode_zero_norm_falls_back_to_global_mode(monkeypatch):
    global_mode = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    local_mode = -global_mode

    monkeypatch.setattr(
        "pamssw.reference_dimer.sample_global_mode",
        lambda positions, masses=None, T_rand=300.0, rng=None: global_mode.copy(),
    )
    monkeypatch.setattr(
        "pamssw.reference_dimer.sample_local_bond_mode",
        lambda positions, min_distance=3.0, cell=None, pbc=None, rng=None: (local_mode.copy(), (0, 1)),
    )

    direction, info = sample_mixed_mode(np.zeros((2, 3), dtype=float), rng=np.random.default_rng(11), lam=1.0)

    np.testing.assert_allclose(direction, global_mode)
    assert info["lambda"] == pytest.approx(1.0)
    assert info["pair"] == (0, 1)


def test_reference_dimer_mode_samplers_accept_reference_t_rand_keyword():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
        ],
        dtype=float,
    )

    global_direction = sample_global_mode(positions, rng=np.random.default_rng(3), T_rand=300.0)
    mixed_direction, _ = sample_mixed_mode(positions, rng=np.random.default_rng(5), lam=0.5, T_rand=300.0)

    assert global_direction.shape == positions.shape
    assert mixed_direction.shape == positions.shape
    assert np.linalg.norm(global_direction) == pytest.approx(1.0)
    assert np.linalg.norm(mixed_direction) == pytest.approx(1.0)


def test_reference_dimer_sample_mixed_mode_rejects_nonfinite_positions():
    positions = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=float)

    with pytest.raises(ValueError, match="positions"):
        sample_mixed_mode(positions, rng=np.random.default_rng(13), lam=0.5)


def test_reference_dimer_rotator_returns_direction_and_curvature():
    class HarmonicSurface:
        def __init__(self):
            self.calls = 0
            self.hessian = np.diag([2.0, 6.0, 10.0, 2.0, 6.0, 10.0])

        def evaluate(self, positions):
            assert positions.shape == (2, 3)
            self.calls += 1
            flat_positions = positions.reshape(-1)
            gradient = self.hessian @ flat_positions
            energy = 0.5 * float(flat_positions @ gradient)
            force = -gradient.reshape(2, 3)
            return energy, force

    surface = HarmonicSurface()
    rotator = ReferenceDimerRotator(
        delta=1e-3,
        bias_strength=0.01,
        max_steps=4,
        rotation_tol=1e-12,
        angular_step=0.05,
    )
    positions = np.zeros((2, 3), dtype=float)
    initial = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
    initial /= np.linalg.norm(initial)

    result = rotator.rotate(positions, initial, surface.evaluate)

    assert result.direction.shape == positions.shape
    assert np.linalg.norm(result.direction) == pytest.approx(1.0)
    assert np.isfinite(result.curvature)
    assert result.curvature < 0.0
    assert result.rotations >= 1
    assert surface.calls >= 2


def test_reference_dimer_rotator_reports_true_and_biased_curvature_separately():
    class FlatSurface:
        def evaluate(self, positions):
            return 0.0, np.zeros_like(positions)

    rotator = ReferenceDimerRotator(
        delta=0.005,
        bias_strength=500.0,
        max_steps=1,
        rotation_tol=1e9,
        angular_step=0.05,
    )
    positions = np.zeros((2, 3), dtype=float)
    initial = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
    initial /= np.linalg.norm(initial)

    result = rotator.rotate(positions, initial, FlatSurface().evaluate)

    assert result.curvature_true == pytest.approx(0.0)
    assert result.curvature_biased == pytest.approx(2000.0)
    assert result.curvature == pytest.approx(result.curvature_biased)


def test_reference_dimer_direction_engine_bypasses_scored_pool(monkeypatch):
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
    )
    config = LSSSWConfig(
        direction_engine="reference_dimer",
        max_steps_per_walk=1,
        proposal_step_mode="bias_relax",
        reference_dimer_max_steps=1,
    )
    walker = SurfaceWalker(calculator=AnalyticCalculator(Quadratic()), config=config, softening_enabled=False)

    def forbidden_choose_direction(*args, **kwargs):
        raise AssertionError("scored pool should not be called")

    monkeypatch.setattr(walker.oracle, "choose_direction", forbidden_choose_direction)
    choice = walker._choose_walk_direction(
        current=state,
        proposal=ProposalPotential(walker.calculator),
        scoring_proposal=ProposalPotential(walker.calculator),
        previous_direction=None,
        anchor_direction=np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0]) / np.sqrt(2.0),
        archive=None,
        step_target=None,
        sigma_scale=1.0,
        previous_relax_outcome=None,
        trial_index=0,
        proposal_index=0,
        seed_entry_id=0,
        step_index=0,
        plateau_evolution_active=False,
    )

    assert choice.kind == DirectionCandidateKind.REFERENCE_DIMER
    assert np.linalg.norm(choice.direction) == pytest.approx(1.0)
    assert np.isfinite(choice.curvature)


def test_reference_dimer_direction_masks_fixed_atoms():
    state = State(
        numbers=np.array([6, 6, 6]),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 1.4, 0.0],
            ],
            dtype=float,
        ),
        fixed_mask=np.array([True, False, False]),
    )
    config = LSSSWConfig(
        direction_engine="reference_dimer",
        max_steps_per_walk=1,
        reference_dimer_max_steps=2,
    )
    walker = SurfaceWalker(calculator=AnalyticCalculator(Quadratic()), config=config, softening_enabled=False)

    choice = walker._choose_reference_dimer_direction(state)

    direction = choice.direction.reshape(-1, 3)
    np.testing.assert_allclose(direction[0], np.zeros(3), atol=1e-12)
    assert np.linalg.norm(choice.direction) == pytest.approx(1.0)


def test_reference_dimer_force_callback_preserves_fixed_positions_and_forces(monkeypatch):
    state = State(
        numbers=np.array([6, 6, 6]),
        positions=np.array(
            [
                [10.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [0.0, 1.4, 0.0],
            ],
            dtype=float,
        ),
        fixed_mask=np.array([True, False, False]),
    )
    evaluated_positions: list[np.ndarray] = []

    class RecordingCalculator:
        def evaluate(self, trial_state):
            evaluated_positions.append(trial_state.positions.copy())
            gradient = np.array(
                [
                    [100.0, 200.0, 300.0],
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ],
                dtype=float,
            )
            return EnergyResult(energy=0.0, gradient=gradient)

        def evaluate_flat(self, flat_positions, template):
            trial_state = template.with_flat_positions(flat_positions)
            result = self.evaluate(trial_state)
            return result.energy, result.gradient.reshape(-1)

    class FakeReferenceDimerRotator:
        def __init__(self, *args, **kwargs):
            pass

        def rotate(
            self,
            positions,
            initial_direction,
            evaluate_forces,
            *,
            initial_forces=None,
            lambda_value=0.0,
            local_pair=(0, 0),
        ):
            np.testing.assert_allclose(initial_direction[0], np.zeros(3), atol=1e-12)
            moved_positions = np.asarray(positions, dtype=float).copy()
            moved_positions[0] = np.array([99.0, 99.0, 99.0])
            moved_positions[1] += np.array([0.1, 0.2, 0.3])

            _, forces = evaluate_forces(moved_positions)

            np.testing.assert_allclose(forces[0], np.zeros(3), atol=1e-12)
            np.testing.assert_allclose(evaluated_positions[-1][0], state.positions[0], atol=1e-12)
            direction = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=float,
            )
            return ReferenceDimerResult(
                direction=direction,
                curvature=-1.0,
                rotations=1,
                dot_initial=1.0,
                converged=True,
                lambda_value=lambda_value,
                local_pair=local_pair,
            )

    monkeypatch.setattr("pamssw.walker.ReferenceDimerRotator", FakeReferenceDimerRotator)
    walker = SurfaceWalker(
        calculator=RecordingCalculator(),
        config=LSSSWConfig(direction_engine="reference_dimer", reference_dimer_max_steps=1),
        softening_enabled=False,
    )

    choice = walker._choose_reference_dimer_direction(state)

    assert evaluated_positions
    assert choice.kind == DirectionCandidateKind.REFERENCE_DIMER
    np.testing.assert_allclose(choice.direction.reshape(-1, 3)[0], np.zeros(3), atol=1e-12)


def test_reference_dimer_walk_reuses_returned_curvature(monkeypatch):
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
    )
    config = LSSSWConfig(
        direction_engine="reference_dimer",
        proposal_step_mode="bias_relax",
        max_steps_per_walk=1,
        reference_dimer_max_steps=1,
    )
    walker = SurfaceWalker(calculator=AnalyticCalculator(Quadratic()), config=config, softening_enabled=False)

    monkeypatch.setattr(
        walker,
        "_true_directional_curvature",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("true curvature probe should not be called")),
    )
    monkeypatch.setattr(
        walker.oracle,
        "_directional_curvature",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("inner curvature probe should not be called")),
    )

    candidate = walker._walk_candidate_from_seed(state)

    assert isinstance(candidate, State)


def test_reference_dimer_direct_qp_uses_choice_curvature_without_true_hvp(monkeypatch):
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
    )
    config = LSSSWConfig(
        direction_engine="reference_dimer",
        proposal_step_mode="direct_qp",
        direct_qp_hessian="rank1",
        direct_qp_kappa=240.0,
        max_steps_per_walk=1,
        reference_dimer_max_steps=1,
    )
    walker = SurfaceWalker(calculator=AnalyticCalculator(Quadratic()), config=config, softening_enabled=False)

    def fail_true_hvp(*args, **kwargs):
        raise AssertionError("reference dimer curvature should be reused for Direct-QP")

    monkeypatch.setattr(walker, "_true_directional_curvature", fail_true_hvp)
    monkeypatch.setattr(walker.oracle, "_directional_curvature", fail_true_hvp)

    result = walker._walk_candidate_from_seed(state)

    assert isinstance(result, State)
    assert walker._direct_qp_stats_summary()["direct_qp_steps"] == 1
    assert walker._reference_dimer_steps == 1


def test_reference_dimer_walk_records_selection_and_stats():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
    )
    config = LSSSWConfig(
        direction_engine="reference_dimer",
        max_steps_per_walk=1,
        reference_dimer_max_steps=1,
    )
    walker = SurfaceWalker(calculator=AnalyticCalculator(Quadratic()), config=config, softening_enabled=False)

    walker._walk_candidate_from_seed(state)
    summary = walker._reference_dimer_stats_summary()

    assert walker._reference_dimer_steps >= 1
    assert walker._direction_selected[DirectionCandidateKind.REFERENCE_DIMER] >= 1
    assert walker._direction_stats_summary()["direction_selected_reference_dimer"] >= 1
    assert summary["reference_dimer_steps"] >= 1
    assert np.isfinite(summary["reference_dimer_mean_rotations"])
    assert np.isfinite(summary["reference_dimer_converged_fraction"])
    assert np.isfinite(summary["reference_dimer_mean_curvature"])
    assert np.isfinite(summary["reference_dimer_mean_abs_dot_initial"])


def test_reference_dimer_walk_reuses_one_initial_mode_across_climb_steps(monkeypatch):
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float),
    )
    config = LSSSWConfig(
        direction_engine="reference_dimer",
        proposal_step_mode="bias_relax",
        max_steps_per_walk=3,
        proposal_relax_steps=0,
        walk_trust_radius=10.0,
        reference_dimer_max_steps=1,
    )
    walker = SurfaceWalker(calculator=AnalyticCalculator(Quadratic()), config=config, softening_enabled=False)

    sampled_initial = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=float)
    sampled_initial /= np.linalg.norm(sampled_initial)
    sample_calls = {"count": 0}
    rotate_initials: list[np.ndarray] = []

    def fake_sample_mixed_mode(*args, **kwargs):
        sample_calls["count"] += 1
        return sampled_initial.copy(), {"pair": (0, 1), "lambda": kwargs.get("lam", 0.0)}

    class FakeReferenceDimerRotator:
        def __init__(self, *args, **kwargs):
            pass

        def rotate(self, positions, initial_direction, evaluate_forces, *, lambda_value=0.0, local_pair=None):
            rotate_initials.append(np.asarray(initial_direction, dtype=float).copy())
            return ReferenceDimerResult(
                direction=np.asarray(initial_direction, dtype=float),
                curvature=1.0,
                rotations=1,
                dot_initial=1.0,
                converged=True,
                lambda_value=lambda_value,
                local_pair=local_pair,
            )

    class NoRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator

        @staticmethod
        def classify_outcome(**kwargs):
            return RelaxOutcomeClass.USEFUL_PROGRESS

        def relax(self, state, **kwargs):
            energy, gradient = self.evaluator(state.flatten_positions(), state)
            return RelaxResult(
                state=state,
                energy=energy,
                gradient_norm=float(np.linalg.norm(gradient)),
                n_iter=0,
            )

    import pamssw.walker as walker_module

    monkeypatch.setattr(walker_module, "sample_mixed_mode", fake_sample_mixed_mode)
    monkeypatch.setattr(walker_module, "ReferenceDimerRotator", FakeReferenceDimerRotator)
    monkeypatch.setattr(walker_module, "Relaxer", NoRelaxer)
    monkeypatch.setattr(walker, "_build_softening", lambda *args, **kwargs: None)
    monkeypatch.setattr(walker, "_execution_step_scale", lambda *args, **kwargs: 0.05)
    monkeypatch.setattr(walker, "_bias_weight", lambda *args, **kwargs: 0.0)

    walker._walk_candidate_from_seed(state)

    assert walker._reference_dimer_steps == 3
    assert sample_calls["count"] == 1
    assert len(rotate_initials) == 3
    for initial in rotate_initials:
        np.testing.assert_allclose(initial, sampled_initial)


def test_reference_dimer_rotator_rejects_nonfinite_positions():
    rotator = ReferenceDimerRotator()
    positions = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], dtype=float)
    initial = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=float)
    initial /= np.linalg.norm(initial)

    with pytest.raises(ValueError, match="positions"):
        rotator.rotate(positions, initial, lambda matrix: (0.0, np.zeros((2, 3), dtype=float)))


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
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), np.random.default_rng(0), candidates=1)
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
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), np.random.default_rng(0), candidates=1)
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
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), np.random.default_rng(0), candidates=1)
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

    def fake_relax_true_minimum(state, trajectory_name=None):
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

    monkeypatch.setattr(walker, "relax_true_minimum", lambda state, trajectory_name=None: next(relax_results))
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

    def fake_relax_true_minimum(state, trajectory_name=None):
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

    monkeypatch.setattr(walker, "relax_true_minimum", lambda state, trajectory_name=None: RelaxResult(state, 0.0, 0.0, 0))
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

    monkeypatch.setattr(walker, "relax_true_minimum", lambda state, trajectory_name=None: RelaxResult(state, 0.0, 0.0, 0))

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

    monkeypatch.setattr(oracle, "_directional_hvp", fake_hvp)
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

    monkeypatch.setattr(oracle, "_directional_hvp", fake_hvp)
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

    monkeypatch.setattr(oracle, "_directional_hvp", lambda state, proposal, direction: np.zeros_like(direction))
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

    monkeypatch.setattr(oracle, "_directional_hvp", lambda state, proposal, direction: np.zeros_like(direction))
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
    monkeypatch.setattr(oracle, "_directional_hvp", lambda state, proposal, direction: 10.0 * direction)

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

    monkeypatch.setattr(oracle, "_directional_hvp", fake_hvp)
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


def test_direction_stats_summary_tracks_curvature_by_kind():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=SSWConfig(),
        softening_enabled=False,
    )
    walker._record_direction_curvatures(
        DirectionCandidateKind.RANDOM,
        choice_curvature=-2.0,
        true_curvature=-1.5,
        inner_curvature=-1.0,
    )
    walker._record_direction_curvatures(
        DirectionCandidateKind.RANDOM,
        choice_curvature=4.0,
        true_curvature=3.0,
        inner_curvature=2.0,
    )
    walker._record_direction_curvatures(
        DirectionCandidateKind.BOND,
        choice_curvature=-8.0,
        true_curvature=-6.0,
        inner_curvature=-5.0,
    )

    summary = walker._direction_stats_summary()

    assert summary["direction_curvature_random_count"] == 2
    assert summary["direction_curvature_random_mean"] == pytest.approx(1.0)
    assert summary["direction_curvature_random_min"] == pytest.approx(-2.0)
    assert summary["direction_curvature_random_max"] == pytest.approx(4.0)
    assert summary["direction_true_curvature_random_mean"] == pytest.approx(0.75)
    assert summary["direction_inner_curvature_random_mean"] == pytest.approx(0.5)
    assert summary["direction_curvature_bond_count"] == 1
    assert summary["direction_true_curvature_bond_mean"] == pytest.approx(-6.0)


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
    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.RANDOM) == 2
    assert DirectionCandidateKind.BOND not in [candidate.kind for candidate in candidates]


def test_direction_generator_can_disable_momentum_candidate():
    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=2, enable_momentum_candidate=False)

    candidates = generator.generate(state, previous_direction=np.array([1.0, 0.0, 0.0]))

    assert DirectionCandidateKind.MOMENTUM not in [candidate.kind for candidate in candidates]
    assert [candidate.kind for candidate in candidates].count(DirectionCandidateKind.RANDOM) == 2


def test_direction_pool_disable_momentum_filters_momentum_candidates():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=float),
    )
    config = LSSSWConfig(direction_pool_disable_momentum=True)
    walker = SurfaceWalker(calculator=AnalyticCalculator(Quadratic()), config=config, softening_enabled=False)
    previous = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    previous /= np.linalg.norm(previous)

    candidates = walker.oracle.generator.generate(state, previous)
    filtered = walker._filter_direction_candidates(candidates)

    assert DirectionCandidateKind.MOMENTUM in {candidate.kind for candidate in candidates}
    assert DirectionCandidateKind.MOMENTUM not in {candidate.kind for candidate in filtered}


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
    generator = CandidateDirectionGenerator(np.random.default_rng(0), n_random=0, bond_pairs=[(0, 1)])

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


def test_reference_dimer_bias_relax_uses_biased_curvature_for_gaussian_weight(monkeypatch):
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(Quadratic()),
        config=LSSSWConfig(
            direction_engine="reference_dimer",
            proposal_step_mode="bias_relax",
            max_steps_per_walk=1,
            proposal_relax_steps=0,
        ),
        softening_enabled=False,
    )
    captured_bias_curvatures = []
    captured_trust_curvatures = []

    monkeypatch.setattr(
        walker,
        "_choose_reference_dimer_direction",
        lambda current, **kwargs: DirectionChoice(
            direction=np.array([1.0, 0.0, 0.0]),
            curvature=2000.0,
            kind=DirectionCandidateKind.REFERENCE_DIMER,
            candidate_count=1,
            true_curvature=-25.0,
            biased_curvature=2000.0,
        ),
    )
    monkeypatch.setattr(walker, "_build_softening", lambda *args, **kwargs: None)
    monkeypatch.setattr(walker, "_execution_step_scale", lambda *args, **kwargs: 0.1)

    def fake_bias_weight(curvature, sigma):
        captured_bias_curvatures.append(curvature)
        return 0.5

    class RecordingTrust:
        def update(self, *, curvature, **kwargs):
            captured_trust_curvatures.append(curvature)
            return TrustRegionUpdate(
                predicted_delta=0.0,
                true_delta=0.0,
                model_error=0.0,
                damaged=False,
                sigma_scale=1.0,
                weight_scale=1.0,
                action="hold",
            )

    class NoRelaxer:
        def __init__(self, evaluator, optimizer):
            self.evaluator = evaluator

        @staticmethod
        def classify_outcome(**kwargs):
            return RelaxOutcomeClass.USEFUL_PROGRESS

        def relax(self, state, **kwargs):
            energy, gradient = self.evaluator(state.flatten_positions(), state)
            return RelaxResult(
                state=state,
                energy=energy,
                gradient_norm=float(np.linalg.norm(gradient)),
                n_iter=0,
            )

    import pamssw.walker as walker_module

    monkeypatch.setattr(walker_module, "Relaxer", NoRelaxer)
    monkeypatch.setattr(walker, "_bias_weight", fake_bias_weight)
    walker.trust_controller = RecordingTrust()

    walker._walk_candidate_from_seed(state)

    assert captured_bias_curvatures == [pytest.approx(2000.0)]
    assert captured_trust_curvatures == [pytest.approx(-25.0)]


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
        n_random=0,
        enable_bond_form_break_split=True,
        n_bond_formation_pairs=1,
        n_bond_breaking_pairs=0,
        bond_distance_threshold=1.0,
        bond_formation_max_distance=4.0,
    )

    candidates = generator.generate(state, previous_direction=None)

    assert candidates == []


def test_bond_form_break_split_generates_break_candidates_for_short_pairs():
    state = State(
        numbers=np.array([6, 6]),
        positions=np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]]),
    )
    generator = CandidateDirectionGenerator(
        np.random.default_rng(0),
        n_random=0,
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
        n_random=0,
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
        n_random=0,
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
        n_random=0,
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
        n_random=0,
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

    def fake_relax_true_minimum(state, trajectory_name=None):
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
    assert "walk_displacement_clips" in result.stats
    assert "fragment_rejections" in result.stats
    assert "direction_bond_pairs_requested" in result.stats
    assert "direction_bond_pairs_generated" in result.stats
    assert "direction_bond_candidates_valid" in result.stats


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

    def fake_relax_true_minimum(state, trajectory_name=None):
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

    shrunk_rms = SurfaceWalker._direction_step_metrics(state, direction, shrunk_sigma, 1e-4)[
        "step_displacement_rms_all"
    ]
    expanded_rms = SurfaceWalker._direction_step_metrics(state, direction, expanded_sigma, 1e-4)[
        "step_displacement_rms_all"
    ]
    assert shrunk_rms == pytest.approx(0.1)
    assert expanded_rms == pytest.approx(0.35)


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
