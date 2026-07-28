from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import pamssw.relax as relax_module
from pamssw.accounting import BudgetExceeded
from pamssw.relax import RelaxEvaluation, Relaxer
from pamssw.result import RelaxOutcomeClass
from pamssw.state import State


def test_relax_evaluation_is_a_frozen_component_value_object():
    evaluation_type = getattr(relax_module, "RelaxEvaluation", None)

    assert evaluation_type is not None
    evaluation = evaluation_type(
        true_energy=1.0,
        true_gradient=np.array([1.0, 2.0, 3.0]),
        bias_energy=2.0,
        bias_gradient=np.array([4.0, 5.0, 6.0]),
        softening_energy=3.0,
        softening_gradient=np.array([7.0, 8.0, 9.0]),
        total_energy=6.0,
        total_gradient=np.array([12.0, 15.0, 18.0]),
    )

    assert evaluation.total_energy == 6.0
    np.testing.assert_allclose(evaluation.total_gradient, [12.0, 15.0, 18.0])
    with pytest.raises(FrozenInstanceError):
        evaluation.total_energy = 0.0


def test_relax_evaluation_defensively_copies_and_locks_all_gradient_arrays():
    gradients = {
        "true_gradient": np.array([1.0, 2.0, 3.0]),
        "bias_gradient": np.array([4.0, 5.0, 6.0]),
        "softening_gradient": np.array([7.0, 8.0, 9.0]),
        "total_gradient": np.array([12.0, 15.0, 18.0]),
    }
    evaluation = RelaxEvaluation(
        true_energy=1.0,
        bias_energy=2.0,
        softening_energy=3.0,
        total_energy=6.0,
        **gradients,
    )
    gradients["true_gradient"][0] = -1.0

    assert evaluation.true_gradient[0] == 1.0
    for name, source in gradients.items():
        component = getattr(evaluation, name)
        assert not np.shares_memory(component, source)
        assert not component.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            component[0] = 0.0


def test_relax_evaluation_rejects_mismatched_component_gradient_shapes():
    with pytest.raises(ValueError, match="same shape"):
        RelaxEvaluation(
            true_energy=1.0,
            true_gradient=np.zeros(3),
            bias_energy=0.0,
            bias_gradient=np.zeros(3),
            softening_energy=0.0,
            softening_gradient=np.zeros(2),
            total_energy=1.0,
            total_gradient=np.zeros(3),
        )


def test_relaxer_maps_per_atom_force_tolerance_to_sufficient_lbfgsb_component_bound(
    monkeypatch,
):
    captured = {}

    class Result:
        x = np.array([0.0, 0.0, 0.0])
        nit = 0

    def fake_minimize(fun, x0, method, jac, bounds=None, options=None):
        captured["options"] = options
        captured["bounds"] = bounds
        return Result()

    monkeypatch.setattr("pamssw.relax.minimize", fake_minimize)

    def evaluator(flat_positions, template):
        return 0.0, np.zeros_like(flat_positions)

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    Relaxer(evaluator).relax(state, fmax=1e-4, maxiter=123)

    assert captured["options"]["gtol"] == pytest.approx(1e-4 / np.sqrt(3.0))
    assert captured["options"]["ftol"] == 0.0
    assert captured["options"]["maxiter"] == 123


def test_relaxer_applies_coordinate_trust_radius(monkeypatch):
    captured = {}

    class Result:
        x = np.array([0.5, 1.0, -0.5])
        nit = 1

    def fake_minimize(fun, x0, method, jac, bounds=None, options=None):
        captured["x0"] = x0.copy()
        captured["bounds"] = bounds
        return Result()

    monkeypatch.setattr("pamssw.relax.minimize", fake_minimize)

    def evaluator(flat_positions, template):
        return 0.0, np.zeros_like(flat_positions)

    state = State(numbers=np.array([1]), positions=np.array([[0.5, 1.0, -0.5]]))
    Relaxer(evaluator).relax(state, fmax=1e-4, maxiter=3, coordinate_trust_radius=0.25)

    assert captured["bounds"] == [(0.25, 0.75), (0.75, 1.25), (-0.75, -0.25)]


def test_relaxer_reports_bound_fraction_and_displacement(monkeypatch):
    class Result:
        x = np.array([0.75, 1.0, -0.25])
        nit = 1

    def fake_minimize(fun, x0, method, jac, bounds=None, options=None):
        return Result()

    monkeypatch.setattr("pamssw.relax.minimize", fake_minimize)

    def evaluator(flat_positions, template):
        return 0.0, np.zeros_like(flat_positions)

    state = State(numbers=np.array([1]), positions=np.array([[0.5, 1.0, -0.5]]))
    result = Relaxer(evaluator).relax(state, fmax=1e-4, maxiter=3, coordinate_trust_radius=0.25)

    assert result.active_bound_fraction == 2 / 6
    assert result.displacement_max == pytest.approx(np.sqrt(0.25**2 + 0.25**2))
    assert result.displacement_rms == pytest.approx(result.displacement_max)


def test_relax_outcome_classifies_stagnated():
    outcome = Relaxer.classify_outcome(
        initial_energy=0.0,
        final_energy=0.0,
        gradient_norm=1.0,
        fmax=0.1,
        displacement_rms=0.0,
        displacement_max=0.0,
        active_bound_fraction=0.0,
    )

    assert outcome == RelaxOutcomeClass.STAGNATED


def test_relax_outcome_fmax_converged_but_unproductive():
    outcome = Relaxer.classify_outcome(
        initial_energy=0.0,
        final_energy=0.0,
        gradient_norm=0.01,
        fmax=0.1,
        displacement_rms=0.0,
        displacement_max=0.0,
        active_bound_fraction=0.0,
    )

    assert outcome == RelaxOutcomeClass.CONVERGED_UNPRODUCTIVE


def test_relax_outcome_true_delta_overrides_biased_displacement_as_stagnated():
    outcome = Relaxer.classify_outcome(
        initial_energy=0.0,
        final_energy=-1.0,
        gradient_norm=1.0,
        fmax=0.1,
        displacement_rms=0.2,
        displacement_max=0.3,
        active_bound_fraction=0.0,
        true_delta=0.01,
    )

    assert outcome == RelaxOutcomeClass.STAGNATED


def test_relax_outcome_true_delta_converged_but_unproductive():
    outcome = Relaxer.classify_outcome(
        initial_energy=0.0,
        final_energy=-1.0,
        gradient_norm=0.01,
        fmax=0.1,
        displacement_rms=0.2,
        displacement_max=0.3,
        active_bound_fraction=0.0,
        true_delta=0.01,
    )

    assert outcome == RelaxOutcomeClass.CONVERGED_UNPRODUCTIVE


def test_relaxer_reports_scipy_trajectory_states(monkeypatch):
    class Result:
        x = np.array([0.25, 0.0, 0.0])
        nit = 1

    def fake_minimize(fun, x0, method, jac, bounds=None, options=None, callback=None):
        if callback is not None:
            callback(np.array([0.25, 0.0, 0.0]))
        return Result()

    monkeypatch.setattr("pamssw.relax.minimize", fake_minimize)

    def evaluator(flat_positions, template):
        return 0.0, np.zeros_like(flat_positions)

    trajectory = []
    state = State(numbers=np.array([1]), positions=np.array([[0.5, 0.0, 0.0]]))
    Relaxer(evaluator).relax(state, fmax=1e-4, maxiter=3, trajectory_callback=trajectory.append)

    assert len(trajectory) >= 2
    np.testing.assert_allclose(trajectory[0].positions, np.array([[0.5, 0.0, 0.0]]))
    np.testing.assert_allclose(trajectory[-1].positions, np.array([[0.25, 0.0, 0.0]]))


def test_relaxer_leaves_periodic_axes_unbounded(monkeypatch):
    captured = {}

    class Result:
        x = np.array([0.5, 1.0, -0.5])
        nit = 1

    def fake_minimize(fun, x0, method, jac, bounds=None, options=None):
        captured["bounds"] = bounds
        return Result()

    monkeypatch.setattr("pamssw.relax.minimize", fake_minimize)

    def evaluator(flat_positions, template):
        return 0.0, np.zeros_like(flat_positions)

    state = State(
        numbers=np.array([1]),
        positions=np.array([[0.5, 1.0, -0.5]]),
        cell=np.eye(3),
        pbc=(True, True, False),
    )
    Relaxer(evaluator).relax(state, fmax=1e-4, maxiter=3, coordinate_trust_radius=0.25)

    assert captured["bounds"] == [(None, None), (None, None), (-0.75, -0.25)]


def test_relaxer_wraps_final_periodic_coordinates(monkeypatch):
    class Result:
        x = np.array([5.2, -0.2, 11.0])
        nit = 1

    def fake_minimize(fun, x0, method, jac, bounds=None, options=None):
        fun(np.asarray(x0, dtype=float))
        fun(Result.x)
        return Result()

    monkeypatch.setattr("pamssw.relax.minimize", fake_minimize)

    def evaluator(flat_positions, template):
        return 0.0, np.zeros_like(flat_positions)

    state = State(
        numbers=np.array([1]),
        positions=np.array([[4.8, 0.2, 10.0]]),
        cell=np.diag([5.0, 5.0, 12.0]),
        pbc=(True, True, False),
    )

    result = Relaxer(evaluator).relax(state, fmax=1e-4, maxiter=3, coordinate_trust_radius=0.25)

    np.testing.assert_allclose(result.state.positions, np.array([[0.2, 4.8, 11.0]]))
    assert result.telemetry.backend_evaluations == 2
    assert result.telemetry.reporting_cache_hits == 1
    assert result.telemetry.reporting_evaluator_calls == 1
    assert result.telemetry.finalization_requests == 1
    assert result.telemetry.explicit_finalization_calls == 1


def test_fully_periodic_scipy_relax_reports_raw_force_when_no_finite_bounds(monkeypatch):
    class Result:
        x = np.array([1.0, 2.0, 3.0])
        nit = 0
        success = True

    def fake_minimize(fun, x0, method, jac, bounds=None, options=None):
        assert bounds == [(None, None), (None, None), (None, None)]
        fun(np.asarray(x0, dtype=float))
        return Result()

    monkeypatch.setattr("pamssw.relax.minimize", fake_minimize)

    def evaluator(flat_positions, template):
        return 0.0, np.zeros_like(flat_positions)

    state = State(
        numbers=np.array([1]),
        positions=np.array([[1.0, 2.0, 3.0]]),
        cell=np.diag([5.0, 5.0, 5.0]),
        pbc=(True, True, True),
    )

    result = Relaxer(evaluator).relax(
        state,
        fmax=1e-4,
        maxiter=3,
        coordinate_trust_radius=0.25,
    )

    assert result.telemetry.gradient_measure == "raw_active_max_force"


def test_relaxer_reports_projected_gradient_for_bound_constrained_optimum(monkeypatch):
    class Result:
        x = np.array([0.25, 0.0, 0.0])
        nit = 1

    def fake_minimize(fun, x0, method, jac, bounds=None, options=None):
        return Result()

    monkeypatch.setattr("pamssw.relax.minimize", fake_minimize)

    def evaluator(flat_positions, template):
        return 0.0, np.array([10.0, 0.0, 0.0])

    state = State(numbers=np.array([1]), positions=np.array([[0.5, 0.0, 0.0]]))
    result = Relaxer(evaluator).relax(state, fmax=1e-4, maxiter=3, coordinate_trust_radius=0.25)

    assert result.gradient_norm == 0.0
    assert result.telemetry.gradient_measure == "projected_active_kkt_residual"


def test_relaxer_can_use_ase_fire_without_scipy_line_search():
    def evaluator(flat_positions, template):
        return 0.5 * float(np.dot(flat_positions, flat_positions)), flat_positions.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(evaluator, optimizer="ase-fire").relax(state, fmax=1e-4, maxiter=200)

    assert result.gradient_norm < 1e-4
    assert result.energy < 1e-8
    assert result.n_iter > 0


def test_relaxer_can_use_ase_fire2_when_available():
    if getattr(relax_module, "_ASE_FIRE2") is None:
        pytest.skip("installed ASE does not provide FIRE2")

    def evaluator(flat_positions, template):
        return 0.5 * float(np.dot(flat_positions, flat_positions)), flat_positions.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(evaluator, optimizer="ase-fire2").relax(state, fmax=1e-4, maxiter=500)

    assert result.gradient_norm <= 1e-4
    assert result.telemetry.backend == "ase-fire2"


def test_relaxer_reports_missing_ase_fire2_capability(monkeypatch):
    monkeypatch.setattr(relax_module, "_ASE_FIRE2", None)

    def evaluator(flat_positions, template):
        return 0.5 * float(np.dot(flat_positions, flat_positions)), flat_positions.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="FIRE2.*not available"):
        Relaxer(evaluator, optimizer="ase-fire2").relax(state, fmax=1e-4, maxiter=5)


@pytest.mark.parametrize("optimizer", ["scipy-lbfgsb", "ase-fire", "ase-lbfgs"])
def test_relaxer_reuses_report_only_endpoint_evaluations_without_changing_backend_path(optimizer):
    evaluated_positions = []

    def evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        evaluated_positions.append(flat.copy())
        return 0.5 * float(np.dot(flat, flat)), flat.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(evaluator, optimizer=optimizer).relax(state, fmax=1e-4, maxiter=200)

    assert result.gradient_norm <= 1e-4
    assert result.telemetry.backend == optimizer
    assert result.telemetry.converged
    assert result.telemetry.termination_reason == "converged"
    assert result.telemetry.evaluator_calls == len(evaluated_positions)
    assert result.telemetry.backend_evaluations == len(evaluated_positions)
    assert result.telemetry.reporting_cache_hits == 2
    assert result.telemetry.reporting_evaluator_calls == 0
    assert result.telemetry.finalization_requests == 1
    assert result.telemetry.explicit_finalization_calls == 0
    assert result.telemetry.gradient_measure == "raw_active_max_force"


def test_relaxer_reports_unified_maxiter_termination():
    def evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(np.dot(flat, flat)), flat.copy()

    state = State(numbers=np.array([1]), positions=np.array([[10.0, 0.0, 0.0]]))
    result = Relaxer(evaluator, optimizer="ase-fire").relax(state, fmax=1e-12, maxiter=1)

    assert not result.telemetry.converged
    assert result.telemetry.termination_reason == "maxiter"
    assert result.gradient_norm > 1e-12


def test_scipy_reporting_does_not_repeat_backend_endpoint_calls(monkeypatch):
    calls = []

    class Result:
        x = np.zeros(3)
        nit = 1
        success = True

    def fake_minimize(fun, x0, method, jac, bounds=None, options=None):
        fun(np.asarray(x0, dtype=float))
        fun(Result.x)
        return Result()

    monkeypatch.setattr("pamssw.relax.minimize", fake_minimize)

    def evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        calls.append(flat.copy())
        return 0.5 * float(np.dot(flat, flat)), flat.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(evaluator, optimizer="scipy-lbfgsb").relax(state, fmax=1e-4, maxiter=5)

    assert len(calls) == 2
    assert result.telemetry.backend_evaluations == 2
    assert result.telemetry.reporting_cache_hits == 2
    assert result.telemetry.reporting_evaluator_calls == 0
    assert result.telemetry.evaluator_calls == (
        result.telemetry.backend_evaluations + result.telemetry.reporting_evaluator_calls
    )


def test_relaxer_applies_ase_trajectory_stride():
    def evaluator(flat_positions, template):
        return 0.5 * float(np.dot(flat_positions, flat_positions)), flat_positions.copy()

    trajectory = []
    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    Relaxer(evaluator, optimizer="ase-fire").relax(
        state,
        fmax=1e-4,
        maxiter=20,
        trajectory_callback=trajectory.append,
        trajectory_stride=5,
    )

    assert len(trajectory) < 20
    assert len(trajectory) >= 2


def test_safe_lbfgs_two_loop_uses_fixed_empty_history_scale_and_latest_inverse_scale():
    inverse_product = getattr(relax_module, "_lbfgs_inverse_product")
    gradient = np.array([2.0, -4.0])

    np.testing.assert_allclose(inverse_product(gradient, []), gradient / 70.0)

    s = np.array([1.0, 0.0])
    y = np.array([2.0, 0.0])
    history = [(s, y, 1.0 / float(np.dot(s, y)))]
    np.testing.assert_allclose(inverse_product(np.array([2.0, 0.0]), history), [1.0, 0.0])


def test_safe_lbfgs_curvature_gate_is_relative_to_secant_norms():
    accepts = getattr(relax_module, "_accept_lbfgs_curvature")

    assert accepts(np.array([1.0]), np.array([1.0]))
    assert not accepts(np.array([1.0]), np.array([0.0]))
    assert not accepts(np.array([1.0]), np.array([-1.0]))


def test_safe_lbfgs_limits_maximum_displacement_of_each_atom():
    limit = getattr(relax_module, "_limit_max_atomic_displacement")
    direction = np.array([3.0, 4.0, 0.0, 0.0, 0.0, 10.0])

    limited = limit(direction)

    atom_norms = np.linalg.norm(limited.reshape(-1, 3), axis=1)
    assert np.max(atom_norms) == pytest.approx(0.2)
    np.testing.assert_allclose(limited, direction * 0.02)


def test_safe_lbfgs_total_converges_on_quadratic_with_armijo_steps():
    calls = []

    def evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        calls.append(flat.copy())
        return 0.5 * float(np.dot(flat, flat)), flat.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
        state,
        fmax=1e-8,
        maxiter=10,
    )

    assert result.telemetry.converged
    assert result.telemetry.termination_reason == "converged"
    assert result.telemetry.backend == "safe-lbfgs-total"
    assert result.telemetry.backend_evaluations == len(calls)
    assert result.telemetry.accepted_steps == result.n_iter
    assert result.telemetry.accepted_secants == result.n_iter
    assert result.telemetry.rejected_secants == 0
    assert result.telemetry.line_search_evaluations == (
        result.telemetry.accepted_steps + result.telemetry.rejected_steps
    )
    assert 1 < result.n_iter <= 10
    np.testing.assert_allclose(result.state.positions, 0.0, atol=1e-12)


def test_safe_lbfgs_total_caps_first_atomic_step():
    trajectory = []

    def evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(np.dot(flat, flat)), flat.copy()

    state = State(numbers=np.array([1]), positions=np.array([[100.0, 0.0, 0.0]]))
    result = Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
        state,
        fmax=1e-12,
        maxiter=1,
        trajectory_callback=trajectory.append,
    )

    assert result.n_iter == 1
    assert result.telemetry.termination_reason == "maxiter"
    assert len(trajectory) == 2
    assert np.linalg.norm(trajectory[1].positions - trajectory[0].positions) == pytest.approx(0.2)


@pytest.mark.parametrize(
    "history_limit",
    [True, False, -1, 2, 9, 11, 1.0, "1", 0.0, 10.0, "10"],
)
def test_safe_lbfgs_history_one_depth_rejects_invalid_values_before_evaluator(history_limit):
    calls = []

    def evaluator(flat_positions, template):
        calls.append(np.asarray(flat_positions, dtype=float).copy())
        return 0.0, np.zeros_like(flat_positions)

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="_safe_lbfgs_history_limit"):
        Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
            state,
            fmax=1e-8,
            maxiter=1,
            _safe_lbfgs_history_limit=history_limit,
        )

    assert calls == []


@pytest.mark.parametrize(
    "optimizer",
    ["ase-fire", "ase-fire2", "ase-lbfgs", "scipy-lbfgsb", "bias-separated-lbfgs"],
)
@pytest.mark.parametrize("history_limit", [0, 1])
def test_safe_lbfgs_history_one_or_zero_rejects_other_optimizers_before_evaluator(
    optimizer,
    history_limit,
):
    calls = []

    def evaluator(flat_positions, template):
        calls.append(np.asarray(flat_positions, dtype=float).copy())
        return 0.0, np.zeros_like(flat_positions)

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="safe-lbfgs-total"):
        Relaxer(evaluator, optimizer=optimizer).relax(
            state,
            fmax=1e-8,
            maxiter=1,
            _safe_lbfgs_history_limit=history_limit,
        )

    assert calls == []


_ADAPTIVE_SCALE_UNSET = object()


def _run_safe_lbfgs_history_limit(
    history_limit,
    *,
    maxiter,
    adaptive_scale_without_history=_ADAPTIVE_SCALE_UNSET,
):
    calls = []
    trajectory = []
    curvature = np.array([1.0, 4.0, 2.0])

    def evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        calls.append(flat.copy())
        return 0.5 * float(np.dot(curvature * flat, flat)), curvature * flat

    state = State(numbers=np.array([1]), positions=np.array([[0.8, -0.6, 0.4]]))
    relax_kwargs = {
        "_safe_lbfgs_history_limit": history_limit,
    }
    if adaptive_scale_without_history is not _ADAPTIVE_SCALE_UNSET:
        relax_kwargs["_safe_lbfgs_adaptive_scale_without_history"] = (
            adaptive_scale_without_history
        )
    result = Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
        state,
        fmax=1e-12,
        maxiter=maxiter,
        trajectory_callback=trajectory.append,
        **relax_kwargs,
    )
    return result, calls, trajectory


def test_safe_lbfgs_history_limit_none_matches_explicit_default_capacity():
    default_result, default_calls, default_trajectory = _run_safe_lbfgs_history_limit(
        None,
        maxiter=4,
    )
    explicit_result, explicit_calls, explicit_trajectory = _run_safe_lbfgs_history_limit(
        10,
        maxiter=4,
    )

    np.testing.assert_array_equal(default_calls, explicit_calls)
    assert len(default_trajectory) == len(explicit_trajectory)
    for default_state, explicit_state in zip(default_trajectory, explicit_trajectory, strict=True):
        np.testing.assert_array_equal(default_state.positions, explicit_state.positions)
    np.testing.assert_array_equal(default_result.state.positions, explicit_result.state.positions)
    assert default_result.energy == explicit_result.energy
    assert default_result.gradient_norm == explicit_result.gradient_norm
    assert default_result.n_iter == explicit_result.n_iter
    assert default_result.telemetry == explicit_result.telemetry


def test_safe_lbfgs_history_one_runs_and_matches_depth_ten_with_one_usable_secant():
    history_one, history_one_calls, history_one_trajectory = _run_safe_lbfgs_history_limit(
        1,
        maxiter=2,
    )
    history_ten, history_ten_calls, history_ten_trajectory = _run_safe_lbfgs_history_limit(
        10,
        maxiter=2,
    )

    assert history_one.n_iter == history_ten.n_iter == 2
    assert history_one.telemetry.accepted_secants == history_ten.telemetry.accepted_secants == 2
    np.testing.assert_array_equal(history_one_calls, history_ten_calls)
    assert len(history_one_trajectory) == len(history_ten_trajectory)
    for one_state, ten_state in zip(history_one_trajectory, history_ten_trajectory, strict=True):
        np.testing.assert_array_equal(one_state.positions, ten_state.positions)
    np.testing.assert_array_equal(history_one.state.positions, history_ten.state.positions)
    assert history_one.energy == history_ten.energy
    assert history_one.gradient_norm == history_ten.gradient_norm
    assert history_one.telemetry == history_ten.telemetry


def test_safe_lbfgs_history_limit_zero_matches_default_capacity_for_first_iteration():
    empty_result, empty_calls, empty_trajectory = _run_safe_lbfgs_history_limit(0, maxiter=1)
    default_result, default_calls, default_trajectory = _run_safe_lbfgs_history_limit(10, maxiter=1)

    np.testing.assert_array_equal(empty_calls, default_calls)
    assert len(empty_trajectory) == len(default_trajectory)
    for empty_state, default_state in zip(empty_trajectory, default_trajectory, strict=True):
        np.testing.assert_array_equal(empty_state.positions, default_state.positions)
    np.testing.assert_array_equal(empty_result.state.positions, default_result.state.positions)
    assert empty_result.energy == default_result.energy
    assert empty_result.gradient_norm == default_result.gradient_norm
    assert empty_result.n_iter == default_result.n_iter == 1
    assert empty_result.telemetry == default_result.telemetry


def test_safe_lbfgs_history_limit_zero_keeps_inverse_product_history_empty(monkeypatch):
    inverse_product = relax_module._lbfgs_inverse_product
    history_lengths = []

    def spy_inverse_product(gradient, history):
        history_lengths.append(len(history))
        return inverse_product(gradient, history)

    monkeypatch.setattr(relax_module, "_lbfgs_inverse_product", spy_inverse_product)

    result, _, _ = _run_safe_lbfgs_history_limit(0, maxiter=3)

    assert result.n_iter == 3
    assert history_lengths == [0, 0, 0]


def test_safe_lbfgs_history_depth_one_retains_one_pair_and_diverges_only_after_second_secant(
    monkeypatch,
):
    inverse_product = relax_module._lbfgs_inverse_product
    observations = {}
    active_history_limit = None

    def spy_inverse_product(gradient, history):
        observations.setdefault(active_history_limit, []).append(
            (len(history), -inverse_product(gradient, history))
        )
        return inverse_product(gradient, history)

    monkeypatch.setattr(relax_module, "_lbfgs_inverse_product", spy_inverse_product)

    active_history_limit = 1
    history_one, _, _ = _run_safe_lbfgs_history_limit(1, maxiter=3)
    active_history_limit = 10
    history_ten, _, _ = _run_safe_lbfgs_history_limit(10, maxiter=3)

    one_observations = observations[1]
    ten_observations = observations[10]
    assert history_one.n_iter == history_ten.n_iter == 3
    assert [length for length, _ in one_observations] == [0, 1, 1]
    assert [length for length, _ in ten_observations] == [0, 1, 2]
    for (_, one_direction), (_, ten_direction) in zip(
        one_observations[:2],
        ten_observations[:2],
        strict=True,
    ):
        np.testing.assert_array_equal(one_direction, ten_direction)
    assert not np.array_equal(one_observations[2][1], ten_observations[2][1])


def test_safe_lbfgs_history_capacity_changes_anisotropic_path_only_after_secant():
    empty_first, _, empty_first_trajectory = _run_safe_lbfgs_history_limit(0, maxiter=1)
    default_first, _, default_first_trajectory = _run_safe_lbfgs_history_limit(10, maxiter=1)
    empty_second, _, _ = _run_safe_lbfgs_history_limit(0, maxiter=2)
    default_second, _, _ = _run_safe_lbfgs_history_limit(10, maxiter=2)

    assert empty_first.telemetry.accepted_secants == 1
    assert default_first.telemetry.accepted_secants == 1
    assert len(empty_first_trajectory) == len(default_first_trajectory)
    for empty_state, default_state in zip(
        empty_first_trajectory,
        default_first_trajectory,
        strict=True,
    ):
        np.testing.assert_array_equal(empty_state.positions, default_state.positions)
    np.testing.assert_array_equal(empty_first.state.positions, default_first.state.positions)
    assert empty_second.telemetry.accepted_secants == 2
    assert default_second.telemetry.accepted_secants == 2
    assert not np.array_equal(empty_second.state.positions, default_second.state.positions)


def test_safe_lbfgs_history_limit_zero_preserves_secant_and_line_search_diagnostics():
    def accepted_components(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        true_energy = 0.5 * float(np.dot(flat, flat))
        bias_energy = 0.25 * float(np.dot(flat, flat))
        return RelaxEvaluation(
            true_energy=true_energy,
            true_gradient=flat.copy(),
            bias_energy=bias_energy,
            bias_gradient=0.5 * flat,
            softening_energy=0.0,
            softening_gradient=np.zeros_like(flat),
            total_energy=true_energy + bias_energy,
            total_gradient=1.5 * flat,
        )

    def accepted_evaluator(flat_positions, template):
        parts = accepted_components(flat_positions, template)
        return parts.total_energy, parts.total_gradient.copy()

    def rejected_evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        return float(flat[0]), np.array([1.0, 0.0, 0.0])

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    accepted = Relaxer(
        accepted_evaluator,
        optimizer="safe-lbfgs-total",
        component_evaluator=accepted_components,
    ).relax(state, fmax=1e-12, maxiter=1, _safe_lbfgs_history_limit=0)
    rejected = Relaxer(rejected_evaluator, optimizer="safe-lbfgs-total").relax(
        state,
        fmax=1e-12,
        maxiter=1,
        _safe_lbfgs_history_limit=0,
    )

    assert accepted.telemetry.accepted_secants == 1
    assert accepted.telemetry.rejected_secants == 0
    assert accepted.telemetry.bias_secant_curvature_sum > 0.0
    assert accepted.telemetry.line_search_evaluations == 1
    assert accepted.telemetry.line_search_evaluations == (
        accepted.telemetry.accepted_steps + accepted.telemetry.rejected_steps
    )
    assert rejected.telemetry.accepted_secants == 0
    assert rejected.telemetry.rejected_secants == 1
    assert rejected.telemetry.line_search_evaluations == 1
    assert rejected.telemetry.line_search_evaluations == (
        rejected.telemetry.accepted_steps + rejected.telemetry.rejected_steps
    )


@pytest.mark.parametrize(
    "invalid",
    [None, 0, 1, 1.0, "true", [], {}, np.bool_(True)],
)
def test_safe_lbfgs_scale_only_flag_rejects_non_booleans_before_evaluation(invalid):
    calls = []

    def evaluator(flat_positions, template):
        calls.append(np.asarray(flat_positions, dtype=float).copy())
        return 0.0, np.zeros_like(flat_positions, dtype=float)

    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    with pytest.raises((TypeError, ValueError), match="adaptive scale"):
        Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
            state,
            fmax=1e-8,
            maxiter=1,
            _safe_lbfgs_history_limit=0,
            _safe_lbfgs_adaptive_scale_without_history=invalid,
        )

    assert calls == []


@pytest.mark.parametrize(
    "optimizer",
    [
        "scipy-lbfgsb",
        "ase-fire",
        "ase-fire2",
        "ase-lbfgs",
        "safe-lbfgs-total",
        "bias-separated-lbfgs",
    ],
)
def test_safe_lbfgs_scale_only_false_is_noop_for_existing_optimizer_paths(optimizer):
    assert (
        relax_module._resolve_safe_lbfgs_adaptive_scale_without_history(
            False,
            optimizer,
            None,
        )
        is False
    )


def test_safe_lbfgs_scale_only_omitted_default_matches_explicit_false_bitwise():
    omitted, omitted_calls, omitted_trajectory = _run_safe_lbfgs_history_limit(
        10,
        maxiter=4,
    )
    explicit, explicit_calls, explicit_trajectory = _run_safe_lbfgs_history_limit(
        10,
        maxiter=4,
        adaptive_scale_without_history=False,
    )

    np.testing.assert_array_equal(omitted_calls, explicit_calls)
    assert len(omitted_trajectory) == len(explicit_trajectory)
    for omitted_state, explicit_state in zip(
        omitted_trajectory,
        explicit_trajectory,
        strict=True,
    ):
        np.testing.assert_array_equal(omitted_state.positions, explicit_state.positions)
    np.testing.assert_array_equal(omitted.state.positions, explicit.state.positions)
    assert omitted.energy == explicit.energy
    assert omitted.gradient_norm == explicit.gradient_norm
    assert omitted.n_iter == explicit.n_iter
    assert omitted.telemetry == explicit.telemetry


@pytest.mark.parametrize(
    ("optimizer", "history_limit"),
    [
        ("safe-lbfgs-total", None),
        ("safe-lbfgs-total", 10),
        ("scipy-lbfgsb", 0),
        ("ase-fire", 0),
        ("bias-separated-lbfgs", 0),
    ],
)
def test_safe_lbfgs_scale_only_rejects_invalid_mode_before_evaluation(
    optimizer,
    history_limit,
):
    calls = []

    def evaluator(flat_positions, template):
        calls.append(np.asarray(flat_positions, dtype=float).copy())
        return 0.0, np.zeros_like(flat_positions, dtype=float)

    state = State(numbers=np.array([1]), positions=np.zeros((1, 3)))
    with pytest.raises(ValueError, match="adaptive scale"):
        Relaxer(evaluator, optimizer=optimizer).relax(
            state,
            fmax=1e-8,
            maxiter=1,
            _safe_lbfgs_history_limit=history_limit,
            _safe_lbfgs_adaptive_scale_without_history=True,
        )

    assert calls == []


def test_safe_lbfgs_scale_only_three_modes_share_the_first_iteration():
    fixed, fixed_calls, _ = _run_safe_lbfgs_history_limit(0, maxiter=1)
    scale_only, scale_calls, _ = _run_safe_lbfgs_history_limit(
        0,
        maxiter=1,
        adaptive_scale_without_history=True,
    )
    history, history_calls, _ = _run_safe_lbfgs_history_limit(10, maxiter=1)

    np.testing.assert_array_equal(fixed_calls, scale_calls)
    np.testing.assert_array_equal(fixed_calls, history_calls)
    np.testing.assert_array_equal(fixed.state.positions, scale_only.state.positions)
    np.testing.assert_array_equal(fixed.state.positions, history.state.positions)


def test_safe_lbfgs_scale_only_is_distinct_after_first_accepted_secant():
    fixed, _, _ = _run_safe_lbfgs_history_limit(0, maxiter=2)
    scale_only, _, _ = _run_safe_lbfgs_history_limit(
        0,
        maxiter=2,
        adaptive_scale_without_history=True,
    )
    history, _, _ = _run_safe_lbfgs_history_limit(10, maxiter=2)

    assert not np.array_equal(fixed.state.positions, scale_only.state.positions)
    assert not np.array_equal(scale_only.state.positions, history.state.positions)


def test_lbfgs_inverse_product_can_use_latest_pair_only_as_scalar_scale():
    gradient = np.array([2.0, -1.0, 0.5])
    s = np.array([1.0, 2.0, 0.0])
    y = np.array([4.0, 1.0, 0.0])
    curvature = float(np.dot(s, y))
    pair = (s, y, 1.0 / curvature)
    expected_gamma = curvature / float(np.dot(y, y))

    product = relax_module._lbfgs_inverse_product(
        gradient,
        [],
        scale_pair=pair,
    )

    np.testing.assert_allclose(product, expected_gamma * gradient)


def test_safe_lbfgs_scale_only_keeps_two_loop_history_empty(monkeypatch):
    inverse_product = relax_module._lbfgs_inverse_product
    observations = []

    def spy_inverse_product(gradient, history, *, scale_pair=None):
        observations.append((len(history), scale_pair is not None))
        return inverse_product(gradient, history, scale_pair=scale_pair)

    monkeypatch.setattr(relax_module, "_lbfgs_inverse_product", spy_inverse_product)

    result, _, _ = _run_safe_lbfgs_history_limit(
        0,
        maxiter=3,
        adaptive_scale_without_history=True,
    )

    assert result.n_iter == 3
    assert observations == [(0, False), (0, True), (0, True)]


def test_safe_lbfgs_scale_only_preserves_latest_pair_after_later_rejection(
    monkeypatch,
):
    inverse_product = relax_module._lbfgs_inverse_product
    scale_gammas = []
    curvature_decisions = iter((True, False, False))

    def spy_inverse_product(gradient, history, *, scale_pair=None):
        if scale_pair is None:
            scale_gammas.append(None)
        else:
            s, y, _ = scale_pair
            scale_gammas.append(float(np.dot(s, y) / np.dot(y, y)))
        return inverse_product(gradient, history, scale_pair=scale_pair)

    monkeypatch.setattr(relax_module, "_lbfgs_inverse_product", spy_inverse_product)
    monkeypatch.setattr(
        relax_module,
        "_accept_lbfgs_curvature",
        lambda s, y: next(curvature_decisions),
    )

    result, _, _ = _run_safe_lbfgs_history_limit(
        0,
        maxiter=3,
        adaptive_scale_without_history=True,
    )

    assert result.telemetry.accepted_secants == 1
    assert result.telemetry.rejected_secants == 2
    assert scale_gammas[0] is None
    assert scale_gammas[1] is not None
    assert scale_gammas[2] == scale_gammas[1]


def test_safe_lbfgs_scale_only_rejected_curvature_does_not_update_scale_pair(
    monkeypatch,
):
    inverse_product = relax_module._lbfgs_inverse_product
    scale_pair_presence = []

    def spy_inverse_product(gradient, history, *, scale_pair=None):
        scale_pair_presence.append(scale_pair is not None)
        return inverse_product(gradient, history, scale_pair=scale_pair)

    monkeypatch.setattr(relax_module, "_lbfgs_inverse_product", spy_inverse_product)

    def linear_evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        return float(flat[0]), np.array([1.0, 0.0, 0.0])

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(linear_evaluator, optimizer="safe-lbfgs-total").relax(
        state,
        fmax=1e-12,
        maxiter=2,
        _safe_lbfgs_history_limit=0,
        _safe_lbfgs_adaptive_scale_without_history=True,
    )

    assert result.telemetry.rejected_secants == 2
    assert scale_pair_presence == [False, False]


def test_safe_lbfgs_scale_only_clears_scale_pair_on_mic_branch_change(monkeypatch):
    inverse_product = relax_module._lbfgs_inverse_product
    scale_pair_presence = []

    def spy_inverse_product(gradient, history, *, scale_pair=None):
        scale_pair_presence.append(scale_pair is not None)
        return inverse_product(gradient, history, scale_pair=scale_pair)

    monkeypatch.setattr(relax_module, "_lbfgs_inverse_product", spy_inverse_product)

    def component_evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        signature = ((0, 0, 0),) if flat[0] >= 0.8 else ((1, 0, 0),)
        gradient = flat.copy()
        return RelaxEvaluation(
            true_energy=0.5 * float(np.dot(flat, flat)),
            true_gradient=gradient,
            bias_energy=0.0,
            bias_gradient=np.zeros_like(flat),
            softening_energy=0.0,
            softening_gradient=np.zeros_like(flat),
            total_energy=0.5 * float(np.dot(flat, flat)),
            total_gradient=gradient,
            bias_image_signature=signature,
        )

    def total_evaluator(flat_positions, template):
        parts = component_evaluator(flat_positions, template)
        return parts.total_energy, parts.total_gradient.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(
        total_evaluator,
        optimizer="safe-lbfgs-total",
        component_evaluator=component_evaluator,
    ).relax(
        state,
        fmax=1e-12,
        maxiter=3,
        _safe_lbfgs_history_limit=0,
        _safe_lbfgs_adaptive_scale_without_history=True,
    )

    assert result.telemetry.mic_branch_resets == 1
    assert scale_pair_presence == [False, True, False]


def test_safe_lbfgs_scale_only_closes_evaluator_and_line_search_accounting():
    result, evaluator_calls, _ = _run_safe_lbfgs_history_limit(
        0,
        maxiter=4,
        adaptive_scale_without_history=True,
    )

    assert len(evaluator_calls) == result.telemetry.evaluator_calls
    assert result.telemetry.backend_evaluations == result.telemetry.evaluator_calls
    assert result.telemetry.reporting_evaluator_calls == 0
    assert result.telemetry.line_search_evaluations == (
        result.telemetry.accepted_steps + result.telemetry.rejected_steps
    )


def test_safe_lbfgs_total_reports_explicit_armijo_failure_without_fallback():
    initial = np.array([1.0, 0.0, 0.0])

    def evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        energy = 0.0 if np.array_equal(flat, initial) else 1.0
        return energy, np.array([1.0, 0.0, 0.0])

    state = State(numbers=np.array([1]), positions=initial.reshape(1, 3))
    result = Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
        state,
        fmax=1e-12,
        maxiter=10,
    )

    assert result.n_iter == 0
    assert not result.telemetry.converged
    assert result.telemetry.termination_reason == "line_search_failed"
    assert result.telemetry.backend_evaluations == 21
    assert result.telemetry.accepted_steps == 0
    assert result.telemetry.rejected_steps == 20
    assert result.telemetry.line_search_evaluations == 20
    np.testing.assert_array_equal(result.state.positions, state.positions)


def test_safe_lbfgs_total_rejects_a_non_descent_direction(monkeypatch):
    monkeypatch.setattr(
        relax_module,
        "_lbfgs_inverse_product",
        lambda gradient, history: -np.asarray(gradient, dtype=float),
    )

    def evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(np.dot(flat, flat)), flat.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
        state,
        fmax=1e-8,
        maxiter=10,
    )

    assert result.n_iter == 0
    assert result.telemetry.termination_reason == "non_descent_direction"
    assert result.telemetry.backend_evaluations == 1


def test_safe_lbfgs_total_stops_on_nonfinite_line_trial():
    initial = np.array([1.0, 0.0, 0.0])

    def evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        if np.array_equal(flat, initial):
            return 0.5, initial.copy()
        return float("nan"), np.full(3, np.nan)

    state = State(numbers=np.array([1]), positions=initial.reshape(1, 3))
    result = Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
        state,
        fmax=1e-8,
        maxiter=10,
    )

    assert result.n_iter == 0
    assert result.telemetry.termination_reason == "nonfinite_evaluation"
    assert result.telemetry.backend_evaluations == 2
    assert result.telemetry.rejected_steps == 1
    assert result.telemetry.line_search_evaluations == 1
    np.testing.assert_array_equal(result.state.positions, state.positions)


def test_safe_lbfgs_total_rejects_zero_curvature_secant_without_fallback():
    def evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        return float(flat[0]), np.array([1.0, 0.0, 0.0])

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
        state,
        fmax=1e-12,
        maxiter=1,
    )

    assert result.telemetry.accepted_steps == 1
    assert result.telemetry.accepted_secants == 0
    assert result.telemetry.rejected_secants == 1


def test_safe_lbfgs_total_keeps_fixed_atoms_out_of_steps_and_secants():
    def evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(np.dot(flat, flat)), flat.copy()

    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[10.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        fixed_mask=np.array([True, False]),
    )
    result = Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
        state,
        fmax=1e-8,
        maxiter=10,
    )

    np.testing.assert_array_equal(result.state.positions[0], state.positions[0])
    np.testing.assert_allclose(result.state.positions[1], 0.0, atol=1e-12)
    assert result.telemetry.converged


def test_safe_lbfgs_total_propagates_budget_exhaustion_from_line_search():
    calls = 0

    def evaluator(flat_positions, template):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise BudgetExceeded("test budget exhausted")
        flat = np.asarray(flat_positions, dtype=float)
        return 0.5 * float(np.dot(flat, flat)), flat.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    with pytest.raises(BudgetExceeded, match="test budget"):
        Relaxer(evaluator, optimizer="safe-lbfgs-total").relax(
            state,
            fmax=1e-8,
            maxiter=10,
        )


def _quadratic_bias_parts(flat_positions, template):
    flat = np.asarray(flat_positions, dtype=float)
    true_energy = 0.5 * float(np.dot(flat, flat))
    true_gradient = flat.copy()
    bias_energy = 0.25 * float(np.dot(flat, flat))
    bias_gradient = 0.5 * flat
    return RelaxEvaluation(
        true_energy=true_energy,
        true_gradient=true_gradient,
        bias_energy=bias_energy,
        bias_gradient=bias_gradient,
        softening_energy=0.0,
        softening_gradient=np.zeros_like(flat),
        total_energy=true_energy + bias_energy,
        total_gradient=true_gradient + bias_gradient,
    )


def _total_from_parts(flat_positions, template):
    parts = _quadratic_bias_parts(flat_positions, template)
    return parts.total_energy, parts.total_gradient.copy()


def test_bias_separated_lbfgs_requires_component_evaluation():
    state = State(numbers=np.array([1]), positions=np.array([[0.1, 0.0, 0.0]]))

    with pytest.raises(ValueError, match="component"):
        Relaxer(_total_from_parts, optimizer="bias-separated-lbfgs").relax(
            state,
            fmax=1e-8,
            maxiter=2,
        )


def test_bias_separated_lbfgs_rejects_local_softening_components():
    def parts_with_softening(flat_positions, template):
        parts = _quadratic_bias_parts(flat_positions, template)
        return RelaxEvaluation(
            true_energy=parts.true_energy,
            true_gradient=parts.true_gradient,
            bias_energy=parts.bias_energy,
            bias_gradient=parts.bias_gradient,
            softening_energy=0.0,
            softening_gradient=np.zeros_like(parts.total_gradient),
            total_energy=parts.total_energy,
            total_gradient=parts.total_gradient,
            softening_present=True,
        )

    def total(flat_positions, template):
        parts = parts_with_softening(flat_positions, template)
        return parts.total_energy, parts.total_gradient.copy()

    state = State(numbers=np.array([1]), positions=np.array([[0.1, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="softening"):
        Relaxer(
            total,
            optimizer="bias-separated-lbfgs",
            component_evaluator=parts_with_softening,
        ).relax(state, fmax=1e-8, maxiter=2)


def test_total_and_bias_separated_modes_have_identical_first_step():
    state = State(numbers=np.array([1]), positions=np.array([[0.1, 0.0, 0.0]]))
    trajectories = {}

    for optimizer in ("safe-lbfgs-total", "bias-separated-lbfgs"):
        trajectory = []
        Relaxer(
            _total_from_parts,
            optimizer=optimizer,
            component_evaluator=_quadratic_bias_parts,
        ).relax(
            state,
            fmax=1e-12,
            maxiter=1,
            trajectory_callback=trajectory.append,
        )
        trajectories[optimizer] = trajectory

    np.testing.assert_array_equal(
        trajectories["safe-lbfgs-total"][1].positions,
        trajectories["bias-separated-lbfgs"][1].positions,
    )


def test_bias_separation_changes_only_post_secant_preconditioning():
    state = State(numbers=np.array([1]), positions=np.array([[0.1, 0.0, 0.0]]))
    results = {}

    for optimizer in ("safe-lbfgs-total", "bias-separated-lbfgs"):
        results[optimizer] = Relaxer(
            _total_from_parts,
            optimizer=optimizer,
            component_evaluator=_quadratic_bias_parts,
        ).relax(state, fmax=1e-12, maxiter=2)

    total = results["safe-lbfgs-total"]
    separated = results["bias-separated-lbfgs"]
    assert total.telemetry.accepted_secants == 2
    assert separated.telemetry.accepted_secants == 2
    assert total.telemetry.bias_secant_curvature_sum != 0.0
    assert separated.telemetry.bias_secant_curvature_sum != 0.0
    assert not np.array_equal(total.state.positions, separated.state.positions)


def test_custom_modes_are_identical_when_bias_gradient_is_zero():
    def unbiased_parts(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        return RelaxEvaluation(
            true_energy=0.5 * float(np.dot(flat, flat)),
            true_gradient=flat.copy(),
            bias_energy=0.0,
            bias_gradient=np.zeros_like(flat),
            softening_energy=0.0,
            softening_gradient=np.zeros_like(flat),
            total_energy=0.5 * float(np.dot(flat, flat)),
            total_gradient=flat.copy(),
        )

    def total(flat_positions, template):
        parts = unbiased_parts(flat_positions, template)
        return parts.total_energy, parts.total_gradient.copy()

    state = State(numbers=np.array([1]), positions=np.array([[0.5, 0.0, 0.0]]))
    results = [
        Relaxer(total, optimizer=optimizer, component_evaluator=unbiased_parts).relax(
            state,
            fmax=1e-12,
            maxiter=4,
        )
        for optimizer in ("safe-lbfgs-total", "bias-separated-lbfgs")
    ]

    np.testing.assert_array_equal(results[0].state.positions, results[1].state.positions)
    assert results[0].telemetry.backend_evaluations == results[1].telemetry.backend_evaluations
    assert results[0].telemetry.accepted_secants == results[1].telemetry.accepted_secants


@pytest.mark.parametrize(
    ("optimizer", "history_limit"),
    [
        ("safe-lbfgs-total", 0),
        ("safe-lbfgs-total", None),
        ("bias-separated-lbfgs", None),
    ],
)
def test_custom_lbfgs_clears_history_on_mic_branch_change(optimizer, history_limit):
    def component_evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        signature = ((0, 0, 0),) if flat[0] >= 0.9 else ((1, 0, 0),)
        return RelaxEvaluation(
            true_energy=100.0 * float(flat[0]),
            true_gradient=np.array([100.0, 0.0, 0.0]),
            bias_energy=0.0,
            bias_gradient=np.zeros(3),
            softening_energy=0.0,
            softening_gradient=np.zeros(3),
            total_energy=100.0 * float(flat[0]),
            total_gradient=np.array([100.0, 0.0, 0.0]),
            bias_image_signature=signature,
        )

    def total(flat_positions, template):
        parts = component_evaluator(flat_positions, template)
        return parts.total_energy, parts.total_gradient.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(
        total,
        optimizer=optimizer,
        component_evaluator=component_evaluator,
    ).relax(
        state,
        fmax=1e-12,
        maxiter=1,
        _safe_lbfgs_history_limit=history_limit,
    )

    assert result.telemetry.accepted_steps == 1
    assert result.telemetry.mic_branch_resets == 1
    assert result.telemetry.accepted_secants == 0
    assert result.telemetry.rejected_secants == 1


@pytest.mark.parametrize(
    ("history_limit", "expected_history_lengths"),
    [(1, [0, 1, 1, 0]), (10, [0, 1, 2, 0])],
)
def test_safe_lbfgs_history_depth_clears_on_mic_branch_change(
    monkeypatch,
    history_limit,
    expected_history_lengths,
):
    inverse_product = relax_module._lbfgs_inverse_product
    history_lengths = []

    def spy_inverse_product(gradient, history):
        history_lengths.append(len(history))
        return inverse_product(gradient, history)

    monkeypatch.setattr(relax_module, "_lbfgs_inverse_product", spy_inverse_product)

    def component_evaluator(flat_positions, template):
        flat = np.asarray(flat_positions, dtype=float)
        signature = ((0, 0, 0),) if flat[0] >= 0.7 else ((1, 0, 0),)
        gradient = flat.copy()
        return RelaxEvaluation(
            true_energy=0.5 * float(np.dot(flat, flat)),
            true_gradient=gradient,
            bias_energy=0.0,
            bias_gradient=np.zeros_like(flat),
            softening_energy=0.0,
            softening_gradient=np.zeros_like(flat),
            total_energy=0.5 * float(np.dot(flat, flat)),
            total_gradient=gradient,
            bias_image_signature=signature,
        )

    def total_evaluator(flat_positions, template):
        parts = component_evaluator(flat_positions, template)
        return parts.total_energy, parts.total_gradient.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(
        total_evaluator,
        optimizer="safe-lbfgs-total",
        component_evaluator=component_evaluator,
    ).relax(
        state,
        fmax=1e-12,
        maxiter=4,
        _safe_lbfgs_history_limit=history_limit,
    )

    assert result.n_iter == 4
    assert result.telemetry.mic_branch_resets == 1
    assert history_lengths == expected_history_lengths
