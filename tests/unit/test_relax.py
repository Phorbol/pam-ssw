import numpy as np
import pytest

from pamssw.relax import Relaxer
from pamssw.result import RelaxOutcomeClass
from pamssw.state import State


def test_relaxer_passes_force_tolerance_to_lbfgsb(monkeypatch):
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

    assert captured["options"]["gtol"] == 1e-4
    assert captured["options"]["ftol"] < 1e-9
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


def test_relaxer_can_use_ase_fire_without_scipy_line_search():
    def evaluator(flat_positions, template):
        return 0.5 * float(np.dot(flat_positions, flat_positions)), flat_positions.copy()

    state = State(numbers=np.array([1]), positions=np.array([[1.0, 0.0, 0.0]]))
    result = Relaxer(evaluator, optimizer="ase-fire").relax(state, fmax=1e-4, maxiter=200)

    assert result.gradient_norm < 1e-4
    assert result.energy < 1e-8
    assert result.n_iter > 0


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
