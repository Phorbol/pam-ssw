import numpy as np

from pamssw.relax import Relaxer
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
