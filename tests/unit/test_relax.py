import numpy as np

from pamssw.relax import Relaxer
from pamssw.state import State


def test_relaxer_passes_force_tolerance_to_lbfgsb(monkeypatch):
    captured = {}

    class Result:
        x = np.array([0.0, 0.0, 0.0])
        nit = 0

    def fake_minimize(fun, x0, method, jac, options):
        captured["options"] = options
        return Result()

    monkeypatch.setattr("pamssw.relax.minimize", fake_minimize)

    def evaluator(flat_positions, template):
        return 0.0, np.zeros_like(flat_positions)

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    Relaxer(evaluator).relax(state, fmax=1e-4, maxiter=123)

    assert captured["options"]["gtol"] == 1e-4
    assert captured["options"]["ftol"] < 1e-9
    assert captured["options"]["maxiter"] == 123
