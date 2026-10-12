import importlib.util
from pathlib import Path
import sys
import types

import numpy as np

RUNNER = Path(__file__).with_name("run.py")
spec = importlib.util.spec_from_file_location("vc_frozen_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def row(q, gradient):
    return {"q": np.asarray(q, dtype=float).copy(), "objective": float(np.dot(q, q)),
            "gradient": np.asarray(gradient, dtype=float).copy(),
            "block_norm": runner.block_norm(gradient, 1)}


def test_block_norm_uses_cell_vector_as_one_block():
    g = np.zeros(9)
    g[-6:] = [0.08, 0.08, 0.08, 0.08, 0.08, 0.08]
    assert runner.block_norm(g, 1) == np.linalg.norm(g[-6:])


def test_initial_shared_target_skips_scipy(monkeypatch):
    scipy = types.ModuleType("scipy")
    optimize = types.ModuleType("scipy.optimize")
    optimize.minimize = lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected optimizer step"))
    scipy.optimize = optimize
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize)
    q = np.zeros(9)
    state = runner.run_scipy_inner(q, lambda x: row(x, np.zeros(9)), 1)
    assert state["status"] == "initial_target_reached"
    assert state["iterations"] == 0


def test_trial_cap_returns_last_callback_accepted_q(monkeypatch):
    scipy = types.ModuleType("scipy")
    optimize = types.ModuleType("scipy.optimize")
    q0, qaccepted, qtrial = np.zeros(9), np.ones(9), np.full(9, 2.0)

    def minimize(fun, x0, **kwargs):
        fun(qaccepted)
        kwargs["callback"](qaccepted)
        fun(qtrial)
        raise AssertionError("trial should be denied")

    optimize.minimize = minimize
    scipy.optimize = optimize
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.optimize", optimize)
    calls = 0
    def evaluate(q):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise runner.PaidCallLimit("test hard cap")
        return row(q, np.full(9, 0.5))

    state = runner.run_scipy_inner(q0, evaluate, 1)
    assert state["status"] == "request_limit"
    np.testing.assert_array_equal(state["q"], qaccepted)


def test_ftol_exit_is_not_shared_target():
    result = types.SimpleNamespace(message="CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH",
                                   success=True)
    status, _ = runner.classify_scipy_result(result, False)
    assert status == "relative_energy_stop"
