import numpy as np

from research.ga_ssw.lbfgs_baselines import (
    ase_lbfgs_linesearch,
    scipy_lbfgsb,
)


def quadratic(q):
    q = np.asarray(q)
    return 0.5 * float(q @ q), q.copy()


def test_scipy_adapter_returns_common_certificate_and_accepted_trace():
    result = scipy_lbfgsb(
        np.array([3.0, -2.0]), quadratic,
        gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
        convergence_norm=lambda q, g: np.linalg.norm(g), gtol=1e-10, maxiter=30,
        max_requests=100,
    )

    assert result.status == "converged"
    assert result.converged
    assert result.requests <= 100
    assert result.certificate["gradient_norm"] <= 1e-10
    assert result.accepted_trace[0]["accepted"] is True
    assert all(row["accepted"] is True for row in result.accepted_trace)
    assert result.trial_trace[0]["accepted"] is True
    assert any(row["accepted"] is False for row in result.trial_trace)
    np.testing.assert_allclose(result.q, 0.0, atol=1e-8)


def test_scipy_budget_returns_last_accepted_state_and_marks_trials():
    calls = []

    def evaluate(q):
        calls.append(np.array(q, copy=True))
        return quadratic(q)

    result = scipy_lbfgsb(
        np.array([3.0, -2.0]), evaluate,
        gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
        gtol=1e-14, maxiter=30, max_requests=2,
    )

    assert result.status == "request_limit"
    assert result.requests == 2
    np.testing.assert_array_equal(result.q, result.accepted_trace[-1]["q"])
    assert result.trial_trace[-1]["accepted"] is False
    assert result.metadata["api_calls"] == 4
    assert result.metadata["cache_hits"] == 1
    assert result.metadata["denied_requests"] == 1
    assert sum(row['charged'] for row in result.trial_trace) == result.requests


def test_scipy_maxiter_zero_is_a_zero_step_maxiter_status():
    result = scipy_lbfgsb(
        np.array([1.0, 2.0]), quadratic,
        gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
        gtol=1e-12, maxiter=0,
    )
    assert result.steps == 0
    assert result.status == "maxiter"
    assert result.requests == 1
    assert result.metadata["scipy_options"]["maxiter"] == 0


def test_scipy_unsuccessful_early_native_return_is_native_failed(monkeypatch):
    import scipy.optimize

    class Failed:
        success = False
        message = "ABNORMAL_TERMINATION_IN_LNSRCH"
        nit = 0
        status = 2

    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **k: Failed())
    result = scipy_lbfgsb(
        np.array([1.0, 0.0]), quadratic,
        gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
        gtol=1e-12, maxiter=10,
    )
    assert result.native_success is False
    assert result.status == "native_failed"
    assert "ABNORMAL" in result.native_message
    assert result.metadata["scipy_options"]["native_status"] == 2


def test_scipy_unsuccessful_status_one_at_iteration_limit_is_maxiter(monkeypatch):
    import scipy.optimize

    class Limited:
        success = False
        message = "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT"
        nit = 10
        status = 1

    monkeypatch.setattr(scipy.optimize, "minimize", lambda *a, **k: Limited())
    result = scipy_lbfgsb(
        np.array([1.0, 0.0]), quadratic,
        gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
        gtol=1e-12, maxiter=10,
    )
    assert result.status == "maxiter"


def test_failed_oracle_attempt_is_in_trial_trace_and_last_accepted_is_safe():
    def failed(q):
        if q[0] != 1.0:
            raise RuntimeError("oracle failure")
        return quadratic(q)

    result = scipy_lbfgsb(
        np.array([1.0, 0.0]), failed,
        gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
        gtol=1e-12, maxiter=10,
    )
    assert result.status == "evaluation_failed"
    assert result.requests == 2
    assert result.trial_trace[-1]["failed"] is True
    assert result.trial_trace[-1]["charged"] is True
    np.testing.assert_array_equal(result.q, np.array([1.0, 0.0]))


def test_ase_linesearch_adapter_uses_flat_evaluator_and_common_certificate():
    result = ase_lbfgs_linesearch(
        np.array([3.0, -2.0, 1.0]), quadratic,
        gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
        gtol=1e-8, maxiter=30, max_requests=100,
    )

    assert result.status == "converged"
    assert result.converged
    assert result.certificate["gradient_norm"] <= 1e-8
    assert result.steps == len(result.accepted_trace) - 1
    assert result.metadata["ase_parameters"]["fmax"] == 1e-8
    assert result.metadata["ase_parameters"]["maxstep"] == 0.2
    np.testing.assert_allclose(result.q, 0.0, atol=1e-6)


def test_ase_native_fmax_is_explicit_and_observer_does_not_duplicate_initial_state():
    result = ase_lbfgs_linesearch(
        np.array([1.0, 0.0, 0.0]), quadratic,
        gradient_norm=np.linalg.norm, step_norm=np.linalg.norm,
        gtol=1e-12, native_fmax=0.1, maxiter=3,
    )
    assert result.metadata["ase_parameters"]["fmax"] == 0.1
    assert result.steps == len(result.accepted_trace) - 1
    assert result.accepted_trace[0]["requests"] == 1
