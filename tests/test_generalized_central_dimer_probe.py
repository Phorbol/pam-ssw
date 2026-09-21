import numpy as np
import pytest

from research.ga_ssw.generalized_central_dimer_probe import solve


def test_central_dimer_matches_nondiagonal_quadratic_mode():
    q0 = np.zeros(4)
    anchor = np.array([1.0, 2.0, -1.0, .5])
    hessian = np.array([[4., 1., 0., .2], [1., 3., .4, 0.],
                        [0., .4, 6., 1.], [.2, 0., 1., 5.]])
    beta = .35
    calls = []

    def evaluate(q):
        q = np.asarray(q)
        calls.append(q.copy())
        return float(.5 * q @ hessian @ q), hessian @ q

    result = solve(q0, anchor, rotation_bias=beta, fd_step=1e-5,
                   max_force_calls=100, tol=1e-10, evaluate=evaluate)
    n0 = anchor / np.linalg.norm(anchor)
    expected_values, expected_vectors = np.linalg.eigh(
        hessian - beta * np.outer(n0, n0))
    expected = expected_vectors[:, 0]
    assert result.force_calls == len(calls) <= 100
    assert result.force_calls % 2 == 0
    assert abs(np.dot(result.direction, expected)) > 1 - 1e-8
    assert result.curvature == pytest.approx(expected_values[0], abs=1e-8)
    assert result.residual_norm < 1e-8
    assert all(np.isclose(np.linalg.norm(q), 1e-5) for q in calls)
    assert all(np.allclose(calls[i] + calls[i + 1], 2 * q0)
               for i in range(0, len(calls) - 1, 2))


def test_central_dimer_budget_has_only_two_point_hvps_and_no_center_call():
    q0 = np.zeros(3)
    anchor = np.array([1.0, 1.0, 1.0])
    hessian = np.diag([1.0, 2.0, 3.0])
    calls = []

    def evaluate(q):
        calls.append(np.array(q, copy=True))
        return 0.5 * float(q @ hessian @ q), hessian @ q

    result = solve(q0, anchor, rotation_bias=0.0, fd_step=1e-4,
                   max_force_calls=6, tol=1e-30, evaluate=evaluate)
    assert result.force_calls == len(calls) == 6
    assert result.force_calls == 2 * result.hvp_calls
    assert all(np.isclose(np.linalg.norm(q), 1e-4) for q in calls)


def test_central_dimer_rejects_budget_without_a_complete_hvp():
    with pytest.raises(ValueError, match="max_force_calls"):
        solve(np.zeros(2), np.array([1., 0.]), rotation_bias=0.,
              fd_step=1e-4, max_force_calls=1, tol=1e-3,
              evaluate=lambda q: (0., q))
