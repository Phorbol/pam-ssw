import numpy as np
import pytest

from research.ga_ssw.generalized_ritz_probe import solve


def test_flat_forward_ritz_reserves_final_hvp_budget_and_returns_direction():
    q0 = np.zeros(4)
    anchor = np.array([1.0, 0.0, 0.0, 0.0])
    hessian = np.diag([2.0, 5.0, 7.0, 11.0])
    calls = []

    def evaluate(q):
        q = np.asarray(q)
        calls.append(q.copy())
        return float(.5 * q @ hessian @ q), hessian @ q

    result = solve(q0, anchor, evaluate, 0.0, 1.0e-4, 1.0e-10, 5, "forward")

    assert result.direction.shape == q0.shape
    assert result.force_calls == len(calls) <= 5
    assert result.hvp_calls == result.force_calls - 1
    assert np.isclose(np.linalg.norm(result.direction), 1.0)
    assert result.curvature == pytest.approx(2.0)
    assert result.residual_norm < 1.0e-8


def test_flat_central_ritz_counts_two_force_requests_per_hvp():
    q0 = np.zeros(3)
    anchor = np.array([0.0, 1.0, 0.0])
    hessian = np.diag([3.0, 2.0, 8.0])
    calls = []

    def evaluate(q):
        calls.append(np.array(q, copy=True))
        return float(.5 * q @ hessian @ q), hessian @ q

    result = solve(q0, anchor, evaluate, 0.0, 1.0e-4, 1.0e-10, 6, "central")

    assert result.force_calls == len(calls) <= 6
    assert result.hvp_calls * 2 == result.force_calls
    assert result.curvature == pytest.approx(2.0)


def test_flat_ritz_bias_is_anchor_projector_and_budget_is_validated():
    q0 = np.zeros(2)
    anchor = np.array([1.0, 0.0])

    def evaluate(q):
        return 0.0, np.zeros(2)

    result = solve(q0, anchor, evaluate, 4.0, 1.0e-4, 1.0e-8, 3, "forward")
    assert result.curvature == pytest.approx(-4.0)
    assert result.direction[0] > 0.99
    with pytest.raises(ValueError, match="max_force_calls"):
        solve(q0, anchor, evaluate, 0.0, 1.0e-4, 1.0e-8, 2, "forward")


@pytest.mark.parametrize("scheme", ["forward", "central"])
def test_rotated_quadratic_matches_exact_shifted_hessian(scheme):
    rng = np.random.default_rng(19)
    u, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    h = u @ np.diag([1., 2., 4., 8., 16., 32., 64.]) @ u.T
    anchor = rng.normal(size=7); anchor /= np.linalg.norm(anchor)
    target = h - 10. * np.outer(anchor, anchor)
    def evaluate(q):
        return .5 * q @ h @ q, h @ q
    result = solve(np.zeros(7), anchor, evaluate, 10., 1e-4, 1e-9, 30, scheme)
    assert result.converged
    assert result.curvature == pytest.approx(np.linalg.eigvalsh(target)[0], abs=1e-9)
    assert np.linalg.norm(target @ result.direction - result.curvature * result.direction) < 1e-9
