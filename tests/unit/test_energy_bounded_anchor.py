from __future__ import annotations

import numpy as np
import pytest

from pamssw.krylov import select_energy_bounded_anchor


def _diagonal_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basis = np.eye(3)
    true_products = np.diag([1.0, 4.0, 9.0])
    anchor = np.ones(3) / np.sqrt(3.0)
    return basis, true_products, anchor


def test_energy_bounded_anchor_returns_exact_anchor_when_feasible() -> None:
    basis, true_products, anchor = _diagonal_problem()

    result = select_energy_bounded_anchor(
        basis=basis,
        true_products=true_products,
        anchor=anchor,
        curvature_limit=5.0,
    )

    np.testing.assert_allclose(result.direction, anchor, atol=1e-12)
    assert result.feasible is True
    assert result.active is False
    assert result.overlap == pytest.approx(1.0, abs=1e-12)
    assert result.true_curvature == pytest.approx(
        result.exact_anchor_curvature,
        abs=1e-12,
    )
    assert result.direction.flags.writeable is False


def test_energy_bounded_anchor_activates_curvature_constraint() -> None:
    basis, true_products, anchor = _diagonal_problem()

    result = select_energy_bounded_anchor(
        basis=basis,
        true_products=true_products,
        anchor=anchor,
        curvature_limit=3.0,
    )

    np.testing.assert_allclose(np.linalg.norm(result.direction), 1.0, atol=1e-12)
    np.testing.assert_allclose(result.true_curvature, 3.0, atol=1e-10)
    assert result.feasible is True
    assert result.active is True
    assert 0.0 < result.overlap < 1.0


def test_energy_bounded_anchor_is_closest_on_dense_three_dimensional_grid() -> None:
    basis, true_products, anchor = _diagonal_problem()
    limit = 3.0

    result = select_energy_bounded_anchor(
        basis=basis,
        true_products=true_products,
        anchor=anchor,
        curvature_limit=limit,
    )

    theta = np.linspace(0.0, np.pi, 721)
    phi = np.linspace(0.0, 2.0 * np.pi, 1440, endpoint=False)
    sin_theta = np.sin(theta)[:, None]
    directions = np.stack(
        np.broadcast_arrays(
            sin_theta * np.cos(phi)[None, :],
            sin_theta * np.sin(phi)[None, :],
            np.cos(theta)[:, None],
        ),
        axis=-1,
    ).reshape(-1, 3)
    curvatures = np.einsum(
        "bi,ij,bj->b",
        directions,
        true_products,
        directions,
    )
    feasible = directions[curvatures <= limit + 1e-12]
    dense_best_overlap = float(np.max(feasible @ anchor))

    assert result.overlap >= dense_best_overlap - 5e-4


def test_energy_bounded_anchor_returns_softest_when_limit_is_infeasible() -> None:
    basis, true_products, anchor = _diagonal_problem()

    result = select_energy_bounded_anchor(
        basis=basis,
        true_products=true_products,
        anchor=anchor,
        curvature_limit=0.5,
    )

    np.testing.assert_allclose(result.direction, np.array([1.0, 0.0, 0.0]))
    assert result.feasible is False
    assert result.active is True
    assert result.true_curvature == 1.0
    assert result.overlap == pytest.approx(float(anchor[0]), abs=1e-12)
