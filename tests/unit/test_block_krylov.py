import numpy as np
import pytest

from pamssw.krylov import IntentBlock, solve_krylov_block


def test_depth_two_single_vector_recovers_lowest_ritz_pair_without_extra_hvp():
    hessian = np.array(
        [
            [2.0, -1.0, 0.0],
            [-1.0, 2.0, 0.0],
            [0.0, 0.0, 5.0],
        ]
    )
    calls = 0

    def hvp(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        product = hessian @ vector
        return product, product

    result = solve_krylov_block(
        IntentBlock(np.array([[1.0], [0.0], [0.0]])),
        hvp,
        depth=2,
    )

    expected = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    np.testing.assert_allclose(result.direction, expected, atol=1e-12)
    assert result.curvature == pytest.approx(1.0)
    assert result.true_curvature == pytest.approx(1.0)
    assert result.residual_norm == pytest.approx(0.0, abs=1e-12)
    assert result.initial_span_overlap == pytest.approx(1.0 / np.sqrt(2.0))
    assert result.dimension == 2
    assert result.hvp_count == calls == 2
    assert not result.direction.flags.writeable


def test_rank_deficient_initial_block_reduces_rank_and_stops_on_krylov_breakdown():
    hessian = np.diag([2.0, 5.0])
    calls = 0

    def hvp(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        product = hessian @ vector
        return product, product

    result = solve_krylov_block(
        IntentBlock(np.array([[1.0, 2.0], [0.0, 0.0]])),
        hvp,
        depth=3,
    )

    assert result.initial_rank == 1
    assert result.dimension == 1
    assert result.hvp_count == calls == 1
    assert result.termination_reason == "krylov_breakdown"


def test_true_hvp_products_provide_distinct_true_curvature_without_extra_calls():
    total_hessian = np.diag([1.0, 3.0])
    true_hessian = np.diag([4.0, 9.0])
    calls = 0

    def hvp(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        return total_hessian @ vector, true_hessian @ vector

    result = solve_krylov_block(IntentBlock(np.eye(2)), hvp, depth=1)

    assert result.curvature == pytest.approx(1.0)
    assert result.true_curvature == pytest.approx(4.0)
    assert result.dimension == 2
    assert result.hvp_count == calls == 2


def test_lowest_ritz_curvature_is_nonincreasing_with_krylov_depth():
    hessian = np.array(
        [
            [3.0, -1.0, 0.0],
            [-1.0, 2.0, -1.0],
            [0.0, -1.0, 1.0],
        ]
    )

    def hvp(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        product = hessian @ vector
        return product, product

    curvatures = [
        solve_krylov_block(
            IntentBlock(np.array([[1.0], [0.0], [0.0]])),
            hvp,
            depth=depth,
        ).curvature
        for depth in (1, 2, 3)
    ]

    assert curvatures[1] <= curvatures[0] + 1e-12
    assert curvatures[2] <= curvatures[1] + 1e-12


def test_nonsymmetric_hvp_exposes_projected_antisymmetry():
    operator = np.array([[2.0, 3.0], [-1.0, 4.0]])

    def hvp(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        product = operator @ vector
        return product, product

    result = solve_krylov_block(
        IntentBlock(np.array([[1.0], [0.0]])),
        hvp,
        depth=2,
    )
    expected = np.linalg.norm(operator - operator.T) / max(1.0, np.linalg.norm(operator))

    assert result.antisymmetry == pytest.approx(expected)
    assert result.antisymmetry > 0.0


def test_inputs_are_defensive_and_solver_rejects_invalid_budget_or_hvp():
    source = np.array([[1.0], [0.0]])
    intent = IntentBlock(source, pair=(0, 1))
    source[0, 0] = 99.0

    assert intent.basis[0, 0] == pytest.approx(1.0)
    assert not intent.basis.flags.writeable
    with pytest.raises(ValueError, match="distinct"):
        IntentBlock(np.eye(2), pair=(0, 0))
    with pytest.raises(ValueError, match="depth"):
        solve_krylov_block(intent, lambda vector: (vector, vector), depth=0)
    with pytest.raises(ValueError, match="rank zero"):
        solve_krylov_block(IntentBlock(np.zeros((2, 1))), lambda vector: (vector, vector), depth=1)
    with pytest.raises(ValueError, match="shape"):
        solve_krylov_block(intent, lambda vector: (vector, np.zeros(3)), depth=1)
    with pytest.raises(ValueError, match="finite"):
        solve_krylov_block(intent, lambda vector: (vector, np.full_like(vector, np.nan)), depth=1)


def test_depth_is_accepted_as_a_positional_argument():
    hessian = np.diag([1.0, 3.0])

    def hvp(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        product = hessian @ vector
        return product, product

    result = solve_krylov_block(IntentBlock(np.array([[1.0], [0.0]])), hvp, 1)

    assert result.curvature == pytest.approx(1.0)
