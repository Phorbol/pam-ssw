"""Budgeted block Krylov--Ritz algebra for directional curvature probes."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, TypeAlias

import numpy as np
from scipy.optimize import brentq


Hvp: TypeAlias = Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]]

_ORTHOGONALIZATION_TOLERANCE = 1e-12


def _readonly_float_copy(values: np.ndarray) -> np.ndarray:
    owned = np.array(values, dtype=float, copy=True)
    owned.setflags(write=False)
    readonly = owned.view()
    readonly.setflags(write=False)
    return readonly


@dataclass(frozen=True, eq=False)
class IntentBlock:
    """A defensively owned collection of initial Krylov directions."""

    basis: np.ndarray
    pair: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        basis = np.array(self.basis, dtype=float, copy=True)
        if basis.ndim != 2 or 0 in basis.shape:
            raise ValueError("basis must be a non-empty two-dimensional column matrix")
        if not np.all(np.isfinite(basis)):
            raise ValueError("basis must contain only finite values")

        pair = self.pair
        if pair is not None:
            if not isinstance(pair, tuple) or len(pair) != 2:
                raise ValueError("pair must be a two-element tuple or None")
            if any(isinstance(index, bool) or not isinstance(index, Integral) for index in pair):
                raise ValueError("pair indices must be integers")
            pair = (int(pair[0]), int(pair[1]))
            if pair[0] < 0 or pair[1] < 0 or pair[0] == pair[1]:
                raise ValueError("pair indices must be distinct and non-negative")

        object.__setattr__(self, "basis", _readonly_float_copy(basis))
        object.__setattr__(self, "pair", pair)


@dataclass(frozen=True, eq=False)
class KrylovRitzPoint:
    """One projected Ritz pair reconstructed from the stored Krylov products."""

    direction: np.ndarray
    curvature: float
    true_curvature: float
    residual_norm: float
    initial_span_overlap: float
    reference_abs_overlap: float | None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "direction",
            _readonly_float_copy(self.direction),
        )


@dataclass(frozen=True, eq=False)
class KrylovResult:
    """The lowest Ritz pair, full spectrum, and budget diagnostics."""

    direction: np.ndarray
    curvature: float
    true_curvature: float
    residual_norm: float
    initial_span_overlap: float
    antisymmetry: float
    dimension: int
    initial_rank: int
    hvp_count: int
    termination_reason: str
    ritz_points: tuple[KrylovRitzPoint, ...] = ()

    def __post_init__(self) -> None:
        direction = np.array(self.direction, dtype=float, copy=True)
        if direction.ndim != 1 or direction.size == 0:
            raise ValueError("direction must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(direction)):
            raise ValueError("direction must contain only finite values")
        object.__setattr__(self, "direction", _readonly_float_copy(direction))


@dataclass(frozen=True, eq=False)
class EnergyBoundedAnchorResult:
    """Closest anchor direction satisfying a projected curvature budget."""

    direction: np.ndarray
    feasible: bool
    active: bool
    overlap: float
    true_curvature: float
    exact_anchor_curvature: float
    curvature_limit: float

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "direction",
            _readonly_float_copy(self.direction),
        )


def select_energy_bounded_anchor(
    basis: np.ndarray,
    true_products: np.ndarray,
    anchor: np.ndarray,
    curvature_limit: float,
) -> EnergyBoundedAnchorResult:
    """Return the paid-subspace direction closest to an anchor below a limit."""

    q = np.asarray(basis, dtype=float)
    true_hq = np.asarray(true_products, dtype=float)
    reference = np.asarray(anchor, dtype=float).reshape(-1)
    limit = float(curvature_limit)
    if (
        q.ndim != 2
        or true_hq.shape != q.shape
        or reference.shape != (q.shape[0],)
        or not np.all(np.isfinite(q))
        or not np.all(np.isfinite(true_hq))
        or not np.all(np.isfinite(reference))
        or not np.isfinite(limit)
    ):
        raise ValueError(
            "basis, true_products, anchor, and curvature_limit must be finite "
            "and shape-compatible"
        )

    reference_norm = float(np.linalg.norm(reference))
    if reference_norm <= _ORTHOGONALIZATION_TOLERANCE:
        raise ValueError("anchor must have nonzero norm")
    reference = reference / reference_norm
    projected_anchor = q.T @ reference
    projected_anchor_norm = float(np.linalg.norm(projected_anchor))
    if projected_anchor_norm <= _ORTHOGONALIZATION_TOLERANCE:
        raise ValueError("anchor must have nonzero projection into the basis")
    projected_anchor = projected_anchor / projected_anchor_norm

    projected_raw = q.T @ true_hq
    projected = 0.5 * (projected_raw + projected_raw.T)
    eigenvalues, eigenvectors = np.linalg.eigh(projected)
    anchor_in_eigenbasis = eigenvectors.T @ projected_anchor

    exact_anchor_curvature = float(
        projected_anchor @ (projected @ projected_anchor)
    )
    numerical_tolerance = 1e-12 * max(
        1.0,
        abs(limit),
        float(np.max(np.abs(eigenvalues))),
    )

    if exact_anchor_curvature <= limit + numerical_tolerance:
        direction = q @ projected_anchor
        direction = direction / np.linalg.norm(direction)
        if float(np.dot(direction, reference)) < 0.0:
            direction = -direction
        return EnergyBoundedAnchorResult(
            direction=direction,
            feasible=True,
            active=False,
            overlap=float(np.dot(direction, reference)),
            true_curvature=float(direction @ (true_hq @ projected_anchor)),
            exact_anchor_curvature=exact_anchor_curvature,
            curvature_limit=limit,
        )

    minimum = float(eigenvalues[0])
    if minimum > limit + numerical_tolerance:
        minimum_mask = np.isclose(
            eigenvalues,
            minimum,
            rtol=0.0,
            atol=numerical_tolerance,
        )
        minimum_space = eigenvectors[:, minimum_mask]
        minimum_projection = minimum_space.T @ projected_anchor
        if float(np.linalg.norm(minimum_projection)) > numerical_tolerance:
            coefficients = minimum_space @ (
                minimum_projection / np.linalg.norm(minimum_projection)
            )
        else:
            coefficients = minimum_space[:, 0]
        direction = q @ coefficients
        direction = direction / np.linalg.norm(direction)
        if float(np.dot(direction, reference)) < 0.0:
            direction = -direction
        return EnergyBoundedAnchorResult(
            direction=direction,
            feasible=False,
            active=True,
            overlap=float(np.dot(direction, reference)),
            true_curvature=float(coefficients @ (projected @ coefficients)),
            exact_anchor_curvature=exact_anchor_curvature,
            curvature_limit=limit,
        )

    lower = float(
        np.nextafter(
            -minimum,
            np.inf,
        )
    )

    def normalized_coefficients(shift: float) -> np.ndarray:
        values = anchor_in_eigenbasis / (eigenvalues + shift)
        return values / np.linalg.norm(values)

    def constrained_curvature(shift: float) -> float:
        coefficients = normalized_coefficients(shift)
        return float(coefficients @ (eigenvalues * coefficients))

    upper = max(1.0, abs(lower) + 1.0)
    while constrained_curvature(upper) < limit:
        upper *= 2.0
    shift = brentq(
        lambda value: constrained_curvature(value) - limit,
        lower,
        upper,
        xtol=1e-14,
        rtol=1e-14,
    )
    eigen_coefficients = normalized_coefficients(float(shift))
    coefficients = eigenvectors @ eigen_coefficients
    direction = q @ coefficients
    direction = direction / np.linalg.norm(direction)
    if float(np.dot(direction, reference)) < 0.0:
        direction = -direction
        coefficients = -coefficients
    true_curvature = float(coefficients @ (projected @ coefficients))
    return EnergyBoundedAnchorResult(
        direction=direction,
        feasible=True,
        active=True,
        overlap=float(np.dot(direction, reference)),
        true_curvature=true_curvature,
        exact_anchor_curvature=exact_anchor_curvature,
        curvature_limit=limit,
    )


def _orthogonalized(vector: np.ndarray, basis: list[np.ndarray]) -> np.ndarray | None:
    candidate = np.array(vector, dtype=float, copy=True)
    for _ in range(2):
        for column in basis:
            candidate -= float(np.dot(column, candidate)) * column
    norm = float(np.linalg.norm(candidate))
    if not np.isfinite(norm) or norm <= _ORTHOGONALIZATION_TOLERANCE:
        return None
    return candidate / norm


def _orthonormal_initial_columns(matrix: np.ndarray) -> list[np.ndarray]:
    columns: list[np.ndarray] = []
    for index in range(matrix.shape[1]):
        column = _orthogonalized(matrix[:, index], columns)
        if column is not None:
            columns.append(column)
    return columns


def _evaluate_hvp(hvp: Hvp, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    returned = hvp(vector.copy())
    if not isinstance(returned, tuple) or len(returned) != 2:
        raise ValueError("hvp must return a (total_hvp, true_hvp) tuple")
    total, true = (np.array(values, dtype=float, copy=True) for values in returned)
    for name, values in (("total_hvp", total), ("true_hvp", true)):
        if values.shape != vector.shape:
            raise ValueError(f"hvp {name} has shape {values.shape}, expected {vector.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"hvp {name} must contain only finite values")
    return total, true


def solve_krylov_block(
    intent: IntentBlock,
    hvp: Hvp,
    depth: int,
    reference_direction: np.ndarray | None = None,
) -> KrylovResult:
    """Return the lowest Ritz vector in a fixed-depth block Krylov space.

    Every basis column that survives into the returned space is evaluated once;
    the stored products provide both curvatures and the explicit residual.
    """

    if not isinstance(intent, IntentBlock):
        raise TypeError("intent must be an IntentBlock")
    if isinstance(depth, bool) or not isinstance(depth, Integral) or depth <= 0:
        raise ValueError("depth must be a positive integer")
    if not callable(hvp):
        raise TypeError("hvp must be callable")

    basis = _orthonormal_initial_columns(intent.basis)
    if not basis:
        raise ValueError("initial basis has numerical rank zero")
    reference = None
    if reference_direction is not None:
        reference = np.asarray(reference_direction, dtype=float)
        if (
            reference.ndim != 1
            or reference.shape[0] != intent.basis.shape[0]
            or not np.all(np.isfinite(reference))
        ):
            raise ValueError(
                "reference_direction must be a finite vector matching the "
                "Krylov dimension"
            )
        reference_norm = float(np.linalg.norm(reference))
        if reference_norm <= _ORTHOGONALIZATION_TOLERANCE:
            raise ValueError("reference_direction must have nonzero norm")
        reference = reference / reference_norm
    initial_basis = list(basis)
    initial_rank = len(initial_basis)

    total_hvps: list[np.ndarray] = []
    true_hvps: list[np.ndarray] = []
    frontier = list(range(initial_rank))
    termination_reason = "depth_reached"

    for level in range(int(depth)):
        next_frontier: list[int] = []
        for index in frontier:
            total_hvp, true_hvp = _evaluate_hvp(hvp, basis[index])
            total_hvps.append(total_hvp)
            true_hvps.append(true_hvp)
            if level < depth - 1:
                next_column = _orthogonalized(total_hvp, basis)
                if next_column is not None:
                    basis.append(next_column)
                    next_frontier.append(len(basis) - 1)

        if level == depth - 1:
            break
        if not next_frontier:
            termination_reason = "krylov_breakdown"
            break
        frontier = next_frontier

    q = np.column_stack(basis)
    total_hq = np.column_stack(total_hvps)
    true_hq = np.column_stack(true_hvps)
    projected_raw = q.T @ total_hq
    antisymmetry = float(
        np.linalg.norm(projected_raw - projected_raw.T)
        / max(1.0, float(np.linalg.norm(projected_raw)))
    )
    projected_symmetric = 0.5 * (projected_raw + projected_raw.T)
    initial_q = np.column_stack(initial_basis)
    _, eigenvectors = np.linalg.eigh(projected_symmetric)
    ritz_points: list[KrylovRitzPoint] = []
    for index in range(eigenvectors.shape[1]):
        coefficients = eigenvectors[:, index]
        direction = q @ coefficients
        direction_norm = float(np.linalg.norm(direction))
        if (
            not np.isfinite(direction_norm)
            or direction_norm <= _ORTHOGONALIZATION_TOLERANCE
        ):
            raise ValueError("projected Ritz vector has numerical rank zero")
        coefficients = coefficients / direction_norm
        direction = direction / direction_norm

        overlaps = initial_q.T @ direction
        if (
            overlaps.size
            and overlaps[int(np.argmax(np.abs(overlaps)))] < 0.0
        ):
            coefficients = -coefficients
            direction = -direction

        total_product = total_hq @ coefficients
        true_product = true_hq @ coefficients
        curvature = float(np.dot(direction, total_product))
        ritz_points.append(
            KrylovRitzPoint(
                direction=direction,
                curvature=curvature,
                true_curvature=float(
                    np.dot(direction, true_product)
                ),
                residual_norm=float(
                    np.linalg.norm(
                        total_product - curvature * direction
                    )
                ),
                initial_span_overlap=float(
                    np.linalg.norm(initial_q.T @ direction)
                ),
                reference_abs_overlap=(
                    None
                    if reference is None
                    else abs(float(np.dot(reference, direction)))
                ),
            )
        )
    selected = ritz_points[0]

    return KrylovResult(
        direction=selected.direction,
        curvature=selected.curvature,
        true_curvature=selected.true_curvature,
        residual_norm=selected.residual_norm,
        initial_span_overlap=selected.initial_span_overlap,
        ritz_points=tuple(ritz_points),
        antisymmetry=antisymmetry,
        dimension=len(basis),
        initial_rank=initial_rank,
        hvp_count=len(total_hvps),
        termination_reason=termination_reason,
    )
