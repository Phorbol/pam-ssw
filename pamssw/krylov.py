"""Budgeted block Krylov--Ritz algebra for directional curvature probes."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Callable, TypeAlias

import numpy as np


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
class KrylovResult:
    """The lowest Ritz pair and budget/accounting diagnostics."""

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

    def __post_init__(self) -> None:
        direction = np.array(self.direction, dtype=float, copy=True)
        if direction.ndim != 1 or direction.size == 0:
            raise ValueError("direction must be a non-empty one-dimensional vector")
        if not np.all(np.isfinite(direction)):
            raise ValueError("direction must contain only finite values")
        object.__setattr__(self, "direction", _readonly_float_copy(direction))


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


def solve_krylov_block(intent: IntentBlock, hvp: Hvp, depth: int) -> KrylovResult:
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
    eigenvalues, eigenvectors = np.linalg.eigh(projected_symmetric)
    coefficients = eigenvectors[:, int(np.argmin(eigenvalues))]
    direction = q @ coefficients
    direction_norm = float(np.linalg.norm(direction))
    if not np.isfinite(direction_norm) or direction_norm <= _ORTHOGONALIZATION_TOLERANCE:
        raise ValueError("projected Ritz vector has numerical rank zero")
    coefficients = coefficients / direction_norm
    direction = direction / direction_norm

    initial_q = np.column_stack(initial_basis)
    overlaps = initial_q.T @ direction
    if overlaps.size and overlaps[int(np.argmax(np.abs(overlaps)))] < 0.0:
        coefficients = -coefficients
        direction = -direction

    total_product = total_hq @ coefficients
    true_product = true_hq @ coefficients
    curvature = float(np.dot(direction, total_product))
    true_curvature = float(np.dot(direction, true_product))
    residual_norm = float(np.linalg.norm(total_product - curvature * direction))
    initial_span_overlap = float(np.linalg.norm(initial_q.T @ direction))

    return KrylovResult(
        direction=direction,
        curvature=curvature,
        true_curvature=true_curvature,
        residual_norm=residual_norm,
        initial_span_overlap=initial_span_overlap,
        antisymmetry=antisymmetry,
        dimension=len(basis),
        initial_rank=initial_rank,
        hvp_count=len(total_hvps),
        termination_reason=termination_reason,
    )
