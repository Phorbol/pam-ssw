"""Pure NumPy BRZERO4 history action (research-only).

This reconstructs the recovered uniform-WI matrix action for an already
prepared history. It is not a CBD/rotation driver and does not choose,
normalize, trim, or reset history. ``native_block_sum`` is the recovered
three-component block bilinear form; ``euclidean`` is an explicit alternative.
The caller must select the metric deliberately.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HistoryAction:
    a: np.ndarray
    beta: np.ndarray
    z: np.ndarray
    x: np.ndarray
    residual: float


def _finite_array(value, name, ndim):
    array = np.asarray(value, dtype=float)
    if array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be nonempty and finite")
    return array


def _bilinear(left, right, metric):
    if metric == "euclidean":
        return left.T @ right
    if metric == "native_block_sum":
        if left.shape[0] % 3:
            raise ValueError("native_block_sum requires ndim divisible by 3")
        left_blocks = left.reshape(-1, 3, left.shape[1]).sum(axis=1)
        right_blocks = right.reshape(-1, 3, right.shape[1]).sum(axis=1)
        return left_blocks.T @ right_blocks
    raise ValueError("metric must be 'native_block_sum' or 'euclidean'")


def reconstruct_history_action(
    df,
    u,
    z_old,
    x,
    force,
    g0,
    *,
    weight,
    metric,
):
    """Return the fixed-uniform-WI BRZERO4 history action.

    For ``DF`` with shape ``(ndim, m)`` this computes ``A = I + w^2 G`` where
    ``G`` is the selected bilinear Gram matrix, ``beta = solve(A, I)``,
    ``Z = (w^2 U + Z_old) beta``, and
    ``X = x + g0*F - Z @ gram(DF, F)``.  ``weight`` and ``metric`` are explicit
    because no physical metric or native default is promoted by this helper.
    """
    df = _finite_array(df, "df", 2)
    u = _finite_array(u, "u", 2)
    z_old = _finite_array(z_old, "z_old", 2)
    x = _finite_array(x, "x", 1)
    force = _finite_array(force, "force", 1)
    g0 = _finite_array(g0, "g0", 1)
    if df.shape[1] == 0:
        raise ValueError("history must have at least one column")
    if u.shape != df.shape or z_old.shape != df.shape:
        raise ValueError("df, u, and z_old must have the same shape")
    if x.shape != force.shape or x.shape != g0.shape or x.shape[0] != df.shape[0]:
        raise ValueError("x, force, g0, and df must share ndim")
    try:
        weight_array = np.asarray(weight, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("weight must be a finite scalar") from exc
    if weight_array.ndim != 0 or not np.isfinite(weight_array):
        raise ValueError("weight must be a finite scalar")
    if metric not in {"native_block_sum", "euclidean"}:
        raise ValueError("metric must be 'native_block_sum' or 'euclidean'")

    weight_squared = float(weight_array) ** 2
    gram = _bilinear(df, df, metric)
    a = np.eye(df.shape[1], dtype=float) + weight_squared * gram
    beta = np.linalg.solve(a, np.eye(df.shape[1], dtype=float))
    z = (weight_squared * u + z_old) @ beta
    force_projection = _bilinear(df, force.reshape(-1, 1), metric)[:, 0]
    x_next = x + g0 * force - z @ force_projection
    residual = float(np.linalg.norm(a @ beta - np.eye(df.shape[1]), ord=np.inf))
    return HistoryAction(a=a, beta=beta, z=z, x=x_next, residual=residual)
