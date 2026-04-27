from __future__ import annotations

import numpy as np

from .state import State


def project_out_rigid_body_modes(state: State, direction: np.ndarray) -> np.ndarray:
    direction = np.asarray(direction, dtype=float).reshape(-1)
    basis = _rigid_body_basis(state)
    if basis.size == 0:
        return direction.copy()
    return direction - basis @ (basis.T @ direction)


def rigid_body_overlap(state: State, direction: np.ndarray) -> float:
    direction = np.asarray(direction, dtype=float).reshape(-1)
    norm_sq = float(direction @ direction)
    if norm_sq <= 1e-24:
        return 0.0
    basis = _rigid_body_basis(state)
    if basis.size == 0:
        return 0.0
    rigid = basis @ (basis.T @ direction)
    return float(np.clip((rigid @ rigid) / norm_sq, 0.0, 1.0))


def _rigid_body_basis(state: State) -> np.ndarray:
    if state.cell is not None or any(state.pbc):
        return np.zeros((state.n_atoms * 3, 0), dtype=float)
    if state.n_atoms < 3:
        return np.zeros((state.n_atoms * 3, 0), dtype=float)

    movable = state.movable_mask
    if np.count_nonzero(movable) < 3:
        return np.zeros((state.n_atoms * 3, 0), dtype=float)

    vectors: list[np.ndarray] = []
    for axis in range(3):
        values = np.zeros((state.n_atoms, 3), dtype=float)
        values[movable, axis] = 1.0
        vectors.append(values.reshape(-1))

    center = state.positions[movable].mean(axis=0, keepdims=True)
    relative = state.positions - center
    axes = np.eye(3)
    for axis in axes:
        values = np.zeros((state.n_atoms, 3), dtype=float)
        values[movable] = np.cross(axis, relative[movable])
        vectors.append(values.reshape(-1))

    matrix = np.stack(vectors, axis=1)
    nonzero = np.linalg.norm(matrix, axis=0) > 1e-12
    if not np.any(nonzero):
        return np.zeros((state.n_atoms * 3, 0), dtype=float)
    q, r = np.linalg.qr(matrix[:, nonzero])
    rank = int(np.count_nonzero(np.abs(np.diag(r)) > 1e-10))
    if rank == 0:
        return np.zeros((state.n_atoms * 3, 0), dtype=float)
    return q[:, :rank]
