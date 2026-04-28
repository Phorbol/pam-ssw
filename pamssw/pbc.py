from __future__ import annotations

import numpy as np


def mic_displacement(
    positions: np.ndarray,
    reference: np.ndarray,
    cell: np.ndarray | None,
    pbc: tuple[bool, bool, bool],
) -> np.ndarray:
    """Return minimum-image displacement from reference positions to positions."""
    delta = np.asarray(positions, dtype=float) - np.asarray(reference, dtype=float)
    if cell is None or not any(pbc):
        return delta
    inverse = np.linalg.inv(cell)
    fractional = delta @ inverse
    for axis, periodic in enumerate(pbc):
        if periodic:
            fractional[..., axis] -= np.round(fractional[..., axis])
    return fractional @ cell


def wrap_positions(
    positions: np.ndarray,
    cell: np.ndarray | None,
    pbc: tuple[bool, bool, bool],
) -> np.ndarray:
    """Wrap periodic coordinates into the unit cell while leaving non-periodic axes unchanged."""
    wrapped = np.asarray(positions, dtype=float).copy()
    if cell is None or not any(pbc):
        return wrapped
    inverse = np.linalg.inv(cell)
    fractional = wrapped @ inverse
    for axis, periodic in enumerate(pbc):
        if periodic:
            fractional[..., axis] -= np.floor(fractional[..., axis])
    return fractional @ cell


def mic_distance_matrix(
    positions: np.ndarray,
    cell: np.ndarray | None,
    pbc: tuple[bool, bool, bool],
) -> np.ndarray:
    """Return pairwise distances using the minimum-image convention on periodic axes."""
    positions = np.asarray(positions, dtype=float)
    delta = positions[None, :, :] - positions[:, None, :]
    if cell is not None and any(pbc):
        inverse = np.linalg.inv(cell)
        fractional = delta @ inverse
        for axis, periodic in enumerate(pbc):
            if periodic:
                fractional[:, :, axis] -= np.round(fractional[:, :, axis])
        delta = fractional @ cell
    return np.linalg.norm(delta, axis=2)
