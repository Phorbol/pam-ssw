"""Reference-Dimer direction sampling utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class ReferenceDimerResult:
    direction: np.ndarray
    curvature: float
    rotations: int
    dot_initial: float
    converged: bool
    lambda_value: float
    local_pair: tuple[int, int]
    curvature_true: float | None = None
    curvature_biased: float | None = None


class ReferenceDimerRotator:
    """Biased reference-Dimer rotation around a fixed center structure."""

    def __init__(
        self,
        delta: float = 0.005,
        bias_strength: float = 500.0,
        max_steps: int = 15,
        rotation_tol: float = 0.03,
        angular_step: float = 0.05,
    ) -> None:
        for name, value in {
            "delta": delta,
            "bias_strength": bias_strength,
            "max_steps": max_steps,
            "rotation_tol": rotation_tol,
            "angular_step": angular_step,
        }.items():
            if float(value) <= 0.0 or not np.isfinite(float(value)):
                raise ValueError(f"{name} must be positive and finite")
        self.delta = float(delta)
        self.bias_strength = float(bias_strength)
        self.max_steps = int(max_steps)
        self.rotation_tol = float(rotation_tol)
        self.angular_step = float(angular_step)

    def rotate(
        self,
        positions: np.ndarray,
        initial_direction: np.ndarray,
        evaluate_forces: Callable[[np.ndarray], tuple[float, np.ndarray]],
        *,
        initial_forces: np.ndarray | None = None,
        lambda_value: float = 0.0,
        local_pair: tuple[int, int] = (0, 0),
    ) -> ReferenceDimerResult:
        positions = _position_matrix(positions)
        direction = np.asarray(initial_direction, dtype=float)
        if direction.shape != positions.shape:
            raise ValueError("initial_direction must have the same shape as positions")
        initial = _normalize_nonzero(direction, "initial_direction")
        direction = initial.copy()

        if initial_forces is None:
            _, center_forces = evaluate_forces(positions)
        else:
            center_forces = np.asarray(initial_forces, dtype=float)
        center_forces = _force_matrix(center_forces, positions.shape)

        converged = False
        rotations = 0
        previous_direction = direction.copy()

        for rotations in range(1, self.max_steps + 1):
            endpoint = positions + self.delta * direction
            _, endpoint_forces = evaluate_forces(endpoint)
            endpoint_forces = _force_matrix(endpoint_forces, positions.shape)
            biased_endpoint_forces = endpoint_forces + self._bias_force(endpoint, positions, initial)

            center_parallel = float(np.sum(center_forces * direction)) * direction
            endpoint_parallel = float(np.sum(biased_endpoint_forces * direction)) * direction
            rotational_force = (biased_endpoint_forces - endpoint_parallel) - (center_forces - center_parallel)
            rot_norm = float(np.linalg.norm(rotational_force))

            if rot_norm / self.delta < self.rotation_tol:
                converged = True
                break

            rotational_direction = rotational_force / rot_norm
            next_direction = (
                direction * np.cos(self.angular_step)
                + rotational_direction * np.sin(self.angular_step)
            )
            direction = _normalize_nonzero(next_direction, "rotated direction")

            angle_change = float(
                np.arccos(np.clip(float(np.sum(direction * previous_direction)), -1.0, 1.0))
            )
            if angle_change < self.rotation_tol:
                converged = True
                break
            previous_direction = direction.copy()

        endpoint = positions + self.delta * direction
        _, endpoint_forces = evaluate_forces(endpoint)
        endpoint_forces = _force_matrix(endpoint_forces, positions.shape)
        biased_endpoint_forces = endpoint_forces + self._bias_force(endpoint, positions, initial)
        curvature_true = float(np.sum((endpoint_forces - center_forces) * direction) / self.delta)
        curvature_biased = float(np.sum((biased_endpoint_forces - center_forces) * direction) / self.delta)

        return ReferenceDimerResult(
            direction=direction,
            curvature=curvature_biased,
            rotations=rotations,
            dot_initial=float(np.sum(direction * initial)),
            converged=converged,
            lambda_value=float(lambda_value),
            local_pair=(int(local_pair[0]), int(local_pair[1])),
            curvature_true=curvature_true,
            curvature_biased=curvature_biased,
        )

    def _bias_force(self, endpoint: np.ndarray, center: np.ndarray, initial: np.ndarray) -> np.ndarray:
        projection = float(np.sum((endpoint - center) * initial))
        return 4.0 * self.bias_strength * projection * initial


def sample_global_mode(
    positions: np.ndarray,
    masses: np.ndarray | None = None,
    T_rand: float = 300.0,
    rng=None,
) -> np.ndarray:
    """Sample a normalized global Gaussian mode."""
    rng = np.random.default_rng() if rng is None else rng
    positions = _position_matrix(positions)

    mode = rng.normal(0.0, 1.0, positions.shape)
    if masses is not None:
        masses = np.asarray(masses, dtype=float)
        if masses.shape != (positions.shape[0],):
            raise ValueError("masses must have shape (n_atoms,)")
        if np.any(masses <= 0.0) or not np.all(np.isfinite(masses)):
            raise ValueError("masses must be positive and finite")
        mode = mode / np.sqrt(masses)[:, np.newaxis]
    return _normalized_matrix(mode, rng)


def sample_local_bond_mode(
    positions: np.ndarray,
    min_distance: float = 3.0,
    cell: np.ndarray | None = None,
    pbc=None,
    rng=None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Sample a normalized equal/opposite local atom-pair mode."""
    rng = np.random.default_rng() if rng is None else rng
    positions = _position_matrix(positions)
    if min_distance < 0.0 or not np.isfinite(min_distance):
        raise ValueError("min_distance must be finite and non-negative")

    n_atoms = positions.shape[0]
    if n_atoms < 2:
        return _normalized_matrix(np.zeros_like(positions), rng), (0, 0)

    valid_pairs: list[tuple[int, int]] = []
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            delta = _mic(positions[i] - positions[j], cell, pbc)
            if np.linalg.norm(delta) > min_distance:
                valid_pairs.append((i, j))

    if valid_pairs:
        pair_index = int(rng.integers(0, len(valid_pairs)))
        atom_a, atom_b = valid_pairs[pair_index]
    else:
        atom_a = int(rng.integers(0, n_atoms))
        atom_b = atom_a
        while atom_b == atom_a:
            atom_b = int(rng.integers(0, n_atoms))

    delta = _mic(positions[atom_b] - positions[atom_a], cell, pbc)
    mode = np.zeros_like(positions)
    mode[atom_a] = delta
    mode[atom_b] = -delta
    return _normalized_matrix(mode, rng), (atom_a, atom_b)


def sample_mixed_mode(
    positions: np.ndarray,
    masses: np.ndarray | None = None,
    T_rand: float = 300.0,
    min_distance: float = 3.0,
    cell: np.ndarray | None = None,
    pbc=None,
    rng=None,
    lam: float | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Sample normalize(global + lambda * local) from reference SSW semantics."""
    rng = np.random.default_rng() if rng is None else rng
    positions = _position_matrix(positions)
    lambda_value = float(rng.uniform(0.1, 1.5) if lam is None else lam)
    if not np.isfinite(lambda_value):
        raise ValueError("lam must be finite")

    global_mode = sample_global_mode(positions, masses=masses, T_rand=T_rand, rng=rng)
    local_mode, pair = sample_local_bond_mode(
        positions,
        min_distance=min_distance,
        cell=cell,
        pbc=pbc,
        rng=rng,
    )
    mixed = global_mode + lambda_value * local_mode
    mixed_norm = float(np.linalg.norm(mixed))
    direction = mixed / mixed_norm if mixed_norm > 1e-15 else global_mode.copy()
    info: dict[str, object] = {
        "lambda": lambda_value,
        "pair": pair,
        "N_global_norm": float(np.linalg.norm(global_mode)),
        "N_local_norm": float(np.linalg.norm(local_mode)),
    }
    return direction, info


def _normalized_matrix(matrix: np.ndarray, rng) -> np.ndarray:
    values = np.asarray(matrix, dtype=float).copy()
    norm = float(np.linalg.norm(values))
    if norm > 1e-15:
        return values / norm
    retry = rng.normal(0.0, 1.0, values.shape)
    retry_norm = float(np.linalg.norm(retry))
    if retry_norm <= 1e-15:
        raise ValueError("cannot normalize zero direction")
    return retry / retry_norm


def _position_matrix(positions: np.ndarray) -> np.ndarray:
    values = np.asarray(positions, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError("positions must be a finite (n_atoms, 3) array")
    return values


def _normalize_nonzero(matrix: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(matrix, dtype=float).copy()
    norm = float(np.linalg.norm(values))
    if norm <= 1e-15 or not np.isfinite(norm):
        raise ValueError(f"{name} cannot be zero")
    return values / norm


def _force_matrix(forces: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    values = np.asarray(forces, dtype=float)
    if values.shape != shape:
        raise ValueError("forces must have the same shape as positions")
    if not np.all(np.isfinite(values)):
        raise ValueError("forces must be finite")
    return values


def _mic(delta: np.ndarray, cell: np.ndarray | None, pbc=None) -> np.ndarray:
    """Minimum-image displacement with wrapping only on periodic axes."""
    delta = np.asarray(delta, dtype=float)
    if cell is None or pbc is None:
        return delta.copy()

    pbc_array = np.asarray(pbc, dtype=bool)
    if pbc_array.shape == ():
        pbc_array = np.repeat(bool(pbc_array), 3)
    if pbc_array.shape != (3,) or not np.any(pbc_array):
        return delta.copy()

    cell_array = np.asarray(cell, dtype=float)
    if cell_array.shape != (3, 3) or abs(float(np.linalg.det(cell_array))) < 1e-12:
        return delta.copy()

    fractional = delta @ np.linalg.inv(cell_array)
    fractional[..., pbc_array] -= np.round(fractional[..., pbc_array])
    return fractional @ cell_array
