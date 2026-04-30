from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pbc import mic_displacement


@dataclass(frozen=True)
class GaussianBiasTerm:
    center: np.ndarray
    direction: np.ndarray
    sigma: float
    weight: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=float).reshape(-1)
        direction = np.asarray(self.direction, dtype=float).reshape(-1)
        norm = np.linalg.norm(direction)
        if norm == 0.0:
            raise ValueError("direction cannot be zero")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "direction", direction / norm)
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

    def evaluate(
        self,
        flat_positions: np.ndarray,
        cell: np.ndarray | None = None,
        pbc: tuple[bool, bool, bool] = (False, False, False),
    ) -> tuple[float, np.ndarray]:
        flat = np.asarray(flat_positions, dtype=float).reshape(-1)
        if flat.shape != self.center.shape:
            raise ValueError("flat_positions must have the same shape as the Gaussian bias center")
        positions = flat.reshape(-1, 3)
        center = self.center.reshape(-1, 3)
        if cell is not None and any(pbc):
            delta = mic_displacement(positions, center, cell, pbc).reshape(-1)
        else:
            delta = flat - self.center
        projection = float(np.dot(delta, self.direction))
        exponent = np.exp(-0.5 * (projection / self.sigma) ** 2)
        energy = self.weight * exponent
        gradient = -(energy * projection / (self.sigma**2)) * self.direction
        return float(energy), gradient

    def directional_curvature_shift(self) -> float:
        return -self.weight / (self.sigma**2)
