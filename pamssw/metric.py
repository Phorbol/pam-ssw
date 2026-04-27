from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class MetricKind(str, Enum):
    EUCLIDEAN = "euclidean"
    MASS_WEIGHTED = "mass_weighted"
    ATOM_CELL_BLOCK = "atom_cell_block"


@dataclass(frozen=True)
class EuclideanMetric:
    kind: MetricKind = MetricKind.EUCLIDEAN

    def dot(self, lhs: np.ndarray, rhs: np.ndarray) -> float:
        return float(np.dot(lhs, rhs))

    def norm(self, vector: np.ndarray) -> float:
        return float(np.sqrt(self.dot(vector, vector)))


@dataclass(frozen=True)
class MassWeightedMetric:
    atomic_masses: np.ndarray
    kind: MetricKind = MetricKind.MASS_WEIGHTED

    def __post_init__(self) -> None:
        masses = np.asarray(self.atomic_masses, dtype=float)
        if masses.ndim != 1 or np.any(masses <= 0.0):
            raise ValueError("atomic_masses must be a one-dimensional positive array")
        object.__setattr__(self, "atomic_masses", masses)

    def dot(self, lhs: np.ndarray, rhs: np.ndarray) -> float:
        weights = np.repeat(self.atomic_masses, 3)
        if lhs.shape != weights.shape or rhs.shape != weights.shape:
            raise ValueError("vectors must have one xyz block per atom")
        return float(np.dot(weights * lhs, rhs))

    def norm(self, vector: np.ndarray) -> float:
        return float(np.sqrt(self.dot(vector, vector)))


class AtomCellBlockMetric:
    kind: MetricKind = MetricKind.ATOM_CELL_BLOCK

    def __init__(self) -> None:
        raise NotImplementedError("atom-cell block metric requires a variable-cell coordinate system")
