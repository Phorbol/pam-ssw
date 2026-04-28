from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class State:
    """Atomistic structure used by PAM-SSW calculators and walkers."""
    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] = (False, False, False)
    fixed_mask: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        numbers = np.asarray(self.numbers, dtype=int)
        positions = np.asarray(self.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (n_atoms, 3)")
        if numbers.ndim != 1 or numbers.shape[0] != positions.shape[0]:
            raise ValueError("numbers and positions must describe the same atoms")

        cell = None if self.cell is None else np.asarray(self.cell, dtype=float)
        if cell is not None and cell.shape != (3, 3):
            raise ValueError("cell must have shape (3, 3)")

        if self.fixed_mask is None:
            fixed_mask = np.zeros(numbers.shape[0], dtype=bool)
        else:
            fixed_mask = np.asarray(self.fixed_mask, dtype=bool)
            if fixed_mask.shape != (numbers.shape[0],):
                raise ValueError("fixed_mask must have shape (n_atoms,)")

        pbc = tuple(bool(x) for x in self.pbc)
        if len(pbc) != 3:
            raise ValueError("pbc must contain three booleans")

        object.__setattr__(self, "numbers", numbers)
        object.__setattr__(self, "positions", positions)
        object.__setattr__(self, "cell", cell)
        object.__setattr__(self, "fixed_mask", fixed_mask)
        object.__setattr__(self, "pbc", pbc)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_atoms(self) -> int:
        return int(self.numbers.shape[0])

    @property
    def movable_mask(self) -> np.ndarray:
        return ~self.fixed_mask

    def flatten_positions(self) -> np.ndarray:
        return self.positions.reshape(-1).copy()

    def flatten_active(self) -> np.ndarray:
        return self.positions[self.movable_mask].reshape(-1).copy()

    def with_flat_positions(self, flat_positions: np.ndarray) -> State:
        new_positions = np.asarray(flat_positions, dtype=float).reshape(self.n_atoms, 3)
        return State(
            numbers=self.numbers.copy(),
            positions=new_positions,
            cell=None if self.cell is None else self.cell.copy(),
            pbc=self.pbc,
            fixed_mask=self.fixed_mask.copy(),
            metadata=self.metadata.copy(),
        )

    def with_active_positions(self, flat_active: np.ndarray) -> State:
        flat_active = np.asarray(flat_active, dtype=float)
        expected = int(np.count_nonzero(self.movable_mask) * 3)
        if flat_active.shape != (expected,):
            raise ValueError("active coordinate vector has the wrong size")
        positions = self.positions.copy()
        positions[self.movable_mask] = flat_active.reshape(-1, 3)
        return State(
            numbers=self.numbers.copy(),
            positions=positions,
            cell=None if self.cell is None else self.cell.copy(),
            pbc=self.pbc,
            fixed_mask=self.fixed_mask.copy(),
            metadata=self.metadata.copy(),
        )

    def displaced(self, full_direction: np.ndarray, step: float) -> State:
        from .coordinates import CartesianCoordinates, TangentVector

        return CartesianCoordinates.from_state(self).displace(TangentVector(full_direction), step)
