from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pbc import wrap_positions
from .state import State


@dataclass(frozen=True)
class TangentVector:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError("tangent values must be one-dimensional")
        object.__setattr__(self, "values", values)

    def normalized(self) -> TangentVector:
        norm = np.linalg.norm(self.values)
        if norm <= 1e-12:
            return TangentVector(self.values.copy())
        return TangentVector(self.values / norm)


@dataclass(frozen=True)
class CartesianCoordinates:
    template: State
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        expected = self.template.n_atoms * 3
        if values.shape != (expected,):
            raise ValueError("coordinate values have the wrong size")
        object.__setattr__(self, "values", values)

    @classmethod
    def from_state(cls, state: State) -> CartesianCoordinates:
        return cls(template=state, values=state.flatten_positions())

    @property
    def size(self) -> int:
        return int(self.values.shape[0])

    @property
    def active_size(self) -> int:
        return int(np.count_nonzero(self.template.movable_mask) * 3)

    def active_values(self) -> np.ndarray:
        return self.template.flatten_active()

    def full_tangent_from_active(self, active_values: np.ndarray) -> TangentVector:
        active_values = np.asarray(active_values, dtype=float)
        if active_values.shape != (self.active_size,):
            raise ValueError("active tangent has the wrong size")
        values = np.zeros(self.size, dtype=float)
        values.reshape(self.template.n_atoms, 3)[self.template.movable_mask] = active_values.reshape(-1, 3)
        return TangentVector(values)

    def to_state(self, values: np.ndarray) -> State:
        return self.template.with_flat_positions(np.asarray(values, dtype=float))

    def displace(self, tangent: TangentVector, step: float) -> State:
        values = self.values + step * tangent.values
        state = self.to_state(values)
        if np.any(self.template.fixed_mask):
            positions = state.positions.copy()
            positions[self.template.fixed_mask] = self.template.positions[self.template.fixed_mask]
            state = State(
                numbers=state.numbers.copy(),
                positions=positions,
                cell=None if state.cell is None else state.cell.copy(),
                pbc=state.pbc,
                fixed_mask=state.fixed_mask.copy(),
                metadata=state.metadata.copy(),
            )
        if state.cell is not None and any(state.pbc):
            state = State(
                numbers=state.numbers.copy(),
                positions=wrap_positions(state.positions, state.cell, state.pbc),
                cell=state.cell.copy(),
                pbc=state.pbc,
                fixed_mask=state.fixed_mask.copy(),
                metadata=state.metadata.copy(),
            )
        return state
