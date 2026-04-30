from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, logm

from .pbc import wrap_positions
from .state import State


_CELL_DOF_MODES = {"fixed_cell", "volume_only", "shape_6", "full_9", "slab_xy"}


@dataclass(frozen=True)
class CellDOFMask:
    mode: str = "fixed_cell"

    def __post_init__(self) -> None:
        if self.mode not in _CELL_DOF_MODES:
            raise ValueError("unsupported cell DOF mode")

    @property
    def dof(self) -> int:
        return {
            "fixed_cell": 0,
            "volume_only": 1,
            "shape_6": 6,
            "full_9": 9,
            "slab_xy": 3,
        }[self.mode]

    def pack(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("cell DOF matrix must have shape (3, 3)")
        if self.mode == "fixed_cell":
            return np.zeros(0, dtype=float)
        if self.mode == "volume_only":
            return np.asarray([np.trace(matrix) / 3.0], dtype=float)
        if self.mode == "shape_6":
            return np.asarray(
                [
                    matrix[0, 0],
                    matrix[1, 1],
                    matrix[2, 2],
                    0.5 * (matrix[0, 1] + matrix[1, 0]),
                    0.5 * (matrix[0, 2] + matrix[2, 0]),
                    0.5 * (matrix[1, 2] + matrix[2, 1]),
                ],
                dtype=float,
            )
        if self.mode == "slab_xy":
            return np.asarray(
                [
                    matrix[0, 0],
                    matrix[1, 1],
                    0.5 * (matrix[0, 1] + matrix[1, 0]),
                ],
                dtype=float,
            )
        return matrix.reshape(-1).astype(float)

    def pack_gradient(self, matrix: np.ndarray) -> np.ndarray:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (3, 3):
            raise ValueError("cell gradient matrix must have shape (3, 3)")
        if self.mode == "fixed_cell":
            return np.zeros(0, dtype=float)
        if self.mode == "volume_only":
            return np.asarray([np.trace(matrix)], dtype=float)
        if self.mode == "shape_6":
            return np.asarray(
                [
                    matrix[0, 0],
                    matrix[1, 1],
                    matrix[2, 2],
                    matrix[0, 1] + matrix[1, 0],
                    matrix[0, 2] + matrix[2, 0],
                    matrix[1, 2] + matrix[2, 1],
                ],
                dtype=float,
            )
        if self.mode == "slab_xy":
            return np.asarray([matrix[0, 0], matrix[1, 1], matrix[0, 1] + matrix[1, 0]], dtype=float)
        return matrix.reshape(-1).astype(float)

    def unpack(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        if values.shape != (self.dof,):
            raise ValueError("cell DOF vector has the wrong size")
        matrix = np.zeros((3, 3), dtype=float)
        if self.mode == "fixed_cell":
            return matrix
        if self.mode == "volume_only":
            np.fill_diagonal(matrix, values[0])
            return matrix
        if self.mode == "shape_6":
            matrix[0, 0], matrix[1, 1], matrix[2, 2] = values[:3]
            matrix[0, 1] = matrix[1, 0] = values[3]
            matrix[0, 2] = matrix[2, 0] = values[4]
            matrix[1, 2] = matrix[2, 1] = values[5]
            return matrix
        if self.mode == "slab_xy":
            matrix[0, 0], matrix[1, 1], matrix[0, 1] = values
            matrix[1, 0] = values[2]
            return matrix
        return values.reshape(3, 3)


@dataclass(frozen=True)
class GeneralizedMetric:
    atom_weight: float
    cell_weight: float
    n_active: int
    cell_dof: int

    def __post_init__(self) -> None:
        if self.atom_weight <= 0 or self.cell_weight <= 0:
            raise ValueError("metric weights must be positive")
        if self.n_active < 0 or self.cell_dof < 0:
            raise ValueError("metric dimensions must be non-negative")

    @property
    def atomic_size(self) -> int:
        return 3 * self.n_active

    def norm_sq(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        atomic = values[: self.atomic_size]
        cell = values[self.atomic_size :]
        atom_term = 0.0
        if atomic.size:
            atom_term = self.atom_weight * float(np.mean(atomic * atomic))
        cell_term = self.cell_weight * float(np.sum(cell * cell))
        return float(atom_term + cell_term)

    def norm(self, values: np.ndarray) -> float:
        return float(np.sqrt(max(self.norm_sq(values), 0.0)))

    def dot(self, lhs: np.ndarray, rhs: np.ndarray) -> float:
        lhs = np.asarray(lhs, dtype=float)
        rhs = np.asarray(rhs, dtype=float)
        if lhs.shape != rhs.shape:
            raise ValueError("metric dot inputs must have the same shape")
        atomic_l = lhs[: self.atomic_size]
        atomic_r = rhs[: self.atomic_size]
        cell_l = lhs[self.atomic_size :]
        cell_r = rhs[self.atomic_size :]
        atom_term = 0.0
        if atomic_l.size:
            atom_term = self.atom_weight * float(np.mean(atomic_l * atomic_r))
        return float(atom_term + self.cell_weight * float(np.dot(cell_l, cell_r)))

    def normalized(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        norm = self.norm(values)
        if norm <= 1e-12:
            return values.copy()
        return values / norm

    def project_onto(self, vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
        vector = np.asarray(vector, dtype=float)
        basis = np.asarray(basis, dtype=float)
        if basis.ndim == 1:
            basis = basis[:, None]
        projection = np.zeros_like(vector)
        for column in basis.T:
            denom = self.dot(column, column)
            if denom > 1e-20:
                projection += self.dot(vector, column) / denom * column
        return projection


@dataclass(frozen=True)
class GeneralizedTangentVector:
    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError("generalized tangent values must be one-dimensional")
        object.__setattr__(self, "values", values)

    def atomic_part(self, gcoord: GeneralizedCoordinates) -> np.ndarray:
        return self.values[: gcoord.atomic_size]

    def cell_part(self, gcoord: GeneralizedCoordinates) -> np.ndarray:
        return self.values[gcoord.atomic_size :]

    def normalized(self, metric: GeneralizedMetric) -> GeneralizedTangentVector:
        return GeneralizedTangentVector(metric.normalized(self.values))


@dataclass(frozen=True)
class GeneralizedCoordinates:
    template: State
    cell_ref: np.ndarray
    cell_dof_mask: CellDOFMask
    metric: GeneralizedMetric
    fixed_atom_cell_semantics: str = "fixed_fractional"
    clamp_abs: float = 5.0

    def __post_init__(self) -> None:
        if self.template.cell is None:
            raise ValueError("variable-cell generalized coordinates require a cell")
        cell_ref = np.asarray(self.cell_ref, dtype=float)
        if cell_ref.shape != (3, 3):
            raise ValueError("cell_ref must have shape (3, 3)")
        if abs(np.linalg.det(cell_ref)) <= 1e-12:
            raise ValueError("cell_ref must be non-singular")
        if self.fixed_atom_cell_semantics not in {"fixed_fractional", "fixed_cartesian"}:
            raise ValueError("invalid fixed atom cell semantics")
        if self.metric.n_active != self.n_active or self.metric.cell_dof != self.cell_dof:
            raise ValueError("metric dimensions do not match generalized coordinates")
        object.__setattr__(self, "cell_ref", cell_ref)

    @classmethod
    def from_state(
        cls,
        state: State,
        cell_dof_mode: str,
        *,
        atom_metric_weight: float | None = None,
        cell_metric_weight: float | None = None,
        fixed_atom_cell_semantics: str = "fixed_fractional",
    ) -> GeneralizedCoordinates:
        if state.cell is None:
            raise ValueError("variable-cell generalized coordinates require a cell")
        mask = CellDOFMask(cell_dof_mode)
        n_active = int(np.count_nonzero(state.movable_mask))
        if n_active == 0 and mask.dof == 0:
            raise ValueError("generalized coordinate has no active degrees of freedom")
        lengths = np.linalg.norm(np.asarray(state.cell, dtype=float), axis=1)
        cell_scale = float(np.mean(lengths[lengths > 1e-12])) if np.any(lengths > 1e-12) else 1.0
        metric = GeneralizedMetric(
            atom_weight=float(cell_scale * cell_scale if atom_metric_weight is None else atom_metric_weight),
            cell_weight=float(max(1, n_active) if cell_metric_weight is None else cell_metric_weight),
            n_active=n_active,
            cell_dof=mask.dof,
        )
        return cls(
            template=state,
            cell_ref=np.asarray(state.cell, dtype=float).copy(),
            cell_dof_mask=mask,
            metric=metric,
            fixed_atom_cell_semantics=fixed_atom_cell_semantics,
        )

    @property
    def n_active(self) -> int:
        return int(np.count_nonzero(self.template.movable_mask))

    @property
    def atomic_size(self) -> int:
        return 3 * self.n_active

    @property
    def cell_dof(self) -> int:
        return self.cell_dof_mask.dof

    @property
    def size(self) -> int:
        return self.atomic_size + self.cell_dof

    @property
    def inv_cell_ref(self) -> np.ndarray:
        return np.linalg.inv(self.cell_ref)

    def to_q(self, state: State) -> np.ndarray:
        if state.cell is None:
            raise ValueError("state has no cell")
        fractional = np.asarray(state.positions, dtype=float) @ self.inv_cell_ref
        active_fractional = fractional[self.template.movable_mask].reshape(-1)
        deformation = np.asarray(state.cell, dtype=float) @ self.inv_cell_ref
        U = _real_matrix(logm(deformation))
        U = np.clip(U, -self.clamp_abs, self.clamp_abs)
        return np.concatenate([active_fractional, self.cell_dof_mask.pack(U)])

    def to_state(self, q: np.ndarray) -> State:
        q = self.fractional_wrap(q)
        active_fractional, U_flat = self._split_q(q)
        U = np.clip(self.cell_dof_mask.unpack(U_flat), -self.clamp_abs, self.clamp_abs)
        cell = _real_matrix(expm(U)) @ self.cell_ref
        fractional = self.template.positions @ self.inv_cell_ref
        fractional = np.asarray(fractional, dtype=float)
        fractional[self.template.movable_mask] = active_fractional.reshape(-1, 3)
        positions = fractional @ cell
        if np.any(self.template.fixed_mask) and self.fixed_atom_cell_semantics == "fixed_cartesian":
            positions[self.template.fixed_mask] = self.template.positions[self.template.fixed_mask]
        state = State(
            numbers=self.template.numbers.copy(),
            positions=positions,
            cell=cell,
            pbc=self.template.pbc,
            fixed_mask=self.template.fixed_mask.copy(),
            metadata=self.template.metadata.copy(),
        )
        if any(state.pbc):
            state = State(
                numbers=state.numbers.copy(),
                positions=wrap_positions(state.positions, state.cell, state.pbc),
                cell=state.cell.copy(),
                pbc=state.pbc,
                fixed_mask=state.fixed_mask.copy(),
                metadata=state.metadata.copy(),
            )
        return state

    def displace(self, q: np.ndarray, tangent: GeneralizedTangentVector | np.ndarray, step: float) -> State:
        values = tangent.values if isinstance(tangent, GeneralizedTangentVector) else np.asarray(tangent, dtype=float)
        return self.to_state(np.asarray(q, dtype=float) + step * values)

    def fractional_wrap(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=float).copy()
        if q.shape != (self.size,):
            raise ValueError("generalized coordinate vector has the wrong size")
        if self.atomic_size:
            fractional = q[: self.atomic_size].reshape(-1, 3)
            for axis, periodic in enumerate(self.template.pbc):
                if periodic:
                    fractional[:, axis] = fractional[:, axis] % 1.0
            q[: self.atomic_size] = fractional.reshape(-1)
        return q

    def delta_q(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        delta = np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float)
        if delta.shape != (self.size,):
            raise ValueError("generalized delta has the wrong size")
        if self.atomic_size:
            fractional = delta[: self.atomic_size].reshape(-1, 3)
            for axis, periodic in enumerate(self.template.pbc):
                if periodic:
                    fractional[:, axis] -= np.round(fractional[:, axis])
            delta[: self.atomic_size] = fractional.reshape(-1)
        return delta

    def _split_q(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = np.asarray(q, dtype=float)
        if q.shape != (self.size,):
            raise ValueError("generalized coordinate vector has the wrong size")
        return q[: self.atomic_size], q[self.atomic_size :]


def generalized_rigid_body_basis(gcoord: GeneralizedCoordinates) -> np.ndarray:
    vectors: list[np.ndarray] = []
    for axis, periodic in enumerate(gcoord.template.pbc):
        if not periodic or gcoord.atomic_size == 0:
            continue
        vector = np.zeros(gcoord.size, dtype=float)
        vector[: gcoord.atomic_size].reshape(-1, 3)[:, axis] = 1.0
        vector = vector - gcoord.metric.project_onto(vector, np.column_stack(vectors)) if vectors else vector
        norm = gcoord.metric.norm(vector)
        if norm > 1e-12:
            vectors.append(vector / norm)
    if not vectors:
        return np.zeros((gcoord.size, 0), dtype=float)
    return np.column_stack(vectors)


def project_out_generalized_rigid_modes(gcoord: GeneralizedCoordinates, direction: np.ndarray) -> np.ndarray:
    basis = generalized_rigid_body_basis(gcoord)
    if basis.shape[1] == 0:
        return np.asarray(direction, dtype=float).copy()
    projected = np.asarray(direction, dtype=float) - gcoord.metric.project_onto(direction, basis)
    norm = gcoord.metric.norm(projected)
    if norm <= 1e-12:
        return np.asarray(direction, dtype=float).copy()
    return projected / norm


def _real_matrix(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix)
    if np.iscomplexobj(matrix):
        if np.max(np.abs(matrix.imag)) > 1e-8:
            raise ValueError("cell deformation produced a complex matrix")
        matrix = matrix.real
    return np.asarray(matrix, dtype=float)
