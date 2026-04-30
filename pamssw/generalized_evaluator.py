from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.linalg import expm_frechet

from .calculators import Calculator
from .generalized_coordinates import GeneralizedCoordinates
from .state import State


# 1 eV / A^3 = 160.21766208 GPa, so GPa -> eV / A^3 is the reciprocal.
GPA_TO_EV_PER_A3 = 1.0 / 160.21766208


class GeneralizedEvaluator(Protocol):
    def evaluate_q(self, q: np.ndarray, gcoord: GeneralizedCoordinates) -> tuple[float, np.ndarray]:
        ...


@dataclass
class CalculatorGeneralizedEvaluator:
    calculator: Calculator
    pressure_gpa: float = 0.0
    requires_stress: bool = True
    finite_diff_cell_gradient: bool = False
    finite_diff_eps: float = 1e-5

    @property
    def pressure_ev_per_a3(self) -> float:
        return float(self.pressure_gpa) * GPA_TO_EV_PER_A3

    def evaluate_q(self, q: np.ndarray, gcoord: GeneralizedCoordinates) -> tuple[float, np.ndarray]:
        state = gcoord.to_state(q)
        result = self.calculator.evaluate(state)
        energy = float(result.energy)
        volume = _volume(state)
        if gcoord.cell_dof > 0:
            energy += self.pressure_ev_per_a3 * volume
        grad_atomic = np.asarray(result.gradient, dtype=float) @ np.asarray(state.cell, dtype=float).T
        grad_q = np.zeros(gcoord.size, dtype=float)
        if gcoord.atomic_size:
            grad_q[: gcoord.atomic_size] = grad_atomic[state.movable_mask].reshape(-1)
        if gcoord.cell_dof > 0:
            if self.finite_diff_cell_gradient:
                grad_q[gcoord.atomic_size :] = self._finite_diff_cell_gradient(q, gcoord)
            else:
                if result.stress is None:
                    if self.requires_stress:
                        raise RuntimeError("Calculator did not provide stress for variable-cell coordinates")
                    grad_U = np.zeros((3, 3), dtype=float)
                else:
                    grad_U = self._stress_log_deformation_gradient(
                        state=state,
                        gcoord=gcoord,
                        stress=np.asarray(result.stress, dtype=float),
                        q=np.asarray(q, dtype=float),
                    )
                grad_q[gcoord.atomic_size :] = gcoord.cell_dof_mask.pack_gradient(grad_U)
        return energy, grad_q

    def _stress_log_deformation_gradient(
        self,
        *,
        state: State,
        gcoord: GeneralizedCoordinates,
        stress: np.ndarray,
        q: np.ndarray,
    ) -> np.ndarray:
        if stress.shape != (3, 3):
            raise RuntimeError("stress must have shape (3, 3)")
        U = gcoord.cell_dof_mask.unpack(q[gcoord.atomic_size :])
        volume = _volume(state)
        stress_for_enthalpy = stress + self.pressure_ev_per_a3 * np.eye(3)
        # ASE cell filters use virial = -V * stress as generalized force.
        # This object returns gradients, so the sign is reversed.
        grad_deformation = volume * stress_for_enthalpy
        grad_U = np.zeros((3, 3), dtype=float)
        for row in range(3):
            for col in range(3):
                basis = np.zeros((3, 3), dtype=float)
                basis[row, col] = 1.0
                d_exp = expm_frechet(U, basis, compute_expm=False)
                grad_U[row, col] = float(np.sum(grad_deformation * d_exp))
        return grad_U

    def _finite_diff_cell_gradient(self, q: np.ndarray, gcoord: GeneralizedCoordinates) -> np.ndarray:
        base = np.asarray(q, dtype=float)
        grad_flat = np.zeros(gcoord.cell_dof, dtype=float)
        start = gcoord.atomic_size
        for index in range(gcoord.cell_dof):
            plus = base.copy()
            minus = base.copy()
            plus[start + index] += self.finite_diff_eps
            minus[start + index] -= self.finite_diff_eps
            e_plus = self._energy_only(plus, gcoord)
            e_minus = self._energy_only(minus, gcoord)
            grad_flat[index] = (e_plus - e_minus) / (2.0 * self.finite_diff_eps)
        return grad_flat

    def _energy_only(self, q: np.ndarray, gcoord: GeneralizedCoordinates) -> float:
        state = gcoord.to_state(q)
        result = self.calculator.evaluate(state)
        return float(result.energy) + self.pressure_ev_per_a3 * _volume(state)


def verify_stress_gradient(
    calculator: Calculator,
    state: State,
    *,
    cell_dof_mode: str = "shape_6",
    pressure_gpa: float = 0.0,
    eps: float = 1e-5,
    relative_tol: float = 5e-2,
) -> dict[str, float | bool]:
    gcoord = GeneralizedCoordinates.from_state(state, cell_dof_mode)
    q = gcoord.to_q(state)
    stress_eval = CalculatorGeneralizedEvaluator(
        calculator,
        pressure_gpa=pressure_gpa,
        requires_stress=True,
        finite_diff_cell_gradient=False,
    )
    fd_eval = CalculatorGeneralizedEvaluator(
        calculator,
        pressure_gpa=pressure_gpa,
        requires_stress=False,
        finite_diff_cell_gradient=True,
        finite_diff_eps=eps,
    )
    _, stress_grad = stress_eval.evaluate_q(q, gcoord)
    _, fd_grad = fd_eval.evaluate_q(q, gcoord)
    lhs = stress_grad[gcoord.atomic_size :]
    rhs = fd_grad[gcoord.atomic_size :]
    denom = np.maximum(np.maximum(np.abs(lhs), np.abs(rhs)), 1e-8)
    rel = np.abs(lhs - rhs) / denom
    max_relative_error = float(np.max(rel, initial=0.0))
    max_absolute_error = float(np.max(np.abs(lhs - rhs), initial=0.0))
    return {
        "max_relative_error": max_relative_error,
        "max_absolute_error": max_absolute_error,
        "components_match": bool(max_relative_error <= relative_tol or max_absolute_error <= 1e-5),
    }


def _volume(state: State) -> float:
    if state.cell is None:
        return 0.0
    return float(abs(np.linalg.det(np.asarray(state.cell, dtype=float))))
