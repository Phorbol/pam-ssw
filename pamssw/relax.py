from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import minimize

from .pbc import wrap_positions
from .result import RelaxResult
from .state import State


class FlatEvaluator(Protocol):
    def __call__(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        ...


@dataclass
class Relaxer:
    evaluator: FlatEvaluator

    def relax(
        self,
        state: State,
        fmax: float,
        maxiter: int,
        coordinate_trust_radius: float | None = None,
    ) -> RelaxResult:
        x0 = state.flatten_active()
        bounds = None
        if coordinate_trust_radius is not None:
            if coordinate_trust_radius <= 0.0:
                raise ValueError("coordinate_trust_radius must be positive")
            bounds = self._coordinate_bounds(state, coordinate_trust_radius)

        def objective(active_flat: np.ndarray) -> tuple[float, np.ndarray]:
            candidate = state.with_active_positions(active_flat)
            energy, full_gradient = self.evaluator(candidate.flatten_positions(), candidate)
            grad_matrix = full_gradient.reshape(candidate.n_atoms, 3)
            return energy, grad_matrix[candidate.movable_mask].reshape(-1)

        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "gtol": fmax, "ftol": 1e-12, "maxls": 50},
        )
        relaxed = state.with_active_positions(np.asarray(result.x, dtype=float))
        if relaxed.cell is not None and any(relaxed.pbc):
            relaxed = State(
                numbers=relaxed.numbers.copy(),
                positions=wrap_positions(relaxed.positions, relaxed.cell, relaxed.pbc),
                cell=relaxed.cell.copy(),
                pbc=relaxed.pbc,
                fixed_mask=relaxed.fixed_mask.copy(),
                metadata=relaxed.metadata.copy(),
            )
        energy, full_gradient = self.evaluator(relaxed.flatten_positions(), relaxed)
        grad_matrix = full_gradient.reshape(relaxed.n_atoms, 3)
        active_gradient = grad_matrix[relaxed.movable_mask].reshape(-1)
        if bounds is not None:
            active_gradient = self._projected_gradient(np.asarray(result.x, dtype=float), active_gradient, bounds)
        gradient_norm = float(
            np.max(np.linalg.norm(active_gradient.reshape(-1, 3), axis=1, ord=2), initial=0.0)
        )
        if gradient_norm > fmax * 20.0:
            # Accept imperfect convergence for rugged proposal surfaces but keep the state.
            n_iter = int(result.nit)
        else:
            n_iter = int(result.nit)
        return RelaxResult(
            state=relaxed,
            energy=float(energy),
            gradient_norm=gradient_norm,
            n_iter=n_iter,
        )

    @staticmethod
    def _projected_gradient(
        active_positions: np.ndarray,
        active_gradient: np.ndarray,
        bounds: list[tuple[float | None, float | None]],
        atol: float = 1e-10,
    ) -> np.ndarray:
        projected = np.asarray(active_gradient, dtype=float).copy()
        for index, (lower, upper) in enumerate(bounds):
            value = active_positions[index]
            grad = projected[index]
            if lower is not None and value <= lower + atol and grad > 0.0:
                projected[index] = 0.0
            elif upper is not None and value >= upper - atol and grad < 0.0:
                projected[index] = 0.0
        return projected

    @staticmethod
    def _coordinate_bounds(state: State, coordinate_trust_radius: float) -> list[tuple[float | None, float | None]]:
        bounds: list[tuple[float | None, float | None]] = []
        for position in state.positions[state.movable_mask]:
            for axis, value in enumerate(position):
                if state.pbc[axis]:
                    bounds.append((None, None))
                else:
                    bounds.append((value - coordinate_trust_radius, value + coordinate_trust_radius))
        return bounds
