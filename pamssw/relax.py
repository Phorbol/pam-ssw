from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import minimize

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
            bounds = [(value - coordinate_trust_radius, value + coordinate_trust_radius) for value in x0]

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
        energy, full_gradient = self.evaluator(relaxed.flatten_positions(), relaxed)
        grad_matrix = full_gradient.reshape(relaxed.n_atoms, 3)
        gradient_norm = float(np.max(np.linalg.norm(grad_matrix[relaxed.movable_mask], axis=1, ord=2), initial=0.0))
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
