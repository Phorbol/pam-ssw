from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .generalized_coordinates import GeneralizedCoordinates
from .generalized_evaluator import GeneralizedEvaluator
from .relax import Relaxer
from .result import RelaxResult
from .state import State


@dataclass
class VariableCellRelaxer:
    generalized_evaluator: GeneralizedEvaluator
    gcoord: GeneralizedCoordinates
    max_atom_step: float = 2.0
    max_cell_strain: float = 0.5
    min_volume: float = 1e-6
    max_volume_ratio: float = 10.0
    min_cell_length: float = 0.5

    def relax(
        self,
        state_or_q: State | np.ndarray,
        fmax: float,
        maxiter: int,
        trajectory_callback: Callable[[State], None] | None = None,
        trajectory_stride: int = 1,
    ) -> RelaxResult:
        q0 = self.gcoord.to_q(state_or_q) if isinstance(state_or_q, State) else np.asarray(state_or_q, dtype=float)
        q0 = self.gcoord.fractional_wrap(q0)
        initial_state = self.gcoord.to_state(q0)
        initial_energy = self._safe_energy(q0)
        bounds = self._bounds(q0)
        if trajectory_stride <= 0:
            raise ValueError("trajectory_stride must be positive")
        if trajectory_callback is not None:
            trajectory_callback(initial_state)

        def objective(q: np.ndarray) -> tuple[float, np.ndarray]:
            q = self.gcoord.fractional_wrap(np.asarray(q, dtype=float))
            state = self.gcoord.to_state(q)
            if not self._valid_cell(state):
                return 1e100, np.zeros_like(q)
            energy, gradient = self.generalized_evaluator.evaluate_q(q, self.gcoord)
            if not np.isfinite(energy) or not np.all(np.isfinite(gradient)):
                return 1e100, np.zeros_like(q)
            return float(energy), np.asarray(gradient, dtype=float)

        def callback(q: np.ndarray) -> None:
            if trajectory_callback is None:
                return
            callback.count += 1
            if callback.count % trajectory_stride == 0:
                trajectory_callback(self.gcoord.to_state(q))

        callback.count = 0
        kwargs = {"callback": callback} if trajectory_callback is not None else {}
        result = minimize(
            objective,
            q0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "gtol": fmax, "ftol": 1e-12, "maxls": 50},
            **kwargs,
        )
        q_final = self.gcoord.fractional_wrap(np.asarray(result.x, dtype=float))
        state_final = self.gcoord.to_state(q_final)
        energy, gradient = objective(q_final)
        projected_gradient = Relaxer._projected_gradient(q_final, gradient, bounds)
        active_bound_fraction = Relaxer._active_bound_fraction(q_final, bounds)
        gradient_norm = float(np.max(np.abs(projected_gradient), initial=0.0))
        displacement_rms, displacement_max = Relaxer._displacement_stats(initial_state, state_final)
        if trajectory_callback is not None:
            trajectory_callback(state_final)
        return RelaxResult(
            state=state_final,
            energy=float(energy),
            gradient_norm=gradient_norm,
            n_iter=int(result.nit),
            active_bound_fraction=active_bound_fraction,
            displacement_rms=displacement_rms,
            displacement_max=displacement_max,
            outcome_class=Relaxer.classify_outcome(
                initial_energy=initial_energy,
                final_energy=energy,
                gradient_norm=gradient_norm,
                fmax=fmax,
                displacement_rms=displacement_rms,
                displacement_max=displacement_max,
                active_bound_fraction=active_bound_fraction,
            ),
        )

    def _bounds(self, q0: np.ndarray) -> list[tuple[float | None, float | None]]:
        bounds: list[tuple[float | None, float | None]] = []
        for _atom in range(self.gcoord.n_active):
            for axis in range(3):
                value = q0[len(bounds)]
                if self.gcoord.template.pbc[axis]:
                    bounds.append((None, None))
                else:
                    bounds.append((value - self.max_atom_step, value + self.max_atom_step))
        for value in q0[self.gcoord.atomic_size :]:
            bounds.append((value - self.max_cell_strain, value + self.max_cell_strain))
        return bounds

    def _safe_energy(self, q: np.ndarray) -> float:
        try:
            energy, _ = self.generalized_evaluator.evaluate_q(q, self.gcoord)
        except Exception:
            return float("inf")
        return float(energy)

    def _valid_cell(self, state: State) -> bool:
        if state.cell is None or not np.all(np.isfinite(state.cell)):
            return False
        volume = abs(float(np.linalg.det(state.cell)))
        ref_volume = abs(float(np.linalg.det(self.gcoord.cell_ref)))
        if volume < self.min_volume:
            return False
        if ref_volume > 0 and volume > self.max_volume_ratio * ref_volume:
            return False
        lengths = np.linalg.norm(state.cell, axis=1)
        if np.any(lengths < self.min_cell_length):
            return False
        return True
