from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.optimize import FIRE, LBFGS
import numpy as np
from scipy.optimize import minimize

from .pbc import mic_displacement, wrap_positions
from .result import RelaxOutcomeClass, RelaxResult
from .state import State


class FlatEvaluator(Protocol):
    def __call__(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        ...


RelaxOptimizer = Literal["scipy-lbfgsb", "ase-fire", "ase-lbfgs"]


class _EvaluatorCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, evaluator: FlatEvaluator, template: State):
        super().__init__()
        self.evaluator = evaluator
        self.template = template

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=all_changes) -> None:
        super().calculate(atoms, properties, system_changes)
        if atoms is None:
            raise ValueError("atoms must be provided")
        state = State(
            numbers=self.template.numbers.copy(),
            positions=np.asarray(atoms.get_positions(), dtype=float),
            cell=None if self.template.cell is None else self.template.cell.copy(),
            pbc=self.template.pbc,
            fixed_mask=self.template.fixed_mask.copy(),
            metadata=self.template.metadata.copy(),
        )
        energy, gradient = self.evaluator(state.flatten_positions(), state)
        self.results["energy"] = float(energy)
        self.results["forces"] = -np.asarray(gradient, dtype=float).reshape(state.n_atoms, 3)


@dataclass
class Relaxer:
    evaluator: FlatEvaluator
    optimizer: RelaxOptimizer = "scipy-lbfgsb"

    def relax(
        self,
        state: State,
        fmax: float,
        maxiter: int,
        coordinate_trust_radius: float | None = None,
        trajectory_callback: Callable[[State], None] | None = None,
        trajectory_stride: int = 1,
    ) -> RelaxResult:
        x0 = state.flatten_active()
        if trajectory_stride <= 0:
            raise ValueError("trajectory_stride must be positive")
        bounds = None
        if coordinate_trust_radius is not None:
            if coordinate_trust_radius <= 0.0:
                raise ValueError("coordinate_trust_radius must be positive")
            if self.optimizer == "scipy-lbfgsb":
                bounds = self._coordinate_bounds(state, coordinate_trust_radius)

        if self.optimizer in {"ase-fire", "ase-lbfgs"}:
            relaxed, n_iter = self._relax_with_ase(
                state,
                fmax=fmax,
                maxiter=maxiter,
                trajectory_callback=trajectory_callback,
                trajectory_stride=trajectory_stride,
            )
            energy, full_gradient = self.evaluator(relaxed.flatten_positions(), relaxed)
            grad_matrix = full_gradient.reshape(relaxed.n_atoms, 3)
            active_gradient = grad_matrix[relaxed.movable_mask].reshape(-1)
            active_bound_fraction = 0.0
            displacement_rms, displacement_max = self._displacement_stats(state, relaxed)
            gradient_norm = float(
                np.max(np.linalg.norm(active_gradient.reshape(-1, 3), axis=1, ord=2), initial=0.0)
            )
            initial_energy, _ = self.evaluator(state.flatten_positions(), state)
            return RelaxResult(
                state=relaxed,
                energy=float(energy),
                gradient_norm=gradient_norm,
                n_iter=n_iter,
                active_bound_fraction=active_bound_fraction,
                displacement_rms=displacement_rms,
                displacement_max=displacement_max,
                outcome_class=self.classify_outcome(
                    initial_energy=initial_energy,
                    final_energy=energy,
                    gradient_norm=gradient_norm,
                    fmax=fmax,
                    displacement_rms=displacement_rms,
                    displacement_max=displacement_max,
                    active_bound_fraction=active_bound_fraction,
                ),
            )
        if self.optimizer != "scipy-lbfgsb":
            raise ValueError(f"unsupported relax optimizer: {self.optimizer}")
        return self._relax_with_scipy(
            state,
            fmax=fmax,
            maxiter=maxiter,
            bounds=bounds,
            trajectory_callback=trajectory_callback,
            trajectory_stride=trajectory_stride,
        )

    def _relax_with_scipy(
        self,
        state: State,
        fmax: float,
        maxiter: int,
        bounds: list[tuple[float | None, float | None]] | None,
        trajectory_callback: Callable[[State], None] | None,
        trajectory_stride: int,
    ) -> RelaxResult:
        x0 = state.flatten_active()
        if trajectory_callback is not None:
            trajectory_callback(state)

        def objective(active_flat: np.ndarray) -> tuple[float, np.ndarray]:
            candidate = state.with_active_positions(active_flat)
            energy, full_gradient = self.evaluator(candidate.flatten_positions(), candidate)
            grad_matrix = full_gradient.reshape(candidate.n_atoms, 3)
            return energy, grad_matrix[candidate.movable_mask].reshape(-1)

        def callback(active_flat: np.ndarray) -> None:
            if trajectory_callback is not None:
                callback.count += 1
                if callback.count % trajectory_stride == 0:
                    trajectory_callback(state.with_active_positions(np.asarray(active_flat, dtype=float)))

        callback.count = 0

        minimize_kwargs = {}
        if trajectory_callback is not None:
            minimize_kwargs["callback"] = callback
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": maxiter, "gtol": fmax, "ftol": 1e-12, "maxls": 50},
            **minimize_kwargs,
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
        active_bound_fraction = self._active_bound_fraction(np.asarray(result.x, dtype=float), bounds)
        displacement_rms, displacement_max = self._displacement_stats(state, relaxed)
        gradient_norm = float(
            np.max(np.linalg.norm(active_gradient.reshape(-1, 3), axis=1, ord=2), initial=0.0)
        )
        if gradient_norm > fmax * 20.0:
            # Accept imperfect convergence for rugged proposal surfaces but keep the state.
            n_iter = int(result.nit)
        else:
            n_iter = int(result.nit)
        if trajectory_callback is not None:
            trajectory_callback(relaxed)
        return RelaxResult(
            state=relaxed,
            energy=float(energy),
            gradient_norm=gradient_norm,
            n_iter=n_iter,
            active_bound_fraction=active_bound_fraction,
            displacement_rms=displacement_rms,
            displacement_max=displacement_max,
            outcome_class=self.classify_outcome(
                initial_energy=objective(x0)[0],
                final_energy=energy,
                gradient_norm=gradient_norm,
                fmax=fmax,
                displacement_rms=displacement_rms,
                displacement_max=displacement_max,
                active_bound_fraction=active_bound_fraction,
            ),
        )

    def _relax_with_ase(
        self,
        state: State,
        fmax: float,
        maxiter: int,
        trajectory_callback: Callable[[State], None] | None = None,
        trajectory_stride: int = 1,
    ) -> tuple[State, int]:
        atoms = Atoms(
            numbers=state.numbers,
            positions=state.positions,
            cell=None if state.cell is None else state.cell,
            pbc=state.pbc,
        )
        if np.any(state.fixed_mask):
            atoms.set_constraint(FixAtoms(mask=state.fixed_mask))
        atoms.calc = _EvaluatorCalculator(self.evaluator, state)
        optimizer_cls = FIRE if self.optimizer == "ase-fire" else LBFGS
        optimizer = optimizer_cls(atoms, logfile=None)
        if trajectory_callback is not None:
            trajectory_callback(state)

            def record_step() -> None:
                trajectory_callback(
                    State(
                        numbers=state.numbers.copy(),
                        positions=np.asarray(atoms.get_positions(), dtype=float),
                        cell=None if state.cell is None else state.cell.copy(),
                        pbc=state.pbc,
                        fixed_mask=state.fixed_mask.copy(),
                        metadata=state.metadata.copy(),
                    )
                )

            optimizer.attach(record_step, interval=trajectory_stride)
        optimizer.run(fmax=fmax, steps=maxiter)
        relaxed = State(
            numbers=state.numbers.copy(),
            positions=np.asarray(atoms.get_positions(), dtype=float),
            cell=None if state.cell is None else state.cell.copy(),
            pbc=state.pbc,
            fixed_mask=state.fixed_mask.copy(),
            metadata=state.metadata.copy(),
        )
        if relaxed.cell is not None and any(relaxed.pbc):
            relaxed = State(
                numbers=relaxed.numbers.copy(),
                positions=wrap_positions(relaxed.positions, relaxed.cell, relaxed.pbc),
                cell=relaxed.cell.copy(),
                pbc=relaxed.pbc,
                fixed_mask=relaxed.fixed_mask.copy(),
                metadata=relaxed.metadata.copy(),
            )
        n_iter = int(getattr(optimizer, "nsteps", 0))
        if trajectory_callback is not None:
            trajectory_callback(relaxed)
        return relaxed, n_iter

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

    @staticmethod
    def _active_bound_fraction(
        active_positions: np.ndarray,
        bounds: list[tuple[float | None, float | None]] | None,
        atol: float = 1e-4,
    ) -> float:
        if not bounds or active_positions.size == 0:
            return 0.0
        hits = 0
        finite_bounds = 0
        for index, (lower, upper) in enumerate(bounds):
            value = active_positions[index]
            if lower is not None:
                finite_bounds += 1
                hits += int(abs(value - lower) <= atol)
            if upper is not None:
                finite_bounds += 1
                hits += int(abs(value - upper) <= atol)
        return float(hits / finite_bounds) if finite_bounds else 0.0

    @staticmethod
    def classify_outcome(
        initial_energy: float,
        final_energy: float,
        gradient_norm: float,
        fmax: float,
        displacement_rms: float,
        displacement_max: float,
        active_bound_fraction: float,
        geometry_valid: bool = True,
        energy_explosion_threshold: float = 5.0,
        displacement_threshold: float = 1e-4,
        true_delta: float | None = None,
        true_delta_stagnation_threshold: float = 0.05,
        bound_damage_threshold: float = 0.5,
    ) -> RelaxOutcomeClass:
        if active_bound_fraction >= bound_damage_threshold:
            return RelaxOutcomeClass.DAMAGED
        energy_delta = final_energy - initial_energy if true_delta is None else true_delta
        if not np.isfinite(final_energy) or not np.isfinite(energy_delta) or energy_delta > energy_explosion_threshold:
            return RelaxOutcomeClass.ENERGY_EXPLODED
        if not geometry_valid:
            return RelaxOutcomeClass.GEOMETRY_INVALID
        converged = gradient_norm <= fmax
        moved = max(displacement_rms, displacement_max) >= displacement_threshold
        if true_delta is not None and abs(true_delta) < true_delta_stagnation_threshold:
            return RelaxOutcomeClass.CONVERGED_UNPRODUCTIVE if converged else RelaxOutcomeClass.STAGNATED
        if not moved:
            return RelaxOutcomeClass.CONVERGED_UNPRODUCTIVE if converged else RelaxOutcomeClass.STAGNATED
        if converged:
            return RelaxOutcomeClass.CONVERGED_PRODUCTIVE
        return RelaxOutcomeClass.USEFUL_PROGRESS

    @staticmethod
    def _displacement_stats(reference: State, relaxed: State) -> tuple[float, float]:
        if relaxed.n_atoms == 0:
            return 0.0, 0.0
        displacement = mic_displacement(relaxed.positions, reference.positions, relaxed.cell, relaxed.pbc)
        movable = relaxed.movable_mask
        if not np.any(movable):
            return 0.0, 0.0
        norms = np.linalg.norm(displacement[movable], axis=1)
        return float(np.sqrt(np.mean(norms * norms))), float(np.max(norms, initial=0.0))
