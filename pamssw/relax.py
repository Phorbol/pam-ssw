from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.optimize import FIRE, LBFGS
import numpy as np
from scipy.optimize import minimize

try:
    from ase.optimize import FIRE2 as _ASE_FIRE2
except ImportError:  # pragma: no cover - depends on the installed ASE version
    _ASE_FIRE2 = None

from .pbc import mic_displacement, wrap_positions
from .result import RelaxOutcomeClass, RelaxResult, RelaxTelemetry
from .state import State


class FlatEvaluator(Protocol):
    def __call__(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        ...


class ComponentEvaluator(Protocol):
    def __call__(self, flat_positions: np.ndarray, template: State) -> RelaxEvaluation:
        ...


@dataclass(frozen=True)
class RelaxEvaluation:
    """One proposal-objective evaluation separated into analytic components."""

    true_energy: float
    true_gradient: np.ndarray
    bias_energy: float
    bias_gradient: np.ndarray
    softening_energy: float
    softening_gradient: np.ndarray
    total_energy: float
    total_gradient: np.ndarray
    bias_image_signature: tuple[tuple[int, ...], ...] = ()
    softening_present: bool = False

    def __post_init__(self) -> None:
        gradient_names = (
            "true_gradient",
            "bias_gradient",
            "softening_gradient",
            "total_gradient",
        )
        expected_shape: tuple[int, ...] | None = None
        for name in gradient_names:
            gradient = np.asarray(getattr(self, name), dtype=float)
            if expected_shape is None:
                expected_shape = gradient.shape
            elif gradient.shape != expected_shape:
                raise ValueError("RelaxEvaluation component gradients must have the same shape")
            gradient = gradient.copy()
            gradient.setflags(write=False)
            object.__setattr__(self, name, gradient)


class _EvaluationTrace:
    """Keep backend calls intact while reusing exact endpoints for reporting."""

    def __init__(self, evaluator: FlatEvaluator, initial_flat: np.ndarray) -> None:
        self.evaluator = evaluator
        self.initial_flat = np.asarray(initial_flat, dtype=float).reshape(-1).copy()
        self.initial_result: tuple[float, np.ndarray] | None = None
        self.last_flat: np.ndarray | None = None
        self.last_result: tuple[float, np.ndarray] | None = None
        self.evaluator_calls = 0
        self.backend_evaluations = 0
        self.reporting_cache_hits = 0
        self.reporting_evaluator_calls = 0
        self.finalization_requests = 0
        self.explicit_finalization_calls = 0

    def backend_evaluate(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        result = self._call(flat_positions, template)
        self._record_backend_result(flat_positions, result)
        return self._copy_result(result)

    def backend_evaluate_parts(
        self,
        flat_positions: np.ndarray,
        template: State,
        component_evaluator: ComponentEvaluator,
    ) -> RelaxEvaluation:
        flat = np.asarray(flat_positions, dtype=float).reshape(-1)
        parts = component_evaluator(flat, template)
        if not isinstance(parts, RelaxEvaluation):
            raise TypeError("component_evaluator must return RelaxEvaluation")
        if parts.total_gradient.shape != flat.shape:
            raise ValueError("component total_gradient must match flat_positions")
        self.evaluator_calls += 1
        result = (float(parts.total_energy), parts.total_gradient)
        self._record_backend_result(flat, result)
        return parts

    def _record_backend_result(
        self,
        flat_positions: np.ndarray,
        result: tuple[float, np.ndarray],
    ) -> None:
        self.backend_evaluations += 1
        flat = np.asarray(flat_positions, dtype=float).reshape(-1)
        if np.array_equal(flat, self.initial_flat):
            self.initial_result = self._copy_result(result)
        self.last_flat = flat.copy()
        self.last_result = self._copy_result(result)

    def report_evaluate(
        self,
        flat_positions: np.ndarray,
        template: State,
        *,
        finalization: bool = False,
    ) -> tuple[float, np.ndarray]:
        self.finalization_requests += int(finalization)
        flat = np.asarray(flat_positions, dtype=float).reshape(-1)
        if np.array_equal(flat, self.initial_flat) and self.initial_result is not None:
            self.reporting_cache_hits += 1
            return self._copy_result(self.initial_result)
        if (
            self.last_flat is not None
            and self.last_result is not None
            and np.array_equal(flat, self.last_flat)
        ):
            self.reporting_cache_hits += 1
            return self._copy_result(self.last_result)
        self.reporting_evaluator_calls += 1
        self.explicit_finalization_calls += int(finalization)
        return self._call(flat, template)

    def telemetry(
        self,
        *,
        backend: str,
        converged: bool,
        termination_reason: str,
        optimizer_success: bool | None,
        gradient_measure: str,
        accepted_steps: int = 0,
        rejected_steps: int = 0,
        accepted_secants: int = 0,
        rejected_secants: int = 0,
        line_search_evaluations: int = 0,
        mic_branch_resets: int = 0,
        bias_secant_curvature_sum: float = 0.0,
    ) -> RelaxTelemetry:
        return RelaxTelemetry(
            backend=backend,
            evaluator_calls=self.evaluator_calls,
            backend_evaluations=self.backend_evaluations,
            reporting_cache_hits=self.reporting_cache_hits,
            reporting_evaluator_calls=self.reporting_evaluator_calls,
            finalization_requests=self.finalization_requests,
            explicit_finalization_calls=self.explicit_finalization_calls,
            gradient_measure=gradient_measure,
            converged=converged,
            termination_reason=termination_reason,
            optimizer_success=optimizer_success,
            accepted_steps=accepted_steps,
            rejected_steps=rejected_steps,
            accepted_secants=accepted_secants,
            rejected_secants=rejected_secants,
            line_search_evaluations=line_search_evaluations,
            mic_branch_resets=mic_branch_resets,
            bias_secant_curvature_sum=bias_secant_curvature_sum,
        )

    def _call(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        flat = np.asarray(flat_positions, dtype=float).reshape(-1)
        energy, gradient = self.evaluator(flat, template)
        gradient = np.asarray(gradient, dtype=float).reshape(-1)
        if gradient.shape != flat.shape:
            raise ValueError("evaluator gradient must have the same shape as flat_positions")
        self.evaluator_calls += 1
        return float(energy), gradient.copy()

    @staticmethod
    def _copy_result(result: tuple[float, np.ndarray]) -> tuple[float, np.ndarray]:
        return float(result[0]), np.asarray(result[1], dtype=float).copy()


_SAFE_LBFGS_MEMORY = 10
_SAFE_LBFGS_EMPTY_HISTORY_SCALE = 1.0 / 70.0
_SAFE_LBFGS_MAX_ATOM_STEP = 0.2
_SAFE_LBFGS_ARMIJO_C1 = 1.0e-4
_SAFE_LBFGS_BACKTRACK = 0.5
_SAFE_LBFGS_MAX_LINE_TRIALS = 20
_SAFE_LBFGS_MIN_ALPHA = 2.0**-20
_SAFE_LBFGS_CURVATURE_REL = float(np.sqrt(np.finfo(float).eps))


def _lbfgs_inverse_product(
    gradient: np.ndarray,
    history: list[tuple[np.ndarray, np.ndarray, float]],
    *,
    scale_pair: tuple[np.ndarray, np.ndarray, float] | None = None,
) -> np.ndarray:
    """Apply the standard limited-memory inverse-BFGS two-loop recursion."""

    q = np.asarray(gradient, dtype=float).reshape(-1).copy()
    alphas: list[float] = []
    for s, y, rho in reversed(history):
        alpha = float(rho * np.dot(s, q))
        alphas.append(alpha)
        q -= alpha * y
    scaling_pair = history[-1] if history else scale_pair
    if scaling_pair is not None:
        latest_s, latest_y, _ = scaling_pair
        gamma = float(np.dot(latest_s, latest_y) / np.dot(latest_y, latest_y))
    else:
        gamma = _SAFE_LBFGS_EMPTY_HISTORY_SCALE
    result = gamma * q
    for (s, y, rho), alpha in zip(history, reversed(alphas), strict=True):
        beta = float(rho * np.dot(y, result))
        result += s * (alpha - beta)
    return result


def _accept_lbfgs_curvature(s: np.ndarray, y: np.ndarray) -> bool:
    """Accept only numerically positive secant curvature."""

    s = np.asarray(s, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    curvature = float(np.dot(s, y))
    threshold = _SAFE_LBFGS_CURVATURE_REL * float(np.linalg.norm(s)) * float(np.linalg.norm(y))
    return bool(np.isfinite(curvature) and curvature > threshold)


def _limit_max_atomic_displacement(direction: np.ndarray) -> np.ndarray:
    """Scale one direction so no movable atom exceeds the shared step limit."""

    direction = np.asarray(direction, dtype=float).reshape(-1).copy()
    if direction.size % 3 != 0:
        raise ValueError("active direction length must be divisible by three")
    max_norm = float(
        np.max(np.linalg.norm(direction.reshape(-1, 3), axis=1), initial=0.0)
    )
    if max_norm > _SAFE_LBFGS_MAX_ATOM_STEP:
        direction *= _SAFE_LBFGS_MAX_ATOM_STEP / max_norm
    return direction


RelaxOptimizer = Literal[
    "scipy-lbfgsb",
    "ase-fire",
    "ase-fire2",
    "ase-lbfgs",
    "safe-lbfgs-total",
    "bias-separated-lbfgs",
]


def _resolve_safe_lbfgs_history_limit(
    history_limit: int | None,
    optimizer: RelaxOptimizer,
) -> int:
    if history_limit is None:
        return _SAFE_LBFGS_MEMORY
    if (
        isinstance(history_limit, bool)
        or not isinstance(history_limit, int)
        or history_limit not in {0, 1, _SAFE_LBFGS_MEMORY}
    ):
        raise ValueError("_safe_lbfgs_history_limit must be None, 0, 1, or 10")
    if optimizer != "safe-lbfgs-total":
        raise ValueError(
            "_safe_lbfgs_history_limit is only supported for optimizer='safe-lbfgs-total'"
        )
    return history_limit


def _resolve_safe_lbfgs_adaptive_scale_without_history(
    enabled: bool,
    optimizer: RelaxOptimizer,
    history_limit: int | None,
) -> bool:
    if type(enabled) is not bool:
        raise TypeError(
            "_safe_lbfgs_adaptive_scale_without_history adaptive scale flag "
            "must be a literal bool"
        )
    if not enabled:
        return False
    if optimizer != "safe-lbfgs-total" or history_limit != 0:
        raise ValueError(
            "adaptive scale without history requires optimizer='safe-lbfgs-total' "
            "and explicit _safe_lbfgs_history_limit=0"
        )
    return True


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


@dataclass(frozen=True)
class CertificateFallbackResult:
    """Primary relaxation and its optional certificate-triggered fallback."""

    primary: RelaxResult
    fallback: RelaxResult | None
    final: RelaxResult
    fallback_used: bool


def has_force_convergence_certificate(result: RelaxResult, fmax: float) -> bool:
    """Return whether the reported active-atom force norm is finite and converged."""

    return bool(np.isfinite(result.gradient_norm) and result.gradient_norm <= fmax)


def relax_with_certificate_fallback(
    primary_relaxer: Relaxer,
    state: State,
    *,
    fmax: float,
    maxiter: int,
    fallback_relaxer: Relaxer | None = None,
    on_fallback_start: Callable[[], None] | None = None,
    trajectory_callback: Callable[[State], None] | None = None,
    trajectory_stride: int = 1,
) -> CertificateFallbackResult:
    """Run one fallback only when the primary result lacks a force certificate."""

    relax_kwargs = {
        "fmax": fmax,
        "maxiter": maxiter,
        "trajectory_callback": trajectory_callback,
        "trajectory_stride": trajectory_stride,
    }
    primary = primary_relaxer.relax(state, **relax_kwargs)
    if fallback_relaxer is None or has_force_convergence_certificate(primary, fmax):
        return CertificateFallbackResult(
            primary=primary,
            fallback=None,
            final=primary,
            fallback_used=False,
        )
    if on_fallback_start is not None:
        on_fallback_start()
    fallback = fallback_relaxer.relax(primary.state, **relax_kwargs)
    return CertificateFallbackResult(
        primary=primary,
        fallback=fallback,
        final=fallback,
        fallback_used=True,
    )


@dataclass
class Relaxer:
    evaluator: FlatEvaluator
    optimizer: RelaxOptimizer = "scipy-lbfgsb"
    component_evaluator: ComponentEvaluator | None = None

    def relax(
        self,
        state: State,
        fmax: float,
        maxiter: int,
        coordinate_trust_radius: float | None = None,
        trajectory_callback: Callable[[State], None] | None = None,
        trajectory_stride: int = 1,
        *,
        _safe_lbfgs_history_limit: int | None = None,
        _safe_lbfgs_adaptive_scale_without_history: bool = False,
    ) -> RelaxResult:
        adaptive_scale_without_history = (
            _resolve_safe_lbfgs_adaptive_scale_without_history(
                _safe_lbfgs_adaptive_scale_without_history,
                self.optimizer,
                _safe_lbfgs_history_limit,
            )
        )
        history_limit = _resolve_safe_lbfgs_history_limit(
            _safe_lbfgs_history_limit,
            self.optimizer,
        )
        trace = _EvaluationTrace(self.evaluator, state.flatten_positions())
        if trajectory_stride <= 0:
            raise ValueError("trajectory_stride must be positive")
        bounds = None
        if coordinate_trust_radius is not None:
            if coordinate_trust_radius <= 0.0:
                raise ValueError("coordinate_trust_radius must be positive")
            if self.optimizer == "scipy-lbfgsb":
                bounds = self._coordinate_bounds(state, coordinate_trust_radius)

        if self.optimizer in {"ase-fire", "ase-fire2", "ase-lbfgs"}:
            relaxed, n_iter, optimizer_success = self._relax_with_ase(
                state,
                trace=trace,
                fmax=fmax,
                maxiter=maxiter,
                trajectory_callback=trajectory_callback,
                trajectory_stride=trajectory_stride,
            )
            energy, full_gradient = trace.report_evaluate(
                relaxed.flatten_positions(),
                relaxed,
                finalization=True,
            )
            grad_matrix = full_gradient.reshape(relaxed.n_atoms, 3)
            active_gradient = grad_matrix[relaxed.movable_mask].reshape(-1)
            active_bound_fraction = 0.0
            displacement_rms, displacement_max = self._displacement_stats(state, relaxed)
            gradient_norm = float(
                np.max(np.linalg.norm(active_gradient.reshape(-1, 3), axis=1, ord=2), initial=0.0)
            )
            initial_energy, _ = trace.report_evaluate(state.flatten_positions(), state)
            converged = gradient_norm <= fmax
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
                telemetry=trace.telemetry(
                    backend=self.optimizer,
                    converged=converged,
                    termination_reason=self._termination_reason(
                        converged=converged,
                        n_iter=n_iter,
                        maxiter=maxiter,
                        optimizer_success=optimizer_success,
                    ),
                    optimizer_success=optimizer_success,
                    gradient_measure="raw_active_max_force",
                ),
            )
        if self.optimizer in {"safe-lbfgs-total", "bias-separated-lbfgs"}:
            if self.optimizer == "bias-separated-lbfgs" and self.component_evaluator is None:
                raise ValueError("bias-separated-lbfgs requires a component evaluator")
            return self._relax_with_safe_lbfgs(
                state,
                fmax=fmax,
                maxiter=maxiter,
                trace=trace,
                trajectory_callback=trajectory_callback,
                trajectory_stride=trajectory_stride,
                history_limit=history_limit,
                adaptive_scale_without_history=adaptive_scale_without_history,
            )
        if self.optimizer != "scipy-lbfgsb":
            raise ValueError(f"unsupported relax optimizer: {self.optimizer}")
        return self._relax_with_scipy(
            state,
            fmax=fmax,
            maxiter=maxiter,
            bounds=bounds,
            trace=trace,
            trajectory_callback=trajectory_callback,
            trajectory_stride=trajectory_stride,
        )

    def _relax_with_scipy(
        self,
        state: State,
        fmax: float,
        maxiter: int,
        bounds: list[tuple[float | None, float | None]] | None,
        trace: _EvaluationTrace,
        trajectory_callback: Callable[[State], None] | None,
        trajectory_stride: int,
    ) -> RelaxResult:
        x0 = state.flatten_active()
        if trajectory_callback is not None:
            trajectory_callback(state)

        def objective(active_flat: np.ndarray) -> tuple[float, np.ndarray]:
            candidate = state.with_active_positions(active_flat)
            energy, full_gradient = trace.backend_evaluate(candidate.flatten_positions(), candidate)
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
            options={
                "maxiter": maxiter,
                "gtol": fmax / np.sqrt(3.0),
                "ftol": 0.0,
                "maxls": 50,
            },
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
        energy, full_gradient = trace.report_evaluate(
            relaxed.flatten_positions(),
            relaxed,
            finalization=True,
        )
        grad_matrix = full_gradient.reshape(relaxed.n_atoms, 3)
        active_gradient = grad_matrix[relaxed.movable_mask].reshape(-1)
        has_finite_bounds = bounds is not None and any(
            lower is not None or upper is not None for lower, upper in bounds
        )
        if has_finite_bounds:
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
        initial_energy, _ = trace.report_evaluate(state.flatten_positions(), state)
        converged = gradient_norm <= fmax
        optimizer_success_value = getattr(result, "success", None)
        optimizer_success = None if optimizer_success_value is None else bool(optimizer_success_value)
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
            telemetry=trace.telemetry(
                backend=self.optimizer,
                converged=converged,
                termination_reason=self._termination_reason(
                    converged=converged,
                    n_iter=n_iter,
                    maxiter=maxiter,
                    optimizer_success=optimizer_success,
                ),
                optimizer_success=optimizer_success,
                gradient_measure=(
                    "projected_active_kkt_residual"
                    if has_finite_bounds
                    else "raw_active_max_force"
                ),
            ),
        )

    def _relax_with_safe_lbfgs(
        self,
        state: State,
        *,
        fmax: float,
        maxiter: int,
        trace: _EvaluationTrace,
        trajectory_callback: Callable[[State], None] | None,
        trajectory_stride: int,
        history_limit: int,
        adaptive_scale_without_history: bool,
    ) -> RelaxResult:
        x = state.flatten_active().copy()
        current = state.with_active_positions(x)
        (
            energy,
            gradient,
            secant_gradient,
            bias_gradient,
            image_signature,
        ) = self._safe_lbfgs_evaluate(current, trace)
        initial_energy = float(energy)
        history: list[tuple[np.ndarray, np.ndarray, float]] = []
        scale_pair: tuple[np.ndarray, np.ndarray, float] | None = None
        n_iter = 0
        rejected_steps = 0
        accepted_secants = 0
        rejected_secants = 0
        line_search_evaluations = 0
        mic_branch_resets = 0
        bias_secant_curvature_sum = 0.0
        termination_reason = "maxiter"
        converged = False
        last_trajectory_x: np.ndarray | None = None
        if trajectory_callback is not None:
            trajectory_callback(current)
            last_trajectory_x = x.copy()

        while n_iter < maxiter:
            if not np.isfinite(energy) or not np.all(np.isfinite(gradient)):
                termination_reason = "nonfinite_evaluation"
                break
            gradient_norm = self._active_max_norm(gradient)
            if gradient_norm <= fmax:
                converged = True
                termination_reason = "converged"
                break

            if adaptive_scale_without_history:
                direction = -_lbfgs_inverse_product(
                    gradient,
                    history,
                    scale_pair=scale_pair,
                )
            else:
                direction = -_lbfgs_inverse_product(gradient, history)
            direction = _limit_max_atomic_displacement(direction)
            directional_derivative = float(np.dot(gradient, direction))
            if not np.all(np.isfinite(direction)) or not np.isfinite(directional_derivative):
                termination_reason = "nonfinite_direction"
                break
            if directional_derivative >= 0.0:
                termination_reason = "non_descent_direction"
                break

            accepted = False
            alpha = 1.0
            trial_x = x
            trial_energy = energy
            trial_gradient = gradient
            trial_secant_gradient = secant_gradient
            trial_bias_gradient = bias_gradient
            trial_image_signature = image_signature
            for _ in range(_SAFE_LBFGS_MAX_LINE_TRIALS):
                trial_x = x + alpha * direction
                trial_state = state.with_active_positions(trial_x)
                (
                    trial_energy,
                    trial_gradient,
                    trial_secant_gradient,
                    trial_bias_gradient,
                    trial_image_signature,
                ) = self._safe_lbfgs_evaluate(trial_state, trace)
                line_search_evaluations += 1
                if not np.isfinite(trial_energy) or not np.all(np.isfinite(trial_gradient)):
                    rejected_steps += 1
                    termination_reason = "nonfinite_evaluation"
                    break
                armijo_bound = energy + _SAFE_LBFGS_ARMIJO_C1 * alpha * directional_derivative
                if trial_energy <= armijo_bound:
                    accepted = True
                    break
                rejected_steps += 1
                alpha *= _SAFE_LBFGS_BACKTRACK
                if alpha < _SAFE_LBFGS_MIN_ALPHA:
                    break
            if termination_reason == "nonfinite_evaluation":
                break
            if not accepted:
                termination_reason = "line_search_failed"
                break

            s = trial_x - x
            branch_changed = trial_image_signature != image_signature
            if branch_changed:
                history.clear()
                scale_pair = None
                rejected_secants += 1
                mic_branch_resets += 1
            else:
                y = trial_secant_gradient - secant_gradient
                bias_y = trial_bias_gradient - bias_gradient
                bias_secant_curvature_sum += float(np.dot(s, bias_y))
            if not branch_changed and _accept_lbfgs_curvature(s, y):
                curvature = float(np.dot(s, y))
                accepted_pair = (s.copy(), y.copy(), 1.0 / curvature)
                history.append(accepted_pair)
                while len(history) > history_limit:
                    history.pop(0)
                if adaptive_scale_without_history:
                    scale_pair = accepted_pair
                accepted_secants += 1
            elif not branch_changed:
                rejected_secants += 1
            x = trial_x.copy()
            energy = float(trial_energy)
            gradient = trial_gradient.copy()
            secant_gradient = trial_secant_gradient.copy()
            bias_gradient = trial_bias_gradient.copy()
            image_signature = trial_image_signature
            n_iter += 1
            current = state.with_active_positions(x)
            if trajectory_callback is not None and n_iter % trajectory_stride == 0:
                trajectory_callback(current)
                last_trajectory_x = x.copy()
        else:
            termination_reason = "maxiter"

        relaxed = state.with_active_positions(x)
        if relaxed.cell is not None and any(relaxed.pbc):
            relaxed = State(
                numbers=relaxed.numbers.copy(),
                positions=wrap_positions(relaxed.positions, relaxed.cell, relaxed.pbc),
                cell=relaxed.cell.copy(),
                pbc=relaxed.pbc,
                fixed_mask=relaxed.fixed_mask.copy(),
                metadata=relaxed.metadata.copy(),
            )
        final_energy, final_full_gradient = trace.report_evaluate(
            relaxed.flatten_positions(),
            relaxed,
            finalization=True,
        )
        final_gradient = (
            final_full_gradient.reshape(relaxed.n_atoms, 3)[relaxed.movable_mask].reshape(-1)
        )
        gradient_norm = self._active_max_norm(final_gradient)
        if gradient_norm <= fmax and np.isfinite(final_energy):
            converged = True
            termination_reason = "converged"
        displacement_rms, displacement_max = self._displacement_stats(state, relaxed)
        final_active_x = relaxed.flatten_active()
        if trajectory_callback is not None and (
            last_trajectory_x is None or not np.array_equal(last_trajectory_x, final_active_x)
        ):
            trajectory_callback(relaxed)
        return RelaxResult(
            state=relaxed,
            energy=float(final_energy),
            gradient_norm=gradient_norm,
            n_iter=n_iter,
            active_bound_fraction=0.0,
            displacement_rms=displacement_rms,
            displacement_max=displacement_max,
            outcome_class=self.classify_outcome(
                initial_energy=initial_energy,
                final_energy=final_energy,
                gradient_norm=gradient_norm,
                fmax=fmax,
                displacement_rms=displacement_rms,
                displacement_max=displacement_max,
                active_bound_fraction=0.0,
            ),
            telemetry=trace.telemetry(
                backend=self.optimizer,
                converged=converged,
                termination_reason=termination_reason,
                optimizer_success=converged,
                gradient_measure="raw_active_max_force",
                accepted_steps=n_iter,
                rejected_steps=rejected_steps,
                accepted_secants=accepted_secants,
                rejected_secants=rejected_secants,
                line_search_evaluations=line_search_evaluations,
                mic_branch_resets=mic_branch_resets,
                bias_secant_curvature_sum=bias_secant_curvature_sum,
            ),
        )

    def _safe_lbfgs_evaluate(
        self,
        state: State,
        trace: _EvaluationTrace,
    ) -> tuple[
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        tuple[tuple[int, ...], ...],
    ]:
        if self.component_evaluator is None:
            energy, full_gradient = trace.backend_evaluate(state.flatten_positions(), state)
            total_gradient = (
                full_gradient.reshape(state.n_atoms, 3)[state.movable_mask].reshape(-1).copy()
            )
            return (
                float(energy),
                total_gradient,
                total_gradient.copy(),
                np.zeros_like(total_gradient),
                (),
            )
        parts = trace.backend_evaluate_parts(
            state.flatten_positions(),
            state,
            self.component_evaluator,
        )
        if self.optimizer == "bias-separated-lbfgs" and (
            parts.softening_present
            or parts.softening_energy != 0.0
            or np.any(parts.softening_gradient != 0.0)
        ):
            raise ValueError("bias-separated-lbfgs does not support local softening")
        total_gradient = (
            parts.total_gradient.reshape(state.n_atoms, 3)[state.movable_mask].reshape(-1).copy()
        )
        bias_gradient = (
            parts.bias_gradient.reshape(state.n_atoms, 3)[state.movable_mask].reshape(-1).copy()
        )
        secant_gradient = (
            total_gradient - bias_gradient
            if self.optimizer == "bias-separated-lbfgs"
            else total_gradient.copy()
        )
        return (
            float(parts.total_energy),
            total_gradient,
            secant_gradient,
            bias_gradient,
            parts.bias_image_signature,
        )

    def _relax_with_ase(
        self,
        state: State,
        trace: _EvaluationTrace,
        fmax: float,
        maxiter: int,
        trajectory_callback: Callable[[State], None] | None = None,
        trajectory_stride: int = 1,
    ) -> tuple[State, int, bool]:
        atoms = Atoms(
            numbers=state.numbers,
            positions=state.positions,
            cell=None if state.cell is None else state.cell,
            pbc=state.pbc,
        )
        if np.any(state.fixed_mask):
            atoms.set_constraint(FixAtoms(mask=state.fixed_mask))
        atoms.calc = _EvaluatorCalculator(trace.backend_evaluate, state)
        if self.optimizer == "ase-fire":
            optimizer_cls = FIRE
        elif self.optimizer == "ase-lbfgs":
            optimizer_cls = LBFGS
        else:
            if _ASE_FIRE2 is None:
                raise ValueError("ASE FIRE2 is not available in the installed ASE version")
            optimizer_cls = _ASE_FIRE2
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
        optimizer_success = bool(optimizer.run(fmax=fmax, steps=maxiter))
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
        return relaxed, n_iter, optimizer_success

    @staticmethod
    def _termination_reason(
        *,
        converged: bool,
        n_iter: int,
        maxiter: int,
        optimizer_success: bool | None,
    ) -> str:
        if converged:
            return "converged"
        if n_iter >= maxiter:
            return "maxiter"
        if optimizer_success is False:
            return "optimizer_stopped"
        return "unconverged"

    @staticmethod
    def _active_max_norm(active_gradient: np.ndarray) -> float:
        gradient = np.asarray(active_gradient, dtype=float).reshape(-1, 3)
        return float(np.max(np.linalg.norm(gradient, axis=1), initial=0.0))

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
