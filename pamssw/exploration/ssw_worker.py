"""Isolated one-action adapter around :class:`pamssw.walker.SurfaceWalker`."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from numbers import Integral
from threading import Lock
from typing import Callable

from ..accounting import BudgetExceeded, EvaluationCounts, EvaluationPurpose
from ..config import LSSSWConfig, SSWConfig
from ..result import SearchResult
from ..state import State
from ..walker import GeometryValidator, SurfaceWalker
from .actions import AttemptResult, AttemptStatus, StarterAction
from .campaign import AttemptDiagnostics


class SSWAttemptWorker:
    """Run exactly one independent SSW trial for a dispatched starter action."""

    def __init__(
        self,
        calculator_factory: Callable[[], object],
        config: SSWConfig,
        *,
        softening_enabled: bool = False,
    ) -> None:
        if not callable(calculator_factory):
            raise ValueError("calculator_factory must be callable")
        if not isinstance(config, SSWConfig):
            raise ValueError("config must be an SSWConfig")
        if not isinstance(softening_enabled, bool):
            raise ValueError("softening_enabled must be a boolean")
        if softening_enabled and not isinstance(config, LSSSWConfig):
            raise ValueError("softening_enabled requires an LSSSWConfig")
        if config.proposal_pool_size != 1:
            raise ValueError("proposal_pool_size must be 1 to avoid internal proposal competition")
        if config.proposal_duplicate_rescue_optimizer is not None:
            raise ValueError("proposal_duplicate_rescue_optimizer enables internal proposal competition")
        if _has_shared_filesystem_output(config):
            raise ValueError("shared filesystem output is not supported by SSWAttemptWorker")

        self.calculator_factory = calculator_factory
        self.config = config
        self.softening_enabled = softening_enabled
        self.geometry_validator = GeometryValidator()
        self._calculator_first_evaluation_lock = Lock()
        self._diagnostics_lock = Lock()
        self._diagnostics: dict[str, AttemptDiagnostics] = {}

    def __call__(self, action: StarterAction, starter_state: State) -> AttemptResult:
        if not isinstance(action, StarterAction):
            raise ValueError("action must be a StarterAction")
        if not isinstance(starter_state, State):
            raise ValueError("starter_state must be a State")
        if not self.geometry_validator.is_valid_state(starter_state):
            result = _failed_result(
                action,
                AttemptStatus.INVALID,
                EvaluationCounts.zero(),
                "invalid_starter_geometry",
            )
            self._record_diagnostics_safely(
                action,
                stage="invalid_starter",
                force_evaluations=0,
            )
            return result

        try:
            calculator = self.calculator_factory()
        except Exception as exc:
            result = _worker_error(action, EvaluationCounts.zero(), "factory_error", exc)
            self._record_diagnostics_safely(
                action,
                stage="factory_error",
                force_evaluations=0,
            )
            return result
        if not _is_calculator(calculator):
            result = _failed_result(
                action,
                AttemptStatus.WORKER_ERROR,
                EvaluationCounts.zero(),
                "calculator_error: missing callable evaluate and evaluate_flat",
            )
            self._record_diagnostics_safely(
                action,
                stage="calculator_error",
                force_evaluations=0,
            )
            return result
        calculator = _FirstEvaluationSerializedCalculator(
            calculator,
            self._calculator_first_evaluation_lock,
        )

        action_config = replace(
            self.config,
            max_trials=1,
            rng_seed=action.random_seed,
            max_force_evals=action.force_budget,
        )
        try:
            walker = SurfaceWalker(calculator, action_config, self.softening_enabled)
        except Exception as exc:
            result = _worker_error(action, EvaluationCounts.zero(), "constructor_error", exc)
            self._record_diagnostics_safely(
                action,
                stage="constructor_error",
                force_evaluations=0,
            )
            return result

        try:
            result = walker.run(
                deepcopy(starter_state),
                initial_quench_purpose=EvaluationPurpose.STARTER_TRUE_QUENCH,
            )
        except BudgetExceeded:
            evaluation_counts = _calculator_snapshot(walker)
            if _counter_exhausted(walker):
                terminal = _failed_result(
                    action,
                    AttemptStatus.BUDGET_EXHAUSTED,
                    evaluation_counts,
                    "budget_exhausted",
                )
                stage = "budget_exhausted"
            else:
                terminal = _failed_result(
                    action,
                    AttemptStatus.INVALID,
                    evaluation_counts,
                    "budget_exception_without_exhaustion",
                )
                stage = "budget_exception_without_exhaustion"
            self._record_diagnostics_safely(
                action,
                walker=walker,
                stage=stage,
                force_evaluations=evaluation_counts.total,
            )
            return terminal
        except Exception as exc:
            evaluation_counts = _calculator_snapshot(walker)
            terminal = _worker_error(action, evaluation_counts, "run_error", exc)
            self._record_diagnostics_safely(
                action,
                walker=walker,
                stage="run_error",
                force_evaluations=evaluation_counts.total,
            )
            return terminal

        evaluation_counts = _calculator_snapshot(walker)
        try:
            terminal = _map_search_result(action, result, evaluation_counts)
        except Exception as exc:
            terminal = _worker_error(
                action,
                evaluation_counts,
                "result_mapping_error",
                exc,
            )
            stage = "result_mapping_error"
        else:
            stage = terminal.status.value
        self._record_diagnostics_safely(
            action,
            walker=walker,
            result=result,
            stage=stage,
            force_evaluations=evaluation_counts.total,
        )
        return terminal

    def diagnostics_snapshot(self) -> tuple[AttemptDiagnostics, ...]:
        with self._diagnostics_lock:
            return tuple(self._diagnostics[key] for key in sorted(self._diagnostics))

    def _record_diagnostics(
        self,
        action: StarterAction,
        walker: SurfaceWalker | None,
        result: SearchResult | None = None,
        *,
        stage: str,
        force_evaluations: int,
    ) -> None:
        if result is not None:
            source = result.stats
        elif walker is None:
            source = {}
        else:
            diagnostic_source = getattr(walker, "relaxation_diagnostics", None)
            source = diagnostic_source() if callable(diagnostic_source) else {}
        retained = {
            name: value
            for name, value in source.items()
            if name.startswith("proposal_relax_")
            or name.startswith("true_quench_")
            or name
            in {
                "force_evaluations",
                "budget_exhausted",
                "fragment_rejections",
                "proposal_optimizer",
                "quench_optimizer",
                "quench_fallback_optimizer",
                "quench_fallback_attempts",
                "quench_fallback_converged",
            }
        }
        if "proposal_optimizer" not in retained:
            retained["proposal_optimizer"] = self.config.proposal_optimizer
        if "quench_optimizer" not in retained:
            retained["quench_optimizer"] = self.config.quench_optimizer
        if "quench_fallback_optimizer" not in retained:
            retained["quench_fallback_optimizer"] = self.config.quench_fallback_optimizer
        retained["force_evaluations"] = force_evaluations
        retained["diagnostic_stage"] = stage
        diagnostic = AttemptDiagnostics(
            action_id=action.action_id,
            stats=tuple(sorted(retained.items())),
        )
        with self._diagnostics_lock:
            self._diagnostics[action.action_id] = diagnostic

    def _record_diagnostics_safely(
        self,
        action: StarterAction,
        *,
        stage: str,
        force_evaluations: int,
        walker: SurfaceWalker | None = None,
        result: SearchResult | None = None,
    ) -> None:
        """Record bounded diagnostics without changing an action's terminal result."""
        try:
            self._record_diagnostics(
                action,
                walker,
                result,
                stage=stage,
                force_evaluations=force_evaluations,
            )
        except Exception as exc:
            fallback = AttemptDiagnostics(
                action_id=action.action_id,
                stats=tuple(
                    sorted(
                        {
                            "diagnostic_stage": stage,
                            "diagnostics_error": type(exc).__name__,
                            "force_evaluations": force_evaluations,
                            "proposal_optimizer": self.config.proposal_optimizer,
                            "quench_optimizer": self.config.quench_optimizer,
                            "quench_fallback_optimizer": self.config.quench_fallback_optimizer,
                        }.items()
                    )
                ),
            )
            try:
                with self._diagnostics_lock:
                    self._diagnostics[action.action_id] = fallback
            except Exception:
                pass


def _has_shared_filesystem_output(config: SSWConfig) -> bool:
    return bool(
        config.accepted_structures_log is not None
        or config.accepted_structures_dir is not None
        or config.write_proposal_minima
        or config.write_relaxation_trajectories
        or config.direction_diagnostics_enabled
        or (config.direction_archive_enabled and config.direction_archive_path is not None)
    )


def _is_calculator(calculator: object) -> bool:
    return callable(getattr(calculator, "evaluate", None)) and callable(
        getattr(calculator, "evaluate_flat", None)
    )


class _FirstEvaluationSerializedCalculator:
    """Serialize lazy calculator initialization without adding evaluations."""

    def __init__(self, calculator: object, shared_lock: Lock) -> None:
        self.calculator = calculator
        self._shared_lock = shared_lock
        self._first_evaluation_complete = False

    def evaluate(self, state: State):
        return self._evaluate_once(self.calculator.evaluate, state)

    def evaluate_flat(self, flat_positions, template: State):
        return self._evaluate_once(
            self.calculator.evaluate_flat,
            flat_positions,
            template,
        )

    def _evaluate_once(self, evaluator, *args):
        if self._first_evaluation_complete:
            return evaluator(*args)
        with self._shared_lock:
            if not self._first_evaluation_complete:
                result = evaluator(*args)
                self._first_evaluation_complete = True
                return result
        return evaluator(*args)


def _calculator_snapshot(walker: SurfaceWalker) -> EvaluationCounts:
    snapshot = getattr(walker.calculator, "snapshot", None)
    if not callable(snapshot):
        raise ValueError("walker calculator must provide a callable snapshot")
    evaluation_counts = snapshot()
    if not isinstance(evaluation_counts, EvaluationCounts):
        raise ValueError("walker calculator snapshot must return EvaluationCounts")
    return EvaluationCounts(tuple(evaluation_counts.values))


def _counter_exhausted(walker: SurfaceWalker) -> bool:
    exhausted = getattr(walker.calculator, "exhausted", None)
    if not callable(exhausted):
        return False
    return bool(exhausted())


def _map_search_result(
    action: StarterAction,
    result: object,
    evaluation_counts: EvaluationCounts,
) -> AttemptResult:
    if not isinstance(result, SearchResult):
        raise ValueError("walker must return a SearchResult")
    if not isinstance(evaluation_counts, EvaluationCounts):
        raise ValueError("evaluation_counts must be an EvaluationCounts")
    force_evaluations = evaluation_counts.total
    reported_force_evaluations = _required_nonnegative_stat(result.stats, "force_evaluations")
    if reported_force_evaluations != force_evaluations:
        raise ValueError("search result force_evaluations must equal calculator snapshot")
    budget_exhausted = _required_nonnegative_stat(result.stats, "budget_exhausted")
    fragment_rejections = _required_nonnegative_stat(result.stats, "fragment_rejections")

    try:
        walk_history = tuple(result.walk_history)
    except TypeError as exc:
        raise ValueError("walk_history must be iterable") from exc
    if len(walk_history) > 1:
        raise ValueError("expected at most one walk record")
    if walk_history:
        discovered_entry_id = _nonnegative_integral(
            "discovered_entry_id",
            getattr(walk_history[0], "discovered_entry_id", None),
        )
        landing = _resolve_discovered_entry(result.archive, discovered_entry_id)
        return AttemptResult(
            action=action,
            landing_state=landing.state,
            landing_energy=landing.energy,
            force_evaluations=force_evaluations,
            status=AttemptStatus.COMPLETED,
            failure_reason=None,
            evaluation_counts=evaluation_counts,
            cost_is_exact=True,
        )
    if budget_exhausted > 0:
        return _failed_result(
            action,
            AttemptStatus.BUDGET_EXHAUSTED,
            evaluation_counts,
            "budget_exhausted_without_landing",
        )
    if fragment_rejections > 0:
        return _failed_result(
            action,
            AttemptStatus.FRAGMENTED,
            evaluation_counts,
            "fragment_rejections_without_landing",
        )
    return _failed_result(action, AttemptStatus.INVALID, evaluation_counts, "no_landing_minimum")


def _required_nonnegative_stat(stats: object, name: str) -> int:
    if not isinstance(stats, dict) or name not in stats:
        raise ValueError(f"search result stats must include {name}")
    return _nonnegative_integral(f"search result stat {name}", stats[name])


def _nonnegative_integral(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _resolve_discovered_entry(archive: object, discovered_entry_id: int):
    entries = getattr(archive, "entries", None)
    try:
        matches = [entry for entry in entries if getattr(entry, "entry_id", None) == discovered_entry_id]
    except TypeError as exc:
        raise ValueError("archive entries must be iterable") from exc
    if len(matches) != 1:
        raise ValueError("discovered entry id must resolve exactly once")
    return matches[0]


def _failed_result(
    action: StarterAction,
    status: AttemptStatus,
    evaluation_counts: EvaluationCounts,
    reason: str,
) -> AttemptResult:
    return AttemptResult(
        action=action,
        landing_state=None,
        landing_energy=None,
        force_evaluations=evaluation_counts.total,
        status=status,
        failure_reason=reason,
        evaluation_counts=evaluation_counts,
        cost_is_exact=True,
    )


def _worker_error(
    action: StarterAction,
    evaluation_counts: EvaluationCounts,
    stage: str,
    exc: Exception,
) -> AttemptResult:
    return _failed_result(
        action,
        AttemptStatus.WORKER_ERROR,
        evaluation_counts,
        f"{stage}: {type(exc).__name__}: {exc}",
    )


__all__ = ["SSWAttemptWorker"]
