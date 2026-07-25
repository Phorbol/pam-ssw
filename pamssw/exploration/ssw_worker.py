"""Isolated one-action adapter around :class:`pamssw.walker.SurfaceWalker`."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from numbers import Integral
from typing import Callable

from ..accounting import BudgetExceeded
from ..config import LSSSWConfig, SSWConfig
from ..result import SearchResult
from ..state import State
from ..walker import GeometryValidator, SurfaceWalker
from .actions import AttemptResult, AttemptStatus, StarterAction


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

    def __call__(self, action: StarterAction, starter_state: State) -> AttemptResult:
        if not isinstance(action, StarterAction):
            raise ValueError("action must be a StarterAction")
        if not isinstance(starter_state, State):
            raise ValueError("starter_state must be a State")
        if not self.geometry_validator.is_valid_state(starter_state):
            return _failed_result(action, AttemptStatus.INVALID, 0, "invalid_starter_geometry")

        try:
            calculator = self.calculator_factory()
        except Exception as exc:
            return _worker_error(action, 0, "factory_error", exc)
        if not _is_calculator(calculator):
            return _failed_result(
                action,
                AttemptStatus.WORKER_ERROR,
                0,
                "calculator_error: missing callable evaluate and evaluate_flat",
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
            return _worker_error(action, 0, "constructor_error", exc)

        try:
            result = walker.run(deepcopy(starter_state))
        except BudgetExceeded:
            force_evaluations = _counter_force_evaluations(walker)
            if _counter_exhausted(walker):
                return _failed_result(
                    action,
                    AttemptStatus.BUDGET_EXHAUSTED,
                    force_evaluations,
                    "budget_exhausted",
                )
            return _failed_result(
                action,
                AttemptStatus.INVALID,
                force_evaluations,
                "budget_exception_without_exhaustion",
            )
        except Exception as exc:
            return _worker_error(action, _counter_force_evaluations(walker), "run_error", exc)

        try:
            return _map_search_result(action, result)
        except Exception as exc:
            return _worker_error(action, _counter_force_evaluations(walker), "result_mapping_error", exc)


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


def _counter_force_evaluations(walker: SurfaceWalker) -> int:
    value = getattr(walker.calculator, "force_evaluations", 0)
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError("walker calculator force_evaluations must be a non-negative integer")
    return int(value)


def _counter_exhausted(walker: SurfaceWalker) -> bool:
    exhausted = getattr(walker.calculator, "exhausted", None)
    if not callable(exhausted):
        return False
    return bool(exhausted())


def _map_search_result(action: StarterAction, result: object) -> AttemptResult:
    if not isinstance(result, SearchResult):
        raise ValueError("walker must return a SearchResult")
    force_evaluations = _required_nonnegative_stat(result.stats, "force_evaluations")
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
        )
    if budget_exhausted > 0:
        return _failed_result(
            action,
            AttemptStatus.BUDGET_EXHAUSTED,
            force_evaluations,
            "budget_exhausted_without_landing",
        )
    if fragment_rejections > 0:
        return _failed_result(
            action,
            AttemptStatus.FRAGMENTED,
            force_evaluations,
            "fragment_rejections_without_landing",
        )
    return _failed_result(action, AttemptStatus.INVALID, force_evaluations, "no_landing_minimum")


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
    force_evaluations: int,
    reason: str,
) -> AttemptResult:
    return AttemptResult(
        action=action,
        landing_state=None,
        landing_energy=None,
        force_evaluations=force_evaluations,
        status=status,
        failure_reason=reason,
    )


def _worker_error(
    action: StarterAction,
    force_evaluations: int,
    stage: str,
    exc: Exception,
) -> AttemptResult:
    return _failed_result(
        action,
        AttemptStatus.WORKER_ERROR,
        force_evaluations,
        f"{stage}: {type(exc).__name__}: {exc}",
    )


__all__ = ["SSWAttemptWorker"]
