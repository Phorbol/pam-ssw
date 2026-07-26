"""Minimal bootstrap primitive for posterior-exploration experiments."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Callable

from ..accounting import EvalCounter, EvaluationCounts, EvaluationPurpose
from ..config import SSWConfig
from ..relax import Relaxer
from ..state import State
from ..walker import GeometryValidator


def _bootstrap_minimum(
    initial_state: State,
    calculator_factory: Callable[[], object],
    ssw_config: SSWConfig,
    *,
    total_force_budget: int,
) -> tuple[State, float, EvaluationCounts]:
    """True-quench one raw state under an exact, local force-evaluation budget."""
    if not isinstance(initial_state, State):
        raise TypeError("initial_state must be a State")

    geometry_validator = GeometryValidator()
    if not geometry_validator.is_valid_state(initial_state):
        raise ValueError("invalid initial geometry")
    if not callable(calculator_factory):
        raise TypeError("calculator_factory must be callable")
    if not isinstance(ssw_config, SSWConfig):
        raise TypeError("ssw_config must be an SSWConfig")
    _positive_force_budget(total_force_budget)

    calculator = calculator_factory()
    if not _is_calculator(calculator):
        raise TypeError("calculator must provide callable evaluate and evaluate_flat")

    counter = EvalCounter(calculator, max_force_evals=total_force_budget)
    relaxer = Relaxer(counter.evaluate_flat, optimizer=ssw_config.quench_optimizer)
    with counter.purpose(EvaluationPurpose.BOOTSTRAP_TRUE_QUENCH):
        relaxed = relaxer.relax(
            deepcopy(initial_state),
            fmax=ssw_config.quench_fmax,
            maxiter=ssw_config.quench_maxiter,
        )
    with counter.purpose(EvaluationPurpose.POST_RELAX_VALIDATION):
        valid_final_evaluation = geometry_validator.is_valid_evaluation(relaxed.state, counter)

    energy = float(relaxed.energy)
    if not valid_final_evaluation or not isfinite(energy):
        raise ValueError("invalid final relaxed state")
    return deepcopy(relaxed.state), energy, counter.snapshot()


def _positive_force_budget(total_force_budget: object) -> int:
    if isinstance(total_force_budget, bool) or not isinstance(total_force_budget, int):
        raise TypeError("total_force_budget must be an integer")
    if total_force_budget <= 0:
        raise ValueError("total_force_budget must be positive")
    return total_force_budget


def _is_calculator(calculator: object) -> bool:
    return callable(getattr(calculator, "evaluate", None)) and callable(
        getattr(calculator, "evaluate_flat", None)
    )
