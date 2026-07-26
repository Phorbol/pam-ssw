from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .state import State

StatsValue = float | int | str | None


class RelaxOutcomeClass(str, Enum):
    DAMAGED = "damaged"
    ENERGY_EXPLODED = "energy_exploded"
    GEOMETRY_INVALID = "geometry_invalid"
    STAGNATED = "stagnated"
    USEFUL_PROGRESS = "useful_progress"
    CONVERGED_PRODUCTIVE = "converged_productive"
    CONVERGED_UNPRODUCTIVE = "converged_unproductive"


@dataclass(frozen=True)
class RelaxTelemetry:
    """Backend-independent accounting for one local relaxation."""

    backend: str = "unknown"
    evaluator_calls: int = 0
    backend_evaluations: int = 0
    reporting_cache_hits: int = 0
    reporting_evaluator_calls: int = 0
    finalization_requests: int = 0
    explicit_finalization_calls: int = 0
    gradient_measure: str = "unknown"
    converged: bool = False
    termination_reason: str = "unknown"
    optimizer_success: bool | None = None
    accepted_steps: int = 0
    rejected_steps: int = 0
    accepted_secants: int = 0
    rejected_secants: int = 0
    line_search_evaluations: int = 0
    mic_branch_resets: int = 0
    bias_secant_curvature_sum: float = 0.0


@dataclass(frozen=True)
class RelaxResult:
    """Result of a local relaxation."""
    state: State
    energy: float
    gradient_norm: float
    n_iter: int
    active_bound_fraction: float = 0.0
    displacement_rms: float = 0.0
    displacement_max: float = 0.0
    outcome_class: RelaxOutcomeClass = RelaxOutcomeClass.USEFUL_PROGRESS
    telemetry: RelaxTelemetry = field(default_factory=RelaxTelemetry)


@dataclass(frozen=True)
class WalkRecord:
    """Archive transition produced by one completed SSW trial."""
    seed_entry_id: int
    discovered_entry_id: int
    energy: float
    accepted_new_basin: bool


@dataclass(frozen=True)
class SearchResult:
    """Result bundle returned by SSW search entry points."""
    best_state: State
    best_energy: float
    archive: Any
    walk_history: list[WalkRecord] = field(default_factory=list)
    stats: dict[str, StatsValue] = field(default_factory=dict)
