from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isclose
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
class UphillStepRecord:
    """Already-observed true-PES endpoints and cost for one uphill micro step."""

    step_index: int
    direction_kind: str
    target_eV: float
    true_energy_before_eV: float
    true_energy_after_eV: float
    requested_sigma: float
    executed_sigma: float
    base_bias_weight: float
    final_bias_weight: float
    true_curvature: float
    inner_curvature: float
    proposal_relax_iterations: int
    proposal_relax_outcome: str
    proposal_relax_termination: str
    direction_oracle_force_evaluations: int
    biased_relax_force_evaluations: int
    true_pes_check_force_evaluations: int
    displacement_clipped: bool
    step_termination_reason: str

    def __post_init__(self) -> None:
        if self.step_index < 0:
            raise ValueError("step_index must be non-negative")
        if self.target_eV <= 0.0:
            raise ValueError("target_eV must be positive")
        if self.proposal_relax_iterations < 0:
            raise ValueError("proposal_relax_iterations must be non-negative")
        counts = (
            self.direction_oracle_force_evaluations,
            self.biased_relax_force_evaluations,
            self.true_pes_check_force_evaluations,
        )
        if any(count < 0 for count in counts):
            raise ValueError("force-evaluation counts must be non-negative")


@dataclass(frozen=True)
class UphillWalkTrace:
    """Physical endpoint observations for one complete SSW uphill walk."""

    target_eV: float
    termination_reason: str
    steps: tuple[UphillStepRecord, ...]
    observed_max_height_eV: float | None = field(init=False)
    observed_terminal_height_eV: float | None = field(init=False)
    target_delivery_ratio: float | None = field(init=False)

    def __post_init__(self) -> None:
        if self.target_eV <= 0.0:
            raise ValueError("target_eV must be positive")
        if tuple(step.step_index for step in self.steps) != tuple(range(len(self.steps))):
            raise ValueError("step indices must be ordered from zero")
        if any(not isclose(step.target_eV, self.target_eV) for step in self.steps):
            raise ValueError("step target_eV must match walk target_eV")
        if not self.steps:
            object.__setattr__(self, "observed_max_height_eV", None)
            object.__setattr__(self, "observed_terminal_height_eV", None)
            object.__setattr__(self, "target_delivery_ratio", None)
            return

        reference = self.steps[0].true_energy_before_eV
        endpoint_energies = tuple(
            energy
            for step in self.steps
            for energy in (step.true_energy_before_eV, step.true_energy_after_eV)
        )
        observed_max = max(endpoint_energies) - reference
        observed_terminal = self.steps[-1].true_energy_after_eV - reference
        object.__setattr__(self, "observed_max_height_eV", float(observed_max))
        object.__setattr__(self, "observed_terminal_height_eV", float(observed_terminal))
        object.__setattr__(
            self,
            "target_delivery_ratio",
            float(observed_max / self.target_eV),
        )


@dataclass(frozen=True)
class ActionRecord:
    """One uphill proposal joined to its true-PES landing outcome and cost."""

    trial_index: int
    proposal_index: int
    seed_entry_id: int
    seed_energy_eV: float
    walk: UphillWalkTrace
    escape_energy_eV: float | None
    landing_energy_eV: float | None
    landing_gradient_norm: float | None
    landing_iterations: int | None
    landing_converged: bool | None
    landing_force_evaluations: int
    accepted_new_basin: bool | None
    is_duplicate: bool | None
    global_improved: bool | None
    status: str

    def __post_init__(self) -> None:
        if self.trial_index < 0 or self.proposal_index < 0:
            raise ValueError("trial and proposal indices must be non-negative")
        if self.landing_force_evaluations < 0:
            raise ValueError("landing_force_evaluations must be non-negative")
        if self.accepted_new_basin is True and self.is_duplicate is True:
            raise ValueError("an action cannot be both new and duplicate")
        if self.escape_energy_eV is not None and self.walk.steps:
            endpoint = self.walk.steps[-1].true_energy_after_eV
            if not isclose(self.escape_energy_eV, endpoint, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("escape_energy_eV must match the final walk endpoint")


@dataclass(frozen=True)
class SearchResult:
    """Result bundle returned by SSW search entry points."""
    best_state: State
    best_energy: float
    archive: Any
    walk_history: list[WalkRecord] = field(default_factory=list)
    action_history: list[ActionRecord] = field(default_factory=list)
    stats: dict[str, StatsValue] = field(default_factory=dict)
