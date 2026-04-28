from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .state import State


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
    stats: dict[str, float | int] = field(default_factory=dict)
