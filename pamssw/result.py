from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .state import State


@dataclass(frozen=True)
class RelaxResult:
    state: State
    energy: float
    gradient_norm: float
    n_iter: int


@dataclass(frozen=True)
class WalkRecord:
    seed_entry_id: int
    discovered_entry_id: int
    energy: float
    accepted_new_basin: bool


@dataclass(frozen=True)
class SearchResult:
    best_state: State
    best_energy: float
    archive: Any
    walk_history: list[WalkRecord] = field(default_factory=list)
    stats: dict[str, float | int] = field(default_factory=dict)
