"""Caller-owned selection among certified fixed-cell SSW observations.

Snapshots contain detached geometry copies. Observation indices are not
permutation-invariant basin identities; deduplication belongs to the selector.
"""
from dataclasses import dataclass
from numbers import Integral
import numpy as np


@dataclass(frozen=True)
class StarterObservation:
    index: int
    atoms: object
    energy: float
    max_force: float


@dataclass(frozen=True)
class StarterPoolSnapshot:
    observations: tuple[StarterObservation, ...]
    current_index: int
    last_landing_index: int | None
    step: int
    cost: int


def snapshot_from_minima(minima, *, current_index, last_landing_index, step, cost):
    observations = tuple(
        StarterObservation(index, minimum.atoms.copy(), float(minimum.energy),
                           float(minimum.max_force))
        for index, minimum in enumerate(minima))
    return StarterPoolSnapshot(observations, current_index, last_landing_index, step, cost)


def validate_starter_index(chosen, size):
    if chosen is None:
        return None
    if isinstance(chosen, (bool, np.bool_)) or not isinstance(chosen, Integral):
        raise TypeError('starter_selector must return an integer observation index or None')
    chosen = int(chosen)
    if chosen < 0 or chosen >= size:
        raise IndexError(f'starter_selector index {chosen} outside observation pool of size {size}')
    return chosen
