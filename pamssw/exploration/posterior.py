from __future__ import annotations

import copy
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral


@dataclass
class _Counts:
    successes: int = 0
    failures: int = 0


class StarterProductivityPosterior:
    """Fixed-prior Beta-Bernoulli productivity counts for completed outcomes only."""

    PRIOR_ALPHA = 1.0
    PRIOR_BETA = 1.0

    def __init__(self) -> None:
        self._counts: dict[int, _Counts] = {}

    @staticmethod
    def _validate_starter_id(starter_id: object) -> int:
        if isinstance(starter_id, bool) or not isinstance(starter_id, Integral) or starter_id < 0:
            raise ValueError("starter_id must be a non-negative integer")
        return int(starter_id)

    def ensure(self, starter_ids: Iterable[int]) -> None:
        for starter_id in starter_ids:
            self._counts.setdefault(self._validate_starter_id(starter_id), _Counts())

    def update(self, starter_id: int, discovered: bool) -> None:
        starter_id = self._validate_starter_id(starter_id)
        if not isinstance(discovered, bool):
            raise ValueError("discovered must be a boolean")
        counts = self._counts.setdefault(starter_id, _Counts())
        if discovered:
            counts.successes += 1
        else:
            counts.failures += 1

    def counts(self, starter_id: int) -> tuple[int, int]:
        counts = self._counts.get(self._validate_starter_id(starter_id))
        if counts is None:
            return 0, 0
        return counts.successes, counts.failures

    def mean(self, starter_id: int) -> float:
        successes, failures = self.counts(starter_id)
        return float(
            (self.PRIOR_ALPHA + successes)
            / (self.PRIOR_ALPHA + self.PRIOR_BETA + successes + failures)
        )

    @property
    def completed_attempts(self) -> int:
        return sum(counts.successes + counts.failures for counts in self._counts.values())

    def clone(self) -> StarterProductivityPosterior:
        return copy.deepcopy(self)


__all__ = ["StarterProductivityPosterior"]
