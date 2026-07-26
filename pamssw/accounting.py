from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
import threading
from typing import Mapping

import numpy as np

from .state import State


class BudgetExceeded(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        evaluation_counts: EvaluationCounts | None = None,
    ) -> None:
        super().__init__(message)
        if evaluation_counts is not None and not isinstance(
            evaluation_counts, EvaluationCounts
        ):
            raise TypeError("evaluation_counts must be an EvaluationCounts or None")
        self.evaluation_counts = evaluation_counts


class EvaluationPurpose(Enum):
    BOOTSTRAP_TRUE_QUENCH = "bootstrap_true_quench"
    STARTER_TRUE_QUENCH = "starter_true_quench"
    DIRECTION_ORACLE = "direction_oracle"
    ESCAPE_TRUE_PES_CHECK = "escape_true_pes_check"
    BIASED_PROPOSAL_RELAX = "biased_proposal_relax"
    LANDING_TRUE_QUENCH = "landing_true_quench"
    POST_RELAX_VALIDATION = "post_relax_validation"
    UNATTRIBUTED = "unattributed"


@dataclass(frozen=True)
class EvaluationCounts:
    values: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.values, tuple):
            raise TypeError("evaluation counts must be a tuple")
        if len(self.values) != len(EvaluationPurpose):
            raise ValueError("evaluation counts must include every purpose")
        for value in self.values:
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("evaluation counts must be integers")
            if value < 0:
                raise ValueError("evaluation counts must be nonnegative")

    @classmethod
    def zero(cls) -> EvaluationCounts:
        return cls((0,) * len(EvaluationPurpose))

    @classmethod
    def unattributed(cls, total: int) -> EvaluationCounts:
        return cls((0,) * (len(EvaluationPurpose) - 1) + (total,))

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[EvaluationPurpose | str, int]
    ) -> EvaluationCounts:
        counts = {purpose: 0 for purpose in EvaluationPurpose}
        seen: set[EvaluationPurpose] = set()
        for key, value in mapping.items():
            purpose = cls._mapping_purpose(key)
            if purpose in seen:
                raise ValueError(f"duplicate evaluation purpose: {purpose.value}")
            seen.add(purpose)
            counts[purpose] = value
        return cls(tuple(counts[purpose] for purpose in EvaluationPurpose))

    @staticmethod
    def _mapping_purpose(key: EvaluationPurpose | str) -> EvaluationPurpose:
        if isinstance(key, EvaluationPurpose):
            return key
        if isinstance(key, str):
            try:
                return EvaluationPurpose(key)
            except ValueError as error:
                raise ValueError(f"unknown evaluation purpose: {key!r}") from error
        raise TypeError("evaluation purpose must be an EvaluationPurpose or string")

    @property
    def total(self) -> int:
        return sum(self.values)

    def count(self, purpose: EvaluationPurpose) -> int:
        if not isinstance(purpose, EvaluationPurpose):
            raise TypeError("evaluation purpose must be an EvaluationPurpose")
        return self.values[list(EvaluationPurpose).index(purpose)]

    def as_dict(self) -> dict[str, int]:
        return {
            purpose.value: self.values[index]
            for index, purpose in enumerate(EvaluationPurpose)
        }

    def __add__(self, other: object) -> EvaluationCounts:
        if not isinstance(other, EvaluationCounts):
            raise TypeError("evaluation counts can only be added to EvaluationCounts")
        return type(self)(tuple(left + right for left, right in zip(self.values, other.values)))

    @classmethod
    def sum(cls, counts: Iterable[EvaluationCounts]) -> EvaluationCounts:
        total = cls.zero()
        for count in counts:
            if not isinstance(count, cls):
                raise TypeError("evaluation counts sum requires EvaluationCounts items")
            total = total + count
        return total


@dataclass
class EvalCounter:
    calculator: object
    max_force_evals: int | None = None
    force_evaluations: int = 0
    energy_evaluations: int = 0
    _purpose_counts: list[int] = field(
        default_factory=lambda: [0] * len(EvaluationPurpose), init=False, repr=False
    )
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _purpose_local: threading.local = field(default_factory=threading.local, init=False, repr=False)

    def evaluate(self, state: State):
        self._start_evaluation()
        return self.calculator.evaluate(state)

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        self._start_evaluation()
        return self.calculator.evaluate_flat(flat_positions, template)

    @contextmanager
    def purpose(self, purpose: EvaluationPurpose) -> Iterator[None]:
        if not isinstance(purpose, EvaluationPurpose):
            raise TypeError("evaluation purpose must be an EvaluationPurpose")
        stack = self._purpose_stack()
        stack.append(purpose)
        try:
            yield
        finally:
            stack.pop()

    def snapshot(self) -> EvaluationCounts:
        with self._lock:
            return EvaluationCounts(tuple(self._purpose_counts))

    def _purpose_stack(self) -> list[EvaluationPurpose]:
        stack = getattr(self._purpose_local, "stack", None)
        if stack is None:
            stack = []
            self._purpose_local.stack = stack
        return stack

    def _start_evaluation(self) -> None:
        with self._lock:
            if self.max_force_evals is not None and self.force_evaluations >= self.max_force_evals:
                raise BudgetExceeded(
                    "force-evaluation budget exhausted",
                    evaluation_counts=EvaluationCounts(tuple(self._purpose_counts)),
                )
            self._record_started()

    def _record_started(self) -> None:
        self.force_evaluations += 1
        self.energy_evaluations += 1
        stack = self._purpose_stack()
        purpose = stack[-1] if stack else EvaluationPurpose.UNATTRIBUTED
        self._purpose_counts[list(EvaluationPurpose).index(purpose)] += 1

    def exhausted(self) -> bool:
        with self._lock:
            return self.max_force_evals is not None and self.force_evaluations >= self.max_force_evals
