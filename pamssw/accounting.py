from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import State


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class EvalCounter:
    calculator: object
    max_force_evals: int | None = None
    force_evaluations: int = 0
    energy_evaluations: int = 0

    def evaluate(self, state: State):
        self._reserve()
        result = self.calculator.evaluate(state)
        self.force_evaluations += 1
        self.energy_evaluations += 1
        return result

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        self._reserve()
        energy, gradient = self.calculator.evaluate_flat(flat_positions, template)
        self.force_evaluations += 1
        self.energy_evaluations += 1
        return energy, gradient

    def exhausted(self) -> bool:
        return self.max_force_evals is not None and self.force_evaluations >= self.max_force_evals

    def _reserve(self) -> None:
        if self.max_force_evals is not None and self.force_evaluations >= self.max_force_evals:
            raise BudgetExceeded("force-evaluation budget exhausted")
