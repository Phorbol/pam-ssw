from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pamssw.calculators import EnergyResult
from pamssw.generalized_coordinates import GeneralizedCoordinates
from pamssw.generalized_evaluator import CalculatorGeneralizedEvaluator
from pamssw.generalized_relax import VariableCellRelaxer
from pamssw.state import State


@dataclass
class InPlaneCellCalculator:
    def evaluate(self, state: State) -> EnergyResult:
        frac = state.positions @ np.linalg.inv(state.cell)
        energy = 0.5 * float(np.sum(frac * frac))
        grad_cart = frac @ np.linalg.inv(state.cell).T
        stress = np.diag([0.1, -0.05, 0.0])
        return EnergyResult(energy=energy, gradient=grad_cart, stress=stress)

    def evaluate_flat(self, flat_positions: np.ndarray, template: State) -> tuple[float, np.ndarray]:
        result = self.evaluate(template.with_flat_positions(flat_positions))
        return result.energy, result.gradient.reshape(-1)


def slab_state() -> State:
    cell = np.diag([5.0, 6.0, 20.0])
    frac = np.array([[0.2, 0.2, 0.1], [0.4, 0.2, 0.1]])
    return State(
        numbers=np.array([6, 6]),
        positions=frac @ cell,
        cell=cell,
        pbc=(True, True, False),
    )


def test_variable_cell_relaxer_leaves_periodic_axes_unbounded_and_preserves_slab_z_cell() -> None:
    state = slab_state()
    gcoord = GeneralizedCoordinates.from_state(state, "slab_xy")
    evaluator = CalculatorGeneralizedEvaluator(InPlaneCellCalculator())
    relaxer = VariableCellRelaxer(evaluator, gcoord, max_atom_step=0.1, max_cell_strain=0.05)
    q0 = gcoord.to_q(state)
    bounds = relaxer._bounds(q0)
    assert bounds[0] == (None, None)
    assert bounds[1] == (None, None)
    assert bounds[2] == (q0[2] - 0.1, q0[2] + 0.1)
    relaxed = relaxer.relax(q0, fmax=1e-2, maxiter=1)
    np.testing.assert_allclose(relaxed.state.cell[2], state.cell[2])
