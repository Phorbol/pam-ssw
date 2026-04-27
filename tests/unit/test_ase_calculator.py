import numpy as np
from ase.calculators.lj import LennardJones

from pamssw.calculators import ASECalculator, EnergyResult
from pamssw.state import State


def test_ase_calculator_returns_gradient_from_forces():
    calc = ASECalculator(LennardJones(sigma=1.0, epsilon=1.0))
    state = State(
        numbers=np.array([18, 18]),
        positions=np.array([[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]]),
    )

    result = calc.evaluate(state)
    energy, gradient = result

    assert isinstance(result, EnergyResult)
    assert np.isfinite(energy)
    assert gradient.shape == (2, 3)
    assert gradient[0, 0] * gradient[1, 0] < 0.0
