import numpy as np

from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import DoubleWell2D
from pamssw.state import State


def test_analytic_calculator_returns_energy_and_gradient():
    calc = AnalyticCalculator(DoubleWell2D())
    state = State(
        numbers=np.array([1]),
        positions=np.array([[0.3, -0.2, 0.0]]),
    )

    energy, gradient = calc.evaluate(state)

    assert np.isfinite(energy)
    assert gradient.shape == (1, 3)
    assert np.all(np.isfinite(gradient))
