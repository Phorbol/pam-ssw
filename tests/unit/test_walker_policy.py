import numpy as np

from pamssw import SSWConfig
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import DoubleWell2D
from pamssw.state import State
from pamssw.walker import ProposalPotential, SoftModeOracle, SurfaceWalker


def test_bias_weight_matches_curvature_inversion_rule():
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(target_negative_curvature=0.2),
        softening_enabled=False,
    )

    assert walker._bias_weight(curvature=0.3, sigma=2.0) == 2.0
    assert walker._bias_weight(curvature=-0.5, sigma=2.0) == 0.0


def test_soft_mode_oracle_returns_best_candidate_without_random_mixing():
    class Quadratic:
        def energy_gradient(self, flat_positions, state):
            hessian = np.diag([1.0, 4.0, 9.0])
            gradient = hessian @ flat_positions
            energy = 0.5 * float(flat_positions @ gradient)
            return energy, gradient

    state = State(numbers=np.array([1]), positions=np.array([[0.0, 0.0, 0.0]]))
    rng = np.random.default_rng(0)
    oracle = SoftModeOracle(AnalyticCalculator(Quadratic()), rng, candidates=1)
    direction = np.array([1.0, 0.0, 0.0])

    choice = oracle.choose_direction(
        state,
        proposal=ProposalPotential(AnalyticCalculator(Quadratic())),
        previous_direction=direction,
    )

    np.testing.assert_allclose(choice.direction, direction)
