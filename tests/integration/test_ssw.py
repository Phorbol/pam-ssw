import numpy as np

from pamssw import SSWConfig, State, run_ssw
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import DoubleWell2D


def test_ssw_finds_multiple_basins_from_single_seed():
    state = State(
        numbers=np.array([1]),
        positions=np.array([[-1.0, 0.0, 0.0]]),
    )
    result = run_ssw(
        initial_state=state,
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(
            max_trials=8,
            max_steps_per_walk=8,
            target_uphill_energy=1.2,
            target_negative_curvature=0.4,
            rng_seed=7,
        ),
    )

    xs = sorted(round(entry.state.positions[0, 0], 1) for entry in result.archive.entries)

    assert len(result.archive.entries) >= 2
    assert xs[0] <= -0.9
    assert xs[-1] >= 0.9
