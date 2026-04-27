import numpy as np

from pamssw import SSWConfig, State, run_ssw
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import DoubleWell2D


def test_multi_proposal_local_relaxation_accounting_is_exact():
    result = run_ssw(
        initial_state=State(
            numbers=np.ones(4, dtype=int),
            positions=np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
        ),
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(
            max_trials=3,
            max_steps_per_walk=2,
            proposal_pool_size=2,
            cluster_reseed_interval=100,
            rng_seed=3,
        ),
    )

    assert result.stats["n_trials"] == 3
    assert result.stats["proposal_pool_size"] == 2
    assert result.stats["local_relaxations"] == 1 + 3 * 2
    assert len(result.walk_history) == 3
