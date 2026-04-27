import numpy as np

from pamssw import SSWConfig, State, run_ssw
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import DoubleWell2D
from pamssw.walker import SurfaceWalker


def test_ssw_local_relaxation_accounting_is_exact():
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
            rng_seed=3,
        ),
    )

    assert result.stats["n_trials"] == 3
    assert result.stats["local_relaxations"] == 1 + 3
    assert len(result.walk_history) == 3
    assert result.stats["coordinate_system"] == "cartesian_fixed_cell"
    assert result.stats["variable_cell_supported"] == 0


def test_default_proposal_pool_uses_only_ssw_walk_for_cluster():
    state = State(
        numbers=np.full(38, 18),
        positions=np.random.default_rng(0).normal(size=(38, 3)),
    )
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, rng_seed=5),
        softening_enabled=False,
    )
    result = run_ssw(state, AnalyticCalculator(DoubleWell2D()), SSWConfig(max_trials=1, rng_seed=5))

    labels = [proposal.label for proposal in walker._proposal_pool(result.archive.entries[0].state, result.archive, 0)]

    assert labels == ["ssw_walk"]


def test_periodic_state_uses_only_ssw_walk_proposal_by_default():
    state = State(
        numbers=np.full(4, 18),
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        cell=np.eye(3) * 6.0,
        pbc=(True, True, True),
    )
    result = run_ssw(state, AnalyticCalculator(DoubleWell2D()), SSWConfig(max_trials=1, rng_seed=7))
    walker = SurfaceWalker(
        calculator=AnalyticCalculator(DoubleWell2D()),
        config=SSWConfig(max_trials=1, rng_seed=7),
        softening_enabled=False,
    )

    proposals = walker._proposal_pool(result.archive.entries[0].state, result.archive, 0)

    assert [proposal.label for proposal in proposals] == ["ssw_walk"]
