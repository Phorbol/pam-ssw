import numpy as np

from pamssw import LSSSWConfig, SSWConfig, State, run_ls_ssw, run_ssw
from pamssw.calculators import AnalyticCalculator
from pamssw.potentials import CoupledPairWell


def test_ls_ssw_crosses_stiff_landscape_more_effectively_than_ssw():
    state = State(
        numbers=np.array([1, 1]),
        positions=np.array([[-0.35, 0.0, 0.0], [0.35, 0.0, 0.0]]),
    )
    calc = AnalyticCalculator(CoupledPairWell())
    base = dict(
        max_trials=1,
        max_steps_per_walk=2,
        target_uphill_energy=0.05,
        rng_seed=11,
    )

    ssw = run_ssw(state, calc, SSWConfig(**base))
    ls = run_ls_ssw(
        state,
        calc,
        LSSSWConfig(
            **base,
            local_softening_strength=0.9,
            local_softening_pairs=[(0, 1)],
        ),
    )

    def pair_distances(result):
        return sorted(
            np.linalg.norm(entry.state.positions[1] - entry.state.positions[0])
            for entry in result.archive.entries
        )

    assert len(ls.archive.entries) >= len(ssw.archive.entries)
    assert ls.best_energy <= ssw.best_energy + 1e-8
    assert pair_distances(ls)[-1] > 1.0
    assert pair_distances(ssw)[-1] < 0.9
