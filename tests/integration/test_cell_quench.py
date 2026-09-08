"""Experimental atom-only SSW propagation followed by true E+pV quenching."""

import numpy as np
import pytest
from ase import units
from ase.build import bulk
from ase.calculators.lj import LennardJones

from pamssw import SSWConfig, LSSSWConfig, QuenchConvergenceError, state_from_atoms
from pamssw.calculators import ASECalculator
from pamssw.walker import SurfaceWalker


class ObservedWalker(SurfaceWalker):
    def _proposal_pool(self, seed_state, archive, *args, **kwargs):
        proposals = super()._proposal_pool(seed_state, archive, *args, **kwargs)
        for proposal in proposals:
            np.testing.assert_array_equal(proposal.state.cell, seed_state.cell)
        self.observed_seed_cell = seed_state.cell.copy()
        return proposals


@pytest.mark.parametrize("softening", [False, True])
def test_periodic_lj_search_retains_relaxed_cell_and_enthalpy(softening, tmp_path):
    initial = state_from_atoms(bulk('Ar', 'fcc', a=1.8, cubic=True))
    cls = LSSSWConfig if softening else SSWConfig
    config = cls(
        max_trials=1, max_steps_per_walk=1, oracle_candidates=2,
        proposal_relax_steps=4, quench_optimizer='ase-lbfgs',
        quench_fallback_optimizer='ase-fire', quench_cell_mode='volume_only',
        quench_fmax=0.005, quench_stress_tol=0.001,
        external_pressure_gpa=1.0, quench_maxiter=150, rng_seed=8,
        write_proposal_minima=True, proposal_minima_dir=str(tmp_path),
    )
    calc = ASECalculator(LennardJones(rc=2.7))
    walker = ObservedWalker(calc, config, softening_enabled=softening)
    result = walker.run(initial)
    assert result.stats['quench_cell_mode'] == 'volume_only'
    assert result.stats['objective'] == 'enthalpy'
    assert result.stats['variable_cell_supported'] == 0  # no joint cell escape
    assert not np.allclose(walker.observed_seed_cell, initial.cell)
    assert len(result.archive.entries) >= 1
    for entry in result.archive.entries:
        evaluation = calc.evaluate(entry.state)
        volume = np.linalg.det(entry.state.cell)
        assert entry.energy == pytest.approx(evaluation.energy + units.GPa * volume, abs=1e-8)
        assert entry.state.metadata['potential_energy'] == pytest.approx(evaluation.energy)
        assert entry.state.metadata['volume'] == pytest.approx(volume)
        assert np.max(np.linalg.norm(evaluation.gradient, axis=1)) <= config.quench_fmax
        assert abs(np.trace(evaluation.stress) / 3 + units.GPa) <= config.quench_stress_tol
    assert walker.calculator.snapshot().total == result.stats['force_evaluations']
    assert walker.calculator.snapshot().as_dict()['unattributed'] == 0
    from ase.io import read
    outputs = list(tmp_path.glob("*.xyz"))
    assert outputs
    frame = read(outputs[0])
    assert frame.info['enthalpy'] == pytest.approx(
        frame.info['potential_energy'] + units.GPa * frame.info['volume'])
    assert frame.info['quench_cell_mode'] == 'volume_only'
    assert 'stress_norm' in frame.info
    assert 'force_max' in frame.info


def test_initial_atomic_force_convergence_cannot_hide_nonzero_stress():
    initial = state_from_atoms(bulk('Ar', 'fcc', a=1.8, cubic=True))
    walker = SurfaceWalker(
        ASECalculator(LennardJones(rc=2.7)),
        SSWConfig(max_trials=1, quench_optimizer='ase-fire',
                  quench_cell_mode='volume_only', quench_maxiter=1,
                  quench_fmax=0.1, quench_stress_tol=1e-9),
        softening_enabled=False,
    )
    with pytest.raises(QuenchConvergenceError) as caught:
        walker.run(initial)
    assert caught.value.relaxation.gradient_norm < 0.1
    assert caught.value.relaxation.stress_norm > 1e-9
    assert caught.value.evaluation_counts.total > 0
