"""Cu2/ASE-EMT interface integration only; not LS scientific validation.

The ASE EMT calculator supplies its normal Cu parameters. The LS tables below
are deliberately explicit synthetic test coefficients (1 eV, 3 Angstrom), NOT
standard Cu bond energies or a proposed physical neighbor/default parameter.
No LASP/Java runtime or external optimizer/calculator process is used.
"""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from ase.optimize import BFGS, LBFGS

from pamssw.standalone.ls_cycle import prepare_ls_step, finish_ls_step
from pamssw.standalone.softening import FrozenBondSoftening
from pamssw.standalone.surface import ASESurface, quench
from pamssw.standalone.paper_reference import LSSettings, SSWConfig, run_ssw

ENERGY_TABLE = {(29,29): 1.0}
LENGTH_TABLE = {(29,29): 3.0}


def test_cu2_emt_soft_preparation_and_bias_free_finish():
    surface = ASESurface(EMT())
    initial = Atoms('Cu2', positions=[[0,0,0], [2.7,0,0]])
    minimum = quench(initial, surface, fmax=1e-6, steps=100, optimizer=BFGS)
    assert minimum.converged
    soft = FrozenBondSoftening.from_atoms(minimum.atoms, bond_energies=ENERGY_TABLE,
                                         bond_lengths=LENGTH_TABLE)
    prepared = prepare_ls_step(minimum.atoms, surface, softening=soft,
                               fmax=1e-6, steps=100, optimizer=LBFGS)
    assert prepared.soft_quench.converged
    independently_evaluated = prepared.atoms.copy()
    independently_evaluated.calc = EMT()
    assert prepared.energy_after == pytest.approx(independently_evaluated.get_potential_energy())
    assert prepared.energy_response == pytest.approx((prepared.energy_after-minimum.energy)/2)
    assert prepared.energy_response > 0
    assert np.linalg.norm(independently_evaluated.get_forces(),axis=1).max() > 1e-4
    finished = finish_ls_step(prepared.atoms, surface, fmax=1e-6, steps=100, optimizer=LBFGS)
    assert finished.converged and finished.surface == 'true'
    assert finished.energy == pytest.approx(minimum.energy, abs=1e-9)
    np.testing.assert_array_equal(initial.positions, [[0,0,0],[2.7,0,0]])


def test_two_cu2_emt_ls_steps_keep_true_archive_and_complete_request_ledger():
    atoms = Atoms('Cu2', positions=[[0,0,0], [2.7,0,0]])
    # A separately specified BFGS pre-equilibration defines this integration
    # fixture. It is not a fallback inside the reference SSW driver.
    fixture = quench(atoms, ASESurface(EMT()), fmax=1e-6, steps=100, optimizer=BFGS)
    assert fixture.converged
    atoms = fixture.atoms.copy()
    source = atoms.positions.copy()
    surface = ASESurface(EMT())
    config = SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e-5, relax_steps=100, fd_step=1e-4,
        rotation_hvp=8, rotation_tol=1e-3, direction_sampling='global')
    settings = LSSettings(bond_energies=ENERGY_TABLE, bond_lengths=LENGTH_TABLE,
                          target_per_atom=.001)
    result = run_ssw(atoms, surface, steps=2, config=config,
                     rng=np.random.default_rng(19), ls=settings)
    assert result.status == 'completed'
    assert len(result.records) == 2
    assert len(result.minima) == 3
    assert all(record.energy_response is not None for record in result.records)
    assert all(record.landing is not None and record.landing.converged for record in result.records)
    assert result.evaluation_requests == surface.requests
    assert result.evaluation_requests == result.initial.evaluation_requests + sum(
        record.evaluation_requests for record in result.records)
    for minimum in result.minima:
        assert minimum.converged and minimum.surface == 'true'
        independent = minimum.atoms.copy()
        independent.calc = EMT()
        assert minimum.energy == pytest.approx(independent.get_potential_energy())
        assert np.linalg.norm(independent.get_forces(),axis=1).max() <= config.fmax
    assert atoms.calc is None
    np.testing.assert_array_equal(atoms.positions, source)


def test_cu2_emt_lbfgs_initial_failure_is_preserved_not_silently_repaired():
    from pamssw.standalone.paper_reference import InitialQuenchError
    atoms = Atoms('Cu2', positions=[[0,0,0], [2.7,0,0]])
    surface = ASESurface(EMT())
    config = SSWConfig(width=.1, rotation_bias=2., max_gaussians=1,
        temperature_K=300., fmax=1e-5, relax_steps=100, fd_step=1e-4,
        rotation_hvp=8, rotation_tol=1e-3, direction_sampling='global')
    with pytest.raises(InitialQuenchError) as info:
        run_ssw(atoms, surface, steps=1, config=config, rng=np.random.default_rng(19))
    assert not info.value.result.converged
    assert info.value.result.optimizer_steps == 100
    assert info.value.result.max_force > config.fmax
    assert info.value.result.evaluation_requests == surface.requests
