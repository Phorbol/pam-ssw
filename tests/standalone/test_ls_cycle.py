"""Lifecycle/force-contract tests on a harmonic model, not physical validation."""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pamssw.standalone.softening import FrozenBondSoftening
from pamssw.standalone.surface import ASESurface, SurfaceCalculator
from pamssw.standalone.gaussian import ProjectedGaussian
from pamssw.standalone.ls_cycle import prepare_ls_step, finish_ls_step, LSCycleError


class AnchoredHarmonic(Calculator):
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        displacement = self.atoms.positions - np.array([[-.5, 0, 0], [.5, 0, 0]])
        self.results = {'energy': float(5*np.sum(displacement**2)), 'forces': -10*displacement}


def setup():
    atoms = Atoms('HH', positions=[[-.5,0,0], [.5,0,0]])
    surface = ASESurface(AnchoredHarmonic())
    soft = FrozenBondSoftening.from_atoms(atoms, bond_energies={(1,1):4}, bond_lengths={(1,1):1.2})
    return atoms, surface, soft


def test_prepare_relaxes_only_soft_surface_and_reports_true_response():
    atoms, surface, soft = setup()
    result = prepare_ls_step(atoms, surface, softening=soft, fmax=1e-7, steps=50)
    assert result.soft_quench.converged
    assert result.soft_quench.surface == 'modified'
    assert result.softening is soft
    assert result.energy_before == pytest.approx(0)
    true_energy, true_force = surface.evaluate(result.atoms)
    assert result.energy_after == pytest.approx(true_energy)
    assert result.energy_response == pytest.approx(true_energy/2)
    assert result.energy_response > 0
    assert np.linalg.norm(true_force,axis=1).max() > 1e-3
    assert result.soft_quench.energy != pytest.approx(true_energy)
    assert result.evaluation_requests == result.soft_quench.evaluation_requests + 2
    assert result.atoms.calc is None
    np.testing.assert_array_equal(atoms.positions, [[-.5,0,0],[.5,0,0]])


def test_finish_removes_attached_ls_and_gaussian():
    atoms, surface, soft = setup()
    prepared = prepare_ls_step(atoms, surface, softening=soft, fmax=1e-7, steps=50)
    candidate = prepared.atoms.copy()
    direction = np.zeros((2,3));direction[0,0]=1
    gaussian = ProjectedGaussian(atoms.positions, direction, sigma=.5, weight=1.)
    candidate.calc = SurfaceCalculator(surface, terms=(soft, gaussian))
    result = finish_ls_step(candidate, surface, fmax=1e-7, steps=50)
    assert result.surface == 'true'
    assert result.converged
    assert result.energy == pytest.approx(0, abs=1e-14)
    assert result.atoms.calc is None
    assert candidate.calc.terms == (soft, gaussian)


def test_prepare_rejects_nonstationary_start_before_optimizer():
    atoms, surface, soft = setup()
    atoms.positions[0,1] = .1
    with pytest.raises(LSCycleError) as info:
        prepare_ls_step(atoms, surface, softening=soft, fmax=1e-7, steps=50)
    assert info.value.stage == 'true_start'
    assert surface.requests == 1


def test_unconverged_stages_fail_explicitly():
    atoms, surface, soft = setup()
    with pytest.raises(LSCycleError) as info:
        prepare_ls_step(atoms, surface, softening=soft, fmax=1e-7, steps=0)
    assert info.value.stage == 'soft_quench'
    assert not info.value.result.converged
    atoms.positions[0,1] = .1
    with pytest.raises(LSCycleError) as info:
        finish_ls_step(atoms, surface, fmax=1e-7, steps=0)
    assert info.value.stage == 'true_finish'
    assert info.value.result.surface == 'true'


def test_reject_stale_frozen_reference_even_at_true_stationary_start():
    atoms, surface, soft = setup()
    stale = atoms.copy();stale.positions[1,0] += .01
    bad_soft = FrozenBondSoftening.from_atoms(stale, bond_energies={(1,1):4}, bond_lengths={(1,1):1.2})
    with pytest.raises(ValueError, match='reference distances'):
        prepare_ls_step(atoms, surface, softening=bad_soft, fmax=1e-7, steps=50)


def test_surface_biases_and_direction_share_force_sign_and_cartesian_metric():
    from pamssw.standalone.direction import reference_soft_mode
    atoms, surface, soft = setup()
    direction = np.array([[-1.,0,0], [1.,0,0]])/np.sqrt(2)
    gaussian = ProjectedGaussian(atoms.positions, direction, sigma=.5, weight=1.)
    modified = ASESurface(SurfaceCalculator(surface, terms=(soft, gaussian)))
    original = atoms.positions.copy()
    mode = reference_soft_mode(atoms, direction, fd_step=1e-5, max_hvp=3,
                               residual_tol=1e-6, evaluate=modified.evaluate)
    # k=10 + pair relative curvature 2*A/(xi*r0)^2 - W/sigma^2 = 12.
    assert mode.curvature == pytest.approx(12., abs=1e-6)
    assert mode.converged
    np.testing.assert_array_equal(atoms.positions, original)
