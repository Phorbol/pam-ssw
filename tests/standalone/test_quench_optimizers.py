import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.surface import ASESurface, quench


class Harmonic(Calculator):
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.positions
        self.results = {'energy': float((x * x).sum()), 'forces': -2.0 * x}


class OffsetAnisotropic(Calculator):
    implemented_properties = ['energy', 'forces']

    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        x = atoms.positions
        coeff = np.array([1e-3, 2e-3, 3e-3])
        self.results = {'energy': 1e12 + float(.5 * (x * x * coeff).sum()),
                        'forces': -x * coeff}


def test_scipy_lbfgsb_uses_force_certificate_and_zero_steps_are_zero_step():
    atoms = Atoms('H', positions=[[1.0, 0.0, 0.0]])
    surface = ASESurface(Harmonic())
    result = quench(atoms, surface, fmax=1e-8, steps=0, optimizer='scipy-lbfgsb')
    np.testing.assert_array_equal(result.atoms.positions, atoms.positions)
    assert result.optimizer_steps == 0
    assert result.optimizer_telemetry.backend == 'scipy-lbfgsb'
    assert result.optimizer_telemetry.nit == 0
    assert result.converged is False
    assert surface.requests == 1


@pytest.mark.parametrize('optimizer', ['scipy-lbfgsb', 'ase-lbfgs-linesearch'])
def test_new_cartesian_baselines_reach_true_force_certificate(optimizer):
    atoms = Atoms('H', positions=[[1.0, 0.0, 0.0]])
    result = quench(atoms, ASESurface(Harmonic()), fmax=1e-6, steps=50,
                    optimizer=optimizer)
    assert result.converged
    assert result.max_force <= 1e-6
    if optimizer == 'scipy-lbfgsb':
        assert result.optimizer_telemetry.nfev >= 1


def test_scipy_lbfgsb_rejects_eckart_frame():
    atoms = Atoms('H3', positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    with pytest.raises(NotImplementedError, match='eckart'):
        SSWConfig(width=.1, rotation_bias=100., max_gaussians=1,
                  temperature_K=150., fmax=1e-3, relax_steps=1,
                  fd_step=1e-4, rotation_hvp=2, rotation_tol=.02,
                  quench_optimizer='scipy-lbfgsb', cluster_frame='eckart')


def test_scipy_success_does_not_override_force_certificate_after_relative_stop():
    atoms = Atoms('H', positions=[[1.0, 1.0, 1.0]])
    result = quench(atoms, ASESurface(OffsetAnisotropic()), fmax=1e-6,
                    steps=50, optimizer='scipy-lbfgsb')
    assert result.optimizer_telemetry.optimizer_success
    assert result.optimizer_telemetry.message
    assert result.max_force > 1e-6
    assert not result.converged


def test_ssw_config_accepts_new_baselines():
    fields = dict(width=.1, rotation_bias=100., max_gaussians=1,
                  temperature_K=150., fmax=.01, relax_steps=2,
                  fd_step=1e-4, rotation_hvp=2, rotation_tol=.02)
    for optimizer in ('scipy-lbfgsb', 'ase-lbfgs-linesearch'):
        assert SSWConfig(**fields, quench_optimizer=optimizer).quench_optimizer == optimizer
