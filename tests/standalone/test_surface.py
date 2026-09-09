import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT


def api():
    from pamssw.standalone import surface
    return surface


def test_independent_emt_energy_forces_and_input_ownership():
    s = api()
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
    source = atoms.copy()
    source.calc = EMT()
    oracle = s.ASESurface(EMT())
    e, f = oracle.evaluate(atoms)
    assert e == pytest.approx(source.get_potential_energy())
    assert np.allclose(f, source.get_forces())
    assert atoms.calc is None
    assert oracle.requests == 1


def test_quench_uses_true_surface_and_reports_nonconvergence():
    s = api()
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
    surface = s.ASESurface(EMT())
    failed = s.quench(atoms, surface, fmax=1e-5, steps=0)
    assert not failed.converged
    result = s.quench(atoms, surface, fmax=1e-5, steps=100)
    assert result.converged
    assert result.max_force <= 1e-5
    assert result.energy < failed.energy
    assert np.allclose(atoms.positions, [[0, 0, 0], [2.7, 0, 0]])
    assert result.atoms.calc is None


def test_biased_calculator_keeps_true_and_modified_energy_separate():
    s = api()
    class LinearTerm:
        def evaluate(self, atoms):
            forces = np.zeros((len(atoms), 3))
            forces[0, 0] = -0.2
            return 0.2 * atoms.positions[0, 0], forces
    atoms = Atoms('Cu2', positions=[[0.3, 0, 0], [2.7, 0, 0]])
    surface = s.ASESurface(EMT())
    e, f = surface.evaluate(atoms)
    atoms.calc = s.SurfaceCalculator(surface, terms=(LinearTerm(),))
    assert atoms.get_potential_energy() == pytest.approx(e + .06)
    assert atoms.get_forces()[0, 0] == pytest.approx(f[0, 0] - .2)
    assert atoms.calc.results['true_energy'] == pytest.approx(e)


def test_constraints_are_not_silently_ignored():
    from ase.constraints import FixAtoms
    s = api()
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(NotImplementedError, match='constraint'):
        s.ASESurface(EMT()).evaluate(atoms)
