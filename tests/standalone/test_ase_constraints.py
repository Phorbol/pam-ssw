import json
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, Hookean, FixBondLength

from pamssw.standalone.ase_constraints import (
    ConstraintSet, HookeanSurface, bind_hookean_surface, normalize_constraints)


class ZeroSurface:
    def __init__(self, fail=False):
        self.requests = 0
        self.exhausted = False
        self.fail = fail

    def evaluate(self, atoms):
        self.requests += 1
        if self.fail:
            raise RuntimeError('physical failure')
        return 3.0, np.zeros((len(atoms), 3))


def test_normalize_clean_attach_is_copy_and_fixatoms_last():
    atoms = Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]])
    atoms.set_constraint([Hookean(0, 1, 2.0, 1.0), FixAtoms(indices=[0])])
    original = atoms.positions.copy()
    spec = normalize_constraints(atoms)
    assert isinstance(spec, ConstraintSet)
    assert spec.fixed_indices == (0,)
    assert len(spec.hookean_specs) == 1
    assert np.array_equal(atoms.positions, original) and len(atoms.constraints) == 2
    clean = spec.clean_atoms(atoms)
    assert not clean.constraints and np.array_equal(clean.positions, original)
    attached = spec.attach(atoms)
    assert isinstance(attached.constraints[0], Hookean)
    assert isinstance(attached.constraints[-1], FixAtoms)
    attached.positions[1, 0] = 9
    assert atoms.positions[1, 0] == 2


def test_hookean_surface_uses_native_pair_energy_and_force_once():
    atoms = Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]])
    physical = ZeroSurface()
    wrapped = HookeanSurface(physical, normalize_constraints(
        atoms.set_constraint(Hookean(0, 1, 2.0, 1.0)) or atoms).hookean_specs)
    energy, forces = wrapped.evaluate(atoms)
    assert energy == pytest.approx(4.0)
    np.testing.assert_allclose(forces, [[2., 0, 0], [-2., 0, 0]])
    assert physical.requests == 1
    assert wrapped.last_evaluation['hookean_energy'] == pytest.approx(1.0)


def test_hookean_surface_pair_uses_mic_and_threshold():
    atoms = Atoms('H2', positions=[[0, 0, 0], [9, 0, 0]], cell=[10, 10, 10], pbc=True)
    atoms.set_constraint(Hookean(0, 1, 2.0, .5))
    wrapped = HookeanSurface(ZeroSurface(), normalize_constraints(atoms).hookean_specs)
    energy, forces = wrapped.evaluate(atoms)
    assert energy == pytest.approx(3.25)
    np.testing.assert_allclose(forces[:, 0], [-1., 1.])
    atoms.positions[1, 0] = .5
    energy, forces = wrapped.evaluate(atoms)
    assert energy == pytest.approx(3.0)
    np.testing.assert_allclose(forces, 0.)


@pytest.mark.parametrize('positions,constraint,expected', [
    ([[4., 0., 0.]], Hookean(0, [2., 0., 0.], 2., 1.), 1.),
    ([[0., 0., 2.]], Hookean(0, (0., 0., 1., -1.), 2.), 1.),
])
def test_hookean_surface_point_and_plane_use_native_adjustments(positions, constraint, expected):
    atoms = Atoms('H', positions=positions)
    atoms.set_constraint(constraint)
    wrapped = HookeanSurface(ZeroSurface(), normalize_constraints(atoms).hookean_specs)
    energy, forces = wrapped.evaluate(atoms)
    assert energy == pytest.approx(3. + expected)
    assert np.isfinite(forces).all()


def test_fixatoms_projection_is_left_to_ase_and_wrapper_keeps_full_hookean_force():
    atoms = Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]])
    atoms.set_constraint([Hookean(0, 1, 2., 1.), FixAtoms(indices=[0])])
    spec = normalize_constraints(atoms)
    wrapped = HookeanSurface(ZeroSurface(), spec.hookean_specs)
    energy, forces = wrapped.evaluate(atoms)
    assert energy == pytest.approx(4.)
    np.testing.assert_allclose(forces[:, 0], [2., -2.])
    attached = spec.attach(atoms)
    attached.calc = _ConstantCalculator()
    np.testing.assert_allclose(attached.get_forces()[:, 0], [0., -2.])


def test_invalid_constraint_and_failure_accounting():
    atoms = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
    atoms.set_constraint(FixBondLength(0, 1))
    with pytest.raises(TypeError, match='FixAtoms|Hookean'):
        normalize_constraints(atoms)
    physical = ZeroSurface(fail=True)
    wrapped = bind_hookean_surface(physical, ())
    assert wrapped is physical
    constrained = Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]])
    constrained.set_constraint(Hookean(0, 1, 1., .5))
    spec = normalize_constraints(constrained).hookean_specs
    with pytest.raises(RuntimeError):
        HookeanSurface(physical, spec).evaluate(atoms)
    assert physical.requests == 1


class _ConstantCalculator:
    implemented_properties = ['energy', 'forces']
    def get_potential_energy(self, atoms, **kwargs):
        return 3.0
    def get_forces(self, atoms):
        return np.zeros((len(atoms), 3))


def test_failed_request_clears_preceding_success_snapshot_and_binding_is_unique():
    atoms = Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]])
    atoms.set_constraint(Hookean(0, 1, 1., .5))
    specs = normalize_constraints(atoms).hookean_specs
    physical = ZeroSurface()
    wrapped = bind_hookean_surface(physical, specs)
    assert bind_hookean_surface(wrapped, specs) is wrapped
    atoms.set_constraint(Hookean(0, 1, 2., .5))
    with pytest.raises(ValueError, match='different'):
        bind_hookean_surface(wrapped, normalize_constraints(atoms).hookean_specs)
    wrapped.evaluate(atoms)
    assert wrapped.last_evaluation is not None
    physical.fail = True
    with pytest.raises(RuntimeError):
        wrapped.evaluate(atoms)
    assert wrapped.last_evaluation is None
    assert physical.requests == 2


@pytest.mark.parametrize('constraint', [
    FixAtoms(indices=[2]), Hookean(0, 1, 1.),
    Hookean(0, [0., 0., 0.], 1.), Hookean(0, 1, -1., .5),
    Hookean(0, (0., 0., 0., 1.), 1.),
])
def test_invalid_specs_fail_at_normalization(constraint):
    atoms = Atoms('H2', positions=[[0, 0, 0], [2, 0, 0]])
    atoms.set_constraint(constraint)
    with pytest.raises(ValueError):
        normalize_constraints(atoms)


def test_periodic_point_and_plane_match_native_ase_forces():
    for constraint in [Hookean(0, [9., 0., 0.], 2., .5),
                       Hookean(0, (1., 2., 3., -1.), 2.)]:
        atoms = Atoms('H', positions=[[0., 0., 2.]], cell=[10, 10, 10], pbc=True)
        atoms.set_constraint(constraint)
        wrapped = HookeanSurface(ZeroSurface(), normalize_constraints(atoms).hookean_specs)
        energy, forces = wrapped.evaluate(atoms)
        atoms.calc = _ConstantCalculator()
        assert energy == pytest.approx(atoms.get_potential_energy())
        np.testing.assert_allclose(forces, atoms.get_forces())
