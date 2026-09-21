"""Focused contract tests for constrained FixAtoms + ASE Hookean support."""
from __future__ import annotations
import json
import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, Hookean
from pamssw.standalone.ase_constraints import bind_hookean_surface, normalize_constraints
from pamssw.standalone.constrained_reference import constrained_quench
from pamssw.standalone.constrained_reference import (
    ConstrainedSSWConfig, run_constrained_ssw, load_constrained_checkpoint,
)
from pamssw.standalone.paper_reference import LSSettings
from pamssw.standalone.ls_native_reference import NativeLSSettings

class QuadraticSurface:
    def __init__(self, anchor=None):
        self.requests = 0; self.last_evaluation = None
        self.anchor = None if anchor is None else np.asarray(anchor, dtype=float)
    @property
    def exhausted(self): return False
    def evaluate(self, atoms):
        self.requests += 1
        x = np.asarray(atoms.positions, dtype=float).copy()
        d = x if self.anchor is None else x - self.anchor
        energy = 0.5 * float(np.square(d).sum())
        forces = -d
        self.last_evaluation = {"energy": energy, "forces": forces.copy()}
        return energy, forces

def _hookean_atoms(*, fixed=False):
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.20, 0.0, 0.0]])
    constraints = [Hookean(0, 1, k=2.0, rt=0.70)]
    if fixed: constraints.insert(0, FixAtoms(indices=[0]))
    atoms.set_constraint(constraints)
    return atoms

def test_constraint_normalization_clean_and_attach_preserves_hookean_and_fixed():
    atoms = _hookean_atoms(fixed=True)
    normalized = normalize_constraints(atoms)
    assert normalized.fixed_indices == (0,)
    assert len(normalized.hookean_specs) == 1
    json.dumps(normalized.hookean_specs)
    assert all(not isinstance(item, (FixAtoms, Hookean)) for item in normalized.hookean_specs)
    clean = normalized.clean_atoms(atoms)
    assert clean.constraints == []
    restored = normalized.attach(clean)
    assert any(isinstance(c, FixAtoms) for c in restored.constraints)
    assert any(isinstance(c, Hookean) for c in restored.constraints)
    assert np.array_equal(restored.positions, atoms.positions)

def test_bound_hookean_surface_adds_native_energy_and_forces_once():
    atoms = _hookean_atoms(fixed=False)
    normalized = normalize_constraints(atoms)
    clean = normalized.clean_atoms(atoms)
    physical = QuadraticSurface()
    bound = bind_hookean_surface(physical, normalized.hookean_specs)
    energy, forces = bound.evaluate(clean)
    base_energy, base_forces = QuadraticSurface().evaluate(clean)
    correction = Hookean(0, 1, k=2.0, rt=0.70)
    expected_forces = base_forces.copy()
    correction.adjust_forces(clean, expected_forces)
    expected_energy = base_energy + correction.adjust_potential_energy(clean)
    assert energy == pytest.approx(expected_energy)
    assert np.allclose(forces, expected_forces)
    assert physical.requests == 1 and bound.requests == 1
    assert bound.last_evaluation["physical_energy"] == pytest.approx(base_energy)
    assert np.allclose(bound.last_evaluation["physical_forces"], base_forces)
    assert bound.last_evaluation["hookean_energy"] == pytest.approx(expected_energy - base_energy)
    assert np.allclose(bound.last_evaluation["hookean_forces"], expected_forces - base_forces)

def test_constrained_quench_accepts_hookean_only_and_returns_physical_plus_hookean():
    atoms = _hookean_atoms(fixed=False)
    result = constrained_quench(atoms, QuadraticSurface(), fmax=0.05, max_step=0.2, maxiter=3)
    assert result.atoms.constraints
    assert any(isinstance(c, Hookean) for c in result.atoms.constraints)
    assert result.certificate["objective"] == "physical_plus_hookean"
    assert np.isfinite(result.full_raw_fmax)
    assert result.active_fmax <= result.full_raw_fmax + 1.0e-12

def test_constrained_quench_preserves_fixatoms_and_hookean_together():
    atoms = _hookean_atoms(fixed=True)
    result = constrained_quench(atoms, QuadraticSurface(), fmax=0.05, max_step=0.2, maxiter=2)
    assert any(isinstance(c, FixAtoms) for c in result.atoms.constraints)
    assert any(isinstance(c, Hookean) for c in result.atoms.constraints)
    assert np.allclose(result.atoms.positions[0], atoms.positions[0])

def test_unsupported_constraint_is_rejected_before_surface_call():
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0, 0]])
    from ase.constraints import ExternalForce
    atoms.set_constraint(ExternalForce(0, 1, [0.1, 0.0, 0.0]))
    with pytest.raises((TypeError, ValueError, NotImplementedError)):
        normalize_constraints(atoms)


@pytest.mark.parametrize("ls_factory", [
    lambda: None,
    lambda: LSSettings(bond_energies={(29, 29): 3.0}, bond_lengths={(29, 29): 1.2}, target_per_atom=.2),
    lambda: NativeLSSettings(bond_energies={(29, 29): 3.0}, bond_lengths={(29, 29): 1.2}, target_mev_per_atom=200.),
], ids=["plain", "paper-ls", "native-ls"])
def test_hookean_outer_boundary_resume_and_ls_state(tmp_path, ls_factory):
    def make_atoms():
        a = Atoms("Cu2", positions=[[0., 0., 0.], [1.20, 0., 0.]], cell=[8., 8., 8.], pbc=True)
        a.set_constraint([FixAtoms(indices=[0]), Hookean(0, 1, k=2., rt=.70)])
        return a
    cfg = ConstrainedSSWConfig(width=.1, rotation_bias=10., max_gaussians=1,
                               relax_steps=100, rotation_hvp=2)
    cp_path = tmp_path / "boundary.pkl"
    surface_factory = lambda: QuadraticSurface(anchor=[[0., 0., 0.], [1.2, 0., 0.]])
    first = run_constrained_ssw(make_atoms(), surface_factory(), steps=1,
                                config=cfg, rng=np.random.default_rng(17),
                                ls=ls_factory(), checkpoint_path=cp_path)
    checkpoint = load_constrained_checkpoint(cp_path)
    assert checkpoint.hookean_specs
    def nested_atoms(value):
        if isinstance(value, Atoms):
            return [value]
        if isinstance(value, dict):
            return sum((nested_atoms(v) for v in value.values()), [])
        if isinstance(value, (tuple, list)):
            return sum((nested_atoms(v) for v in value), [])
        if hasattr(value, "atoms") and isinstance(value.atoms, Atoms):
            return [value.atoms]
        return []
    assert any(v.constraints for v in nested_atoms(checkpoint.records[0]))
    resumed_surface = surface_factory()
    resumed = run_constrained_ssw(make_atoms(), resumed_surface, steps=0,
                                  config=cfg, rng=np.random.default_rng(17),
                                  ls=ls_factory(), checkpoint=checkpoint)
    assert resumed.requests == checkpoint.evaluation_requests
    assert resumed.current.atoms.constraints
    bad = make_atoms()
    bad.set_constraint([FixAtoms(indices=[0]), Hookean(0, 1, k=3., rt=.70)])
    zero = surface_factory()
    with pytest.raises(ValueError, match="Hookean"):
        run_constrained_ssw(bad, zero, steps=0, config=cfg,
                            rng=np.random.default_rng(17), ls=ls_factory(),
                            checkpoint=checkpoint)
    assert zero.requests == 0


@pytest.mark.parametrize("ls_factory", [
    lambda: None,
    lambda: LSSettings(bond_energies={(29, 29): 3.0}, bond_lengths={(29, 29): 1.2}, target_per_atom=.2),
    lambda: NativeLSSettings(bond_energies={(29, 29): 3.0}, bond_lengths={(29, 29): 1.2}, target_mev_per_atom=200.),
], ids=["plain", "paper-ls", "native-ls"])
@pytest.mark.parametrize("fixed", [True, False], ids=["fixed-hookean", "hookean-only"])
def test_hookean_resume_one_more_outer_step_matches_continuous(tmp_path, ls_factory, fixed):
    def make_atoms():
        a = Atoms("Cu2", positions=[[0., 0., 0.], [1.20, 0., 0.]], cell=[8., 8., 8.], pbc=True)
        constraints = [Hookean(0, 1, k=2., rt=.70)]
        if fixed:
            constraints.insert(0, FixAtoms(indices=[0]))
        a.set_constraint(constraints)
        return a

    def surface():
        return QuadraticSurface(anchor=[[0., 0., 0.], [1.2, 0., 0.]])

    cfg = ConstrainedSSWConfig(width=.1, rotation_bias=10., temperature_K=0.,
                               max_gaussians=1, relax_steps=100, rotation_hvp=2)
    continuous_path = tmp_path / "continuous.pkl"
    split_path = tmp_path / "split.pkl"
    continuous = run_constrained_ssw(make_atoms(), surface(), steps=2, config=cfg,
                                     rng=np.random.default_rng(31), ls=ls_factory(),
                                     checkpoint_path=continuous_path)
    split_first = run_constrained_ssw(make_atoms(), surface(), steps=1, config=cfg,
                                      rng=np.random.default_rng(31), ls=ls_factory(),
                                      checkpoint_path=split_path)
    boundary = load_constrained_checkpoint(split_path)
    assert boundary.next_index == 1
    resumed = run_constrained_ssw(make_atoms(), surface(), steps=1, config=cfg,
                                  rng=np.random.default_rng(31), ls=ls_factory(),
                                  checkpoint=boundary, checkpoint_path=split_path)

    assert continuous.status == resumed.status
    assert continuous.requests == resumed.requests
    assert len(continuous.records) == len(resumed.records)
    np.testing.assert_allclose(continuous.current.atoms.positions,
                               resumed.current.atoms.positions, atol=1e-12, rtol=0.)
    assert continuous.current.energy == pytest.approx(resumed.current.energy, abs=1e-12)
    continuous_boundary = load_constrained_checkpoint(continuous_path)
    resumed_boundary = load_constrained_checkpoint(split_path)
    assert continuous_boundary.next_index == resumed_boundary.next_index
    assert continuous_boundary.evaluation_requests == resumed_boundary.evaluation_requests
    assert continuous_boundary.rng_state == resumed_boundary.rng_state
    assert continuous_boundary.ls_state == resumed_boundary.ls_state
