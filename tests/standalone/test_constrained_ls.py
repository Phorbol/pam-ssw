import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT

from pamssw.standalone.constrained_ls import (ConstrainedLSRuntime,
                                               ConstrainedNativeLSRuntime)
from pamssw.standalone.constrained_reference import ReducedCartesianChart
from pamssw.standalone.paper_reference import LSSettings
from pamssw.standalone.ls_native_reference import NativeLSSettings
from pamssw.standalone.periodic_softening import FrozenPeriodicBondSoftening


def _full_cu():
    return Atoms("Cu2", positions=[[0, 0, 0], [2.4, 0, 0]],
                 cell=np.diag([4., 4., 4.]), pbc=True)


def _ls():
    return LSSettings(bond_energies={(29, 29): 3.0},
                      bond_lengths={(29, 29): 2.0}, target_per_atom=.2)


def _native_ls():
    return NativeLSSettings(bond_energies={(29, 29): 3.0},
                            bond_lengths={(29, 29): 5.0})


class EF:
    def __init__(self):
        self.calculator = EMT()
        self.requests = 0

    def evaluate(self, atoms):
        self.requests += 1
        trial = atoms.copy(); trial.calc = self.calculator
        return trial.get_potential_energy(), trial.get_forces()


def test_full_and_partial_pbc_use_image_resolved_ls():
    atoms = _full_cu()
    runtime = ConstrainedLSRuntime.initialize(atoms, EF(), _ls(), fixed_indices=[0])
    assert isinstance(runtime.softening, FrozenPeriodicBondSoftening)
    assert any(tuple(shift) != (0, 0, 0) for shift in runtime.softening.image_shifts)

    partial = atoms.copy(); partial.pbc = [True, True, False]
    partial_runtime = ConstrainedLSRuntime.initialize(partial, EF(), _ls(), fixed_indices=[0])
    assert isinstance(partial_runtime.softening, FrozenPeriodicBondSoftening)
    assert all(shift[2] == 0 for shift in partial_runtime.softening.image_shifts)


def test_fixed_neighbor_contributes_active_ls_force():
    atoms = _full_cu()
    physical = EF()
    runtime = ConstrainedLSRuntime.initialize(atoms, physical, _ls(), fixed_indices=[0])
    chart = ReducedCartesianChart(atoms, fixed_indices=[0])
    active = chart.active_indices[0]
    displaced = chart.atoms(np.array([0.1, 0., 0.]))
    _, full_force = runtime.soft_surface.evaluate(displaced)
    _, physical_force = physical.evaluate(displaced)
    _, ls_force = runtime.softening.evaluate(displaced)
    np.testing.assert_allclose(full_force-physical_force,ls_force,atol=1e-13)
    assert np.linalg.norm(ls_force[active]) > 0
    assert np.array_equal(displaced.positions[0], atoms.positions[0])


def test_response_uses_all_atoms_and_preserves_prepared_state():
    atoms = _full_cu()
    runtime = ConstrainedLSRuntime.initialize(atoms, EF(), _ls(), fixed_indices=[0])
    old_total = sum(runtime.softening.strengths)
    expected = old_total - len(atoms)*runtime.settings.learning_rate*(.1/len(atoms)-runtime.settings.target_per_atom)
    updated = runtime.update(atoms.copy(), energy_before=0., energy_after=.1)
    assert sum(updated.strengths) == pytest.approx(expected)
    assert runtime.response.steps == 1
    assert updated is runtime.softening
    assert len(updated.numbers) == 2


def test_native_adapter_keeps_fixed_endpoint_bonds_and_reuses_cycle_runtime():
    atoms = _full_cu()
    physical = EF()
    runtime = ConstrainedNativeLSRuntime.initialize(
        atoms, physical, _native_ls(), fixed_indices=[0])
    assert runtime.native is not None
    assert len(runtime.softening.pairs) > 0
    # The frozen native potential is evaluated on a constraint-free copy, while
    # the active chart decides which force components reach the optimizer.
    chart = ReducedCartesianChart(atoms, fixed_indices=[0])
    _, force = runtime.soft_surface.evaluate(chart.atoms(np.array([.1, 0., 0.])))
    assert np.linalg.norm(force[1]) > 0
    before = runtime.native.steps
    runtime.update(atoms.copy(), energy_before=0., energy_after=.1)
    assert runtime.native.steps == before + 1
    assert runtime.native.last_update['bond_count'] == len(runtime.native.frozen.pairs)


def _public_input():
    from ase.build import bulk
    from ase.constraints import FixAtoms
    a=bulk('Cu','fcc',a=3.6,cubic=True)
    a.set_constraint(FixAtoms(indices=[0]))
    return a


def _public_run(surface, settings, **kwargs):
    from pamssw.standalone.constrained_reference import run_constrained_ssw,ConstrainedSSWConfig
    return run_constrained_ssw(_public_input(),surface,steps=2,
        config=ConstrainedSSWConfig(width=.1,rotation_bias=100.,max_gaussians=1),
        rng=np.random.default_rng(17),ls=settings,**kwargs)


def test_initialization_failure_does_not_count_paid_initial_twice():
    from pamssw.standalone.surface import ASESurface
    s=ASESurface(EMT())
    r=_public_run(s,LSSettings(bond_energies={(29,29):1.},bond_lengths={(29,29):.1},target_per_atom=.001))
    assert len(r.minima)==1 and r.current is r.initial
    assert r.requests==s.requests==sum(e['requests'] for e in r.records)
    assert r.status=='ls_initialization_failed'


def test_preparation_failure_preserves_last_accepted_soft_geometry(monkeypatch):
    from pamssw.standalone import constrained_ls
    from pamssw.standalone.generalized_numerics import GeneralizedRelaxResult
    from pamssw.standalone.surface import ASESurface
    def limited(q,evaluate,**kwargs):
        q=q+.05;e,g=evaluate(q)
        return GeneralizedRelaxResult(q,e,g,'maxiter',1,1,(),0,0,0)
    monkeypatch.setattr(constrained_ls,'safe_lbfgs',limited)
    s=ASESurface(EMT())
    r=_public_run(s,LSSettings(bond_energies={(29,29):1.},bond_lengths={(29,29):2.9},target_per_atom=.001))
    event=r.records[-1]
    assert r.status=='ls_prequench_failed' and len(r.minima)==1
    np.testing.assert_array_equal(event['last_work'].positions,event['ls_preparation'].atoms.positions)
    assert event['ls_preparation'].energy_after is None
    assert r.requests==s.requests==sum(e['requests'] for e in r.records)
