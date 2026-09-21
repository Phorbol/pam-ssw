import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.paper_reference import LSSettings, LSResponseState
from pamssw.standalone.softening import FrozenBondSoftening
from pamssw.standalone.periodic_softening import FrozenPeriodicBondSoftening
from pamssw.standalone.vc_softening import FrozenPeriodicCellSoftening
from pamssw.standalone.constrained_ls import ConstrainedLSRuntime


E = {(6, 6): 4.0, (6, 26): 5.0, (26, 26): 6.0}
L = {(6, 6): 2.0, (6, 26): 2.0, (26, 26): 2.0}
FILTER = {(26, 26): 0.0}


def _atoms(pbc=False):
    return Atoms('FeFeC', positions=[(0, 0, 0), (1, 0, 0), (0, 1, 0)],
                 cell=np.diag([8., 8., 8.]), pbc=pbc)


def test_fixed_filter_keeps_geometric_pair_but_zeroes_fe_fe_and_rebuild():
    a = _atoms()
    soft = FrozenBondSoftening.from_atoms(a, bond_energies=E, bond_lengths=L,
                                          energy_filter=FILTER)
    assert len(soft.pairs) == 3
    assert soft.energy_filter == (((26, 26), 0.0),)
    assert soft.strengths[soft.pairs.index((0, 1))] == 0.0
    state = LSResponseState(.1)
    rebuilt = state.update(soft, a.copy(), energy_before=0., energy_after=.3,
                           bond_energies=E, bond_lengths=L)
    assert rebuilt.energy_filter == soft.energy_filter
    assert rebuilt.strengths[rebuilt.pairs.index((0, 1))] == 0.0


def test_periodic_and_vc_forward_filter_without_deleting_pairs():
    a = _atoms(pbc=True)
    periodic = FrozenPeriodicBondSoftening.from_atoms(
        a, bond_energies=E, bond_lengths=L, energy_filter=FILTER)
    assert periodic.energy_filter == (((26, 26), 0.0),)
    assert any(tuple(a.numbers[i:i+1]) == (26,) and tuple(a.numbers[j:j+1]) == (26,)
               for i, j in periodic.pairs)
    vc = FrozenPeriodicCellSoftening.from_atoms(
        a, bond_energies=E, bond_lengths=L, energy_filter=FILTER)
    assert vc.energy_filter == periodic.energy_filter
    rebuilt = LSResponseState(.1).update(
        vc, a.copy(), energy_before=0., energy_after=0.,
        bond_energies=E, bond_lengths=L)
    assert isinstance(rebuilt, FrozenPeriodicCellSoftening)
    assert rebuilt.energy_filter == vc.energy_filter


def test_all_zero_filter_rejects_without_dropping_geometric_pairs():
    a = _atoms()
    with np.testing.assert_raises_regex(ValueError, 'all eligible'):
        FrozenBondSoftening.from_atoms(
            a, bond_energies=E, bond_lengths=L,
            energy_filter={(6, 6): 0., (6, 26): 0., (26, 26): 0.})


def test_constrained_runtime_forwards_filter():
    a = _atoms()
    class EF:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0.0, np.zeros((len(atoms), 3))
    settings = LSSettings(E, L, .1, energy_filter=FILTER)
    runtime = ConstrainedLSRuntime.initialize(a, EF(), settings, fixed_indices=())
    assert runtime.softening.energy_filter == (((26, 26), 0.0),)


def test_public_fixed_entrypoint_forwards_filter(monkeypatch):
    from pamssw.standalone import paper_reference
    from pamssw.standalone.surface import ASESurface
    from ase.calculators.calculator import Calculator, all_changes
    class Anchored(Calculator):
        implemented_properties = ['energy', 'forces']
        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            delta = atoms.positions - np.array([[0., 0., 0.], [1., 0., 0.]])
            self.results = {'energy': .5 * float(np.sum(delta**2)), 'forces': -delta}
    seen = []
    original = paper_reference.FrozenBondSoftening.from_atoms
    def capture(cls, atoms, **kwargs):
        seen.append(kwargs['energy_filter'])
        return original(atoms, **kwargs)
    monkeypatch.setattr(paper_reference.FrozenBondSoftening, 'from_atoms', classmethod(capture))
    a = Atoms('H2', positions=[(0., 0., 0.), (1., 0., 0.)])
    settings = LSSettings({(1, 1): 1.}, {(1, 1): 2.}, .1,
                           energy_filter={(1, 1): .5})
    from pamssw.standalone.paper_reference import run_ssw
    from pamssw.standalone.paper_reference import SSWConfig
    cfg = SSWConfig(width=.2, rotation_bias=2., max_gaussians=2,
                    temperature_K=300., fmax=1e-4, relax_steps=10,
                    fd_step=.001, rotation_hvp=8, rotation_tol=1e-5,
                    direction_sampling='global')
    run_ssw(a, ASESurface(Anchored()), steps=0, config=cfg,
            rng=np.random.default_rng(1), ls=settings)
    assert seen == [(((1, 1), .5),)]


def test_public_vc_and_constrained_entrypoints_forward_filter(monkeypatch):
    from pamssw.standalone import vc_reference
    from pamssw.standalone import vc_softening
    from pamssw.standalone.vc_reference import VCSSWConfig, run_vc_ssw
    class ZeroStress:
        requests = 0
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions), np.zeros((3, 3))
    from ase.build import bulk
    a = bulk('Cu', 'fcc', a=3.6, cubic=True)
    seen_vc = []
    original_vc = vc_softening.FrozenPeriodicCellSoftening.from_atoms
    def capture_vc(cls, atoms, **kwargs):
        seen_vc.append(kwargs['energy_filter'])
        return original_vc(atoms, **kwargs)
    monkeypatch.setattr(vc_softening.FrozenPeriodicCellSoftening, 'from_atoms', classmethod(capture_vc))
    settings = LSSettings({(29, 29): 1.}, {(29, 29): 2.9}, .1,
                           energy_filter={(29, 29): .5})
    run_vc_ssw(a, ZeroStress(), steps=0,
               config=VCSSWConfig(strain_length=3.6, width=.1, rotation_bias=1.),
               rng=np.random.default_rng(2), ls=settings)
    assert seen_vc == [(((29, 29), .5),)]

    from pamssw.standalone import constrained_reference
    from pamssw.standalone.constrained_reference import run_constrained_ssw, ConstrainedSSWConfig
    from ase.constraints import FixAtoms
    a = Atoms('Cu2', positions=[(3., 0., 0.), (0., 0., 0.)],
              cell=[8., 8., 8.], pbc=[True, True, False], constraint=FixAtoms([0]))
    seen_constrained = []
    original_init = ConstrainedLSRuntime.initialize_at
    def capture_init(self, atoms):
        seen_constrained.append(self.settings.energy_filter)
        return original_init(self, atoms)
    monkeypatch.setattr(ConstrainedLSRuntime, 'initialize_at', capture_init)
    class ZeroFixed(ZeroStress):
        def evaluate(self, atoms):
            self.requests += 1
            return 0., np.zeros_like(atoms.positions)
    run_constrained_ssw(a, ZeroFixed(), steps=0,
        config=ConstrainedSSWConfig(width=.1, rotation_bias=1., relax_steps=2),
        rng=np.random.default_rng(3), fixed_indices=[0], ls=LSSettings(
            {(29, 29): 1.}, {(29, 29): 4.}, .1, energy_filter={(29, 29): .5}))
    assert seen_constrained == [(((29, 29), .5),)]


def test_filter_input_is_canonical_and_rejects_bad_keys():
    mapping = {(26, 26): 0.}
    settings = LSSettings(E, L, .1, energy_filter=mapping)
    mapping[(6, 6)] = .2
    assert settings.energy_filter == (((26, 26), 0.),)
    for bad in ({(True, 26): 0.}, {(26.5, 26): 0.}, {(26, 26): -1.}):
        try:
            LSSettings(E, L, .1, energy_filter=bad)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid energy filter key/value accepted')


def test_filtered_vc_energy_force_stress_remain_chart_consistent():
    from pamssw.standalone.vc_geometry import SymmetricLogStrainChart
    a = Atoms('CuAgCu', positions=[(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)],
              cell=np.diag([8., 8., 8.]), pbc=True)
    e = {(29, 29): 4., (29, 47): 5., (47, 47): 6.}
    l = {(29, 29): 2., (29, 47): 2., (47, 47): 2.}
    soft = FrozenPeriodicCellSoftening.from_atoms(
        a, bond_energies=e, bond_lengths=l,
        energy_filter={(29, 29): 0., (29, 47): .5})
    chart = SymmetricLogStrainChart(a, strain_length=8.)
    q = chart.pack(a); q[-6:] = [.02, -.01, .015, .01, -.02, .012]
    got = chart.evaluate(q, soft.evaluate_stress)
    for k in range(len(q)):
        plus, minus = q.copy(), q.copy()
        plus[k] += 1e-6; minus[k] -= 1e-6
        fd = (chart.evaluate(plus, soft.evaluate_stress).energy -
              chart.evaluate(minus, soft.evaluate_stress).energy) / 2e-6
        assert fd == pytest.approx(got.gradient[k], abs=2e-8)
