"""Numerical adapter checks, not scientific validation of search performance."""
from dataclasses import replace
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone.surface import ASESurface, quench


def test_safe_quench_total_objective_certificate_and_count():
    class PairTerm:
        def evaluate(self, atoms):
            delta = atoms.positions[1] - atoms.positions[0]
            distance = np.linalg.norm(delta)
            force = .8 * (distance - 3.) * delta / distance
            return .4 * (distance - 3.)**2, np.array([force, -force])
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
    original = atoms.positions.copy()
    surface = ASESurface(EMT())
    term = PairTerm()
    result = quench(atoms, surface, fmax=1e-5, steps=100,
                    terms=(term,), optimizer='safe-lbfgs-total')
    assert result.converged and result.surface == 'modified'
    assert result.evaluation_requests == surface.requests
    e, f = surface.evaluate(result.atoms)
    de, df = term.evaluate(result.atoms)
    assert result.energy == pytest.approx(e + de)
    assert result.max_force == pytest.approx(np.linalg.norm(f + df, axis=1).max())
    assert np.linalg.norm(f, axis=1).max() > 1e-3
    true = quench(result.atoms, surface, fmax=1e-5, steps=100,
                  optimizer='safe-lbfgs-total')
    assert true.converged and true.surface == 'true'
    assert true.energy < e
    assert np.array_equal(atoms.positions, original)
    assert atoms.calc is None and result.atoms.calc is None


def test_safe_quench_zero_budget_and_rejected_frame():
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
    surface = ASESurface(EMT())
    result = quench(atoms, surface, fmax=1e-8, steps=0,
                    optimizer='safe-lbfgs-total')
    assert not result.converged and result.optimizer_steps == 0
    assert result.evaluation_requests == surface.requests == 1
    before = surface.requests
    with pytest.raises(NotImplementedError, match='frame'):
        quench(atoms, surface, fmax=1e-5, steps=10, terms=(object(),),
               frame=object(), optimizer='safe-lbfgs-total')
    assert surface.requests == before


def test_safe_quench_exposes_termination_telemetry_on_real_cu_emt():
    atoms = Atoms('Cu2', positions=[[0, 0, 0], [2.7, 0, 0]])
    surface = ASESurface(EMT())
    result = quench(atoms, surface, fmax=1e-12, steps=1,
                    optimizer='safe-lbfgs-total')
    assert result.optimizer_telemetry is not None
    assert result.optimizer_telemetry.backend == 'safe-lbfgs-total'
    assert result.optimizer_telemetry.termination_reason == 'maxiter'
    assert result.optimizer_telemetry.converged is False


def test_ssw_backend_selection_validation():
    from test_paper_reference import configuration
    config = configuration()
    assert config.quench_optimizer == 'ase-lbfgs'
    assert replace(config, quench_optimizer='safe-lbfgs-total').quench_optimizer == 'safe-lbfgs-total'
    with pytest.raises(ValueError, match='quench_optimizer'):
        replace(config, quench_optimizer='unknown')
    with pytest.raises(NotImplementedError, match='eckart'):
        replace(config, quench_optimizer='safe-lbfgs-total', cluster_frame='eckart')


def test_safe_ssw_complete_quenches_use_selected_backend(monkeypatch):
    from test_paper_reference import configuration, Harmonic
    from pamssw.standalone import paper_reference as paper
    seen = []
    original = paper.quench
    def observe(*args, **kwargs):
        seen.append(kwargs['optimizer'])
        return original(*args, **kwargs)
    monkeypatch.setattr(paper, 'quench', observe)
    surface = ASESurface(Harmonic())
    result = paper.run_ssw(Atoms('H', positions=[[.1, .2, .3]]), surface,
        steps=1, config=replace(configuration(), quench_optimizer='safe-lbfgs-total'),
        rng=np.random.default_rng(9))
    assert len(seen) == 4  # Initial, two Gaussian stages, true landing.
    assert set(seen) == {'safe-lbfgs-total'}
    assert all(m.converged and m.surface == 'true' for m in result.minima)
    assert result.evaluation_requests == surface.requests


def test_safe_ls_walk_forwards_backend_and_keeps_both_biases(monkeypatch):
    from test_paper_reference import configuration
    from test_ls_cycle import setup
    from pamssw.standalone import paper_reference as paper, ls_cycle
    from pamssw.standalone.softening import FrozenBondSoftening
    from pamssw.standalone.gaussian import ProjectedGaussian
    atoms, surface, _ = setup()
    seen = []
    original = quench
    def observe(*args, **kwargs):
        seen.append((kwargs['optimizer'], kwargs.get('terms', ())))
        return original(*args, **kwargs)
    monkeypatch.setattr(paper, 'quench', observe)
    monkeypatch.setattr(ls_cycle, 'quench', observe)
    # Exact bond-stretch eigenvector isolates quench wiring from rotation error.
    monkeypatch.setattr(paper, 'sample_initial_direction',
                        lambda *a, **kw: np.array([[-1., 0., 0.], [1., 0., 0.]])/np.sqrt(2))
    result = paper.run_ssw(atoms, surface, steps=1,
        config=replace(configuration(), quench_optimizer='safe-lbfgs-total'),
        ls=paper.LSSettings({(1, 1): 4.}, {(1, 1): 1.2}, target_per_atom=.02),
        rng=np.random.default_rng(9))
    assert all(backend == 'safe-lbfgs-total' for backend, _ in seen)
    assert any(len(terms) == 1 and isinstance(terms[0], FrozenBondSoftening) for _, terms in seen)
    assert any(any(isinstance(t, FrozenBondSoftening) for t in terms)
               and any(isinstance(t, ProjectedGaussian) for t in terms) for _, terms in seen)
    assert seen[-1][1] == ()  # True landing removes both biases.
    assert result.records[0].landing.converged
    assert result.evaluation_requests == surface.requests


def test_safe_quench_preserves_physical_pbc_and_unwrapped_coordinates():
    from ase.calculators.calculator import Calculator, all_changes
    class MetadataOracle(Calculator):
        implemented_properties = ['energy', 'forces']
        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            assert atoms.pbc.all()
            assert np.array_equal(atoms.cell.array, np.eye(3) * 4)
            assert atoms.get_initial_charges()[0] == .3
            delta = atoms.positions - np.array([[5., 0., 0.]])
            self.results = {'energy': float((delta**2).sum()/2), 'forces': -delta}
    atoms = Atoms('H', positions=[[5.2, 0, 0]], cell=[4, 4, 4], pbc=True)
    atoms.set_initial_charges([.3])
    surface = ASESurface(MetadataOracle())
    result = quench(atoms, surface, fmax=1e-6, steps=50, optimizer='safe-lbfgs-total')
    assert result.converged
    assert result.atoms.positions[0, 0] == pytest.approx(5., abs=1e-6)
    assert result.atoms.pbc.all()
    assert result.evaluation_requests == surface.requests
