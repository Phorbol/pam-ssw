import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes


class Harmonic(Calculator):
    implemented_properties = ['energy', 'forces']
    def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = dict(energy=float((atoms.positions**2).sum()/2),
                            forces=-atoms.positions.copy())


def configuration():
    from pamssw.standalone.paper_reference import SSWConfig
    return SSWConfig(width=.2, rotation_bias=2., max_gaussians=2,
                     temperature_K=300., fmax=1e-4, relax_steps=100,
                     fd_step=.001, rotation_hvp=8, rotation_tol=1e-5,
                     direction_sampling='global')


def test_complete_walk_removes_bias_and_keeps_caller_owned_geometry():
    from pamssw.standalone.paper_reference import run_ssw
    from pamssw.standalone.surface import ASESurface
    atoms = Atoms('H', positions=[[.1, .2, .3]])
    original = atoms.positions.copy()
    result = run_ssw(atoms, ASESurface(Harmonic()), steps=2,
                      config=configuration(), rng=np.random.default_rng(9))
    assert len(result.records) == 2
    assert len(result.minima) == 3
    assert all(m.surface == 'true' and m.converged for m in result.minima)
    assert np.linalg.norm(result.current.positions) < 1e-4
    assert np.array_equal(atoms.positions, original)
    assert atoms.calc is None and result.current.calc is None
    assert all(len(record.climb) == 2 for record in result.records)
    assert result.evaluation_requests > 0


def test_budget_failure_does_not_enter_minima_archive():
    from dataclasses import replace
    from pamssw.standalone.paper_reference import run_ssw
    from pamssw.standalone.surface import ASESurface
    result = run_ssw(Atoms('H', positions=[[0., 0., 0.]]), ASESurface(Harmonic()),
                    steps=1, config=replace(configuration(), relax_steps=0),
                    rng=np.random.default_rng(9))
    assert len(result.minima) == 1
    assert not result.records[0].accepted
    assert result.records[0].status == 'biased_quench_failed'


def test_paper_direction_requires_nonlocal_pair_and_is_seed_reproducible():
    import pytest
    from pamssw.standalone.paper_reference import sample_initial_direction
    small = Atoms('H2', positions=[[0, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError, match='more than 3'):
        sample_initial_direction(small, np.random.default_rng(1), mode='paper')
    small.positions[1, 0] = 4.
    a = sample_initial_direction(small, np.random.default_rng(1), mode='paper')
    b = sample_initial_direction(small, np.random.default_rng(1), mode='paper')
    assert np.array_equal(a, b)
    assert np.linalg.norm(a) == pytest.approx(1.)
    assert a[0, 0] > 0 and a[1, 0] < 0
