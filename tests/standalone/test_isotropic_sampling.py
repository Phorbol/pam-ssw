"""Mass independence of the explicit Cartesian direction distribution."""
from dataclasses import replace
import numpy as np
from ase import Atoms
from ase.cluster import Icosahedron
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface, SSWConfig, run_ssw
from pamssw.standalone.paper_reference import sample_initial_direction


def test_isotropic_sampling_is_independent_of_isotope_masses():
    atoms = Atoms('CH', positions=[[0., 0., 0.], [1., 0., 0.]])
    isotope = atoms.copy(); isotope.set_masses([13., 2.])
    a = sample_initial_direction(atoms, np.random.default_rng(11), mode='isotropic')
    b = sample_initial_direction(isotope, np.random.default_rng(11), mode='isotropic')
    np.testing.assert_array_equal(a, b)
    g = sample_initial_direction(atoms, np.random.default_rng(11), mode='global')
    h = sample_initial_direction(isotope, np.random.default_rng(11), mode='global')
    assert not np.allclose(g, h)


def test_emt_walk_replays_exactly_under_mass_changes():
    atoms = Icosahedron('Cu', 2)
    isotope = atoms.copy(); mass = isotope.get_masses(); mass[0] *= 2
    isotope.set_masses(mass)
    cfg = SSWConfig(width=.1, rotation_bias=None, pre_rotation_hvp=5,
        max_gaussians=3, temperature_K=150., fmax=.01, bias_fmax=.1,
        relax_steps=400, fd_step=1e-4, rotation_hvp=100, rotation_tol=.02,
        direction_sampling='isotropic', rotation_solver='ritz',
        cluster_frame='direction_only', quench_optimizer='safe-lbfgs-total')
    results = [run_ssw(a, ASESurface(EMT()), steps=1, config=cfg,
                      rng=np.random.default_rng(11)) for a in (atoms, isotope)]
    first, second = results
    assert first.evaluation_requests == second.evaluation_requests
    assert [r.status for r in first.records] == [r.status for r in second.records]
    np.testing.assert_array_equal(first.records[0].last_atoms.positions,
                                  second.records[0].last_atoms.positions)
    assert [q.energy for q in first.minima] == [q.energy for q in second.minima]
