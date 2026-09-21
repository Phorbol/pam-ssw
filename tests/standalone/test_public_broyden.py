import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.paper_reference import SSWConfig, run_ssw
from pamssw.standalone.broyden_direction import paper_broyden_direction
from pamssw.standalone.atomic_climb import atomic_climb, resume_atomic_climb, AtomicClimbCheckpoint


class QuadraticSurface:
    def __init__(self, matrix):
        self.matrix = np.asarray(matrix, dtype=float)
        self.calls = 0

    def __call__(self, atoms):
        self.calls += 1
        x = atoms.positions.ravel()
        return .5 * x @ self.matrix @ x, (-(self.matrix @ x)).reshape(atoms.positions.shape)

    @property
    def requests(self):
        return self.calls

    evaluate = __call__


def test_public_broyden_uses_euclidean_fixed_factor_and_certifies_nonzero_center():
    hessian = np.diag([1., 2., 4.])
    atoms = Atoms('H', positions=[[.3, -.2, .4]])
    surface = QuadraticSurface(hessian)
    result = paper_broyden_direction(
        atoms, np.array([[1., 0., 0.]]), rotation_bias=.5, fd_step=1e-3,
        max_hvp=1, tol=1e-10, evaluate=surface,
    )
    assert result.hvp_calls == 1
    assert result.force_calls == 2
    assert result.converged
    direction = result.direction.ravel()
    biased_hessian = hessian - .5 * np.outer([1., 0., 0.], [1., 0., 0.])
    residual = biased_hessian @ direction - result.curvature * direction
    assert np.linalg.norm(residual) < 1e-10


def test_config_allows_broyden_budget_one_and_staged_budget_one():
    kwargs = dict(width=.1, rotation_bias=.5, max_gaussians=1,
                  temperature_K=300., fmax=.01, relax_steps=1, fd_step=1e-3,
                  rotation_tol=.02, direction_sampling='global',
                  cluster_frame='translation_only', quench_optimizer='safe-lbfgs-total')
    config = SSWConfig(rotation_solver='broyden-euclidean', rotation_hvp=1, **kwargs)
    staged = SSWConfig(rotation_solver='broyden-euclidean', rotation_hvp=3,
                       pre_rotation_hvp=1, rotation_bias=None, **{k: v for k, v in kwargs.items() if k != 'rotation_bias'})
    assert config.rotation_solver == staged.rotation_solver
    with pytest.raises(ValueError):
        SSWConfig(rotation_solver='ritz', rotation_hvp=1, **kwargs)


def test_staged_broyden_is_reachable_without_public_solver_fallback():
    atoms = Atoms('H', positions=[[.2, -.1, .3]])
    surface = QuadraticSurface(np.diag([1., 2., 3.]))
    config = SSWConfig(width=.1, rotation_bias=None, pre_rotation_hvp=1,
                       max_gaussians=1, temperature_K=300., fmax=100.,
                       relax_steps=0, fd_step=1e-3, rotation_hvp=3,
                       rotation_tol=10., rotation_solver='broyden-euclidean',
                       direction_sampling='global', cluster_frame='cartesian',
                       quench_optimizer='ase-lbfgs')
    result = run_ssw(atoms, surface, steps=1, config=config,
                     rng=np.random.default_rng(11))
    assert result.records
    event = result.records[0].climb[0]
    assert event['rotation_solver'] == 'broyden-euclidean'
    assert event['pre_rotation']['force_calls'] == 2
    assert event['main_rotation']['force_calls'] == 2
    assert event['rotation_force_requests'] == 4


def test_public_broyden_matches_frozen_research_runner_on_multistep_quadratic():
    from research.ga_ssw.broyden_direction_reconstruction import broyden_direction

    atoms = Atoms('H', positions=[[.3, -.2, .4]])
    anchor = np.array([[.7, .4, -.2]])
    surface = QuadraticSurface(np.diag([1., 2., 4.]))
    public = paper_broyden_direction(atoms, anchor, rotation_bias=.5, fd_step=1e-3,
                                     max_hvp=4, tol=1e-12, evaluate=surface)
    surface.calls = 0
    research = broyden_direction(atoms, anchor, rotation_bias=.5, fd_step=1e-3,
                                 max_hvp=4, tol=1e-12, initial_factor=.05,
                                 metric='euclidean', evaluate=surface)
    np.testing.assert_allclose(public.direction, research.direction)
    assert public.hvp_calls == research.hvp_calls
    assert public.force_calls == research.force_calls
    assert public.trace == research.trace


def test_atomic_climb_rejects_broyden_before_surface_requests():
    atoms = Atoms('Cu', positions=[[0., 0., 0.]], cell=np.eye(3) * 4., pbc=True)
    config = SSWConfig(width=.1, rotation_bias=.5, max_gaussians=1,
                       temperature_K=300., fmax=.01, relax_steps=1, fd_step=1e-3,
                       rotation_hvp=1, rotation_tol=.02,
                       rotation_solver='broyden-euclidean', direction_sampling='global',
                       cluster_frame='translation_only', quench_optimizer='safe-lbfgs-total')
    surface = QuadraticSurface(np.eye(3))
    with pytest.raises(NotImplementedError, match='run_ssw'):
        atomic_climb(atoms, surface, reference_energy=0., config=config,
                     rng=np.random.default_rng(2))
    checkpoint = AtomicClimbCheckpoint(0, atoms.copy(), np.ones((1, 3)), (), 0., config)
    with pytest.raises(NotImplementedError, match='run_ssw'):
        resume_atomic_climb(checkpoint, surface)
    assert surface.requests == 0
