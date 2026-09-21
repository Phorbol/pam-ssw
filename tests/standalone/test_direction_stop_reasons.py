import numpy as np
from ase import Atoms

from pamssw.standalone.direction import reference_soft_mode
from pamssw.standalone.dimer import paper_dimer_direction
from pamssw.standalone.broyden_direction import broyden_direction
from pamssw.standalone.staged_direction import two_stage_dimer_direction


def harmonic(matrix):
    matrix = np.asarray(matrix, dtype=float)

    def evaluate(atoms):
        x = atoms.positions.ravel()
        return 0.5 * x @ matrix @ x, -(matrix @ x).reshape(atoms.positions.shape)

    return evaluate


def test_reference_marks_residual_convergence():
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    result = reference_soft_mode(
        atoms, [[1., 0., 0.]], fd_step=1e-4, max_hvp=4,
        residual_tol=1e-8, evaluate=harmonic(np.diag([1., 2., 3.])),
    )

    assert result.converged
    assert result.stop_reason == 'residual_converged'


def test_reference_marks_budget_when_certificate_cannot_start_new_basis():
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    result = reference_soft_mode(
        atoms, [[1., 1., 0.]], fd_step=1e-4, max_hvp=2,
        residual_tol=1e-12, evaluate=harmonic(np.diag([1., 2., 3.])),
    )

    assert not result.converged
    assert result.stop_reason == 'budget_exhausted'


def test_reference_distinguishes_nonlinear_subspace_exhaustion_from_budget():
    atoms = Atoms('H',positions=[[.2,.1,.3]])
    evaluated = []
    def quartic(trial):
        x=trial.positions.ravel()
        evaluated.append(trial.positions.copy())
        h=np.array([1.,2.,4.])
        return float(np.sum(.5*h*x*x+x**4)), (-(h*x+4*x**3)).reshape(1,3)
    result=reference_soft_mode(atoms,[[1.,1.,1.]],fd_step=.01,max_hvp=30,
        residual_tol=1e-12,evaluate=quartic,finite_difference='forward')
    assert not result.converged
    assert result.hvp_calls < 30
    assert result.stop_reason == 'subspace_exhausted'
    assert result.force_calls == len(evaluated)
    assert any(np.allclose(atoms.positions+.01*result.direction,x,rtol=0,atol=1e-14)
               for x in evaluated)


def test_dimer_marks_budget_when_two_hvps_are_unavailable():
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    result = paper_dimer_direction(
        atoms, [[1., 1., 0.]], rotation_bias=0., fd_step=1e-4,
        max_hvp=2, tol=1e-12,
        evaluate=harmonic(np.diag([1., 2., 3.])),
    )

    assert not result.converged
    assert result.stop_reason == 'budget_exhausted'


def test_broyden_marks_endpoint_budget():
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    result = broyden_direction(
        atoms, [[1., 1., 0.]], rotation_bias=0., fd_step=1e-4,
        max_hvp=1, tol=1e-12, initial_factor=.05, metric='euclidean',
        evaluate=harmonic(np.diag([1., 2., 3.])),
    )

    assert not result.converged
    assert result.stop_reason == 'budget_exhausted'


def test_two_stage_copies_main_reason_and_preserves_pre_reason():
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    result = two_stage_dimer_direction(
        atoms, [[1., 1., 0.]], fd_step=1e-4, max_hvp=4,
        pre_rotation_hvp=1, tol=1e-12,
        evaluate=harmonic(np.diag([1., 2., 3.])), main_solver='dimer',
    )

    assert result.stop_reason == result.main.stop_reason
    assert result.pre.stop_reason == 'budget_exhausted'
    assert result.direction.shape == atoms.positions.shape
