"""Numerical contracts only; these checks do not establish PES-search efficacy."""
import numpy as np
import pytest
from ase import Atoms
from pamssw.standalone.dimer import paper_dimer_direction


def solve(hessian, anchor, **kwargs):
    atoms = Atoms('H', positions=[[0.3, -0.2, 0.4]])
    seen = []
    def evaluate(trial):
        x = trial.positions.ravel()
        seen.append(x.copy())
        return .5*x@hessian@x, (-hessian@x).reshape(1, 3)
    result = paper_dimer_direction(atoms, np.array(anchor).reshape(1, 3),
        rotation_bias=kwargs.pop('rotation_bias', 0.), fd_step=1e-4,
        max_hvp=kwargs.pop('max_hvp', 301), tol=kwargs.pop('tol', 1e-8),
        evaluate=evaluate)
    return result, seen


def test_biased_direction_matches_independent_dense_eigenproblem():
    h = np.array([[3., .4, .1], [.4, 2., .2], [.1, .2, 1.]])
    anchor = np.array([1., 2., 3.]); anchor /= np.linalg.norm(anchor)
    result, calls = solve(h, anchor, rotation_bias=4.)
    values, vectors = np.linalg.eigh(h-4*np.outer(anchor, anchor))
    assert result.converged
    assert abs(result.curvature-values[0]) < 1e-8
    assert abs(result.direction.ravel()@vectors[:, 0]) > 1-1e-10
    assert result.direction.ravel()@anchor >= 0
    assert result.force_calls == len(calls) == result.hvp_calls+1


@pytest.mark.parametrize('budget', [1, 2, 3, 4, 5])
def test_budget_returns_only_directly_evaluated_direction(budget):
    h = np.diag([1., 4., 9.])
    result, calls = solve(h, [1., 1., 1.], max_hvp=budget, tol=1e-12)
    assert result.hvp_calls <= budget
    n = result.direction.ravel()
    assert np.allclose(calls[-1], calls[0]+1e-4*n)
    assert np.isclose(result.residual_norm, np.linalg.norm(h@n-(n@h@n)*n))
    assert not result.converged


def test_stationarity_does_not_certify_lowest_eigenmode():
    result, _ = solve(np.diag([1., 4., 9.]), [0., 0., 1.])
    assert result.converged
    assert result.curvature == pytest.approx(9.)
    assert result.hvp_calls == 1


def test_rotation_equivariance():
    h = np.array([[3., .4, .1], [.4, 2., .2], [.1, .2, 1.]])
    q, _ = np.linalg.qr(np.array([[1., 2., 3.], [4., -3., 2.], [2., 5., 1.]]))
    anchor = np.array([1., 2., 3.])
    one, _ = solve(h, anchor, rotation_bias=2.)
    two, _ = solve(q@h@q.T, q@anchor, rotation_bias=2.)
    assert np.allclose(two.direction.ravel(), q@one.direction.ravel(), atol=1e-8)
    assert abs(one.curvature-two.curvature) < 1e-8


def test_rejects_constraints_before_evaluation():
    from ase.constraints import FixAtoms
    atoms = Atoms('H', positions=[[0., 0., 0.]], pbc=True)
    atoms.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match='unconstrained'):
        paper_dimer_direction(atoms, [[1., 0., 0.]], rotation_bias=0,
            fd_step=1e-3, max_hvp=3, tol=1e-3, evaluate=None)


def test_nonlinear_return_residual_is_fresh_and_reports_secant_asymmetry():
    atoms = Atoms('H', positions=[[.3, .5, .2]])
    anchor = np.array([[1., 2., 3.]])
    h = np.diag([1., 3., 7.])
    calls = []
    def evaluate(trial):
        x = trial.positions.ravel()
        calls.append(x.copy())
        return .5*x@h@x+np.sum(x**3)/3, (-h@x-x*x).reshape(1, 3)
    result = paper_dimer_direction(atoms, anchor, rotation_bias=2.,
        fd_step=.02, max_hvp=9, tol=1e-12, evaluate=evaluate)
    n = result.direction.ravel()
    n0 = anchor.ravel()/np.linalg.norm(anchor)
    x = atoms.positions.ravel()
    y = x+.02*n
    hn = (h@(y-x)+y*y-x*x)/.02-2*(n@n0)*n0
    assert np.allclose(calls[-1], y)
    assert result.curvature == pytest.approx(n@hn)
    assert result.residual_norm == pytest.approx(np.linalg.norm(hn-(n@hn)*n))
    assert result.projected_symmetry_error > 0
    assert np.array_equal(atoms.positions, [[.3, .5, .2]])
    assert np.array_equal(anchor, [[1., 2., 3.]])
