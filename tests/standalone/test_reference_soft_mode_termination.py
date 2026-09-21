import numpy as np
import pytest
from ase import Atoms

from pamssw.standalone.direction import reference_soft_mode


@pytest.mark.parametrize('finite_difference', ('forward', 'central'))
@pytest.mark.parametrize('max_hvp', range(2, 9))
def test_reference_soft_mode_respects_hvp_budget_and_certifies_linear_case(
        finite_difference, max_hvp):
    atoms = Atoms('H2', positions=[[0., 0., 0.], [1., 0., 0.]])
    hessian = np.diag(np.arange(1., 7.))

    def evaluate(candidate):
        x = candidate.positions.ravel()
        return 0., (-hessian @ x).reshape(2, 3)

    result = reference_soft_mode(
        atoms, np.ones((2, 3)), fd_step=1e-4, max_hvp=max_hvp,
        residual_tol=1e-10, evaluate=evaluate,
        finite_difference=finite_difference)
    assert result.hvp_calls <= max_hvp
    assert np.isfinite(result.residual_norm)
    if max_hvp >= 7:
        assert result.converged
    if finite_difference == 'forward':
        assert result.force_calls == result.hvp_calls + 1
    else:
        assert result.force_calls == 2 * result.hvp_calls


def test_reference_soft_mode_continues_after_failed_direct_certificate():
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    desired_hvp = [np.array([1., .001, 0.]), np.array([1., 1., 0.]),
                   np.array([2., .5, 0.]), np.array([3., .2, 0.]),
                   np.array([4., .1, 0.])]
    calls = []

    def evaluate(candidate):
        index = len(calls)
        calls.append(candidate.positions.copy())
        force = np.zeros(3) if index == 0 else -1e-4 * desired_hvp[index - 1]
        return 0., force.reshape(1, 3)

    result = reference_soft_mode(
        atoms, [[1., 0., 0.]], fd_step=1e-4, max_hvp=4,
        residual_tol=.02, evaluate=evaluate, finite_difference='forward')
    assert not result.converged
    assert result.hvp_calls == 4
    assert result.force_calls == len(calls) == 5
