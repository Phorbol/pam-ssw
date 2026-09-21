import numpy as np
from ase import Atoms

from research.ga_ssw.verified_ritz_research import reference_soft_mode


def test_linear_hessian_keeps_early_certificate_cost():
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    hessian = np.diag([1., 2., 3.])

    def evaluate(candidate):
        x = candidate.positions.ravel()
        return 0., (-hessian @ x).reshape(1, 3)

    result = reference_soft_mode(atoms, [[1., 0., 0.]], fd_step=1e-4,
                                 max_hvp=6, residual_tol=1e-8,
                                 evaluate=evaluate, finite_difference='central')
    assert result.converged
    assert result.hvp_calls == 2
    assert result.force_calls == 4


def test_failed_direct_certificate_continues_within_hvp_budget():
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    desired_hvp = [np.array([1., .001, 0.]),
                   np.array([1., 1., 0.]),
                   np.array([2., .5, 0.]),
                   np.array([3., .2, 0.]),
                   np.array([4., .1, 0.])]
    calls = []

    def evaluate(candidate):
        calls.append(candidate.positions.copy())
        return 0., np.zeros((1, 3))

    # Replace the callback's force output by a deterministic forward-HVP
    # sequence through candidate position, preserving the real finite
    # difference and budget machinery.
    def sequenced(candidate):
        index = len(calls)
        calls.append(candidate.positions.copy())
        if index == 0:
            force = np.zeros(3)
        else:
            force = -1e-4 * desired_hvp[index - 1]
        return 0., force.reshape(1, 3)

    result = reference_soft_mode(atoms, [[1., 0., 0.]], fd_step=1e-4,
                                 max_hvp=4, residual_tol=.02,
                                 evaluate=sequenced, finite_difference='forward')
    assert not result.converged
    assert result.hvp_calls == 4
    assert result.force_calls == len(calls) == 5
