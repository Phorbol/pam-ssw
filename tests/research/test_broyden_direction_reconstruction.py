import numpy as np
from ase import Atoms

from research.ga_ssw.broyden_direction_reconstruction import broyden_direction


def quadratic(hessian):
    def evaluate(candidate):
        x = candidate.positions.ravel()
        return float(0.5 * x @ hessian @ x), (-hessian @ x).reshape(candidate.positions.shape)
    return evaluate


def test_evaluated_eigen_direction_and_rank_one_bias():
    atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]])
    result = broyden_direction(
        atoms, [[1.0, 0.0, 0.0]], rotation_bias=0.25, fd_step=1e-3,
        max_hvp=4, tol=1e-10, initial_factor=1.0, metric='euclidean',
        evaluate=quadratic(np.diag([1.0, 2.0, 3.0])),
    )
    assert result.converged
    assert result.hvp_calls == 1
    assert result.force_calls == 2
    np.testing.assert_allclose(result.direction, [[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(result.curvature, 0.75)
    assert result.trace[0]['event'] == 'endpoint'


def test_retry_is_algebraic_and_does_not_add_evaluator_calls():
    atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]])
    result = broyden_direction(
        atoms, [[1.0, 1.0, 0.0]], rotation_bias=0.0, fd_step=1e-3,
        max_hvp=2, tol=1e-12, initial_factor=1e5, metric='euclidean',
        evaluate=quadratic(np.diag([1.0, 4.0, 9.0])),
    )
    proposals = [item for item in result.trace if item['event'] == 'proposal']
    assert proposals and proposals[0]['retries'] > 0
    assert result.hvp_calls <= 2
    assert result.force_calls == result.hvp_calls + 1
    assert all(item['event'] in {'endpoint', 'proposal'} for item in result.trace)


def test_native_block_metric_is_explicit_and_validated():
    atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]])
    result = broyden_direction(
        atoms, [[0.0, 1.0, 0.0]], rotation_bias=0.0, fd_step=1e-3,
        max_hvp=1, tol=1e-10, initial_factor=1.0, metric='native_block_sum',
        evaluate=quadratic(np.eye(3)),
    )
    assert result.converged
    np.testing.assert_allclose(result.direction, [[0.0, 1.0, 0.0]])

    with np.testing.assert_raises(ValueError):
        broyden_direction(
            atoms, [[1.0, 0.0, 0.0]], rotation_bias=0.0, fd_step=1e-3,
            max_hvp=1, tol=1e-10, initial_factor=1.0, metric='bad',
            evaluate=quadratic(np.eye(3)),
        )


def test_retry_is_only_first_endpoint_iteration():
    atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]])
    result = broyden_direction(
        atoms, [[1.0, 1.0, 0.0]], rotation_bias=0.0, fd_step=1e-3,
        max_hvp=3, tol=1e-12, initial_factor=1e5, metric='euclidean',
        evaluate=quadratic(np.diag([1.0, 4.0, 9.0])),
    )
    proposals = [item for item in result.trace if item['event'] == 'proposal']
    assert len(proposals) == 2
    assert proposals[0]['retries'] > 0
    assert proposals[1]['retries'] == 0
    assert result.hvp_calls == 3
    assert result.force_calls == 4


def test_failing_evaluator_request_is_counted():
    atoms = Atoms('H', positions=[[0.0, 0.0, 0.0]])
    calls = []

    def evaluate(candidate):
        calls.append(candidate.positions.copy())
        if len(calls) == 2:
            raise RuntimeError('controlled endpoint failure')
        return 0.0, np.zeros((1, 3))

    try:
        broyden_direction(
            atoms, [[1.0, 0.0, 0.0]], rotation_bias=0.0, fd_step=1e-3,
            max_hvp=2, tol=1e-8, initial_factor=1.0, metric='euclidean',
            evaluate=evaluate,
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError('expected controlled evaluator failure')
    assert len(calls) == 2


def test_returned_residual_is_for_the_evaluated_endpoint_at_nonzero_center():
    hessian = np.diag([1.0, 3.0, 5.0])
    atoms = Atoms('H', positions=[[0.2, -0.1, 0.3]])
    anchor = np.array([[1.0, 1.0, 0.0]])
    result = broyden_direction(
        atoms, anchor, rotation_bias=0.4, fd_step=2e-4, max_hvp=2,
        tol=1e-14, initial_factor=0.1, metric='euclidean',
        evaluate=quadratic(hessian),
    )
    center = atoms.positions.ravel()
    direction = result.direction.ravel()
    endpoint = center + 2e-4 * direction
    center_force = -hessian @ center
    endpoint_force = -hessian @ endpoint
    raw_h = (center_force - endpoint_force) / 2e-4
    biased_h = raw_h - 0.4 * np.dot(anchor.ravel() / np.linalg.norm(anchor), direction) * (anchor.ravel() / np.linalg.norm(anchor))
    curvature = direction @ biased_h
    residual = np.linalg.norm(biased_h - curvature * direction)
    np.testing.assert_allclose(result.curvature, curvature)
    np.testing.assert_allclose(result.residual_norm, residual)
    assert result.force_calls == result.hvp_calls + 1
