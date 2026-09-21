"""Numerical/lifecycle checks, not evidence of material-search efficacy."""
import numpy as np
from ase import Atoms

from pamssw.standalone.recovered_cbd import recovered_cbd_direction


class Surface:
    def __init__(self, diagonal):
        self.diagonal = np.array(diagonal)
        self.positions = []

    def __call__(self, atoms):
        x = atoms.positions.ravel()
        self.positions.append(x.copy())
        return .5 * np.dot(x, self.diagonal*x), (-self.diagonal*x).reshape(-1, 3)


def solve(atoms, anchor, surface, **kwargs):
    options = dict(fd_step=1e-3, max_force_calls=8, pre_rotmax=2,
                   rotmax=4, pre_ftol=1e-8, ftol=1e-8,
                   metric='euclidean', evaluate=surface)
    options.update(kwargs)
    return recovered_cbd_direction(atoms, anchor, **options)


def test_positive_prerotation_reuses_endpoint_force_for_biased_stage():
    atoms = Atoms('H', positions=[[.3, -.2, .4]])
    original = atoms.positions.copy()
    surface = Surface([2., 3., 4.])
    result = solve(atoms, [[1., 0., 0.]], surface, max_force_calls=2)
    assert result.stage_complete and result.converged
    assert result.force_calls == len(surface.positions) == 2
    assert result.stop_reason == 'force_tolerance'
    np.testing.assert_array_equal(atoms.positions, original)
    np.testing.assert_allclose(result.direction, [[1., 0., 0.]])
    stages = [row for row in result.trace if row['event'] == 'rotation']
    assert [row['stage'] for row in stages] == ['CBD_PreRot', 'CBD_biasedRot']
    assert [row['force_calls'] for row in stages] == [2, 2]
    assert stages[1]['history_size_before'] == 0
    assert abs(result.rotation_weight - 2.) < 1e-10
    assert abs(result.real_curvature - 2.) < 1e-10
    assert abs(result.curvature) < 1e-10


def test_negative_prerotation_override_cannot_escape_external_budget():
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    surface = Surface([-2., 3., 4.])
    result = solve(atoms, [[1., 0., 0.]], surface,
                   pre_rotmax=0, max_force_calls=2)
    assert not result.stage_complete and not result.converged
    assert result.stop_reason == 'force_budget'
    assert result.force_calls == len(surface.positions) == 2
    assert result.stage == 'CBD_UnbiasedRot'
    np.testing.assert_array_equal(result.direction, [[1., 0., 0.]])
    assert result.trace[0]['prerot_override']


def test_rotation_limit_is_not_reported_as_force_convergence():
    atoms = Atoms('H', positions=[[0., 0., 0.]])
    surface = Surface([1., 3., 5.])
    initial = np.array([[1., 1., 0.]]) / np.sqrt(2.)
    result = solve(atoms, initial, surface, pre_rotmax=0, rotmax=0)
    assert result.stage_complete and not result.converged
    assert result.stop_reason == 'rotation_limit'
    assert result.force_calls == 2
    np.testing.assert_allclose(result.direction, initial)


def test_returned_direction_was_evaluated_when_force_budget_exhausts():
    atoms = Atoms('H', positions=[[.2, -.1, .1]])
    surface = Surface([1., 3., 5.])
    result = solve(atoms, [[1., .4, .2]], surface, max_force_calls=4,
                   pre_rotmax=100, rotmax=100)
    assert result.force_calls == len(surface.positions) == 4
    assert result.stop_reason == 'force_budget'
    np.testing.assert_allclose(
        surface.positions[-1], atoms.positions.ravel()+1e-3*result.direction.ravel())



def test_exports_actual_prerotation_anchor_for_gaussian_curvature_accounting():
    atoms=Atoms('H',positions=[[0.,0.,0.]])
    surface=Surface([1.,3.,5.])
    result=solve(atoms,[[1.,.4,.2]],surface,pre_rotmax=1,rotmax=1,max_force_calls=8)
    biased=[row for row in result.trace if row['stage']=='CBD_biasedRot']
    assert biased
    actual=(surface.positions[biased[0]['force_calls']-1]-atoms.positions.ravel())/1e-3
    np.testing.assert_allclose(result.bias_reference.ravel(),actual,atol=1e-12,rtol=0)
    expected=result.curvature+result.rotation_weight*np.sum(result.direction*result.bias_reference)**2
    np.testing.assert_allclose(expected,result.real_curvature,atol=1e-10,rtol=0)


def test_cluster_projection_applies_to_every_requested_rotation_endpoint():
    from pamssw.standalone.cluster_frame import ClusterFrame
    atoms = Atoms('Cu3', positions=[[0.,0.,0.],[2.3,.1,.2],[.4,2.1,-.1]])
    frame = ClusterFrame(atoms)
    anchor = frame.project(np.arange(9).reshape(3,3)+1.)
    anchor /= np.linalg.norm(anchor)
    surface = Surface(np.arange(1.,10.))
    result = solve(atoms, anchor, surface, max_force_calls=6,
                   pre_rotmax=100, rotmax=100, project=frame.project)
    assert result.force_calls == 6
    for endpoint in surface.positions[1:]:
        direction = (endpoint.reshape(3,3)-atoms.positions)/1e-3
        np.testing.assert_allclose(direction, frame.project(direction), atol=1e-11, rtol=0)
        assert np.isclose(np.linalg.norm(direction), 1.)
    np.testing.assert_allclose(result.direction, frame.project(result.direction), atol=1e-11, rtol=0)
