"""Recovered branch/geometry contracts; no search-efficiency claim."""
import numpy as np
import pytest
from pamssw.standalone.native_rotation_control import cap_rotation, retry_factor, finish_rotation


def test_cap_preserves_tangent_plane_and_limits_to_forty_degrees():
    old=np.array([[1.,0.,0.]])
    proposal=np.array([[0.,3.,4.]])
    capped=cap_rotation(old,proposal)
    assert np.linalg.norm(capped)==pytest.approx(1)
    assert np.degrees(np.arccos(np.vdot(old,capped)))==pytest.approx(40)
    assert capped[0,1]/capped[0,2]==pytest.approx(3/4)
    np.testing.assert_array_equal(proposal,[[0,3,4]])


def test_cap_leaves_small_rotation_but_normalizes_candidate():
    proposal=np.array([[2.,.1,0.]])
    np.testing.assert_allclose(cap_rotation([[1,0,0]],proposal),proposal/np.linalg.norm(proposal))
    with pytest.raises(ValueError,match='tangent'):
        cap_rotation([[1,0,0]],[[-1,0,0]])


def test_retry_requires_all_three_native_conditions():
    assert retry_factor(1,1,1.03,2.)==pytest.approx(1.6)
    assert retry_factor(1,14,1.03,2.)==pytest.approx(1.6)
    for rotnum,attempt,norm in ((2,1,2.),(1,15,2.),(1,1,1.02)):
        assert retry_factor(rotnum,attempt,norm,2.) is None


def test_stop_returns_evaluated_direction_not_unqueried_proposal():
    result=finish_rotation([[1,0,0]],[[0,1,0]],rotnum=3,rotmax=20,
                           reported_force=.001,ftol=.01,infor='CBD',curv_real=1.)
    assert result.stop and result.force_converged and not result.budget_exceeded
    assert result.next_rotnum==4
    np.testing.assert_array_equal(result.direction,[[1,0,0]])


def test_prerot_negative_curvature_overrides_native_budget():
    result=finish_rotation([[1,0,0]],[[0,1,0]],rotnum=21,rotmax=20,
                           reported_force=.001,ftol=.01,infor='CBD_PreRot   ',curv_real=-.1)
    assert not result.stop and result.prerot_override and result.budget_exceeded
    np.testing.assert_array_equal(result.direction,[[0,1,0]])
    assert not result.direction.flags.writeable


def test_native_strict_boundaries_do_not_stop_at_equality():
    result=finish_rotation([[1,0,0]],[[0,1,0]],rotnum=20,rotmax=20,
                           reported_force=.01,ftol=.01,infor='CBD_PreRot',curv_real=-1e-6)
    assert not result.stop and not result.prerot_override


@pytest.mark.parametrize('infor,curvature',[('CBD_PreRot',-1e-6),('CBD',-.1),('CBD_PreRot\t',-.1)])
def test_budget_rollback_without_exact_prerot_override(infor,curvature):
    result=finish_rotation([[1,0,0]],[[0,1,0]],rotnum=21,rotmax=20,
                           reported_force=1.,ftol=.01,infor=infor,curv_real=curvature)
    assert result.stop and result.budget_exceeded and not result.force_converged
    assert not result.prerot_override
    np.testing.assert_array_equal(result.direction,[[1,0,0]])
