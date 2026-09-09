"""Static kernel reconstruction checks; no PES calls or scientific validation."""
import numpy as np
import pytest
from research.ga_ssw.native_weight import adjust_native_weight


def adjust(**changes):
    args=dict(fa0=np.array([[-100.,1.,0.]]),fa2=np.zeros((1,3)),n=np.array([[1.,0.,0.]]),
              d1=1.,d2=1.,e2=-10.,w=1.,maxw=1.5,step=1.,scalefact0=2.)
    args.update(changes)
    return adjust_native_weight(**args)


def test_maxw_is_exit_after_update_not_a_weight_clip():
    result=adjust()
    assert result.weight==2. and result.updates==1
    assert result.stop_reason=='maxw_exceeded'
    assert result.angle_degrees>87
    assert result.energy==-8.


def test_scale_growth_and_additive_ceiling_both_act():
    result=adjust(w=.1,maxw=1.,step=2.)
    assert result.weight==pytest.approx(2.8)
    assert result.updates==3


@pytest.mark.parametrize('angle,updates',[(86.999,0),(87.001,1)])
def test_force_angle_straddles_native_87_degree_boundary(angle,updates):
    radians=angle*np.pi/180
    # Unit force at specified angle after the initial Gaussian contribution.
    result=adjust(fa0=np.array([[np.cos(radians)-1,np.sin(radians),0.]]),maxw=10.)
    assert result.updates==updates
    assert result.stop_reason=='angle_satisfied'


def test_recorded_water_force_excerpt_preserves_transverse_force_and_inputs():
    # First three atoms, original uploaded H30O15 NN single point, eval-00000,
    # water-ase-quench; allfor.arc lines 3-5, not a new evaluation or native helper oracle.
    force=np.array([[-.0000701078,-.0003741950,.0203528949],
                    [-.0094348319,-.0037231760,-.0069127364],
                    [-.0031284610,.0055420761,-.0009787887]])
    direction=force/np.linalg.norm(force)
    fa2=np.full_like(force,.0001)
    incoming=force+fa2
    saved=incoming.copy()
    result=adjust(fa0=incoming,fa2=fa2,n=direction,d1=.8,d2=.3,w=.4,e2=-220.8993988)
    assert result.updates==0
    assert result.energy==pytest.approx(-220.5793988)
    delta=result.force-force
    assert np.sum(delta*direction)==pytest.approx(.096)
    np.testing.assert_allclose(delta-np.sum(delta*direction)*direction,0,atol=1e-16)
    np.testing.assert_array_equal(incoming,saved)


def test_degenerate_force_and_nonunit_direction_are_explicitly_outside_domain():
    with pytest.raises(ValueError,match='zero resultant'):
        adjust(fa0=np.array([[-1.,0.,0.]]))
    with pytest.raises(ValueError,match='unit'):
        adjust(n=np.array([[2.,0.,0.]]))


def test_angle_satisfied_entry_does_not_apply_maxw_check():
    result=adjust(fa0=np.array([[1.,1.,0.]]),w=2.,maxw=1.)
    assert result.weight==2. and result.updates==0
    assert result.stop_reason=='angle_satisfied'


def test_post_update_limit_precedes_angle_satisfaction():
    result=adjust(fa0=np.array([[-1.,1.,0.]]))
    assert result.weight==2. and result.angle_degrees<87.
    assert result.stop_reason=='maxw_exceeded'
