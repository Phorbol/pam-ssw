"""Numerical contracts, not native execution or scientific validation."""
import numpy as np
import pytest
from ase import Atoms
from pamssw.standalone.native_rotation import RotationQuadraticBias, rotation_observation, dimer_endpoint


def test_rotation_bias_negative_rank_one_hessian_and_consistent_force():
    n=np.array([[.6,.8,0.]])
    bias=RotationQuadraticBias(np.zeros((1,3)),n,3.)
    atoms=Atoms('H',positions=[[.2,-.1,.7]])
    h=1e-6
    for j in range(3):
        plus=atoms.copy();minus=atoms.copy()
        plus.positions[0,j]+=h;minus.positions[0,j]-=h
        derivative=(bias.evaluate(plus)[0]-bias.evaluate(minus)[0])/(2*h)
        assert bias.evaluate(atoms)[1][0,j]==pytest.approx(-derivative,abs=1e-10)
    shifted=atoms.copy();shifted.positions+=h*n
    h_n=-(bias.evaluate(shifted)[1]-bias.evaluate(atoms)[1])/h
    np.testing.assert_allclose(h_n,-3*n,atol=1e-9)


def test_one_sided_curvature_tangent_and_cost_contract():
    hessian=np.array([[2.,1.,0.],[1.,4.,0.],[0.,0.,7.]])
    n=np.array([[1.,0.,0.]])
    r0=np.array([[.3,-.2,.4]]); dr=.01
    endpoint=dimer_endpoint(r0,n,dr)
    f0=-r0@hessian;f1=-endpoint@hessian
    result=rotation_observation(f0,f1,n,dr)
    assert result.curvature==pytest.approx(2)
    np.testing.assert_allclose(result.hessian_vector,[[2,1,0]],atol=1e-13)
    np.testing.assert_allclose(result.tangent_force,[[0,-.01,0]],atol=1e-13)
    assert np.vdot(result.tangent_force,n)==pytest.approx(0)
    np.testing.assert_array_equal(r0,[[.3,-.2,.4]])


def test_endpoint_rejects_nonunit_and_invalid_separation():
    with pytest.raises(ValueError,match='unit'):
        dimer_endpoint([[0,0,0]],[[2,0,0]],.1)
    with pytest.raises(ValueError,match='positive'):
        dimer_endpoint([[0,0,0]],[[1,0,0]],0)
