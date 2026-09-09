import numpy as np
import pytest
from ase import Atoms
from pamssw.standalone.gaussian import ProjectedGaussian, GaussianSum, adjust_native_weight


def test_projection_energy_force_finite_difference_and_transverse_invariance():
    atoms=Atoms('HH',positions=[[.3,.2,0.],[0.,0.,0.]])
    n=np.array([[1.,0.,0.],[0.,0.,0.]])
    bias=ProjectedGaussian(np.zeros((2,3)),n,.4,2.)
    energy,force=bias.evaluate(atoms)
    h=1e-6;plus=atoms.copy();minus=atoms.copy()
    plus.positions[0,0]+=h;minus.positions[0,0]-=h
    derivative=(bias.evaluate(plus)[0]-bias.evaluate(minus)[0])/(2*h)
    assert force[0,0]==pytest.approx(-derivative)
    atoms.positions[0,1]+=100
    assert bias.evaluate(atoms)[0]==energy
    assert force[0,0]>0


def test_gaussian_sum_and_no_mic_force_jump():
    atoms=Atoms('H',positions=[[2.99,0.,0.]],cell=[6.,6.,6.],pbc=True)
    bias=ProjectedGaussian(np.zeros((1,3)),np.array([[1.,0.,0.]]),3.,2.)
    first=bias.evaluate(atoms)[1][0,0]
    atoms.positions[0,0]=3.01
    second=bias.evaluate(atoms)[1][0,0]
    assert first>0 and second>0
    e,f=GaussianSum([bias,bias]).evaluate(atoms)
    assert e==2*bias.evaluate(atoms)[0]
    np.testing.assert_allclose(f,2*bias.evaluate(atoms)[1])


def test_height_maxw_is_exit_not_clip():
    result=adjust_native_weight(fa0=[[-100,1,0]],fa2=[[0,0,0]],n=[[1,0,0]],
                                d1=1,d2=1,e2=-10,w=1,maxw=1.5,step=1,scalefact0=2)
    assert result.weight==2 and result.stop_reason=='maxw_exceeded'


def test_nonunit_direction_rejected():
    with pytest.raises(ValueError,match='unit'):
        ProjectedGaussian(np.zeros((1,3)),np.array([[2.,0.,0.]]),1.,1.)
