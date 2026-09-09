import numpy as np
import pytest
from ase import Atoms
from ase.constraints import FixAtoms, FixBondLengths
from pamssw.standalone.direction import reference_soft_mode


def harmonic(matrix,calls):
    def evaluate(atoms):
        calls.append(atoms.positions.copy())
        x=atoms.positions.ravel()
        return .5*x@matrix@x,(-matrix@x).reshape(-1,3)
    return evaluate


def test_off_diagonal_hessian_low_mode_and_evaluation_budget():
    matrix=np.array([[2.,1.,0.],[1.,2.,0.],[0.,0.,5.]])
    atoms=Atoms('H',positions=[[.2,.1,-.3]])
    saved=atoms.positions.copy();calls=[]
    result=reference_soft_mode(atoms,np.array([[1.,.2,.4]]),fd_step=1e-4,
                               max_hvp=4,residual_tol=1e-8,evaluate=harmonic(matrix,calls))
    assert result.converged
    assert result.curvature==pytest.approx(1.)
    assert abs(result.direction.ravel()@np.array([1.,-1.,0.])/np.sqrt(2))==pytest.approx(1.)
    assert result.hvp_calls<=4 and result.force_calls==len(calls)==2*result.hvp_calls
    np.testing.assert_array_equal(atoms.positions,saved)


def test_small_budget_reports_unconverged_residual():
    calls=[];matrix=np.diag([1.,2.,8.])
    result=reference_soft_mode(Atoms('H'),np.ones((1,3)),fd_step=1e-4,max_hvp=2,
                               residual_tol=1e-10,evaluate=harmonic(matrix,calls))
    assert not result.converged and result.residual_norm>1
    assert len(calls)==4


def test_fixed_atoms_excluded_from_hessian_subspace():
    atoms=Atoms('HH',positions=np.zeros((2,3)),constraint=FixAtoms(indices=[0]))
    calls=[];matrix=np.diag([-10.,-9.,-8.,1.,2.,3.])
    result=reference_soft_mode(atoms,np.ones((2,3)),fd_step=1e-4,max_hvp=4,
                               residual_tol=1e-8,evaluate=harmonic(matrix,calls))
    np.testing.assert_array_equal(result.direction[0],0)
    assert result.curvature==pytest.approx(1.)
    assert all(np.all(position[0]==0) for position in calls)


def test_unknown_constraint_rejected_before_evaluation():
    atoms=Atoms('HH',constraint=FixBondLengths([(0,1)]));calls=[]
    with pytest.raises(ValueError,match='FixAtoms'):
        reference_soft_mode(atoms,np.ones((2,3)),fd_step=1e-4,max_hvp=4,
                            residual_tol=1e-8,evaluate=harmonic(np.eye(6),calls))
    assert not calls


def test_paper_biased_direction_uses_original_anchor_and_one_sided_budget():
    from pamssw.standalone.direction import paper_biased_direction
    atoms=Atoms('H',positions=[[.2,.3,0.]])
    anchor=np.array([[1.,1.,0.]])/np.sqrt(2)
    calls=[]
    hessian=np.diag([2.,5.,8.])
    def evaluate(a):
        calls.append(a.positions.copy())
        x=a.positions.ravel()
        return .5*x@hessian@x,(-hessian@x).reshape(1,3)
    result=paper_biased_direction(atoms,anchor,rotation_bias=4.,fd_step=.001,
                                max_hvp=5,tol=1e-9,evaluate=evaluate)
    shifted=hessian-4*np.outer(anchor.ravel(),anchor.ravel())
    vals,vecs=np.linalg.eigh(shifted)
    assert result.converged
    assert result.curvature==pytest.approx(vals[0],abs=1e-9)
    assert abs(np.dot(result.direction.ravel(),vecs[:,0]))==pytest.approx(1.)
    assert len(calls)==result.force_calls==result.hvp_calls+1
    np.testing.assert_array_equal(calls[0],atoms.positions)
