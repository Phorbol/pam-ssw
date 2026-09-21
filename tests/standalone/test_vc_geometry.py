"""Finite-strain E/F/stress consistency using real Cu EMT, not VC efficacy."""
import numpy as np
import pytest
from ase.build import bulk
from ase.calculators.emt import EMT
from pamssw.standalone.vc_geometry import SymmetricLogStrainChart, ASEStressSurface, SYMMETRIC_BASIS


def reference():
    atoms = bulk('Cu', 'fcc', a=3.6, cubic=True)
    atoms.set_cell([[3.6,.18,-.07],[.06,3.7,.13],[-.11,.04,3.55]], scale_atoms=True)
    atoms.positions += np.random.default_rng(12).normal(scale=.027,size=(4,3))
    return atoms


def test_orthonormal_basis_chart_roundtrip_and_projection():
    atoms=reference(); chart=SymmetricLogStrainChart(atoms,strain_length=2.7)
    np.testing.assert_allclose(np.einsum('iab,jab->ij',SYMMETRIC_BASIS,SYMMETRIC_BASIS),np.eye(6),atol=1e-15)
    q=chart.pack(atoms);q[-6:]=2.7*np.array([.12,-.06,.04,.09,-.05,.08]);q[:3]+=atoms.cell[0]*3
    trial=chart.unpack(q)
    np.testing.assert_allclose(chart.pack(trial),q,atol=1e-13)
    assert np.linalg.det(trial.cell)>0
    assert np.max(trial.positions)>8.  # No wrapping a stored lifted coordinate.
    vector=np.random.default_rng(2).normal(size=chart.ndof);projected=chart.project(vector)
    np.testing.assert_allclose(projected[:-6].reshape(-1,3).sum(0),0.,atol=1e-15)
    np.testing.assert_array_equal(projected[-6:],vector[-6:])
    np.testing.assert_allclose(chart.project(projected),projected,atol=1e-15)
    np.testing.assert_array_equal(atoms.cell.array,reference().cell.array)


@pytest.mark.parametrize('strain',[[.12,-.06,.04,.09,-.05,.08],[-.08,.11,-.03,-.07,.04,.06]])
@pytest.mark.parametrize('pressure',[0.,.013])
@pytest.mark.parametrize('length',[1.4,5.2])
def test_cu_emt_all_coordinate_gradients_at_finite_triclinic_strain(strain,pressure,length):
    atoms=reference();chart=SymmetricLogStrainChart(atoms,strain_length=length);surface=ASEStressSurface(EMT())
    q=chart.pack(atoms);q[-6:]=length*np.array(strain)
    result=chart.evaluate(q,surface.evaluate,pressure=pressure)
    # Atomic coordinate plus all six cell basis derivatives; nonzero offdiagonal
    # strain is essential to expose expm/gradient transposition mistakes.
    for i in (0,4,8,*range(12,18)):
        direction=np.zeros(chart.ndof);direction[i]=1.;h=2e-6
        plus=chart.evaluate(q+h*direction,surface.evaluate,pressure=pressure).objective
        minus=chart.evaluate(q-h*direction,surface.evaluate,pressure=pressure).objective
        assert (plus-minus)/(2*h)==pytest.approx(result.gradient[i],abs=2e-7,rel=2e-7)
    assert surface.requests==19
    assert result.objective==pytest.approx(result.energy+pressure*result.volume)
    assert result.atoms.calc is None


def test_pressure_gradient_and_metric_rescaling_are_exact():
    atoms=reference();a=SymmetricLogStrainChart(atoms,strain_length=2.);b=SymmetricLogStrainChart(atoms,strain_length=7.)
    q=a.pack(atoms);q[-6:]=[.2,-.1,.08,.15,-.04,.07];other=b.pack(a.unpack(q));s=ASEStressSurface(EMT())
    ra=a.evaluate(q,s.evaluate,pressure=.02);rb=b.evaluate(other,s.evaluate,pressure=.02)
    np.testing.assert_allclose(ra.atoms.positions,rb.atoms.positions,atol=1e-13)
    assert ra.objective==pytest.approx(rb.objective,abs=1e-12)
    np.testing.assert_allclose(ra.gradient[:-6],rb.gradient[:-6],atol=1e-12)
    np.testing.assert_allclose(ra.gradient[-6:]*2,rb.gradient[-6:]*7,atol=1e-12)
    r0=a.evaluate(q,s.evaluate)
    np.testing.assert_allclose(ra.gradient[-6:]-r0.gradient[-6:],.02*ra.volume/2*np.array([1,1,1,0,0,0]),atol=1e-12)


def test_rotation_and_partial_pbc_rejected():
    atoms=reference();chart=SymmetricLogStrainChart(atoms,strain_length=2.)
    rotated=atoms.copy();rotated.rotate(20,'z',rotate_cell=True)
    with pytest.raises(ValueError,match='outside symmetric'):
        chart.pack(rotated)
    atoms.pbc[2]=False
    with pytest.raises(ValueError,match='fully periodic'):
        SymmetricLogStrainChart(atoms,strain_length=2.)
