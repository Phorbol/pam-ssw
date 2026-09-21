"""VC-RC exact derivatives; EMT here is a force oracle, not molecular physics."""
import numpy as np
import pytest
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone.rc_vc_geometry import RigidForestCellChart
from pamssw.standalone.vc_geometry import ASEStressSurface


def reference():
    a=Atoms('Cu6',positions=[[.5,.4,.3],[2.8,.4,.3],[1.5,2.5,.6],[5.,4.,3.],[7.3,4.,3.],[6.,6.1,3.3]],cell=[[9,.4,.3],[0,9,.2],[.1,0,9]],pbc=True)
    trees=[dict(bodies=[(0,1,2)],parents=(-1,),joints=(None,)),dict(bodies=[(3,4,5)],parents=(-1,),joints=(None,))]
    return a,trees


def test_finite_vc_all_geometric_jacobians_and_rigidity():
    a,trees=reference();chart=RigidForestCellChart(a,trees,rotation_length=2.1,torsion_length=1.7,strain_length=4.)
    assert chart.dimension==15  # all6 root rotations + relative3 translation + cell6
    q=np.array([.3,-.5,.7,.2,-.3,.4,-.6,.5,.8,.15,-.12,.2,.11,-.17,.09])
    b,J,C=chart.geometry(q)
    for ids in (range(3),range(3,6)):
        ids=list(ids);np.testing.assert_allclose(b.get_all_distances()[np.ix_(ids,ids)],a.get_all_distances()[np.ix_(ids,ids)],atol=3e-15)
    assert b.pbc.all() and not np.allclose(a.cell.array,b.cell.array)
    for k in range(chart.dimension):
        d=np.eye(chart.dimension)[k]*1e-6
        plus=chart.unpack(q+d);minus=chart.unpack(q-d)
        np.testing.assert_allclose((plus.positions-minus.positions)/2e-6,J[:,:,k],atol=2e-9)
        np.testing.assert_allclose((plus.cell.array-minus.cell.array)/2e-6,C[:,:,k],atol=2e-9)
    # Anchor rotation is retained and actually moves its atoms.
    assert np.linalg.norm(J[:3,:,:3])>0
    np.testing.assert_allclose(a.positions,reference()[0].positions)


def test_actual_emt_enthalpy_all_dofs_and_nonaffine_correction():
    a,trees=reference();chart=RigidForestCellChart(a,trees,rotation_length=2.1,torsion_length=1.7,strain_length=4.)
    q=np.array([.3,-.5,.7,.2,-.3,.4,-.6,.5,.8,.15,-.12,.2,.11,-.17,.09])
    oracle=ASEStressSurface(EMT());pressure=.003
    ev=chart.evaluate(q,oracle.evaluate,pressure=pressure)
    for k in range(chart.dimension):
        d=np.eye(chart.dimension)[k]*1e-5
        fd=(chart.evaluate(q+d,oracle.evaluate,pressure=pressure).objective-chart.evaluate(q-d,oracle.evaluate,pressure=pressure).objective)/2e-5
        assert ev.gradient[k]==pytest.approx(fd,abs=3e-7)
    b,J,C=chart.geometry(q)
    naive=[]
    for k in range(chart.dimension-6,chart.dimension):
        A=np.linalg.solve(b.cell.array,C[:,:,k]);naive.append(b.get_volume()*np.sum((ev.stress+pressure*np.eye(3))*A))
    assert np.linalg.norm(ev.gradient[-6:]-naive)>1e-3
    assert oracle.requests==31


def test_lifted_coordinates_preserved_and_invalid_domain():
    a,trees=reference();a.positions[3:]+=a.cell.array[0]
    chart=RigidForestCellChart(a,trees,rotation_length=2,torsion_length=2,strain_length=4)
    np.testing.assert_allclose(chart.unpack(np.zeros(chart.dimension)).positions,a.positions,atol=1e-14)
    a.pbc=False
    with pytest.raises(ValueError,match='periodic'):RigidForestCellChart(a,trees,rotation_length=2,torsion_length=2,strain_length=4)


def test_joint_torsion_and_strain_preserve_shared_endpoints_with_actual_emt():
    a=Atoms('Cu4',positions=[[.2,.3,.4],[2.5,.3,.4],[1.4,2.3,.7],[3.7,2.4,1.8]],cell=[9,10,11],pbc=True)
    tree=dict(bodies=[(0,1,2),(1,2,3)],parents=(-1,0),joints=(None,(1,2)))
    c=RigidForestCellChart(a,[tree],rotation_length=1.8,torsion_length=2.3,strain_length=5.)
    assert c.dimension==10
    q=np.array([.6,-.7,.3,1.1,.2,-.1,.3,.15,-.2,.11]);b,J,C=c.geometry(q)
    for ids in tree['bodies']:
        np.testing.assert_allclose(a.get_all_distances()[np.ix_(ids,ids)],b.get_all_distances()[np.ix_(ids,ids)],atol=2e-15)
    oracle=ASEStressSurface(EMT());ev=c.evaluate(q,oracle.evaluate,pressure=-.001)
    for k in range(c.dimension):
        d=np.eye(c.dimension)[k]*1e-5
        ap=c.unpack(q+d);am=c.unpack(q-d)
        np.testing.assert_allclose((ap.positions-am.positions)/2e-5,J[:,:,k],atol=1e-9)
        fd=(c.evaluate(q+d,oracle.evaluate,pressure=-.001).objective-c.evaluate(q-d,oracle.evaluate,pressure=-.001).objective)/2e-5
        assert ev.gradient[k]==pytest.approx(fd,abs=3e-7)
    assert oracle.requests==21
