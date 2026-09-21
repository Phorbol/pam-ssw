import numpy as np
import pytest
from ase import Atoms
from ase.data.s22 import create_s22_system
from ase.io import read
from pathlib import Path
from pamssw.standalone.rc_periodic_input import unwrap_rigid_molecules
from pamssw.standalone.rc_topology import read_rigid_topology


def test_wrapped_water_dimer_triclinic_images():
    a=create_s22_system('Water_dimer');a.cell=[[8.,0,0],[2.,9.,0],[1.,2.,10.]];a.pbc=True
    bonds=[(0,1),(0,2),(3,4),(3,5)]
    original=a.copy();offset=np.array([[0,0,0],[2,-1,0],[-1,0,2],[1,1,-1],[-2,1,0],[0,-1,1]])
    a.positions+=offset@a.cell.array
    r=unwrap_rigid_molecules(a,bonds)
    assert r.components==((0,1,2),(3,4,5))
    for component in r.components:
        ids=list(component);root=ids[0]
        np.testing.assert_allclose(r.atoms.positions[ids]-r.atoms.positions[root],original.positions[ids]-original.positions[root],atol=1e-12)
    np.testing.assert_allclose(r.atoms.positions,a.positions+r.images@a.cell.array,atol=1e-12)
    np.testing.assert_array_equal(r.atoms.cell.array,a.cell.array)
    np.testing.assert_array_equal(r.atoms.pbc,a.pbc)
    np.testing.assert_array_equal(r.atoms.numbers,a.numbers)


def test_half_cell_ambiguity_and_polymer_winding_rejected():
    a=Atoms('HH',positions=[[0,0,0],[5.,0,0]],cell=[10.]*3,pbc=True)
    with pytest.raises(ValueError,match='ambiguous'):unwrap_rigid_molecules(a,[(0,1)])
    b=Atoms('HHH',positions=[[0,0,0],[3.,0,0],[6.,0,0]],cell=[9.,10.,10.],pbc=True)
    with pytest.raises(ValueError,match='winding'):unwrap_rigid_molecules(b,[(0,1),(1,2),(2,0)])


def test_actual_xxxii_172atoms_four_molecules_finite_forest():
    p=Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII')
    if not p.exists():pytest.skip('uploaded fixture unavailable')
    a=read(p/'addition/add.arc',index=0,format='dmol-arc');a.pbc=True
    top=read_rigid_topology(p/'mc/rigidbody',p/'mc/blist',natoms=len(a))
    r=unwrap_rigid_molecules(a,top.bonds)
    assert len(a)==172 and sorted(map(len,r.components))==[43]*4
    wrapped=a.copy();wrapped.wrap()
    rewrapped=unwrap_rigid_molecules(wrapped,top.bonds)
    for ids in r.components:
        ids=list(ids)
        np.testing.assert_allclose(rewrapped.atoms.positions[ids]-rewrapped.atoms.positions[ids[0]],r.atoms.positions[ids]-r.atoms.positions[ids[0]],atol=1e-11)
    from pamssw.standalone.rc_forest import RigidForestChart
    lifted=r.atoms.copy();lifted.pbc=False # chart itself is an isolated internal-geometry view
    chart=RigidForestChart(lifted,top.components)
    out,jac=chart.evaluate(np.zeros(chart.dimension))
    assert np.isfinite(jac).all() and np.isfinite(out.positions).all()
    np.testing.assert_allclose(out.positions,lifted.positions,atol=1e-12)
    for i,j in top.bonds:assert np.linalg.norm(out.positions[j]-out.positions[i])>1e-8
