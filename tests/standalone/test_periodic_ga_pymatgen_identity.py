"""Optional real AlOH identity controls, zero PES calls."""
from pathlib import Path
import numpy as np
import pytest
pytest.importorskip('pymatgen.analysis.structure_matcher')
from ase.io import read
from pamssw.standalone.periodic_ga_reference import pymatgen_identity


def test_aloh_representation_controls_and_no_implicit_density_scaling():
    p=Path(__file__).resolve().parents[2]/'research/ga_ssw/evidence/block-aloh26-seed3/input.extxyz'
    a=read(p)
    match=pymatgen_identity(ltol=.2,stol=.3,angle_tol=5.)
    b=a[np.random.default_rng(17).permutation(len(a))]
    b.positions += [.17,-.31,.23]
    b.positions[0] += 2*b.cell[0]-b.cell[2]
    b.rotate(31,'z',rotate_cell=True)
    assert match(a,b)
    c=a.copy();c.set_cell(np.array([[1,1,0],[0,1,0],[0,0,1]])@a.cell.array,scale_atoms=False)
    assert match(a,c)
    assert match(a,a.repeat((2,1,1)))
    dilated=a.copy();dilated.set_cell(a.cell.array*1.5,scale_atoms=True)
    assert not match(a,dilated)
    changed=a.copy();changed.numbers[0]=1 if changed.numbers[0]!=1 else 8
    assert not match(a,changed)
