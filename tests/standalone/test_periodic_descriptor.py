"""Periodic routing geometry only; no PES and no identity/kinetic claims."""
import numpy as np
import pytest
from ase.build import bulk
from ase.io import read
from pathlib import Path
from pamssw.standalone.periodic_descriptor import periodic_descriptor,periodic_projection


def tables(a):
    elements=set(a.numbers.tolist())
    return {tuple(sorted((i,j))):2. for i in elements for j in elements}


def test_primitive_fcc_self_images_and_repeated_local_environments():
    a=bulk('Cu','fcc',a=3.6);b=a.repeat((2,2,2));table={(29,29):2.6}
    d=periodic_descriptor(a,table,1.05);e=periodic_descriptor(b,table,1.05)
    assert d['n1']==[[12]]
    for key in ['n1','n2','n3','d1','d2','d3']:
        np.testing.assert_allclose(e[key],np.repeat(d[key],len(b),axis=0),atol=1e-12)
    assert d['d1'][0][0]==pytest.approx(3.6/np.sqrt(2)-1.3)


def test_real_aloh_rotation_translation_image_and_permutation_invariance():
    a=read(Path(__file__).resolve().parents[2]/'research/ga_ssw/evidence/block-aloh26-seed3/input.extxyz')
    d=periodic_descriptor(a,tables(a),1.1)
    b=a.copy();b.positions[0]+=3*b.cell[0]-2*b.cell[2];b.positions +=[.17,-.38,.24]
    b.rotate(37,'z',rotate_cell=True);b=b[np.random.default_rng(11).permutation(len(b))]
    e=periodic_descriptor(b,tables(b),1.1)
    for key in ['n1','n2','n3','d1','d2','d3']:
        np.testing.assert_allclose(d[key],e[key],atol=1e-10)
    weights=[.3,.2,.2,.1,.1,.1]
    np.testing.assert_allclose(periodic_projection(d,[d,d,d],weights),[1,1,1])


def test_basis_count_and_composition_contract():
    a=bulk('Cu','fcc',a=3.6);d=periodic_descriptor(a,{(29,29):2.6},1.05)
    with pytest.raises(ValueError):periodic_projection(d,[d,d],[1,0,0,0,0,0])
    e=periodic_descriptor(a.repeat((2,1,1)),{(29,29):2.6},1.05)
    with pytest.raises(ValueError):periodic_projection(d,[e,e,e],[1,0,0,0,0,0])


def test_unimodular_cell_basis_change_preserves_periodic_shells():
    a=bulk('Cu','fcc',a=3.6);b=a.copy()
    b.set_cell(np.array([[1,2,0],[0,1,0],[0,0,1]])@a.cell.array,scale_atoms=False)
    d=periodic_descriptor(a,{(29,29):2.6},1.05)
    e=periodic_descriptor(b,{(29,29):2.6},1.05)
    for key in ['n1','n2','n3','d1','d2','d3']:
        np.testing.assert_allclose(d[key],e[key],atol=1e-12)
