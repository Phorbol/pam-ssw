"""Uploaded XXXII molecular geometry, source contracts, no PES evaluation."""
from pathlib import Path
import numpy as np
import pytest
from ase.io import read
from pamssw.standalone.molecular_periodic_ga import reconstruct_type2,propose_type2


def parent():
    return read(Path(__file__).with_name('fixtures')/'type2_xxxii.extxyz')


def groups():return tuple(tuple(range(43*i,43*(i+1))) for i in range(4))


def distances(x):return np.linalg.norm(x[:,None]-x[None,:],axis=2)


def verify(c,parents):
    assert c.atoms.pbc.all() and np.linalg.det(c.atoms.cell.array)>0
    np.testing.assert_allclose(c.atoms.cell.array,np.diag(np.ptp(c.atoms.positions,axis=0)+.5))
    for g,p in zip(c.groups,c.group_parent_indices):
        number=c.groups.index(g)
        np.testing.assert_allclose(distances(c.atoms.positions[list(g)]),distances(parents[p].positions[list(groups()[number])]),atol=1e-11)
    np.testing.assert_array_equal(c.atoms.numbers,parents[0].numbers)


def test_reconstruction_uses_complete_molecules_and_new_periodic_cell():
    a=parent();old=a.copy();results=reconstruct_type2(a,groups(),2,np.random.default_rng(3))
    assert len(results)==2
    for c in results:
        verify(c,[a]);assert c.details['docking_accuracy']==3
    np.testing.assert_array_equal(a.positions,old.positions)


def test_complete_type2_batch_contains_crossover_and_reconstruction():
    a=parent();parents=[a.copy() for _ in range(3)]
    result=propose_type2(parents,[0.,.1,.2],groups(),np.random.default_rng(5),min_ga=2,
        bond_limits={},max_batches=1,max_cut_attempts=100,max_pair_attempts=1000)
    assert result.status=='target_reached' and len(result.candidates)==2
    assert [c.operation for c in result.candidates]==['crossover','periodic_reconstruction']
    for c in result.candidates:verify(c,parents)


def test_overlapping_rc_groups_rejected_and_explicit_images_supported():
    a=parent();bad=list(groups());bad[1]=(0,)+bad[1]
    with pytest.raises(ValueError):reconstruct_type2(a,bad,1,np.random.default_rng(1))
    b=a.copy();b.positions[:43]+=b.cell[0]
    shift=np.zeros((172,3),dtype=int);shift[:43,0]=-1
    left=reconstruct_type2(a,groups(),1,np.random.default_rng(2))[0]
    right=reconstruct_type2(b,groups(),1,np.random.default_rng(2),image_shifts=shift)[0]
    np.testing.assert_allclose(left.atoms.positions,right.atoms.positions,atol=1e-11)
