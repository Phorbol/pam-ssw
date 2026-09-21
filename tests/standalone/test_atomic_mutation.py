"""Original-bytecode primitive fixtures and corrected reinsertion contracts."""
import json
from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.cluster.icosahedron import Icosahedron
from pamssw.standalone.atomic_ga import (disturb_atoms, exchange_atoms,
    reinsert_undercoordinated_atoms, mutate_type0, propose_type0)
from pamssw.standalone.ga_operators import SamplingExhausted

class JavaRandom:
    def __init__(self, seed):self.seed=(seed ^ 0x5DEECE66D)&((1<<48)-1)
    def next(self,bits):
        self.seed=(self.seed*0x5DEECE66D+0xB)&((1<<48)-1)
        return self.seed>>(48-bits)
    def random(self):return ((self.next(26)<<27)+self.next(27))/float(1<<53)

def mindistance(a):
    d=np.linalg.norm(a.positions[:,None]-a.positions[None,:],axis=2)
    np.fill_diagonal(d,np.inf)
    return d.min()

def test_original_bytecode_primitives_and_collision_anomaly():
    fixture=json.loads((Path(__file__).parent/'fixtures/type0_mutation.json').read_text())
    a=Atoms(numbers=fixture['numbers'],positions=fixture['positions'])
    for row in fixture['cases']:
        if row['method']=='interMu':
            original=Atoms(numbers=row['numbers'],positions=row['positions'])
            assert not row['no_close_pair'] and mindistance(original)<.3
            corrected,details=reinsert_undercoordinated_atoms(a,5,JavaRandom(row['seed']),max_insertion_attempts=10000)
            assert mindistance(corrected)>=.3
            assert sorted(corrected.numbers)==sorted(a.numbers)
            continue
        result,_=(disturb_atoms(a,6,.7,JavaRandom(row['seed'])) if row['method']=='disturbance' else exchange_atoms(a,JavaRandom(row['seed'])))
        np.testing.assert_array_equal(result.numbers,row['numbers'])
        np.testing.assert_allclose(result.positions,row['positions'],atol=2e-14,rtol=0)
    np.testing.assert_array_equal(a.positions,fixture['positions'])

def test_complete_quotas_lineage_and_parent_preservation():
    for alloy in (False,True):
        a=Icosahedron('Cu',2)
        if alloy:a.numbers[::2]=47
        parents=[a.copy() for _ in range(3)];energies=[1.,-1.,0.]
        result=mutate_type0(parents,energies,np.random.default_rng(7),n=16,max_insertion_attempts=10000)
        assert len(result)==(3*(16//4)+5*(16//8) if alloy else 3*(16//4+1)+2*(16//2)+1)
        for c in result:
            assert sorted(c.atoms.numbers)==sorted(a.numbers)
            assert len(c.group_parent_indices)==len(a)
            assert len(set(c.group_parent_indices))==1
            assert c.atoms.calc is None
        for p in parents:np.testing.assert_array_equal(p.positions,a.positions)
        if not alloy:assert result[-1].group_parent_indices==(1,)*13

def test_reinsertion_exhaustion_never_pads_origin():
    a=Atoms('Cu11',positions=np.zeros((11,3)))
    with pytest.raises(SamplingExhausted):
        reinsert_undercoordinated_atoms(a,5,np.random.default_rng(1),max_insertion_attempts=2)
    with pytest.raises(ValueError,match='surviving'):
        reinsert_undercoordinated_atoms(a,11,np.random.default_rng(1),max_insertion_attempts=2)

def test_whole_batch_regions_and_no_truncation():
    a=Icosahedron('Cu',2);parents=[a.copy() for _ in range(4)]
    result=propose_type0(parents,[0.,1.,2.,3.],np.random.default_rng(5),min_ga=8,bond_limits={},max_batches=1,max_cut_attempts=100,max_pair_attempts=100,parent_regions=[[0,1],[2,3]],max_insertion_attempts=1000)
    # cross2 + pure(n4)=11 + pure(n1)=4 => 17, not target8.
    assert result.status=='target_reached' and len(result.candidates)==17
    for c in result.candidates:
        if 'parent_region' in c.details:
            expected={0,1} if c.details['parent_region']==0 else {2,3}
            assert set(c.group_parent_indices)<=expected
    with pytest.raises(ValueError,match='partition'):
        propose_type0(parents,[0.,1.,2.,3.],np.random.default_rng(5),min_ga=8,bond_limits={},max_batches=1,max_cut_attempts=100,max_pair_attempts=100,parent_regions=[[0,1],[1,2]])
