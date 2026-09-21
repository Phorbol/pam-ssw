"""TYPE1 source arithmetic and real-input geometry contracts; zero PES calls."""
from pathlib import Path
import numpy as np
import pytest
from ase.io import read
from pamssw.standalone.periodic_ga import mutate_type1


def real_parent():
    root=Path(__file__).resolve().parents[2]
    return read(root/'research/ga_ssw/evidence/block-aloh26-seed3/input.extxyz')


class ConstantRandom:
    def random(self):return .75


def test_doping_exact_batch_counts_parent_sort_and_cartesian_draws():
    a=real_parent(); b=a.copy();b.cell*=1.04
    olda=a.copy();oldb=b.copy()
    children=mutate_type1([a,b],[2.,1.],8,ConstantRandom())
    assert len(children)==10
    assert [c.operation for c in children]==['exchange']*4+['disturb_best_sparse','disturb_best_half']+['disturb_random_sparse']*2+['disturb_random_half']*2
    # Random sorted-index 1 selects caller index 0; best selects caller index 1.
    assert [c.parent_index for c in children]==[0]*4+[1]*2+[0]*4
    c=children[4]
    expected=b.positions.copy();expected[19]+=5*.3*.25
    np.testing.assert_allclose(c.atoms.positions,expected,atol=1e-13)
    np.testing.assert_array_equal(c.atoms.cell.array,b.cell.array)
    for c in children:
        assert c.atoms.pbc.all() and c.atoms.calc is None
        np.testing.assert_array_equal(np.sort(c.atoms.numbers),np.sort(a.numbers))
    np.testing.assert_array_equal(a.positions,olda.positions)
    np.testing.assert_array_equal(b.positions,oldb.positions)


def test_pure_source_batch_count_and_zero_n_quirk():
    a=real_parent();a.numbers[:]=29
    children=mutate_type1([a],[1.],8,np.random.default_rng(8))
    assert len(children)==13
    zero=mutate_type1([a],[1.],0,np.random.default_rng(8))
    assert len(zero)==1 and zero[0].details['displacement_draws']==len(a)//10
    assert mutate_type1([real_parent()],[1.],0,np.random.default_rng(8))==[]


def test_pure_small_system_retains_source_noop_without_invented_fallback():
    a=real_parent()[:4];a.numbers[:]=29
    child=mutate_type1([a],[0.],0,np.random.default_rng(0))[0]
    np.testing.assert_array_equal(child.atoms.positions,a.positions)
    assert child.details['displacement_draws']==0


def test_periodic_metadata_and_composition_validation():
    a=real_parent();b=a.copy();b.numbers[0]=6
    with pytest.raises(ValueError,match='composition'):mutate_type1([a,b],[0,1],4,np.random.default_rng(0))
    a.pbc=False
    with pytest.raises(ValueError):mutate_type1([a],[0],4,np.random.default_rng(0))
    with pytest.raises(ValueError):mutate_type1([],[],4,np.random.default_rng(0))


def test_exchange_swaps_species_not_sites_with_exact_10n_draws():
    a=real_parent(); draws=iter([0.,0.,.99]+[0.,0.]*(10*len(a)-1))
    class Replay:
        def random(self):return next(draws)
    c=mutate_type1([a],[0.],2,Replay())[0]
    expected=a.numbers.copy();expected[0],expected[-1]=int(expected[-1]),int(expected[0])
    np.testing.assert_array_equal(c.atoms.numbers,expected)
    np.testing.assert_array_equal(c.atoms.positions,a.positions)
    assert c.details['exchange_draws']==10*len(a)
    with pytest.raises(StopIteration):next(draws)


def test_periodic_crossover_uses_one_selected_cell_for_both_halves():
    from pamssw.standalone.periodic_ga import PeriodicGamete, PeriodicPool, cross_periodic_pool
    from ase import Atoms
    a=Atoms('HO',positions=[[.3,.4,.5],[1,1,1]],cell=[[4,0,0],[.5,5,0],[.2,.3,6]],pbc=True)
    b=a.copy();b.set_cell(a.cell.array*1.4,scale_atoms=True)
    son=PeriodicGamete(a[:1],0,(0,),a.cell.array.copy())
    daughters=(PeriodicGamete(a[1:],0,(1,),a.cell.array.copy()),PeriodicGamete(b[1:],1,(1,),b.cell.array.copy()))
    pool=PeriodicPool((son,),daughters,(1,8),(),1,1)
    draws=iter([0.,.75,.25])
    class Replay:
        def random(self):return next(draws)
    child=cross_periodic_pool(pool,Replay(),max_pair_attempts=1)
    np.testing.assert_allclose(child.atoms.cell.array,b.cell.array)
    np.testing.assert_allclose(child.atoms.positions,np.vstack([b.positions[1],a.positions[0]*1.4]))
    assert child.details['cell_parent_index']==1
    assert child.details['native_ova_index_defect_corrected']
    assert child.atom_parent_indices==(1,0)


def test_full_periodic_filter_self_images_and_skew_short_vectors():
    from ase import Atoms
    from pamssw.standalone.periodic_ga import periodic_collision_free
    a=Atoms('Cu',positions=[[0,0,0]],cell=[[3,0,0],[6,.4,0],[0,0,3]],pbc=True)
    assert not periodic_collision_free(a,{(29,29):.5}) # -2a+b omitted by native 2x2x2
    assert periodic_collision_free(a,{(29,29):.3})
    assert periodic_collision_free(a,{})


def test_periodic_pool_and_complete_proposal_keep_composition_and_sources():
    from pamssw.standalone.periodic_ga import build_periodic_pool,propose_type1
    a=real_parent();parents=[a.copy() for _ in range(3)]
    for i,b in enumerate(parents):b.set_cell(b.cell.array*(1+.02*i),scale_atoms=True)
    pool=build_periodic_pool(parents,[0.,.1,.2],np.random.default_rng(6),
        max_cut_attempts=100,slots_per_parent=1,cuts_per_slot=1)
    assert len(pool.sons)==len(pool.daughters)==3
    result=propose_type1(parents,[0.,.1,.2],[(0,1),(2,)],np.random.default_rng(8),
        min_ga=8,bond_limits={},max_batches=2,max_cut_attempts=100,max_pair_attempts=1000,
        slots_per_parent=1,cuts_per_slot=1)
    assert result.status=='target_reached' and len(result.candidates)>=8
    assert {'crossover','exchange','disturb_random_half'}<=set(c.operation for c in result.candidates)
    for c in result.candidates:
        np.testing.assert_array_equal(np.sort(c.atoms.numbers),np.sort(a.numbers))
        assert c.atoms.pbc.all() and np.linalg.det(c.atoms.cell.array)>0
        assert all(0<=i<3 for i in c.atom_parent_indices)
        np.testing.assert_array_equal(c.atoms.numbers,[parents[p].numbers[j] for p,j in zip(c.atom_parent_indices,c.source_atom_indices)])
