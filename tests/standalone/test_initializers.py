from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from pamssw.standalone.initializers import initialize_type2,initialize_type3,initialize_type4

FIX=Path(__file__).parent/'fixtures'


def test_type2_supplied_seed_plus_whole_molecular_reconstruction():
    a=read(FIX/'type2_xxxii.extxyz');groups=[range(i,i+43) for i in range(0,172,43)]
    result=initialize_type2([a],groups,count=1,rng=np.random.default_rng(6))
    assert result.status=='completed' and len(result.structures)==2
    assert result.physical_requests==result.auxiliary_evaluations==0
    np.testing.assert_array_equal(result.structures[0].positions,a.positions)
    for group in groups:
        np.testing.assert_allclose(result.structures[1][list(group)].get_all_distances(),a[list(group)].get_all_distances(),atol=1e-11)
    assert all(not o['certified'] for o in result.origins)


def test_type3_complete_expansion_filter_shuffle_cap():
    path=Path(__file__).resolve().parents[2]/'research/ga_ssw/evidence/staged-water/final-arc/0.arc'
    rows=[line.split() for line in path.read_text().splitlines() if 'CORE' in line]
    a=Atoms([r[0] for r in rows],positions=[[float(x) for x in r[1:4]] for r in rows])
    groups=[range(i,i+3) for i in range(0,len(a),3)]
    result=initialize_type3([a,a,a],[0.,.1,.2],groups,[0]*len(groups),count=1,rng=np.random.default_rng(7),
        bond_limits={},max_cut_attempts=100,max_pair_attempts=100)
    assert result.status=='completed' and len(result.structures)==1
    assert result.ledger['raw_count']==7 # 3 seeds + cross + 3 mutation families
    assert len(result.ledger['generated_candidates'])==4
    assert not result.structures[0].pbc.any() and result.physical_requests==0
    assert all(not o['certified'] for o in result.origins)
    with pytest.raises(ValueError,match='explicit LJ pair sigmas'):
        initialize_type3([a],[0.],groups,[0]*len(groups),count=1,rng=np.random.default_rng(7),
            bond_limits={},max_cut_attempts=100,max_pair_attempts=100,lj_monomer_optimization=True)


def test_type4_keeps_seed_and_exact_one_batch_with_auxiliary_cost():
    a=read(FIX/'type4_tio2_au24o4.extxyz');symbols=set(a.get_chemical_symbols())
    limits={tuple(sorted((x,y))):.001 for x in symbols for y in symbols}
    result=initialize_type4([a,a,a],[0.,.1,.2],range(486),range(486,514),count=4,rng=np.random.default_rng(9),
        bond_limits=limits,atomic_radii={8:1.269578/2,79:2.574144/2},site_fractional=(.5,.5),
        max_cut_attempts=100,max_pair_attempts=100,max_face_attempts=1000,max_insertion_attempts=10000,
        auxiliary_evaluations=3,cuts_per_parent_slot=2)
    assert result.status=='completed' and len(result.structures)==10
    assert result.auxiliary_evaluations==90 and result.physical_requests==0
    assert all(not o['certified'] for o in result.origins)
    for structure in result.structures:
        np.testing.assert_array_equal(structure.positions[:486],a.positions[:486])
        np.testing.assert_array_equal(structure.cell,a.cell)


def test_type1_forced_doping_keeps_crossing_batch_instead_of_dropping_it():
    from pamssw.standalone.initializers import initialize_type1
    a=read(FIX/'type2_xxxii.extxyz') # actual periodic input; atomic TYPE1 ignores molecular groups
    result=initialize_type1([a],[0.],rng=np.random.default_rng(8),bond_limits={},max_batches=1)
    assert result.status=='completed' and len(result.structures)==36
    assert result.ledger['batches'][0]['generated']==35
    assert result.physical_requests==result.auxiliary_evaluations==0
    for b in result.structures:
        np.testing.assert_array_equal(b.cell,a.cell)
        assert sorted(b.numbers)==sorted(a.numbers)
    assert all(o.get('energy_is_inherited_not_evaluated',True) for o in result.origins)


def test_type3_optional_auxiliary_branch_sorts_then_caps_and_counts_every_run():
    from test_molecular_auxiliary import water,SIGMA
    a,groups=water()
    result=initialize_type3([a,a,a],[0.,.1,.2],groups,[0]*len(groups),count=1,rng=np.random.default_rng(19),
        bond_limits={},max_cut_attempts=100,max_pair_attempts=100,lj_monomer_optimization=True,
        lj_pair_sigma=SIGMA,lj_max_evaluations=3)
    assert len(result.ledger['auxiliary_runs'])==7 and result.auxiliary_evaluations==21
    energies=[item['result'].energy_aux for item in result.ledger['auxiliary_runs']]
    selected=result.ledger['selected_indices'][0]
    best=next(x for x in result.ledger['auxiliary_runs'] if x['raw_index']==selected)
    assert best['result'].energy_aux==min(energies)
    assert not result.origins[0]['certified'] and result.physical_requests==0


def test_type0_explicit_regular_families_match_source_length_formulas():
    from pamssw.standalone.initializers import initialize_type0_regular
    a=read(FIX/'type4_tio2_au24o4.extxyz');numbers=a.numbers[486:]
    radii={8:1.269578/2,79:2.574144/2};radius=np.mean([radii[z] for z in numbers])
    result=initialize_type0_regular(numbers,rng=np.random.default_rng(11),atomic_radii=radii,
        ring_sizes=[4,7],ring_multiplicity=1,cage_count=1,tangent_count=1)
    assert len(result.structures)==4 and result.physical_requests==0
    for structure in result.structures:assert sorted(structure.numbers)==sorted(numbers)
    ring=result.structures[0]
    np.testing.assert_allclose(np.linalg.norm(ring.positions[1]-ring.positions[0]),2*radius)
    np.testing.assert_allclose(ring.positions[4,2]-ring.positions[0,2],radius*np.sqrt(3))
    cage=result.structures[2]
    np.testing.assert_allclose(np.mean(np.linalg.norm(np.diff(cage.positions,axis=0),axis=1)),2*radius)
    assert all(not o['certified'] for o in result.origins)


def test_type3_rejects_noncontiguous_groups_before_auxiliary_work(monkeypatch):
    from ase.data.s22 import create_s22_system
    import pamssw.standalone.molecular_auxiliary as auxiliary
    a=create_s22_system('Water_dimer')
    # Interleave actual two water molecules, retaining valid group membership.
    a=a[[0,3,1,4,2,5]];groups=((0,2,4),(1,3,5))
    def unexpected(*args,**kw):raise AssertionError('must reject before auxiliary setup')
    monkeypatch.setattr(auxiliary,'MolecularLJChart',unexpected)
    with pytest.raises(ValueError,match='contiguous'):
        initialize_type3([a],[0.],groups,[0,0],count=1,rng=np.random.default_rng(3),bond_limits={},
            max_cut_attempts=10,max_pair_attempts=10,lj_monomer_optimization=True,lj_pair_sigma={})
