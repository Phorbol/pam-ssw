from pathlib import Path
import numpy as np
import pytest
from ase import Atoms
from ase.io import read
from ase.constraints import FixAtoms
from pamssw.standalone.surface_ga import (reload_surface,disturb_surface,cross_surface,
    surface_collision_free,surface_topology,suit_orientation)


def fixture():
    return read(Path(__file__).parent/'fixtures/type4_tio2_au24o4.extxyz')


def assert_frozen(a,c):
    np.testing.assert_array_equal(a.positions[:486],c.atoms.positions[:486])
    np.testing.assert_array_equal(a.numbers[:486],c.atoms.numbers[:486])
    np.testing.assert_array_equal(a.cell,c.atoms.cell)
    assert not c.atoms.pbc[2]
    assert set(c.atoms.constraints[0].get_indices())==set(range(486))
    assert c.atoms.calc is None


def test_actual_supported_cluster_reload_and_disturbance():
    a=fixture();xyz=a.positions.copy();group=range(486,514)
    c=reload_surface(a,range(486),group,np.random.default_rng(8),site_fractional=(.5,.5),accuracy=10)
    assert_frozen(a,c)
    d=lambda x:np.linalg.norm(x[:,None]-x[None,:],axis=2)
    np.testing.assert_allclose(d(a.positions[486:]),d(c.atoms.positions[486:]),atol=2e-12)
    e=disturb_surface(a,range(486),group,np.random.default_rng(6));assert_frozen(a,e)
    assert len(e.details['moved_adsorbate_indices'])==5
    assert sorted(e.atoms.numbers[486:])==sorted(a.numbers[486:])
    for i in group:assert e.atoms.numbers[i]==a.numbers[e.source_atom_indices[i]]
    np.testing.assert_array_equal(a.positions,xyz)


def test_cross_actual_support_composition_and_lineage():
    a=fixture()
    children=cross_surface([a,a,a],[0.,1.,2.],range(486),range(486,514),np.random.default_rng(42),
        n=1,site_fractional=(.5,.5),max_cut_attempts=100,max_pair_attempts=100,cuts_per_parent_slot=2)
    assert len(children)==1;c=children[0];assert_frozen(a,c)
    assert sorted(c.atoms.numbers)==sorted(a.numbers)
    for i in range(514):assert c.atoms.numbers[i]==a.numbers[c.source_atom_indices[i]]
    assert c.details['orientation_accuracy'] is None


def test_partial_pbc_filter_and_topology():
    # A vacuum image would overlap, but there is no physical z periodicity.
    a=Atoms('Cu3',positions=[[0,0,0],[2,0,0],[0,2,0]],cell=[10,10,.1],pbc=[True,True,False])
    assert surface_collision_free(a,{('Cu','Cu'):1.})
    a.cell[0,0]=.5
    assert not surface_collision_free(a,{('Cu','Cu'):1.}) # periodic self image
    a=fixture();a.set_constraint(FixAtoms(indices=[0]))
    with pytest.raises(ValueError,match='exactly match'):surface_topology(a,range(486),range(486,514))
    a.set_constraint();a.pbc=True
    with pytest.raises(ValueError,match='PBC'):surface_topology(a,range(486),range(486,514))


def test_suit_orientation_retains_pair_geometry():
    x=fixture().positions[486:];y,info=suit_orientation(x,10)
    assert info['orientation_count']==60
    np.testing.assert_allclose(np.linalg.norm(x[:,None]-x[None,:],axis=2),np.linalg.norm(y[:,None]-y[None,:],axis=2),atol=1e-12)


def test_reload_contact_and_frame_covariance():
    from ase.geometry import find_mic
    from scipy.spatial.transform import Rotation
    a=fixture();kwargs=dict(site_fractional=(.5,.5),accuracy=10)
    c=reload_surface(a,range(486),range(486,514),np.random.default_rng(9),**kwargs)
    delta=c.atoms.positions[:486,None,:]-c.atoms.positions[None,486:,:]
    _,dist=find_mic(delta.reshape(-1,3),a.cell,pbc=a.pbc)
    assert dist.min()>=2.-1e-12
    assert dist.min()<2.125+1e-12
    rot=Rotation.from_rotvec([.3,-.2,.7]).as_matrix();b=a.copy()
    b.positions=b.positions@rot.T;b.cell=b.cell.array@rot.T
    d=reload_surface(b,range(486),range(486,514),np.random.default_rng(9),**kwargs)
    np.testing.assert_allclose(d.atoms.positions,c.atoms.positions@rot.T,atol=2e-12)


# Actual ElementPara.getAtomR divides these tabulated diameters by two.
RADII={8:1.269578/2,79:2.574144/2}


def test_reconstruction_and_all_four_rebuild_families_actual_514():
    from pamssw.standalone.surface_ga import reconstruct_surface,rebuild_surface
    a=fixture();kwargs=dict(atomic_radii=RADII,max_face_attempts=1000)
    reconstructed=reconstruct_surface(a,range(486),range(486,514),np.random.default_rng(4),**kwargs)
    assert len(reconstructed)==2
    for c in reconstructed:
        assert_frozen(a,c);assert sorted(c.atoms.numbers)==sorted(a.numbers)
        lower=c.details['lower_indices']
        np.testing.assert_array_equal(c.atoms.positions[list(lower)],a.positions[list(lower)])
        for i in range(514):assert c.atoms.numbers[i]==a.numbers[c.source_atom_indices[i]]
    rebuilt=rebuild_surface(a,range(486),range(486,514),np.random.default_rng(5),
        site_fractional=(.5,.5),max_insertion_attempts=10000,**kwargs)
    assert len(rebuilt)==4
    assert [c.details['family'] for c in rebuilt]==['tangent','cubic','cubic','cubic']
    assert [c.details['space'] for c in rebuilt[1:]]==[(1.,1.,1.),(2.,2.,1.),(3.,3.,1.)]
    for c in rebuilt:
        assert_frozen(a,c);assert sorted(c.atoms.numbers)==sorted(a.numbers)
        for i in range(514):assert c.atoms.numbers[i]==a.numbers[c.source_atom_indices[i]]
    for c in rebuilt[1:]:
        assert len(c.details['auxiliary_runs'])==10
        assert 10<=c.details['auxiliary_evaluations']<=1000
        assert all(r['auxiliary_evaluations']<=100 for r in c.details['auxiliary_runs'])


def test_auxiliary_wall_energy_gradient_consistency():
    from pamssw.standalone.surface_ga import cubic_auxiliary_energy_gradient
    # Samples both walls; source's upper-wall gradient sign was wrong.
    x=np.array([[.13,.4,.6],[2.82,1.1,1.8]])
    numbers=[8,79];box=np.array([3.,3.,3.]);rng=np.random.default_rng(2)
    direction=rng.normal(size=x.shape);h=1e-6
    energy,g=cubic_auxiliary_energy_gradient(x,numbers,box,RADII)
    plus=cubic_auxiliary_energy_gradient(x+h*direction,numbers,box,RADII)[0]
    minus=cubic_auxiliary_energy_gradient(x-h*direction,numbers,box,RADII)[0]
    assert np.isfinite(energy)
    np.testing.assert_allclose((plus-minus)/(2*h),np.sum(g*direction),rtol=1e-7)


def test_complete_native_batch_counts_with_explicit_auxiliary_budget():
    from pamssw.standalone.surface_ga import propose_type4
    a=fixture();symbols=set(a.get_chemical_symbols())
    # Permissive thresholds test batch accounting, not physical validity.
    limits={tuple(sorted((x,y))):.001 for x in symbols for y in symbols}
    proposal=propose_type4([a,a,a],[0.,1.,2.],range(486),range(486,514),np.random.default_rng(23),
        min_ga=8,bond_limits=limits,atomic_radii=RADII,site_fractional=(.5,.5),max_batches=1,
        max_cut_attempts=100,max_pair_attempts=100,max_face_attempts=1000,
        max_insertion_attempts=10000,auxiliary_evaluations=5,cuts_per_parent_slot=2)
    assert proposal.status=='target_reached'
    batch=proposal.batches[0]
    assert batch['generated']==16 # 2 cross +2 reload +2 disturbance +2 reconstruction +8 rebuild
    assert batch['passed']+batch['rejected']==16
    assert len(proposal.candidates)==batch['passed']
    assert proposal.auxiliary_evaluations==300 # 2 rebuilds *3 cubes *10 starts *5 evals
    assert not batch['failures']
    assert all(c.details.get('substrate_parent_index',0) in (0,1,2) for c in proposal.candidates)


def test_type4_min_ga_below_family_batch_is_rejected_before_generation():
    from pamssw.standalone.surface_ga import propose_type4
    a = Atoms('Cu3', positions=[[0, 0, 0], [2, 0, 1], [0, 2, 1]],
              cell=[10, 10, 10], pbc=[True, True, False],
              constraint=FixAtoms(indices=[0]))
    with pytest.raises(ValueError, match='min_ga.*4'):
        propose_type4([a, a.copy(), a.copy()], [0., 1., 2.], [0], [1, 2],
                      np.random.default_rng(1), min_ga=1,
                      bond_limits={('Cu', 'Cu'): .1}, atomic_radii={29: 1.},
                      site_fractional=(.5, .5), max_batches=1,
                      max_cut_attempts=10, max_pair_attempts=10,
                      max_face_attempts=10, max_insertion_attempts=10)
