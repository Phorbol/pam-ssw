from pathlib import Path
import numpy as np
import pytest
from ase.io import read
from pamssw.standalone.packing import pack_type0,source_cage_occupancy,source_cubic_boxes

FIX=Path(__file__).parent/'fixtures'
RADII={8:1.269578/2,79:2.574144/2}


def composition():return read(FIX/'type4_tio2_au24o4.extxyz').numbers[486:]


@pytest.mark.parametrize('family,extra',[
    ('unlimited',{}),('simple_cubic',dict(box=(3,3,4))),('irregular_ball',{}),
    ('irregular_ball_ori',dict(occupancy=.5)),
    ('irregular_cage',dict(occupancy=source_cage_occupancy(28)))])
def test_actual_au24o4_composition_all_source_families(family,extra):
    numbers=composition();r=pack_type0(numbers,family,np.random.default_rng(7),atomic_radii=RADII,max_attempts=10000,**extra)
    if family in ('irregular_ball_ori','irregular_cage'):
        # Retain native-density failure at this bounded seed/cap; don't tune it away.
        assert r.status=='insertion_budget_exhausted' and len(r.atoms)<len(numbers)
        assert r.details['total_attempts']>=10000
        sparse=pack_type0(numbers,family,np.random.default_rng(7),atomic_radii=RADII,max_attempts=10000,occupancy=.2)
        assert sparse.status=='completed'
        r=sparse
    else:assert r.status=='completed',r.details
    assert sorted(r.atoms.numbers)==sorted(numbers)
    assert np.isfinite(r.atoms.positions).all() and r.physical_requests==0
    for index,origin in enumerate(r.source_atom_indices):assert r.atoms.numbers[index]==numbers[origin]
    if family=='unlimited':
        radius=np.array([RADII[z] for z in r.atoms.numbers]);i,j=np.triu_indices(len(numbers),1)
        assert np.all(r.atoms.get_all_distances()[i,j]>=radius[i]+radius[j]-1e-12)


def test_source_custom_template_and_172_atom_regular_shell_are_explicit():
    a=read(FIX/'type4_tio2_au24o4.extxyz')[486:]
    r=pack_type0(a.numbers,'custom_template',np.random.default_rng(2),atomic_radii=RADII,
        max_attempts=1,template=a,template_atomic_radii=RADII)
    np.testing.assert_allclose(r.atoms.positions,a.positions)
    b=read(FIX/'type2_xxxii.extxyz');radii={1:1.45548/2,6:1.278556/2,7:1.133232/2,8:1.269578/2,17:2.54154/2}
    result=pack_type0(b.numbers,'irregular_ball',np.random.default_rng(4),atomic_radii=radii,max_attempts=1)
    assert result.status=='completed' and len(result.atoms)==172
    assert sorted(result.atoms.numbers)==sorted(b.numbers)


def test_single_point_shell_and_failure_budget_are_not_fake_complete_outputs():
    # Native golden-angle denominator n-1 is zero for a one-atom final shell.
    r=pack_type0(np.array([79,79]),'irregular_ball',np.random.default_rng(4),atomic_radii=RADII,max_attempts=1)
    assert r.status=='completed' and np.isfinite(r.atoms.positions).all()
    assert r.details['corrections']
    r=pack_type0(composition(),'irregular_cage',np.random.default_rng(3),atomic_radii=RADII,
        max_attempts=1,occupancy=source_cage_occupancy(28))
    assert r.status=='insertion_budget_exhausted' and len(r.atoms)<28
    assert (1,1,2) in source_cubic_boxes(2)
    assert all(np.prod(box)>=28 for box in source_cubic_boxes(28))
