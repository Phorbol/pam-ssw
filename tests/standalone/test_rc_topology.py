from pathlib import Path
import pytest
from pamssw.standalone.rc_topology import read_rigid_topology


def test_si_topology_retains_shared_endpoints_and_comment_format(tmp_path):
    # RC SI section7, phenylacetaldehyde: memberships use one-based atom indices.
    rb=tmp_path/'rigidbody';bl=tmp_path/'blist'
    rb.write_text('3 # groups\n12\n1 2 3 4 5 6 7 8 9 10 11 12\n5\n4 12 13 14 15\n4\n12 15 16 17\n')
    pairs=[(1,2),(2,3),(3,4),(4,5),(5,6),(6,1),(1,7),(2,8),(3,9),(5,10),(6,11),(4,12),(12,13),(12,14),(12,15),(15,16),(15,17)]
    bl.write_text(''.join(f'{a} {b}\n' for a,b in pairs))
    r=read_rigid_topology(rb,bl,natoms=17)
    assert len(r.components)==1
    c=r.components[0]
    assert c['parents']==(-1,0,1) and c['joints']==(None,(3,11),(11,14))
    assert c['source_body_indices']==(0,1,2)
    assert len(r.bonds)==17
    bl.write_text('1 2\n')
    with pytest.raises(ValueError,match='joint'):read_rigid_topology(rb,bl,natoms=17)


def test_uploaded_xxxii_four_chains():
    folder=Path('/home/gengjianrui/bin/pam-ssw-research/ga-ssw-20260909/GA-SSW_examples_run/global_exploration/input-templates/TYPE2-XXXII/mc')
    if not folder.exists():pytest.skip('uploaded example not available')
    r=read_rigid_topology(folder/'rigidbody',folder/'blist',natoms=172)
    assert len(r.components)==4 and len(r.bonds)==180
    assert [len(c['bodies']) for c in r.components]==[8]*4
    assert [len(set(i for g in c['bodies'] for i in g)) for c in r.components]==[43]*4
