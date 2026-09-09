"""Real input structures, original-JAR oracle; these are behavior tests, not GO validation."""
import json
from pathlib import Path
import numpy as np
import pytest

from research.ga_ssw.behavior import cluster_descriptor, descriptor_similarity

FIXTURES=Path(__file__).parents[1]/'fixtures/ga_ssw'

@pytest.mark.parametrize('case',['water','aloh','molecular_crystal'])
def test_similarity_matches_original_nna_on_real_structures(case):
    data=json.loads((FIXTURES/(case+'.json')).read_text())
    frames=data['frames']
    got=[[descriptor_similarity(a['descriptor'],b['descriptor'],[.3,.2,.2,.1,.1,.1]) for b in frames] for a in frames]
    np.testing.assert_allclose(got,data['similarity_matrix'],rtol=0,atol=2e-13)


def test_cluster_descriptor_matches_original_on_water_and_rigid_transforms():
    data=json.loads((FIXTURES/'water.json').read_text())
    bonds={(int(a),int(b)):v for a,b,v in data['bond_lengths']}
    for frame in data['frames']:
        got=cluster_descriptor(frame['numbers'],frame['positions'],bonds,data['neighbor_range'])
        for key,expected in frame['descriptor'].items():
            np.testing.assert_allclose(got[key],expected,rtol=0,atol=2e-12,err_msg=frame['label']+':'+key)


def test_archive_and_parent_weights_match_original_sgn():
    from research.ga_ssw.behavior import same_projection, remove_duplicates, merge_archive, energy_window, compete_cumulative
    data=json.loads((FIXTURES/'core-input.json').read_text())
    expected=json.loads((FIXTURES/'core-output.json').read_text())
    rows=data['rows'];tol=data['tolerance']
    assert [[same_projection(a,b,tol) for b in rows] for a in rows]==expected['same']
    assert [r['id'] for r in remove_duplicates(rows,tol)]==expected['dedup']
    assert [r['id'] for r in merge_archive(rows[:2],rows[2:],tol)]==expected['merged']
    assert [r['id'] for r in energy_window(rows,data['window'])]==expected['window']
    np.testing.assert_allclose(compete_cumulative([r['energy'] for r in rows]),expected['compete_cumulative'],atol=2e-14,rtol=0)


def test_frozen_virtual_projection_matches_original_and_exposes_order_dependence():
    from research.ga_ssw.behavior import same_projection, remove_duplicates
    data=json.loads((FIXTURES/'water.json').read_text())
    expected=json.loads((FIXTURES/'water-classification.json').read_text())
    projections=[[descriptor_similarity(f['descriptor'],ref,[.3,.2,.2,.1,.1,.1])
                  for ref in data['reference_descriptors']] for f in data['frames']]
    np.testing.assert_allclose(projections,data['projections'],rtol=0,atol=2e-13)
    rows=[dict(id=i,energy=f['energy'],sims=p) for i,(f,p) in enumerate(zip(data['frames'],projections))]
    assert same_projection(rows[0],rows[1],.0001)==expected['original_reverse_same']==False
    assert same_projection(rows[0],rows[2],.0001)==expected['original_rigid_same']==True
    assert len(remove_duplicates(rows,.0001))==expected['dedup_count']==2


def test_replacement_changes_the_representative_used_by_later_comparisons():
    from research.ga_ssw.behavior import remove_duplicates, same_projection
    data=json.loads((FIXTURES/'core-input.json').read_text()); rows=data['rows']; tol=data['tolerance']
    expected=json.loads((FIXTURES/'core-output.json').read_text())
    assert same_projection(rows[0],rows[8],tol)
    assert same_projection(rows[8],rows[9],tol)
    assert not same_projection(rows[0],rows[9],tol)
    ids=[r['id'] for r in remove_duplicates(rows,tol)]
    assert ids==expected['dedup']
    assert 8 in ids and 0 not in ids and 9 not in ids
