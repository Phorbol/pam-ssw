"""Zero-PES input/descriptor qualification for a frozen comparison protocol."""
import json
from pathlib import Path
import numpy as np
from ase.io import read
from pamssw.standalone.paper_ga import PaperGAConfig
from pamssw.standalone.paper_reference import SSWConfig
from pamssw.standalone.legacy_descriptor import cluster_descriptor, descriptor_similarity, _full_fingerprint_order

out=Path(__file__).parent
results={}
for system in ('c60','cu13'):
    plan=json.loads((out/f'{system}-seed3.json').read_text())
    atoms=[read(path) for path in plan['inputs']]
    PaperGAConfig(**plan['ga']); SSWConfig(**plan['ssw'])
    assert all(not a.pbc.any() and not a.constraints and np.isfinite(a.positions).all() for a in atoms)
    assert all(np.array_equal(a.numbers,atoms[0].numbers) for a in atoms)
    spec=plan['descriptor']; bonds={(a,b):v for a,b,v in spec['bond_lengths']}
    descriptors=[_full_fingerprint_order(cluster_descriptor(a.numbers,a.positions,bonds,spec['neighbor_range'])) for a in atoms]
    projections=np.array([[descriptor_similarity(d,r,spec['weights']) for r in descriptors[:3]] for d in descriptors])
    assert np.isfinite(projections).all()
    distinct=[]
    for p in projections:
        if not any(np.all(np.abs(p-q)<=plan['ga']['projection_tolerance']) for q in distinct):distinct.append(p)
    results[system]=dict(ninputs=len(atoms),projections=projections.tolist(),distinct_raw_projections=len(distinct),
        raw_projection_gate=len(distinct)>=3,
        interpretation='Raw geometry/reference separation only. Post-quench diversity and GA parent eligibility not yet known.')
    if system=='c60':
        assert Path(plan['backend']['model']).is_file()
        reference=read(plan['reference_geometry'])
        assert len(reference)==60 and np.all(reference.numbers==6)
        from research.ga_ssw.analyze_c60_random_development import graph_row
        results[system]['reference_graph']=graph_row(reference.numbers,reference.positions,1.8)
        assert results[system]['reference_graph']['graph_cage_candidate']
(out/'preflight.json').write_text(json.dumps(results,indent=2)+'\n')
assert all(r['raw_projection_gate'] for r in results.values()),results
print(json.dumps(results,indent=2))
