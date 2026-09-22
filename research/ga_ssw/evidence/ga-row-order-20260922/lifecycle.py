"""Bounded engineering lifecycle, no search-efficiency or default promotion claim."""
import json
import runpy
from dataclasses import replace
from pathlib import Path
from ase.cluster.octahedron import Octahedron
from ase.io import write

fixture = runpy.run_path('tests/standalone/test_ga_row_order.py')
output = Path(__file__).parent
records = []
for element in ('Cu', 'Al'):
    for mode in ('legacy_counts', 'full_fingerprint'):
        kwargs = fixture['inputs'](element)
        kwargs['initial'].append(Octahedron(element, 3, cutoff=1))
        kwargs['groups'] = None
        kwargs['config'] = replace(kwargs['config'], proposal_type=0, generations=1)
        kwargs['proposal_bond_limits'] = kwargs['descriptor_bonds']
        result = fixture['run'](kwargs, descriptor_row_order=mode)
        name = f'{element}13-{mode}'
        write(output / f'{name}-initial.extxyz', kwargs['initial'])
        write(output / f'{name}-observations.extxyz', [o.result.atoms for o in result.observations])
        records.append(dict(name=name, status=result.status, requests=result.evaluation_requests,
            stages=[dict(phase=s.phase, requests=s.evaluation_requests) for s in result.stages],
            failures=[dict(phase=f.phase, reason=f.reason) for f in result.failures],
            observations=[dict(phase=o.phase, eligible=o.eligible_for_archive, energy=o.result.energy,
                max_force=o.result.max_force, projection=o.projection, parent_ids=o.parent_ids,
                operator=o.operator) for o in result.observations],
            archive=[dict(energy=a['energy'], sims=a['sims']) for a in result.archive]))
(output / 'lifecycle.json').write_text(json.dumps(records, indent=2))
