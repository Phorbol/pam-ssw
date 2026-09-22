"""Reuse archived Cu13 EMT full-loop protocol; only descriptor order changes."""
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from ase.calculators.emt import EMT
from ase.io import read, write
from pamssw.standalone import ASESurface, SSWConfig, PaperGAConfig, run_ga_ssw
from pamssw.standalone.legacy_descriptor import cluster_descriptor

base=Path('research/ga_ssw/evidence/atomic-cu13-ga')
out=Path(__file__).parent
plan=json.loads((base/'plan.json').read_text())
initial=read(base/'initial.extxyz', ':')
bonds={(int(a),int(b)):v for a,b,v in plan['bond_lengths']}
refs=[cluster_descriptor(a.numbers,a.positions,bonds,plan['neighbor_range']) for a in initial]
kwargs=dict(initial=initial,groups=None,references=refs,descriptor_bonds=bonds,
    descriptor_weights=plan['descriptor_weights'],neighbor_range=plan['neighbor_range'],
    proposal_bond_limits={},config=PaperGAConfig(**plan['ga']),ssw_config=SSWConfig(**plan['ssw']),
    max_evaluations=4000)
(out/'protocol.json').write_text(json.dumps(dict(parent_protocol=str(base/'plan.json'),
    input=str(base/'initial.extxyz'),seed=3,max_evaluations_per_mode=4000,
    modes=['legacy_counts','full_fingerprint'],ga=plan['ga'],ssw=plan['ssw'],
    stop='4000 E/F per arm or five-minute CPU job; no retries or tuning',
    purpose='GA offspring integration and independent force validation, not efficiency ranking'),indent=2))
for mode in ('legacy_counts','full_fingerprint'):
    surface=ASESurface(EMT());start=time.monotonic()
    result=run_ga_ssw(**kwargs,surface=surface,rng=np.random.default_rng(3),descriptor_row_order=mode)
    (out/f'{mode}.pkl').write_bytes(pickle.dumps(result, protocol=4))
    checks=[]
    for o in result.observations:
        e,f=ASESurface(EMT()).evaluate(o.result.atoms)
        checks.append(dict(id=o.id,phase=o.phase,eligible=o.eligible_for_archive,
            energy=e,max_force=float(np.linalg.norm(f,axis=1).max()),
            force_pass=bool(np.linalg.norm(f,axis=1).max() <= kwargs['config'].quench_fmax),
            composition_pass=bool(len(o.result.atoms)==13 and np.all(o.result.atoms.numbers==29)),
            projection=o.projection,parent_ids=o.parent_ids,operator=o.operator))
    row=dict(mode=mode,status=result.status,requests=result.evaluation_requests,
        fresh_requests=len(checks),seconds=time.monotonic()-start,observations=checks,
        stages=[dict(phase=s.phase, status=s.status, requests=s.evaluation_requests) for s in result.stages],failures=[asdict(f) for f in result.failures],
        archive=[dict(energy=a['energy'],sims=a['sims']) for a in result.archive])
    (out/f'{mode}.json').write_text(json.dumps(row,indent=2))
    write(out/f'{mode}.extxyz',[o.result.atoms for o in result.observations])
