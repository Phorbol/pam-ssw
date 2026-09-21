"""Reconstruct full modified forces at failed biased-quench endpoints."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.calculators.emt import EMT
from pamssw.standalone import ASESurface
from pamssw.standalone.gaussian import ProjectedGaussian
from pamssw.standalone.cluster_frame import ClusterFrame


def main():
    base=Path('research/ga_ssw/evidence/cu13-direction-only');rows=[]
    for source in sorted(base.glob('[0-9]*-*.json')):
        d=json.loads(source.read_text())
        for record in d['result']['records']:
            if record['status']!='biased_quench_failed':continue
            a=Atoms(**record['last_atoms']);s=ASESurface(EMT());e,f=s.evaluate(a)
            for g in record['climb']:
                de,df=ProjectedGaussian(np.array(g['center']),np.array(g['direction']),g['width'],g['weight']).evaluate(a)
                e+=de;f+=df
            internal=ClusterFrame(a).project(f);rigid=f-internal
            rows.append(dict(source=source.name,step=record['index'],gaussian_count=len(record['climb']),
                recorded_max_force=record['climb'][-1]['max_force'],max_force=float(np.linalg.norm(f,axis=1).max()),
                internal_max_force=float(np.linalg.norm(internal,axis=1).max()),
                rigid_squared_force_fraction=float(np.sum(rigid**2)/np.sum(f**2)),
                quench_requests=record['climb'][-1]['quench_requests'],validation_requests=s.requests))
    result=dict(scope='post hoc instantaneous force decomposition; not causal work decomposition; no failed structure is added to successful minima',records=rows,
        summary=dict(count=len(rows),median_rigid_squared_force_fraction=float(np.median([r['rigid_squared_force_fraction'] for r in rows])),
            max_force_range=[min(r['max_force'] for r in rows),max(r['max_force'] for r in rows)],
            internal_max_force_range=[min(r['internal_max_force'] for r in rows),max(r['internal_max_force'] for r in rows)],
            full_force_reconstruction_error=max(abs(r['max_force']-r['recorded_max_force']) for r in rows),requests=sum(r['validation_requests'] for r in rows)))
    (base/'failure-force-diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary'],indent=2))
if __name__=='__main__':main()
