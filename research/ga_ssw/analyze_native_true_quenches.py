"""Audit isolated native-kernel replay against the same frozen starts."""
import json,argparse
from pathlib import Path
import numpy as np


def analyze(root):
    plan=json.loads((root/'plan.json').read_text());rows=[];issues=[]
    previous=Path(plan['parent_protocol']).parent
    for start in plan['starts']:
        folder=root/start['name']
        if not (folder/'summary.json').exists():continue
        r=json.loads((folder/'summary.json').read_text())
        ledger=[json.loads(l) for l in (folder/'requests.jsonl').read_text().splitlines()]
        paid=[e for e in ledger if e['kind'] in ('search','search_failure')]
        if len(paid)!=r['requests'] or [e['request'] for e in paid]!=list(range(1,len(paid)+1)):issues.append(start['name']+': request mismatch')
        if not paid or paid[0]['atoms']!=start['atoms']:issues.append(start['name']+': start mismatch')
        old=json.loads((previous/(start['name']+'-safe10')/'requests.jsonl').open().readline())
        r['initial_force_difference_from_safe10']=float(np.max(np.abs(np.array(old['forces'])-np.array(paid[0]['forces']))))
        r['initial_energy_difference_from_safe10']=paid[0]['energy']-old['energy']
        accepted=json.loads((folder/'accepted.json').read_text());lookup={e['request']:e for e in paid if e['kind']=='search'}
        states=[paid[0]]+[lookup[e['request']] for e in accepted]
        curvature=[]
        for a,b in zip(states,states[1:]):
            s=np.array(b['atoms']['positions'])-np.array(a['atoms']['positions'])
            y=np.array(a['forces'])-np.array(b['forces'])
            curvature.append(float(np.sum(s*y)))
        r['nonpositive_accepted_secants']=sum(c<=0 for c in curvature)
        r['secant_count']=len(curvature)
        r['minimum_secant_curvature']=min(curvature) if curvature else None
        r['maximum_accepted_coordinate_error']=max((e['accepted_coordinate_error'] for e in accepted),default=0)
        if r['maximum_accepted_coordinate_error']>1e-12:issues.append(start['name']+': accepted coordinate mismatch')
        if r['status']=='force_qualified' and not r['fresh']['qualified']:issues.append(start['name']+': force qualification mismatch')
        rows.append(r)
    return dict(expected=len(plan['starts']),finished=len(rows),force_qualified=sum(r['status']=='force_qualified' for r in rows),
        total_search=sum(r['requests'] for r in rows),total_fresh=sum(r['fresh_requests'] for r in rows),issues=issues,rows=rows,
        scope='Original isolated LBFGS/MCSRCH/MCSTEP with consistent physical gradient; not complete LASP driver or implementation speed comparison.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=analyze(a.root);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('rows','scope')},indent=2))
