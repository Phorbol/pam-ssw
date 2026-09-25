"""Cost-aligned readout of the frozen two direction arms; no PES calls."""
import json
from pathlib import Path
import numpy as np
from ase.io import read
HERE=Path(__file__).resolve().parent

def load(root):
    summary=json.loads((root/'summary.json').read_text())
    result={}
    for row in summary['trajectories']:
        key=(row['n'],row['seed'])
        folder=root/f'lj{key[0]}-seed{key[1]}'
        log=folder/'outer-steps.jsonl'
        steps=[json.loads(s) for s in log.read_text().splitlines()] if log.exists() else []
        completed=[s for s in steps if isinstance(s.get('step'),int) and s['step']>=0]
        costs=[s.get('cumulative_requests',0) for s in steps]
        prefix_cost=max(costs,default=0)
        result[key]=(row,steps,dict(status=row['status'],search_requests=row['search_requests'],
            complete_outer_callbacks=len(completed),accepted=sum(s.get('accepted',False) for s in completed),
            requests_after_last_logged_boundary=row['search_requests']-prefix_cost,
            best_energy_eV=row.get('best_energy_eV'),first_candidate=row.get('first_hit'),
            fresh_checks=row.get('fresh_checks',[])))
    return result

def best_at(steps,budget):
    observations=[m['energy_eV'] for s in steps if s.get('cumulative_requests',float('inf'))<=budget
                  for m in s.get('new_minima',[]) if m['converged']]
    return min(observations,default=None)

def main():
    a,b=load(HERE/'runs'),load(HERE/'paper-direction-runs')
    pairs=[]
    for key in sorted(set(a)|set(b)):
        if key not in a or key not in b:
            pairs.append(dict(n=key[0],seed=key[1],status='missing_arm'));continue
        ra,sa,da=a[key];rb,sb,db=b[key]
        common=min(ra['search_requests'],rb['search_requests'])
        name=f'lj{key[0]}-seed{key[1]}'
        pa,pb=HERE/'runs'/name/'initial.extxyz',HERE/'paper-direction-runs'/name/'initial.extxyz'
        same=None
        if pa.exists() and pb.exists(): same=bool(np.array_equal(read(pa).positions,read(pb).positions))
        pairs.append(dict(n=key[0],seed=key[1],identical_initial_coordinates=same,
            common_search_budget=common,global_arm=da,paper_arm=db,
            global_best_at_common_budget=best_at(sa,common),paper_best_at_common_budget=best_at(sb,common)))
    output=dict(scope='Development panel, not published success-rate replication. Energy candidates require geometry-analysis.json. Prefix energies use completed qualified observations only; all partial costs remain charged.',pairs=pairs)
    (HERE/'pilot-comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
if __name__=='__main__': main()
