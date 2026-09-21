"""Offline request, input-identity and force-qualification audit of replays."""
import argparse,json
from pathlib import Path
import numpy as np


def analyze(root):
    plan=json.loads((root/'plan.json').read_text());rows=[];issues=[]
    for start in plan['starts']:
        for method in plan['methods']:
            name=start['name']+'-'+method['name'];folder=root/name
            if not (folder/'summary.json').exists(): continue
            row=json.loads((folder/'summary.json').read_text())
            ledger=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
            paid=[e for e in ledger if e['kind'] in ('search','search_failure')]
            if len(paid)!=row['requests'] or [e['request'] for e in paid]!=list(range(1,len(paid)+1)):
                issues.append(name+': request ledger mismatch')
            if not paid or paid[0]['atoms']!=start['atoms']:
                issues.append(name+': input mismatch')
            if row['status']=='force_qualified' and row['max_force']>plan['fmax']:
                issues.append(name+': false force qualification')
            row['original_start_equal']=bool(paid and paid[0]['atoms']==start['atoms'])
            if method['name']=='safe10':
                original=json.loads(Path(start['source']).read_text())['records'][start['record']]['landing']
                target=np.array(original['atoms']['positions'])
                errors=[float(np.max(np.abs(np.array(e['atoms']['positions'])-target))) for e in paid if 'atoms' in e]
                row['closest_original_300step_endpoint_max_coordinate_error']=min(errors) if errors else None
                original_ledger=[json.loads(line) for line in Path(start['source']).with_name('requests.jsonl').read_text().splitlines()]
                original_paid=[e for e in original_ledger if e['kind']=='search']
                matches=[i for i,e in enumerate(original_paid) if e['atoms']==start['atoms']]
                row['prefix_comparison']=[]
                if matches:
                    offset=matches[-1]
                    for k in (0,1,10,50,100,200):
                        if offset+k>=len(original_paid) or k>=len(paid) or 'forces' not in paid[k]: continue
                        a,b=original_paid[offset+k],paid[k]
                        row['prefix_comparison'].append(dict(request_offset=k,
                            max_coordinate_error=float(np.max(np.abs(np.array(a['atoms']['positions'])-np.array(b['atoms']['positions'])))),
                            max_force_error=float(np.max(np.abs(np.array(a['forces'])-np.array(b['forces']))))))
            rows.append(row)
    by_method={}
    for method in plan['methods']:
        selected=[r for r in rows if r['method']['name']==method['name']]
        by_method[method['name']]=dict(denominator=len(selected),force_qualified=sum(r['status']=='force_qualified' for r in selected),
            requests=sum(r['requests'] for r in selected),statuses={status:sum(r['status']==status for r in selected) for status in sorted({r['status'] for r in selected})})
    return dict(expected=len(plan['starts'])*len(plan['methods']),finished=len(rows),issues=issues,rows=rows,by_method=by_method,
        total_search=sum(r['requests'] for r in rows),total_validation=sum(r['fresh_requests'] for r in rows),
        scope='Matched original-start developmental diagnostic, not new SSW trajectories or global optimizer ranking.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();report=analyze(a.root);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('expected','finished','issues','by_method','total_search','total_validation')},indent=2))
