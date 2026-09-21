"""Read-only source-result comparison and saved E/F accounting; no calculators."""
from pathlib import Path
import json
import numpy as np


def read(p):return json.loads(p.read_text())
def ledger(p):return [json.loads(s) for s in p.read_text().splitlines()]

def audit(root):
    baseline=root/'constrained-gaussian-reference-20260912'
    pam=root/'constrained-gaussian-pam-20260912'
    previous=root/'hookean-multicase-20260912'
    a,b=read(baseline/'plan.json'),read(pam/'plan.json')
    for field in ('seed','steps','inputs','ls_settings','config','search_cap','fresh_cap'):
        assert a[field]==b[field],field
    rows=[]
    for case in a['inputs']:
      for variant in a['variants']:
       for tag,directory in [('reference',baseline),('pam',pam)]:
        arm=directory/f'{case}-{variant}';s=read(arm/'summary.json');r=read(arm/'result.json')
        calls=ledger(arm/'evaluations.jsonl')
        assert s['status']==r['status'] and s['ledger_consistent']
        assert r['requests']==s['search_requests']==len(calls)==sum(x['requests'] for x in r['records'])
        assert [x['request'] for x in calls]==list(range(1,len(calls)+1))
        assert len(s['checks'])==len(r['minima'])==s['fresh_requests']
        assert all(x['qualified'] for x in s['checks'])
        for m,c in zip(r['minima'],s['checks']):
            assert abs(m['energy']-c['energy'])<1e-7
            assert c['active_fmax']<=a['config']['fmax']
        old_equal=None
        if tag=='reference':
            old_equal=calls==ledger(previous/f'{case}-{variant}'/'evaluations.jsonl')
            assert old_equal, 'default trajectory changed'
        outer=[x for x in r['records'] if 'index' in x];assert len(outer)==1
        widths=[x['width'] for x in outer[0].get('frozen_gaussians',[])]
        best_delta=min(m['energy'] for m in r['minima'])-r['initial']['energy']
        rows.append(dict(case=case,variant=variant,policy=tag,status=s['status'],
            outer_status=outer[0]['status'],search=s['search_requests'],fresh=s['fresh_requests'],
            qualified=len(s['checks']),widths=widths,best_objective_delta=best_delta,
            default_full_ledger_matches_previous=old_equal))
    report=dict(rows=rows,search=sum(x['search'] for x in rows),fresh=sum(x['fresh'] for x in rows),
                qualified=sum(x['qualified'] for x in rows),claims='short lifecycle and default regression only; no efficiency or distinct-basin claim')
    return report

if __name__=='__main__':
    import sys
    root=Path(sys.argv[1]);r=audit(root)
    (root/'constrained-gaussian-comparison-20260912.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
