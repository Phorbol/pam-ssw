"""Offline accounting and paired-trajectory audit; no calculator calls."""
import hashlib
import json
from pathlib import Path


def audit(root):
    rows=json.loads((root/'summary.json').read_text())
    assert len(rows)==12
    manifest=json.loads((root/'source-manifest.json').read_text())
    for path,digest in manifest['sha256'].items():
        assert hashlib.sha256((root/'source'/path).read_bytes()).hexdigest()==digest
    assert hashlib.sha256((root/'verified_ritz_research.py').read_bytes()).hexdigest()==manifest['research_solver_sha256']
    checks=[]
    for row in rows:
        folder=root/f"{row['case']}-{row['arm']}-seed{row['seed']}"
        result=json.loads((folder/'result.json').read_text())
        ledger=[json.loads(s) for s in (folder/'evaluations.jsonl').read_text().splitlines()]
        paid=[x for x in ledger if x['kind'] in ('search','search_failure')]
        assert [x['request'] for x in paid]==list(range(1,6001))
        assert all(x['kind']=='search' for x in paid)
        assert len(ledger)==6001 and ledger[-1]['kind']=='search_denial'
        assert row['boundary']=='request_cap' and row['denied']==1
        assert result['evaluation_requests']==6000==result['initial']['evaluation_requests']+sum(x['evaluation_requests'] for x in result['records'])
        rotation_costs=[]
        for record in result['records']:
            for event in record['climb']:
                if 'rotation_force_requests' in event:
                    rotation_costs.append(event['rotation_force_requests'])
                    assert event['rotation_force_requests']<=101
                    if event.get('main_rotation'):
                        assert event['main_rotation']['force_calls']+event.get('pre_rotation',{}).get('force_calls',0)==event['rotation_force_requests']
        fresh=row['fresh_checks']
        assert len(fresh)==row['fresh_requests']==len(result['minima'])
        assert all(x.get('force_qualified') and x.get('cell_unchanged') and abs(x['energy_error'])<1e-9 for x in fresh)
        checks.append(dict(case=row['case'],seed=row['seed'],arm=row['arm'],search=6000,fresh=len(fresh),rotation_failures=row['record_statuses'].get('rotation_failed',0),max_rotation_cost=max(rotation_costs),best_delta=row['best_delta']))
    paired=[]
    for case in ('cu13','cu31_fixed','bicyclobutane'):
        for seed in (11,29):
            a=(root/f'{case}-original-seed{seed}/evaluations.jsonl').read_bytes()
            b=(root/f'{case}-verified-seed{seed}/evaluations.jsonl').read_bytes()
            paired.append(dict(case=case,seed=seed,ledger_byte_identical=a==b))
            if case!='bicyclobutane': assert a==b
    return dict(search_requests=sum(x['search'] for x in checks),fresh_requests=sum(x['fresh'] for x in checks),all_fresh_qualified=True,all_ledger_sequences_valid=True,all_source_hashes_valid=True,rows=checks,pairs=paired)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args()
    result=audit(a.root)
    (a.root/'root-final-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
