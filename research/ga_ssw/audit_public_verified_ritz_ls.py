"""Offline final accounting of public verified Ritz SSW/LS integration."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def main(root):
    manifest=json.loads((root/'source-manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==digest
    initial=json.loads((root/'initial.json').read_text())
    rows=[]
    for arm in ('ssw','paper_ls','native_ls'):
        for seed in (11,29):
            folder=root/f'butadiene-{arm}-seed{seed}'
            result=json.loads((folder/'result.json').read_text())
            summary=json.loads((folder/'summary.json').read_text())
            ledger=[json.loads(s) for s in (folder/'evaluations.jsonl').read_text().splitlines()]
            paid=[x for x in ledger if x['kind'] in ('search','search_failure')]
            assert [x['request'] for x in paid]==list(range(1,summary['search_requests']+1))
            assert result['evaluation_requests']==len(paid)==result['initial']['evaluation_requests']+sum(r['evaluation_requests'] for r in result['records'])
            fresh=json.loads((folder/'fresh-checks.json').read_text())
            assert len(fresh)==summary['fresh_requests']==len(result['minima'])
            assert all(c.get('force_qualified') and abs(c['energy_error'])<1e-9 for c in fresh)
            assert all(m['atoms']['cell']==initial['cell'] and m['atoms']['pbc']==initial['pbc'] for m in result['minima'])
            rotations=[c for r in result['records'] for c in r['climb'] if 'rotation_force_requests' in c]
            assert all(c['rotation_force_requests']<=101 for c in rotations)
            assert all(c['rotation_force_requests']==c.get('pre_rotation',{}).get('force_calls',0)+c['main_rotation']['force_calls'] for c in rotations if c.get('main_rotation'))
            counts=Counter(r['status'] for r in result['records'])
            responses=[r['energy_response'] for r in result['records'] if r.get('energy_response') is not None]
            rows.append(dict(arm=arm,seed=seed,search=len(paid),fresh=len(fresh),boundary=summary['boundary'],backend_failures=sum(x['kind']=='search_failure' for x in ledger),denials=sum(x['kind']=='search_denial' for x in ledger),record_statuses=dict(counts),ls_responses=len(responses),last_response=responses[-1] if responses else None,best_delta=min(m['energy'] for m in result['minima'])-result['initial']['energy'],max_rotation_cost=max(c['rotation_force_requests'] for c in rotations)))
    audit=dict(rows=rows,search_requests=sum(r['search'] for r in rows),fresh_requests=sum(r['fresh'] for r in rows),source_hashes_checked=True,all_fresh_qualified=True,all_accounted=True)
    (root/'root-final-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();main(args.root)
