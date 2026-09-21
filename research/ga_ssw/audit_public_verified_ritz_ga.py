"""Offline qualification and complete stage-cost audit for public GA replay."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root):
    manifest=json.loads((root/'source-manifest.json').read_text())
    for name,digest in manifest['sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    rows=[]
    for seed in (11,29):
        folder=root/f'bicyclobutane-seed{seed}-plain'
        result=json.loads((folder/'result.json').read_text())
        summary=json.loads((folder/'summary.json').read_text())
        ledger=[json.loads(s) for s in (folder/'evaluations.jsonl').read_text().splitlines()]
        paid=[x for x in ledger if x['kind'] in ('search','search_failure')]
        assert [x['request'] for x in paid]==list(range(1,result['evaluation_requests']+1))
        assert len(paid)==sum(s['evaluation_requests'] for s in result['stages'])==summary['search_requests']
        observations={o['id']:o for o in result['observations']}
        for item in result['archive']:
            o=observations[item['observation_id']]
            assert o['eligible_for_archive'] and o['result']['converged'] and o['result']['surface']=='true'
            assert item['atoms']==o['result']['atoms'] and item['energy']==o['result']['energy']
        for walk in result['walks']:
            assert walk['evaluation_requests']==walk['initial']['evaluation_requests']+sum(r['evaluation_requests'] for r in walk['records'])
            assert all(e['rotation_force_requests']<=101 for r in walk['records'] for e in r['climb'] if 'rotation_force_requests' in e)
        fresh=json.loads((folder/'fresh-checks.json').read_text())
        assert len(fresh)==len(result['archive'])==summary['fresh_requests']
        assert {x['id'] for x in fresh}=={x['id'] for x in result['archive']}
        assert all(x.get('force_qualified') and abs(x['energy_error'])<1e-9 for x in fresh)
        phases=[dict(phase=s['phase'],status=s['status'],requests=s['evaluation_requests']) for s in result['stages']]
        rows.append(dict(seed=seed,status=result['status'],search=len(paid),fresh=len(fresh),archive=len(result['archive']),best_energy=min(x['energy'] for x in result['archive']),fragmented_archive=sum(x['molecular_components'][0]>1 for x in fresh),backend_failures=sum(x['kind']=='search_failure' for x in ledger),external_denials=sum(x['kind']=='search_denial' for x in ledger),stages=phases))
    report=dict(rows=rows,search=sum(r['search'] for r in rows),fresh=sum(r['fresh'] for r in rows),source_checked=True,stage_and_walk_costs_checked=True,all_archive_fresh_qualified=True)
    (root/'root-final-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args();audit(args.root)
