"""Offline audit for the staged Broyden SSW campaign; never calls a calculator."""
import argparse, hashlib, json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

def read(path): return json.loads(path.read_text())

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evidence',type=Path,required=True); ap.add_argument('--output',type=Path)
    a=ap.parse_args(); out=a.evidence.resolve(); rows=read(out/'summary.json')
    expected={(c,s,m) for c in ('cu13','cu31_fixed','bicyclobutane') for s in (11,29)
              for m in ('ritz','broyden_euclidean','broyden_native')}
    actual={(r['case'],r['seed'],r['arm']) for r in rows}
    if actual != expected: raise AssertionError(f'campaign incomplete: missing={sorted(expected-actual)} extra={sorted(actual-expected)}')
    manifest=read(out/'source-manifest.json')
    for name,digest in manifest['sha256'].items():
        p=out/'source'/name
        if hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise AssertionError(f'source hash mismatch {name}')
    inputs=read(out/'inputs.json'); sources=read(out/'input-sources.json')
    for case,src in sources.items():
        p=Path(src['path']); assert hashlib.sha256(p.read_bytes()).hexdigest()==src['sha256']
        assert read(p)['initial']['atoms']==inputs[case]
    report=[]; groups=defaultdict(lambda: {'arms':0,'search_requests':0,'fresh_requests':0,'minima':0,'landings':0,'rotation_failures':0})
    for row in rows:
        folder=out/f"{row['case']}-{row['arm']}-seed{row['seed']}"
        result=read(folder/'result.json'); ledger=[readline for readline in map(json.loads,(folder/'evaluations.jsonl').read_text().splitlines())]
        paid=[x for x in ledger if x['kind'] in ('search','search_failure')]; denials=[x for x in ledger if x['kind']=='search_denial']
        assert [x['request'] for x in paid]==list(range(1,len(paid)+1))
        assert len(paid)==row['search_requests']==result['evaluation_requests']
        assert result['initial']['evaluation_requests'] + sum(x['evaluation_requests'] for x in result['records']) == result['evaluation_requests']
        assert all('atoms' in x and 'forces' in x for x in paid if x['kind']=='search')
        if row['arm']=='ritz': assert not (folder/'broyden-traces.jsonl').exists()
        else:
            traces=(folder/'broyden-traces.jsonl').read_text().splitlines()
            assert traces, 'research arm has no Broyden trace'
            for line in traces:
                trace=json.loads(line)
                assert trace['metric'] == ('euclidean' if row['arm']=='broyden_euclidean' else 'native_block_sum')
                detail=trace['result']; assert detail['force_calls']==detail['hvp_calls']+1
                factor=read(out/'plan.json')['broyden_initial_factor']
                for event in detail['trace']:
                    if event['event']=='endpoint': np.testing.assert_allclose(event['factor1'],factor,rtol=1e-14)
                    else:
                        assert event['hvp']==1 or event['retries']==0
                        factor*=.8**event['retries']
                        np.testing.assert_allclose(event['factor1'],factor,rtol=1e-14)
        rotation_calls=[]; factor_changes=[]
        for rec in result['records']:
            for climb in rec.get('climb',[]):
                if 'rotation_force_requests' in climb:
                    rotation_calls.append(climb['rotation_force_requests']); assert climb['rotation_force_requests']<=101
                main=climb.get('main_rotation'); pre=climb.get('pre_rotation')
                if main is not None:
                    assert main['force_calls']+pre['force_calls']<=101
                    assert pre['force_calls']<=6
                    assert main['hvp_calls']<=100-pre['force_calls']
        fresh=read(folder/'fresh-checks.json'); assert len(fresh)==row['fresh_requests']
        assert len(fresh)==len(result['minima'])
        assert all(isinstance(x.get('error'),(float,int)) and x.get('qualified') is True and x['fmax']<=.01 for x in fresh)
        assert all(abs(x.get('error',0.))<=1e-8 and x.get('cell_unchanged') is True for x in fresh)
        g=groups[row['arm']]; g['arms']+=1; g['search_requests']+=row['search_requests']; g['fresh_requests']+=row['fresh_requests']; g['minima']+=len(result['minima'])
        g['landings']+=sum(1 for r in result['records'] if r.get('landing') is not None)
        g['rotation_failures']+=sum(r['status']=='rotation_failed' for r in result['records'])
        report.append({'case':row['case'],'seed':row['seed'],'arm':row['arm'],'status':row['status'],'search_requests':row['search_requests'],'fresh_requests':row['fresh_requests'],'minima':len(result['minima']),'record_statuses':dict(Counter(r['status'] for r in result['records'])),'rotation_calls':rotation_calls,'best_delta':row['best_delta'],'denials':len(denials),'best_energy':min(x['energy'] for x in result['minima']) if result['minima'] else None})
    output={'arms':report,'groups':groups,'checked':'18 complete rows; source hashes; paid/denied ledger sequences; result cost accounting; research traces; fresh force/energy/cell checks','scope':'offline accounting and staged integration audit; no scientific superiority claim'}
    target=a.output or out/'audit-broyden-ssw-multicase.json'; target.write_text(json.dumps(output,indent=2)+'\n'); print(json.dumps(dict(groups=groups,total_requests=sum(g['search_requests']+g['fresh_requests'] for g in groups.values())),indent=2))
if __name__=='__main__': main()
