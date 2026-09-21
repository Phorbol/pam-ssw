"""Offline endpoint/cost comparison; no calculator calls."""
import json
from pathlib import Path
import numpy as np

ROOT=Path('research/ga_ssw/evidence')
BH=ROOT/'ase-basin-hopping-baseline-20260912-v2'
SSW=ROOT/'two-stage-ritz-comparison-20260912'

def same(a,b):
    return (a['numbers']==b['numbers'] and a['pbc']==b['pbc']
            and np.array_equal(a['positions'],b['positions'])
            and np.array_equal(a['cell'],b['cell']))

def main():
    rows=[]
    for case in ('cu13','cu31_fixed','bicyclobutane'):
      for seed in (11,29):
        folder=BH/f'{case}-seed{seed}';summary=json.loads((folder/'summary.json').read_text())
        qs=json.loads((folder/'local-results.json').read_text())
        ledger=[json.loads(line) for line in (folder/'evaluations.jsonl').open()]
        paid=[x for x in ledger if x['kind'] in ('search','search_failure')]
        assert len(paid)==summary['search_requests']
        assert [x['request'] for x in paid]==list(range(1,len(paid)+1))
        checks={x['index']:x for x in summary['fresh_checks']}
        costs=[];cursor=0;extra=0
        for i,q in enumerate(qs):
            cursor+=q['evaluation_requests'];assert cursor<=len(paid)
            endpoint=paid[cursor-1]
            assert same(endpoint['atoms'],q['atoms']) and abs(endpoint['energy']-q['energy'])<1e-10
            costs.append(cursor)
            # Unmodified ASE BH obtains endpoint energy after each local optimizer.
            if cursor<len(paid):
                callback=paid[cursor]
                assert same(callback['atoms'],q['atoms']) and abs(callback['energy']-q['energy'])<1e-10
                cursor+=1;extra+=1
        parent=SSW/f'{case}-two_stage_ritz-{seed}'
        sr=json.loads((parent/'result.json').read_text());sf=json.loads((parent/'fresh-checks.json').read_text())
        cap=min(summary['search_requests'],sr['evaluation_requests'])
        sc=[sr['initial']['evaluation_requests']];next_index=1;total=sc[0]
        for record in sr['records']:
            total+=record['evaluation_requests'];q=record.get('landing')
            if q and q['converged'] and q['surface']=='true':
                assert q==sr['minima'][next_index];next_index+=1;sc.append(total)
        assert next_index==len(sr['minima']) and total==sr['evaluation_requests']
        bq=[(q,checks[i]) for i,q in enumerate(qs) if q['converged'] and costs[i]<=cap and checks[i].get('force_qualified')]
        sq=[(q,f) for q,f,c in zip(sr['minima'],sf,sc) if c<=cap and f['fmax']<=.01]
        rows.append(dict(case=case,seed=seed,common_search_requests=cap,
            bh_status=summary['status'],bh_boundary=summary['boundary'],bh_error=summary.get('controller_error'),
            bh_search=len(paid),bh_fresh=summary['fresh_requests'],bh_qualified_prefix_landings=len(bq),
            bh_best_fresh=min(x[1]['energy'] for x in bq),bh_best_delta=min(x[1]['energy'] for x in bq)-checks[0]['energy'],
            bh_local_requests=sum(q['evaluation_requests'] for q in qs),bh_endpoint_callbacks=extra,
            bh_incomplete_local_requests=len(paid)-cursor,bh_backend_failure_calls=sum(x['kind']=='search_failure' for x in paid),
            bh_fresh_all_qualified=all(x.get('force_qualified') for x in checks.values()),
            bh_fresh_max_energy_error=max(abs(x['energy_error']) for x in checks.values()),
            ssw_search=sr['evaluation_requests'],ssw_fresh=len(sf),ssw_qualified_prefix_landings=len(sq),
            ssw_best_fresh=min(x[1]['energy'] for x in sq),ssw_best_delta=min(x[1]['energy'] for x in sq)-sf[0]['energy'],
            bh_initial_vs_ssw_initial=checks[0]['energy']-sf[0]['energy']))
    output=dict(scope='unmodified ASE BH controller / Safe-total versus frozen staged Ritz SSW; matched realized search prefix, posthoc developmental comparison, no statistical superiority claim',
        bh_search_total=sum(x['bh_search'] for x in rows),bh_fresh_total=sum(x['bh_fresh'] for x in rows),rows=rows)
    (BH/'root-final-audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
