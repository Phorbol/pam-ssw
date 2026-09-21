"""Read-only, calculator-free audit of bounded multi-case paid ledgers."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();rows=[]
    for path in sorted(a.directory.glob('*/summary.json')):
        s=json.loads(path.read_text());folder=path.parent
        ledger=[json.loads(x) for x in (folder/'evaluations.jsonl').read_text().splitlines()] if (folder/'evaluations.jsonl').exists() else []
        paid=[r for r in ledger if r['kind'] in ('paid','failure')]
        seq=[r['request'] for r in paid]
        count=s.get('requests',0)
        assert seq==list(range(1,count+1)),(folder,seq[-4:],count)
        fresh=json.loads((folder/'fresh.json').read_text()) if (folder/'fresh.json').exists() else []
        r=json.loads((folder/'result.json').read_text()) if (folder/'result.json').exists() else None
        certified=[];timeline=[]
        if r is not None:
            assert r['evaluation_requests']==count
            initial=r['initial'];cumulative=initial['evaluation_requests']
            timeline=[dict(request=cumulative,energy=initial['energy'])] if initial['converged'] else []
            for record in r['records']:
                cumulative+=record['evaluation_requests']
                landing=record.get('landing')
                if landing and landing['converged']:timeline.append(dict(request=cumulative,energy=landing['energy']))
            assert cumulative==count,(folder,cumulative,count)
            for f in fresh:
                m=r['minima'][f['index']];atoms=m['atoms'];base=initial['atoms']
                unchanged=atoms['numbers']==base['numbers'] and atoms['pbc']==base['pbc'] and np.array_equal(atoms['cell'],base['cell'])
                valid=('error' not in f and np.isfinite(f['energy_error']) and abs(f['energy_error'])<=1e-7 and f['fmax']<=.03 and unchanged)
                certified.append(dict(index=f['index'],qualified=bool(valid),geometry_invariant=unchanged))
        rows.append(dict(arm=folder.name,case=s['case'],variant=s['variant'],seed=s['seed'],raw_status=s['status'],
            budget_denied=any(x['kind']=='denial' for x in ledger),search_requests=count,fresh_attempts=len(fresh),
            paid_backend_failures=sum(x['kind']=='failure' for x in paid),fresh_checks=certified,timeline=timeline))
    comparisons=[]
    for case in sorted({r['case'] for r in rows}):
        for seed in (11,29):
            pair=[r for r in rows if r['case']==case and r['seed']==seed]
            if len(pair)!=2:continue
            prefix=min(x['search_requests'] for x in pair);entries=[]
            for r in pair:
                states=[x for x in r['timeline'] if x['request']<=prefix]
                entries.append(dict(variant=r['variant'],best_energy_at_prefix=min(x['energy'] for x in states) if states else None,
                    valid_observations_at_prefix=len(states)))
            comparisons.append(dict(case=case,seed=seed,common_paid_prefix=prefix,entries=entries))
    out=a.directory/'root-cost-audit.json'
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(dict(planned_arms=12,recorded_arms=len(rows),rows=rows,comparisons=comparisons,
        limit='Common-prefix energies are force-qualified observations, not proven minima identities or GM hits. Budget denials are not paid evaluations. No PES calls.'),indent=2)+'\n')
    print(json.dumps(dict(recorded_arms=len(rows),search_requests=sum(x['search_requests'] for x in rows),fresh_attempts=sum(x['fresh_attempts'] for x in rows))))

if __name__=='__main__':main()
