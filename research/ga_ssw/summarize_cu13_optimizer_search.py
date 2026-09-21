"""Compare complete Cu13 trajectories and strict structural certificates."""
import json
from pathlib import Path


def summarize(base):
    base=Path(base)
    validation=json.loads((base/'strict-validation/summary.json').read_text())
    runs=[]
    for source in sorted(base.glob('[0-9]*-*.json')):
        data=json.loads(source.read_text())
        result=data['result']
        strict=json.loads((base/'strict-validation'/source.name).read_text())
        ids=[r['fingerprint_group'] if r['qualified'] else None for r in strict]
        current=ids[0];cursor=1;accepted_changes=0;proposals=[]
        cost=result['initial']['evaluation_requests']
        for record in result['records']:
            cost+=record['evaluation_requests']
            landing=record['landing']
            if landing is not None and landing['converged']:
                identity=ids[cursor];cursor+=1
                proposals.append(dict(cost=cost,group=identity,energy=landing['energy']))
                if record['accepted']:
                    if identity is not None and current is not None and identity!=current:
                        accepted_changes+=1
                    current=identity
        assert cursor==len(ids)
        assert cost==result['evaluation_requests']
        runs.append(dict(source=source.name,requests=cost,steps=len(result['records']),
            failed=sum('failed' in r['status'] for r in result['records']),
            accepted_changes=accepted_changes,groups=len(set(i for i in ids if i is not None)),
            proposals=proposals,initial_group=ids[0],
            true_certified=sum(c['passed'] for c in data['fresh_checks'])))
    return dict(runs=runs,requests=sum(r['requests'] for r in runs),
        failed=sum(r['failed'] for r in runs),accepted_changes=sum(r['accepted_changes'] for r in runs),
        groups=validation['fingerprint_groups'],strict_validation_requests=validation['total_validation_requests'],
        positive_internal_spectra=all(s['eigenvalues'][0]>0 for r in validation['representatives'] for s in r['spectra']))


def main():
    root=Path('research/ga_ssw/evidence')
    results={key:summarize(root/name) for key,name in [('ase','cu13-direction-only'),('safe','cu13-safe-total')]}
    # Common per-trajectory completed-landing prefixes: report budget selection
    # and spent full costs, never discard a failed move's work from the budget.
    budget=min(r['requests'] for result in results.values() for r in result['runs'])
    for result in results.values():
        for run in result['runs']:
            run['groups_at_common_budget']=len(set([run['initial_group']]+[p['group'] for p in run['proposals'] if p['cost']<=budget and p['group'] is not None]))
    output=dict(results=results,common_request_budget=budget,
        limits='same Cu13 geometry used in optimizer selection; fingerprint groups are not exhaustive basins; common budget selected from total costs, completed landings only; no global or cross-system superiority claim')
    (root/'cu13-safe-total/comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps({key:{k:v for k,v in result.items() if k!='runs'} for key,result in results.items()},indent=2))
    print('common_request_budget',budget)
    print([(key,[(r['source'],r['groups_at_common_budget']) for r in result['runs']]) for key,result in results.items()])
if __name__=='__main__':main()
