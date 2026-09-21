"""Audit saved constraint results; no new calculator calls or overwritten parents."""
from pathlib import Path
import json
import numpy as np


def audit(root):
    plan=json.loads((root/'plan.json').read_text());rows=[]
    for name in plan['inputs']:
      initial=plan['inputs'][name]
      for variant in plan['variants']:
        directory=root/f'{name}-{variant}'
        original=json.loads((directory/'summary.json').read_text())
        result=json.loads((directory/'result.json').read_text())
        ledger=[json.loads(line) for line in (directory/'evaluations.jsonl').read_text().splitlines()]
        assert original.get('error') == 'AttributeError("\'ConstrainedSSWResult\' object has no attribute \'evaluation_requests\'")'
        assert len(ledger)==result['requests']==original['search_requests']==sum(r['requests'] for r in result['records'])
        assert [entry['request'] for entry in ledger]==list(range(1,len(ledger)+1))
        checks=original['checks'];assert len(checks)==len(result['minima'])
        for minimum,check in zip(result['minima'],checks):
            assert check['qualified']
            assert np.isclose(check['energy'],minimum['energy'],atol=1e-7,rtol=0)
            assert minimum['atoms']['constraints']==initial['constraints']
            assert check['active_fmax'] <= plan['config']['fmax']
            assert minimum['active_fmax'] <= plan['config']['fmax']
        outer=[r for r in result['records'] if 'index' in r]
        assert len(outer)==1
        rows.append(dict(case=name,variant=variant,status=result['status'],
            outer_status=outer[0]['status'],ls_update=outer[0].get('ls_update'),
            search_requests=result['requests'],fresh_requests=original['fresh_requests'],
            qualified=len(checks),all_constraints_retained=all(c['constraint_metadata'] for c in checks),
            max_fresh_fmax_difference=max(abs(c['active_fmax']-m['active_fmax']) for c,m in zip(checks,result['minima'])),
            hookean_energy_range=[min(c['hookean_energy'] for c in checks),max(c['hookean_energy'] for c in checks)],
            recording_error='original summary used evaluation_requests instead of requests; source results/ledgers/fresh certificates intact',
            records_cost_checked=True))
    report=dict(rows=rows,search_requests=sum(r['search_requests'] for r in rows),
        fresh_requests=sum(r['fresh_requests'] for r in rows),qualified=sum(r['qualified'] for r in rows),
        new_pes_calls=0,originals_preserved=True)
    (root/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

if __name__=='__main__':
    import sys
    print(json.dumps(audit(Path(sys.argv[1])),indent=2))
