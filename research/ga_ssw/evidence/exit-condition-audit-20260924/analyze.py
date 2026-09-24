"""Read-only energy-exit triage; no potential calls or basin claims."""
import json
from pathlib import Path
from collections import Counter
from statistics import median
root=Path(__file__).resolve().parent
base=root.parent
paths=sorted((base/'c4h6-mh1-coverage-20260924').glob('*/result.json'))
paths+=sorted((base/'periodic-rotation-priority-20260923').glob('*/result.json'))
assert len(paths)==10, len(paths)
rows=[]
for path in paths:
    d=json.loads(path.read_text());current=float(d['initial']['energy']);events=[]
    assert d['initial']['converged']
    for r in d['records']:
        assert not r.get('starter_selection'), 'pool requires explicit start reconstruction'
        landing=r.get('landing');climb=r['climb']
        if r['status']=='lower_true_energy':
            last=climb[-1];drop=current-last['true_energy']
            assert drop>0, (path,r['index'],drop)
            events.append(dict(index=r['index'],stages=len(climb),climb_drop_eV=drop,
                landing_drop_eV=None if not landing else current-landing['energy'],
                landing_converged=bool(landing and landing['converged']),accepted=r['accepted'],
                requests=r['evaluation_requests']))
        if r['accepted']:
            assert landing and landing['converged']
            current=float(landing['energy'])
    def stats(key):
        vals=[e[key] for e in events if e[key] is not None]
        return dict(n=len(vals),minimum=min(vals),median=median(vals),maximum=max(vals)) if vals else dict(n=0)
    rows.append(dict(path=str(path.relative_to(base)),records=len(d['records']),
        statuses=dict(Counter(r['status'] for r in d['records'])),
        early_exit_climb_drop_eV=stats('climb_drop_eV'),early_exit_landing_drop_eV=stats('landing_drop_eV'),
        early_exit_stages=stats('stages'),early_exit_events=events))
(root/'analysis.json').write_text(json.dumps(dict(rows=rows,scope='existing trajectories; energy-exit triage, not basin identity or independent runs'),indent=2)+'\n')
for r in rows:print(r['path'],r['statuses'],r['early_exit_climb_drop_eV'],r['early_exit_landing_drop_eV'])
