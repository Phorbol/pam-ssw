"""Read saved summaries; distinguish requested restart, committed restart, and continuation."""
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = json.loads((root / 'summary.json').read_text())
summary = []
for row in rows:
    selections = row.get('committed_selections', [])
    updates = row.get('ls_updates', [])
    restarts = []
    for event in selections:
        if not event.get('restarted'):
            continue
        index = event['step']
        following = updates[index+1] if index+1 < len(updates) else None
        restarts.append(dict(step=index, chosen=event['chosen_index'],
            ls_reinitialized=event.get('ls_reinitialized', False),
            next_update_step=None if following is None else following.get('step'),
            next_local_step_one=bool(following is not None and following.get('step') == 1)))
    fresh = row.get('fresh', [])
    qualified = sum(all(check.get(key, False) for key in
        ('finite_energy','finite_forces','force_qualified','composition_unchanged',
         'cell_unchanged','pbc_unchanged')) for check in fresh)
    summary.append(dict(case=row['case'],arm=row['arm'],status=row['status'],
        search_requests=row.get('search_requests'),fresh_requests=row.get('fresh_requests',0),
        total_ef=row['total_ef'],restarts=restarts,fresh_count=len(fresh),
        fresh_qualified=qualified,boundary=row.get('boundary'),error=row.get('error')))
output=dict(expected_arms=4,observed_arms=len(summary),complete=len(summary)==4,runs=summary,total_ef=sum(r['total_ef'] for r in summary),
    budget=12000,within_budget=sum(r['total_ef'] for r in summary)<=12000,
    interpretation='Interface and numerical qualification only. No global-search efficiency claim.')
(root/'analysis.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(output,indent=2))
