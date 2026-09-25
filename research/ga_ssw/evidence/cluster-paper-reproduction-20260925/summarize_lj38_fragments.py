"""Join precomputed connectivity with observed decisions/costs; zero PES."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = HERE/'lj38-connectivity.json'
rows = []
for run in json.loads(source.read_text())['runs']:
    if run.get('status') == 'missing_run':
        rows.append(run)
        continue
    landings = {row['step']: row for row in run['landing_minima']}
    events = [json.loads(line) for line in
              (HERE/run['run']/'outer-steps.jsonl').read_text().splitlines()]
    by_cutoff = {}
    for scale in (1.3, 1.5):
        groups = {key: dict(landings=0, accepted=0, search_requests=0)
                  for key in ('connected', 'fragmented', 'unclassified')}
        for event in events:
            if event.get('step') not in landings:
                continue
            geometry = landings[event['step']].get(f'cutoff_{scale:.1f}sigma_A')
            key = ('unclassified' if geometry is None else
                   ('connected' if geometry['single_cluster'] else 'fragmented'))
            group = groups[key]
            group['landings'] += 1
            group['accepted'] += int(event['accepted'])
            group['search_requests'] += event['step_requests']
        by_cutoff[str(scale)] = groups
    rows.append(dict(run=run['run'], total_search_requests=run['search_requests'],
        initial_quench_requests=run['initial_quench_event']['requests'],
        by_cutoff=by_cutoff))
result = dict(source=str(source), PES_requests=0, rows=rows,
    scope='Costs of completed outer steps with saved landings, including their climbing. '
          'Initial quench and terminal partial attempts are outside these bins but remain in total cost. '
          'Connectivity is not basin identity or causal proof of performance.')
(HERE/'lj38-fragment-costs.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
