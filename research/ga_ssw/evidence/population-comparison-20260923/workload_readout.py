"""Read stored runs to distinguish outer moves from PES requests; zero new PES."""
import json
import pickle
from collections import Counter
from pathlib import Path
root = Path(__file__).resolve().parent
rows = []
for folder in sorted((root/'runs').glob('c60-*-v2')):
    if not (folder/'summary.json').exists():
        raise RuntimeError(f'incomplete: {folder}')
    summary = json.loads((folder/'summary.json').read_text())
    with (folder/'result.pkl').open('rb') as stream:
        result = pickle.load(stream)
    walks = result.walks if summary['arm']=='ga' else [r['result'] for r in result['walks'] if r['result'] is not None]
    records = [r for walk in walks for r in walk.records]
    qualified = [r for r in records if getattr(r.landing,'converged',False) and getattr(r.landing,'surface',None)=='true' and not r.error]
    rows.append(dict(run=folder.name, requests=summary['search_requests'],
        calculator_calls=summary['search_calculator_calculate_calls'],
        outer_attempts=len(records),force_qualified_landings=len(qualified),
        outer_status_counts=dict(Counter(r.status for r in records)),
        qualified_move_mean_requests=sum(r.evaluation_requests for r in qualified)/len(qualified) if qualified else None,
        recorded_move_requests=sum(r.evaluation_requests for r in records),
        search_wall_seconds=summary['wall_seconds'],
        seconds_per_search_request=summary['wall_seconds']/summary['search_requests']))
report=dict(runs=rows,
    paper=dict(source='Uploaded GA-SSW-user.pdf Table1 and rendered author244-SI.pdf S14',
        c60_ssw_mean_steps=10807,c60_ga_mean_normalized_steps=6908,
        c60_mean_energy_evaluations_per_ssw_step=509.9,
        approximate_ga_evaluations=6908*509.9,
        approximate_ssw_evaluations=10807*509.9,
        caveat='Order-of-magnitude conversion only: paper charges offspring quench0.5 step; its C60-specific NN differs from MH1. These numbers do not predict MH1 success or exact GA calls.'),
    interpretation='Outer attempts, qualified landings, requests, actual calculator calls and successful target discovery are different quantities.')
(root/'workload-readout.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
