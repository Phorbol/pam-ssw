"""Read-only shared first-walk prefix audit, no model or PES calls."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
a = parser.parse_args()
root = Path(__file__).resolve().parent / 'runs'
rows = []
for seed in (3, 17):
    paths = [root / f'c60-seed{seed}-{arm}-v2' / 'result.pkl' for arm in ('ssw', 'ga')]
    if not all(p.exists() for p in paths):
        rows.append(dict(seed=seed, status='pending'))
        continue
    with paths[0].open('rb') as f:
        baseline = pickle.load(f)['walks'][0]['result']
    with paths[1].open('rb') as f:
        genetic = pickle.load(f).walks[0]
    steps = []
    for left, right in zip(baseline.records, genetic.records):
        record = dict(index=left.index, status_equal=left.status == right.status,
                      accepted_equal=left.accepted == right.accepted,
                      requests=[left.evaluation_requests, right.evaluation_requests])
        if left.initial_direction is not None and right.initial_direction is not None:
            record['direction_max_difference'] = float(np.max(np.abs(np.asarray(left.initial_direction)-np.asarray(right.initial_direction))))
        if hasattr(left.landing, 'energy') and hasattr(right.landing, 'energy'):
            record['landing_energy_difference'] = float(left.landing.energy-right.landing.energy)
            record['landing_position_max_difference'] = float(np.max(np.abs(left.landing.atoms.positions-right.landing.atoms.positions)))
        steps.append(record)
    rows.append(dict(seed=seed,status='analyzed',
        initial_energy_difference=float(baseline.initial.energy-genetic.initial.energy),
        initial_position_max_difference=float(np.max(np.abs(baseline.initial.atoms.positions-genetic.initial.atoms.positions))),
        baseline_records=len(baseline.records),ga_quick_records=len(genetic.records),steps=steps))
a.output.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
print(json.dumps(rows,indent=2))
