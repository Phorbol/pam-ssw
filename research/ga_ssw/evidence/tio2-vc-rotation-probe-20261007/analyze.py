"""Independent raw-cost and paired-direction readout of the saved-state probe."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def ledger(path):
    events = [json.loads(line) for line in path.open()]
    return dict(path=str(path), requests=sum(e['event'] in ('evaluation', 'failure') for e in events),
                calculator_calls=sum(e.get('calculator_calls', 0) for e in events),
                failures=sum(e['event'] == 'failure' for e in events),
                denials=sum(e['event'] == 'denial' for e in events))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    records_path = args.run / 'mode-records.jsonl'
    rows = [json.loads(line) for line in records_path.open()] if records_path.exists() else []
    plan = json.loads((Path(__file__).parent / 'plan.json').read_text())
    arrays = np.load(args.run / 'mode-arrays.npz', allow_pickle=False)
    costs = [ledger(p) for p in sorted(args.run.glob('*.jsonl'))
             if p.name.startswith(('requests-', 'fresh-'))]
    total = sum(row['requests'] for row in costs)
    cap = sum(plan['budget'][key] for key in
              ('total_search_requests', 'total_reference_requests', 'total_fresh_requests'))
    if total > cap:
        raise AssertionError('probe exceeds predeclared total cap')
    for row in rows:
        if not row['search_ledger_closure']['closed'] or row['search_cost']['requests'] > 40:
            raise AssertionError('iterative mode cost contract failed')
    summary = json.loads((args.run / 'result.json').read_text())
    if total != summary['total_paid_requests'] or not summary['all_ledgers_closed']:
        raise AssertionError('aggregate/raw-ledger closure failed')
    pairs = []
    for case in plan['cases']:
        for index in case['outer_anchor_indices']:
            pair = {r['method']: r for r in rows if r['case'] == case['id'] and r['anchor_index'] == index}
            result = dict(case=case['id'], anchor_index=index, methods=pair,
                          both_directions_present=False)
            a, b = pair.get('plane_dimer'), pair.get('central_ritz')
            if a and b and a.get('search_status') == b.get('search_status') == 'completed':
                aa, ab = arrays[a['anchor_array_key']], arrays[b['anchor_array_key']]
                if not np.array_equal(aa, ab):
                    raise AssertionError('paired anchors differ')
                na, nb = arrays[a['direction_array_key']], arrays[b['direction_array_key']]
                result.update(both_directions_present=True,
                    unsigned_direction_angle_deg=float(np.degrees(np.arccos(np.clip(abs(na @ nb), 0., 1.)))))
            pairs.append(result)
    reference_path = args.run / 'reference.json'
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else None
    payload = dict(scope='fixed-center direction diagnosis; no quench, phase-search or minimum-stability claim',
                   planned_modes=8, recorded_modes=len(rows), pairs=pairs,
                   reference=reference, raw_ledgers=costs, total_paid_requests=total,
                   total_cap=cap, analysis_pes_requests=0,
                   analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.out / 'analysis.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(recorded_modes=len(rows), total_paid_requests=total, total_cap=cap), indent=2))


if __name__ == '__main__':
    main()
