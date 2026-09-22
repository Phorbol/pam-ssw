"""Read saved trusted-local results. No potential requests."""
import json
import pickle
from pathlib import Path


def signature(result):
    return dict(status=result.status, requests=result.evaluation_requests,
        stages=[(s.phase, s.status, s.evaluation_requests) for s in result.stages],
        observations=[(o.id, o.phase, o.eligible_for_archive, o.result.energy,
            o.result.atoms.positions.tolist()) for o in result.observations])


def analyze(out):
    rows = {name: json.loads((out / f'{name}.json').read_text())
            for name in ('baseline', 'full', 'first', 'resumed')}
    full = pickle.loads((out / 'full.pkl').read_bytes())
    resumed = pickle.loads((out / 'resumed.pkl').read_bytes())
    equal = signature(full) == signature(resumed)
    phases = {'quick', 'offspring_ssw', 'generation_short', 'fine'}
    errors = []
    for name, row in rows.items():
        if name != 'first' and row['status'] != 'completed':
            errors.append(f'{name}: incomplete')
        if sum(s['requests'] for s in row['stages']) != row['cumulative_requests']:
            errors.append(f'{name}: stage ledger mismatch')
        if not all(c['force_pass'] and c['composition_pass'] and c['boundary_pass']
                   and c['energy_agreement'] < 1e-10 for c in row['checks']):
            errors.append(f'{name}: fresh qualification failure')
        if name in ('baseline', 'full') and not phases.issubset(s['phase'] for s in row['stages']):
            errors.append(f'{name}: missing walk phase')
        if name != 'baseline' and not all(s['fresh'] for s in row['starts']):
            errors.append(f'{name}: reused controller state')
    if rows['first']['segment_requests'] + rows['resumed']['segment_requests'] != rows['full']['segment_requests']:
        errors.append('split ledger mismatch')
    if not equal:
        errors.append('full/resumed trajectory mismatch')
    summary = dict(errors=errors, continuous_resume_equal=equal,
        search_requests=sum(row['segment_requests'] for row in rows.values()),
        fresh_requests=sum(row['fresh_requests'] for row in rows.values()),
        arms={name: dict(status=r['status'], search_requests=r['segment_requests'],
            fresh_requests=r['fresh_requests'], initialized_controllers=len(r['starts']),
            escape_attempts=len(r['escapes']), failures=len(r['failures'])) for name,r in rows.items()},
        rng_boundary='Terminal RNG not captured in completed results; no strict RNG-state equality claim',
        scope='Interface/numerical qualification; duplicated endpoints retained in fresh denominator; no global-optimum or general efficiency claim')
    (out / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary, indent=2))
    assert not errors, errors

if __name__ == '__main__':
    analyze(Path(__file__).parent)
