"""Analyze the frozen allocation diagnostic; never evaluate a potential."""
import argparse
import json
from pathlib import Path

from analyze_c60_cost_allocation import one_run
from analyze_c60_random_development import EREF, ETOL, graph_row


def analyze(root):
    rows = []
    for group in ('c60', 'controls'):
        folder = root / group
        plan = json.loads((folder / 'plan.json').read_text())
        for case in plan['cases']:
            for arm in plan['arms']:
                run = folder / f"{case}-{arm['name']}"
                if not (run / 'summary.json').exists():
                    rows.append(dict(case=case, arm=arm['name'], state='missing'))
                    continue
                summary = json.loads((run / 'summary.json').read_text())
                if not (run / 'result.json').exists():
                    rows.append(dict(case=case, arm=arm['name'], state='no_result', summary=summary))
                    continue
                result = json.loads((run / 'result.json').read_text())
                qualification = json.loads((run / 'qualification.json').read_text())
                assert len(qualification) == len(result['minima'])
                assert all(q['index'] == i for i, q in enumerate(qualification))
                eligible = [q for q in qualification if q.get('qualified')
                            and q.get('composition_match') and q.get('fixed_cell')
                            and abs(q.get('energy_error', 1e99)) < 1e-6]
                costs = one_run(run)
                assert costs['ledger_reconciliation_residual'] == 0
                assert costs['ledger_search_requests'] == summary['search_requests']
                assert costs['result_evaluation_requests'] == summary['search_requests']
                initial = result['initial']['energy']
                best = min((q['energy'] for q in eligible), default=None)
                row = dict(case=case, arm=arm['name'], state='analyzed',
                           execution=summary['execution'], boundary=summary['boundary'],
                           requests=summary['search_requests'], fresh_requests=summary['fresh_requests'],
                           elapsed=summary['elapsed'], initial_energy=initial, best_energy=best,
                           delta_energy=None if best is None else best-initial,
                           qualified_including_initial=len(eligible),
                           qualified_landings=sum(q['index'] != 0 for q in eligible),
                           outer_statuses=summary['outer_statuses'], stage=costs['stage'],
                           gaussian_events=costs['gaussian_event_count'],
                           qualification_failures=len(qualification)-len(eligible))
                if case.startswith('c60'):
                    cages = []
                    for q in eligible:
                        atom = result['minima'][q['index']]['atoms']
                        flags = [graph_row(atom['numbers'], atom['positions'], cutoff)
                                 ['graph_cage_candidate'] for cutoff in (1.64, 1.7, 1.8)]
                        cages.append(dict(index=q['index'], cage_at_cutoffs=flags,
                                          target=q['energy'] <= EREF+ETOL))
                    row['c60_candidates'] = cages
                rows.append(row)
    return dict(scope='Saved-minimum development diagnostic; not random-cloud acceptance or independent validation.',
                rows=rows, definitions={'delta_energy': 'qualified best minus initial, eV; lower is better',
                                        'landings': 'force-qualified returned candidates, not deduplicated basins',
                                        'stage_residuals': 'unassigned requests retained; no forced causal attribution'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = analyze(args.input)
    with args.output.open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write('\n')
    for row in data['rows']:
        print(row['case'], row['arm'], row['state'], row.get('requests'),
              row.get('delta_energy'), row.get('qualified_landings'))
