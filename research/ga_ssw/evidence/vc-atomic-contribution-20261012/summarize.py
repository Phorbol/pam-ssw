"""Compact, read-only accounting for the frozen four-arm block experiment."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def summarize(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    rows = []
    for path in sorted(root.glob('seed-*/result.json')):
        d = json.loads(path.read_text())
        ledger = read_rows(path.parent / 'search-ledger.jsonl')
        fresh = read_rows(path.parent / 'fresh-ledger.jsonl')
        events = read_rows(path.parent / 'outer-records.jsonl')
        paid = sum(x.get('event') in ('evaluation', 'failure') for x in ledger)
        actual = sum(x.get('calculator_calls', 0) for x in ledger)
        assert paid == d['search_requests'] == d['search_ledger_paid']
        assert actual == d['search_calculator_calls']
        initial_cost = events[0]['request_start'] if events else paid
        assert initial_cost + sum(x['requests'] for x in events) == paid
        fresh_paid = sum(x.get('event') in ('evaluation', 'failure') for x in fresh)
        assert fresh_paid == d['fresh_requests'] == d['fresh_ledger_paid']
        rows.append(dict(seed=d['seed'], arm=d['arm'], status=d['status'],
            paid_search=paid, actual_search=actual, paid_fresh=fresh_paid,
            actual_fresh=sum(x.get('calculator_calls', 0) for x in fresh),
            initial_paid=initial_cost,
            outcomes=dict(Counter(x['execution_outcome'] for x in events)),
            atomic_paths_paid_attempts=dict(Counter(x['atomic_path_diagnostic'] for x in events if x['requests'] > 0)),
            zero_request_records=sum(x['requests'] == 0 for x in events),
            stage_unassigned_paid=sum(x['stage_cost_unassigned_difference'] for x in events),
            valid_atomic_landings=[dict(index=x['index'], accepted=x['accepted'],
                target=x.get('landing_qualification',{}).get('target_candidate'))
                for x in events if x['atomic_scheduled'] and x['status']=='valid_landing'],
            completed_gaussians=sum(x['completed_gaussians'] for x in events),
            budget_released_rotations=sum(x['rotation_budget_released'] for x in events),
            target_candidates=[dict(index=x['index'], cost=x['request_start']+x['requests'])
                for x in events if x.get('landing_qualification',{}).get('target_candidate')],
            first_target=d['first_target_candidate'],
            fresh_checks=d['fresh_checks'], best=d['best_physical_search_endpoint'],
            stage_costs={key:sum(x['stage_diagnostic_costs'].get(key,0) for x in events)
                for key in sorted({k for x in events for k in x['stage_diagnostic_costs']})}))
    return dict(run=str(root.resolve()), status=manifest['status'],
        code=manifest['source_head'], arms=rows, planned_arm_statuses=manifest['arm_results'],
        paired_trace_checks=manifest.get('paired_trace_checks'),
        total_search_paid=sum(x['paid_search'] for x in rows),
        total_search_actual=sum(x['actual_search'] for x in rows),
        fresh_plus_reference_paid=manifest.get('total_fresh_and_reference_paid'),
        fresh_plus_reference_actual=manifest.get('total_fresh_and_reference_calculator_calls'),
        limits=manifest['limits'],
        interpretation='Two seeds of one TiO2 starting geometry; bounded component diagnostic, not a success-rate estimate.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    payload=summarize(args.run)
    with args.output.open('x') as stream:
        json.dump(payload,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print(json.dumps({k:payload[k] for k in ('status','total_search_paid','total_search_actual','fresh_plus_reference_paid')}))
