"""Audit frozen shared-direction campaigns without further PES evaluations."""
import argparse
import json
import math
from pathlib import Path


def analyze(root):
    plan = json.loads((root/'plan.json').read_text())
    rows, issues = [], []
    for case in plan['cases']:
        for seed in plan.get('seeds', [plan.get('seed')]):
            for arm in plan['arms']:
                name = f'{case}-{arm}-seed{seed}'
                if name in plan.get('excluded_runs', {}):
                    continue  # Exclusions retain provenance in plan.json, not unfinished arms.
                folder = root/name
                row = dict(run=name, execution='not_finished', numerical='not_assessed',
                           physical='not_independently_validated', scientific='implementation_check')
                summary_path = folder/'summary.json'
                ledger_path = folder/'requests.jsonl'
                if not summary_path.exists():
                    rows.append(row)
                    continue  # Do not parse a concurrently written ledger.
                summary = json.loads(summary_path.read_text())
                ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()] if ledger_path.exists() else []
                paid = [entry for entry in ledger if entry['kind'] in ('search', 'search_failure')]
                denials = [entry for entry in ledger if entry['kind'] == 'search_denial']
                if [entry['request'] for entry in paid] != list(range(1, len(paid)+1)):
                    issues.append(f'{name}: nonsequential paid ledger')
                if len(paid) != summary['search_requests'] or len(denials) != summary['denials']:
                    issues.append(f'{name}: ledger/summary count mismatch')
                boundary = summary.get('boundary')
                row.update(execution='censored' if boundary else summary['execution'],
                    program_status=summary['execution'], boundary=boundary,
                    search_requests=len(paid), failed_requests=sum(e['kind']=='search_failure' for e in paid),
                    denied_requests=len(denials), validation_requests=summary['fresh_requests'])
                result_path = folder/'result.json'
                if result_path.exists():
                    result = json.loads(result_path.read_text())
                    if result['evaluation_requests'] != len(paid):
                        issues.append(f'{name}: result/ledger cost mismatch')
                    initial_cost = result['initial']['evaluation_requests']
                    outer_cost = sum(record['evaluation_requests'] for record in result['records'])
                    if initial_cost + outer_cost != len(paid):
                        issues.append(f'{name}: initial+outer cost mismatch')
                    checks = json.loads((folder/'qualification.json').read_text())
                    qualified = sum(c.get('qualified', False) and c.get('fixed_cell', False)
                        and c.get('composition_match', True) and c.get('pbc_unchanged', True) for c in checks)
                    denominator = len(result['minima'])
                    row.update(initial_requests=initial_cost, outer_requests=outer_cost,
                        attempted_outer_steps=len(result['records']),
                        minima_including_initial=denominator, validated_frames=len(checks),
                        qualified_frames=qualified, unvalidated_frames=denominator-len(checks),
                        numerical='all_returned_minima_qualified' if qualified==denominator else 'incomplete_or_failed_qualification',
                        attempt_errors=[r['error'] for r in result['records'] if r.get('error')],
                        outer_statuses=summary.get('outer_statuses'))
                    if summary['fresh_requests'] != len(checks):
                        issues.append(f'{name}: independent qualification cost mismatch')
                    diagnostics = folder/'native-ls-diagnostics.json'
                    if diagnostics.exists():
                        ls = json.loads(diagnostics.read_text())
                        updates = ls['updates']
                        row.update(ls_nonzero_initialization=(ls['initial_pair_count']>0 and ls['initial_strength_sum']>0),
                            ls_initial_pair_count=ls['initial_pair_count'], ls_initial_strength_sum=ls['initial_strength_sum'],
                            ls_outer_update_events=len(updates),
                            ls_response_updates=sum('normal_update' in u['actions'] for u in updates),
                            ls_preparation_qualifications=[p['qualification'] for p in ls['preparations']])
                        paired = [r for r in result['records'] if r.get('ls_preparation') and r.get('ls_update')]
                        natoms = len(result['initial']['atoms']['numbers'])
                        responses = [r['ls_update']['observed_response_mev_per_atom'] for r in paired]
                        response_consistent = all(math.isclose(
                            r['ls_update']['observed_response_mev_per_atom'],
                            1000*(r['ls_preparation']['true_energy_after']-r['ls_preparation']['true_energy_before'])/natoms,
                            rel_tol=1e-12, abs_tol=1e-12) for r in paired)
                        row.update(ls_response_pairs=len(paired), ls_response_units_consistent=response_consistent,
                            ls_nonzero_response_events=sum(math.isfinite(x) and x != 0 for x in responses),
                            ls_table_changed=any(u['table'] != ls['initial_table'] for u in updates),
                            ls_response_range_mev_per_atom=[min(responses),max(responses)] if responses else None,
                            recovered_rotation_events=sum(e.get('rotation_solver')=='recovered-cbd'
                                for r in result['records'] for e in r['climb']))
                        if not response_consistent: issues.append(f'{name}: true-energy response mismatch')
                        # Older frozen runner checks the wrong observed_response key.
                        # Recompute from raw events without rewriting that evidence.
                        row['ls_real_response_update'] = any('normal_update' in u['actions'] and
                            math.isfinite(u['observed_response_mev_per_atom']) for u in updates)

                else:
                    row['error'] = summary.get('error')
                rows.append(row)
    return dict(protocol='plan.json', excluded_runs=plan.get('excluded_runs', {}), expected_arms=len(rows),
        finished_arms=sum(row['execution']!='not_finished' for row in rows),
        total_search_requests=sum(row.get('search_requests',0) for row in rows),
        total_validation_requests=sum(row.get('validation_requests',0) for row in rows),
        audit_issues=issues, rows=rows,
        scope='Numerical and lifecycle audit only; no deduplication, physical certification or efficiency ranking.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key: report[key] for key in ('expected_arms','finished_arms','total_search_requests',
                                                   'total_validation_requests','audit_issues')}))
