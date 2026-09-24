#!/usr/bin/env python3
"""Compare saved global+CBD and full-direction C60 runs; never evaluates PES."""
import json
from pathlib import Path

from analyze_escape import events, qualified

HERE = Path(__file__).resolve().parent
CUTOFFS = (1.64, 1.7, 1.8)


def read_rows(path):
    return json.loads(path.read_text()) if path.is_file() else []


def load_arm(runs, seed, method):
    folder = runs / f'{method}-{seed}'
    path = folder / 'summary.json'
    root_rows = read_rows(runs / 'summary.json')
    fallback = next((row for row in root_rows
                     if row.get('seed') == seed and row.get('method') == method), None)
    if path.is_file():
        raw = json.loads(path.read_text())
        source = str(path)
        summary_present = True
    elif fallback is not None:
        raw = fallback
        source = str(runs / 'summary.json')
        summary_present = False
    else:
        return dict(seed=seed, method=method, present=False,
                    status='no_completed_summary', source=str(path), checks=[], records=[])
    checks = raw.get('checks', [])
    record_path = folder / 'record-costs.json'
    records = json.loads(record_path.read_text()) if record_path.is_file() else []
    requests = raw.get('search_requests')
    full_events = events(checks, requests) if isinstance(requests, int) else None
    return dict(seed=seed, method=method, present=True,
        arm_summary_file_present=summary_present, source=source,
        status=raw.get('status', 'status_missing'),
        search_requests=requests, fresh_requests=raw.get('fresh_requests'),
        total_requests=raw.get('total_requests',
            (requests + raw['fresh_requests']) if isinstance(requests, int)
            and isinstance(raw.get('fresh_requests'), int) else None),
        search_calculations=raw.get('search_calculations'),
        fresh_calculations=raw.get('fresh_calculations'),
        search_boundary=raw.get('search_boundary'), search_denials=raw.get('search_denials'),
        fresh_boundary=raw.get('fresh_boundary'), fresh_denials=raw.get('fresh_denials'),
        elapsed_seconds=raw.get('elapsed_seconds'),
        record_statuses=[row.get('status') for row in records],
        record_costs=[{key: row.get(key) for key in
            ('index', 'status', 'accepted', 'requests', 'cumulative_requests')}
            for row in records],
        record_cost_sum_matches_requests=raw.get('record_cost_sum_matches_requests'),
        requests_outside_saved_records=raw.get('requests_outside_saved_records'),
        fresh_missing_or_failed=sum(check.get('status') != 'checked' for check in checks),
        check_failures=[{key: check.get(key) for key in
            ('role', 'status', 'error', 'reported_converged', 'force_qualified')}
            for check in checks if check.get('status') != 'checked'],
        initial_check_present=any(check.get('role') == 'initial' for check in checks),
        landing_check_count=sum(check.get('role', '').startswith('landing-') for check in checks),
        full_arm_events=full_events,
        full_arm_best_including_initial=best_including_initial(
            {'checks': checks}, ceiling=None), checks=checks)


def best_including_initial(arm, ceiling=None):
    checks = arm.get('checks', [])
    candidates = []
    for check in checks:
        role = check.get('role')
        if role == 'initial':
            include = True
        elif role and role.startswith('landing-'):
            include = ceiling is None or check.get('search_cost', float('inf')) <= ceiling
        else:
            include = False
        if include and qualified(check):
            candidates.append(check)
    if not candidates:
        return dict(candidate_count=0, best=None)
    best = min(candidates, key=lambda row: row.get('delta_ih_eV', float('inf')))
    graph_rows = best.get('graphs', {})
    cage = all(graph_rows.get(str(cutoff), {}).get('graph_cage_candidate', False)
               for cutoff in CUTOFFS)
    ih = all(graph_rows.get(str(cutoff), {}).get('ih_graph_match', False)
             for cutoff in CUTOFFS)
    return dict(candidate_count=len(candidates), best=dict(
        role=best.get('role'), search_cost=best.get('search_cost'),
        energy_eV=best.get('energy_eV'), delta_ih_eV=best.get('delta_ih_eV'),
        energy_target_met=best.get('energy_window_met'),
        cage_candidate_all_cutoffs=cage, ih_graph_match_all_cutoffs=ih,
        non_cage=not cage))


def paired_prefix(global_arm, direction_arm):
    if not global_arm['present'] or not direction_arm['present']:
        return dict(status='missing_arm', common_search_prefix=None,
                    global_cbd=None if not global_arm['present'] else global_arm,
                    full_direction=None if not direction_arm['present'] else direction_arm)
    left, right = global_arm.get('search_requests'), direction_arm.get('search_requests')
    if not isinstance(left, int) or not isinstance(right, int):
        return dict(status='search_cost_missing', common_search_prefix=None,
                    global_cbd=global_arm, full_direction=direction_arm)
    ceiling = min(left, right)
    return dict(status='compared', common_search_prefix=ceiling,
        global_cbd=dict(status=global_arm['status'], costs=costs(global_arm),
            common_prefix_events=events(global_arm['checks'], ceiling),
            best_including_initial=best_including_initial(global_arm, ceiling)),
        full_direction=dict(status=direction_arm['status'], costs=costs(direction_arm),
            common_prefix_events=events(direction_arm['checks'], ceiling),
            best_including_initial=best_including_initial(direction_arm, ceiling)),
        interpretation='Prefix outcomes are retrospective from saved landings; fresh checks and total cost were paid for the completed arm, not at the prefix.')


def costs(arm):
    return {key: arm.get(key) for key in (
        'search_requests', 'fresh_requests', 'total_requests', 'search_calculations',
        'fresh_calculations', 'search_boundary', 'search_denials', 'fresh_boundary',
        'fresh_denials', 'elapsed_seconds', 'record_statuses', 'record_costs',
        'record_cost_sum_matches_requests', 'requests_outside_saved_records',
        'fresh_missing_or_failed', 'check_failures', 'initial_check_present',
        'landing_check_count')}


def main():
    parent_plan = json.loads((HERE / 'escape-plan.json').read_text())
    direction_plan_path = HERE / 'direction-probe' / 'plan.json'
    direction_plan = json.loads(direction_plan_path.read_text())
    output = HERE / 'direction-probe' / 'comparison.json'
    if output.exists():
        raise FileExistsError(output)
    parent_runs = Path(parent_plan['output']['runs_dir'])
    direction_runs = Path(direction_plan['output']['runs_dir'])
    arms = []
    pairs = []
    for seed in parent_plan['arms']['seeds']:
        for method in parent_plan['methods']:
            global_arm = load_arm(parent_runs, seed, method)
            direction_arm = load_arm(direction_runs, seed, method)
            arms.extend([dict(configuration='global_plus_recovered_CBD', **global_arm),
                         dict(configuration='full_recovered_direction', **direction_arm)])
            pairs.append(dict(seed=seed, ls_enabled=(method == 'native_ls'),
                method=method, comparison=paired_prefix(global_arm, direction_arm)))
    report = dict(scope='One qualified C60 defect input; two seeds are paired probes, not independent structural validation.',
        plan_sources={'global_cbd': str(HERE / 'escape-plan.json'),
                      'full_direction': str(direction_plan_path)},
        definitions={'full_direction': 'existing recovered pair/group/local/displacement controller plus recovered CBD; replaces the separate recovered_rotation route.',
            'event_counts': 'qualified landing observations from analyze_escape.events; not unique minima.',
            'best': 'lowest Ih-relative energy among qualified checked endpoints, explicitly including the initial structure; target, cage and Ih flags are reported separately.',
            'missing_and_failures': 'missing arms, run status, record statuses, fresh failures, request denials and total costs are retained where saved.',
            'prefix_boundary': 'minimum completed/charged search request count of the pair; fresh validation costs correspond to each full run.'},
        arms=arms, paired_comparisons=pairs)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(output)


if __name__ == '__main__':
    main()
