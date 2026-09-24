"""Read saved checks only; never evaluate a potential or alter raw runs."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def qualified(check):
    return (check.get('status') == 'checked' and check.get('reported_converged')
            and check.get('force_qualified') and check.get('composition_preserved')
            and check.get('cell_preserved') and check.get('pbc_preserved'))


def events(checks, ceiling):
    landed = [c for c in checks if c['role'].startswith('landing-')
              and c['search_cost'] <= ceiling and qualified(c)]
    def at_all_cutoffs(c, field):
        return all(c['graphs'][str(h)][field] for h in (1.64, 1.7, 1.8))
    ih = [c for c in landed if at_all_cutoffs(c, 'ih_graph_match')]
    energy = [c for c in landed if c['energy_window_met']]
    cages = [c for c in landed if at_all_cutoffs(c, 'graph_cage_candidate')]
    changed = [c for c in landed if all(not c['graphs'][str(h)]['source_defect_graph_match']
                                       for h in (1.64, 1.7, 1.8))]
    fragmented = [c for c in landed if c['graphs']['1.8']['components'] > 1]
    return dict(search_prefix_ceiling=ceiling, qualified_landing_observations=len(landed),
        changed_graph_observations=len(changed), intact_cage_observations=len(cages),
        fragmented_observations=len(fragmented), ih_recovery_observations=len(ih),
        energy_window_observations=len(energy),
        first_ih_search_requests=min((c['search_cost'] for c in ih), default=None),
        first_energy_window_search_requests=min((c['search_cost'] for c in energy), default=None),
        best_delta_ih_eV=min((c['delta_ih_eV'] for c in landed), default=None),
        best_intact_cage_delta_ih_eV=min((c['delta_ih_eV'] for c in cages), default=None))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, default=HERE / 'escape-plan.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = json.loads(args.plan.read_text())
    runs = Path(plan['output']['runs_dir'])
    global_path = runs / 'summary.json'
    global_rows = json.loads(global_path.read_text()) if global_path.exists() else []
    rows = []
    for seed in plan['arms']['seeds']:
        for method in plan['methods']:
            path = runs / f'{method}-{seed}' / 'summary.json'
            if not path.exists():
                state = next((r['status'] for r in global_rows if r['seed'] == seed and r['method'] == method),
                             'no_completed_summary')
                rows.append(dict(seed=seed, method=method, summary_present=False, status=state))
                continue
            raw = json.loads(path.read_text())
            record_path = path.parent / 'record-costs.json'
            records = json.loads(record_path.read_text()) if record_path.exists() else []
            rows.append(dict(seed=seed, method=method, summary_present=True, source=str(path),
                status=raw['status'], search_requests=raw['search_requests'],
                fresh_requests=raw['fresh_requests'], total_requests=raw['search_requests']+raw['fresh_requests'],
                search_calculations=raw['search_calculations'], fresh_calculations=raw['fresh_calculations'],
                boundary=raw['search_boundary'], elapsed_seconds=raw['elapsed_seconds'],
                record_statuses=[r['status'] for r in records],
                record_cost_sum_matches_requests=raw.get('record_cost_sum_matches_requests'),
                requests_outside_saved_records=raw.get('requests_outside_saved_records'),
                fresh_missing_or_failed=sum(c['status'] != 'checked' for c in raw['checks']),
                full_observed_prefix=events(raw['checks'], raw['search_requests']),
                checks=raw['checks']))
    pairs = []
    for seed in plan['arms']['seeds']:
        pair = [r for r in rows if r['seed'] == seed]
        if not all(r['summary_present'] for r in pair):
            pairs.append(dict(seed=seed, comparison='missing_arm'))
            continue
        ceiling = min(r['search_requests'] for r in pair)
        pairs.append(dict(seed=seed, common_search_prefix=ceiling,
            methods={r['method']: events(r['checks'], ceiling) for r in pair}))
    execution_path = runs / 'execution.json'
    output = dict(scope='Two paired RNG probes of one input; no global success-rate or general algorithm ranking.',
        interpretation='Counts are landing observations, not unique basins. First-hit costs are retrospective search requests; fresh validation was paid after each full arm. All fresh and failed work is also reported. Positive source Hessians do not qualify every searched landing.',
        execution=json.loads(execution_path.read_text()) if execution_path.exists() else None,
        observed_search_requests=sum(r.get('search_requests', 0) for r in rows),
        observed_fresh_requests=sum(r.get('fresh_requests', 0) for r in rows),
        arms=rows, paired_common_prefixes=pairs)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
