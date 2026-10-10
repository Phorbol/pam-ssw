"""Summarize fixed one-step LS comparisons without conflating landing and acceptance."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def analyze(source, output):
    if output.exists():
        raise FileExistsError(output)
    data = json.loads(source.read_text())
    if data['status'] != 'all_24_arms_recorded':
        raise ValueError('campaign incomplete: preserve raw partial evidence, do not rank')
    rows = []
    groups = defaultdict(lambda: dict(attempts=0, completed_physical_landings=0,
        fresh_force_qualified=0, accepted_candidates=0, truncated=0,
        graph_change_candidates=0, disconnected_candidates=0, requests=0, fresh_requests=0,
        wall_seconds=0.0))
    for case in data['runs']:
        fresh = case.get('candidate_fresh_check', {})
        diagnostics = case.get('candidate_geometry_diagnostics', {})
        refs = diagnostics.get('reference_comparisons', {})
        labels = [name for name, ref in refs.items() if ref['graph_isomorphic_with_element_labels']]
        prep = case.get('ls_preparation') or {}
        soft = prep.get('soft_quench') or {}
        physical = bool(case.get('physical_landing_completed'))
        qualified = physical and bool(fresh.get('force_qualified'))
        connected = diagnostics.get('component_count') == 1
        graph_change = qualified and connected and not any(x in labels for x in ('gauche', 'trans'))
        row = {key: case.get(key) for key in ('run_id','start','seed','arm','status','run_ssw_status',
            'outer_step_status','error','search_requests','fresh_requests','search_truncated',
            'search_truncation_reason','elapsed_seconds','mc_accepted','candidate_produced',
            'physical_landing_completed','search_count_matches_result','checkpoint_read_error')}
        row.update(fresh_force_qualified=bool(fresh.get('force_qualified')),
            fresh_energy_eV=fresh.get('energy_eV'), fresh_fmax_eV_A=fresh.get('fmax_eV_A'),
            component_count=diagnostics.get('component_count'), graph_labels=labels,
            connected_graph_change_candidate=graph_change,
            ls_prequench_qualification=prep.get('qualification'),
            ls_prequench_fmax_eV_A=soft.get('max_force'),
            ls_prequench_requests=prep.get('evaluation_requests'),
            physical_response_eV=(prep['true_energy_after']-prep['true_energy_before']) if prep else None,
            reference_comparisons=refs)
        rows.append(row)
        group = groups[(case['start'], case['arm'])]
        group['attempts'] += 1
        group['completed_physical_landings'] += physical
        group['fresh_force_qualified'] += qualified
        group['accepted_candidates'] += qualified and bool(case.get('mc_accepted'))
        group['truncated'] += bool(case.get('search_truncated'))
        group['graph_change_candidates'] += graph_change
        group['disconnected_candidates'] += qualified and not connected
        group['requests'] += case['search_requests']
        group['fresh_requests'] += case['fresh_requests']
        group['wall_seconds'] += case['elapsed_seconds']
    report = dict(source=str(source.resolve()), status=data['status'],
        search_requests=data['search_requests_total'], fresh_requests=data['fresh_requests_total'],
        actual_calculate_calls=data['actual_calculate_calls_total'], rows=rows,
        groups=[dict(start=k[0],arm=k[1],**v) for k,v in groups.items()],
        limits='Graph changes are candidates, not independently validated basin identities; completed landing may return to original basin; four seeds are development evidence only.')
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = analyze(args.source, args.output)
    print(json.dumps(result['groups'], indent=2))
