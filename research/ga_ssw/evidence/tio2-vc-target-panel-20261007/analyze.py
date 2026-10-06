"""Audit four bounded TiO2 VC arms; no calculator calls or geometric retuning."""
import argparse
import json
from collections import Counter
from pathlib import Path


def ledger_cost(path):
    rows = [json.loads(line) for line in path.open()] if path.exists() else []
    paid = [r for r in rows if r['event'] in ('evaluation', 'failure')]
    return dict(present=path.exists(), requests=len(paid),
                calculator_calls=sum(r.get('calculator_calls', 0) for r in rows),
                failures=sum(r['event'] == 'failure' for r in rows),
                denials=dict(Counter(r.get('reason') for r in rows if r['event'] == 'denial')))


def read_arm(path, slot):
    costs = {kind: ledger_cost(path / name) for kind, name in
             [('search', 'requests.jsonl'), ('fresh', 'fresh/requests.jsonl'),
              ('reference', 'shared-reference/requests.jsonl')]}
    if not (path / 'result.json').exists():
        return dict(slot=slot, path=str(path), result_present=False, raw_costs=costs)
    result = json.loads((path / 'result.json').read_text())
    provenance = json.loads((path / 'provenance.json').read_text())
    expected_case, expected_arm = [('phase87_12', 'joint_vc'), ('phase87_12', 'block'),
                                   ('phase87_48', 'joint_vc'), ('phase87_48', 'block')][slot]
    if (result['case'], result['arm']) != (expected_case, expected_arm):
        raise AssertionError(f'{path}: unexpected slot identity')
    if provenance['head_pamssw_tree'] != 'c572a1cc0766f8d3ff534a467ad64613aaa78ce0':
        raise AssertionError('unexpected core tree')
    for prefix, fields in [('search', {'requests': 'search_paid_requests', 'calculator_calls': 'search_calculator_calls',
                                      'failures': 'search_failures'}),
                           ('fresh', {'requests': 'fresh_requests', 'calculator_calls': 'fresh_calculator_calls',
                                     'failures': 'fresh_failures'})]:
        for key, field in fields.items():
            if costs[prefix][key] != result[field]:
                raise AssertionError(f'{path}: raw {prefix}/{key} mismatch')
    for key in ('ledger_matches_budget', 'walker_matches_budget',
                'records_match_budget', 'minima_rows_match_walker'):
        if result['cost_closure'][key] is False:
            raise AssertionError(f'{path}: {key} failed')
    if slot == 0:
        reference = result.get('shared_reference_cold_check', {})
        for key in ('requests', 'calculator_calls', 'failures'):
            if reference.get(key, 0) != costs['reference'][key]:
                raise AssertionError(f'{path}: shared reference/{key} mismatch')
    minima_path = path / 'minima.jsonl'
    minima = [json.loads(line) for line in minima_path.open()] if minima_path.exists() else []
    outer_path = path / 'outer-records.jsonl'
    outer = [json.loads(line) for line in outer_path.open()] if outer_path.exists() else []
    # Preserve all raw observations. Different observations can be the same basin.
    qualified = [r for r in minima if r.get('physical_gate_passed')]
    best = min(qualified, key=lambda r: r['energy_per_atom_eV'], default=None)
    return dict(slot=slot, path=str(path), result_present=True, raw_costs=costs,
                result=result, provenance=provenance, qualified_observations=qualified,
                best_qualified_observation=best,
                outer_statuses=dict(Counter(r['status'] for r in outer)),
                accepted_count=sum(bool(r.get('accepted')) for r in outer),
                partial_stage_failures=[r for r in outer if r['status'] != 'valid_landing' and not r.get('accepted')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arms', nargs=4, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    arms = [read_arm(path, slot) for slot, path in enumerate(args.arms)]
    ref = arms[0].get('result', {}).get('shared_reference_cold_check', {})
    ref_passed = bool(ref.get('qualified'))
    pairs = []
    for i in (0, 2):
        pair = arms[i:i+2]
        horizon = min(a['raw_costs']['search']['requests'] for a in pair)
        pairs.append(dict(slots=[i, i+1], common_paid_horizon=horizon,
            best_energy_per_atom_at_common_cost={a.get('result', {}).get('arm', f'missing-{a["slot"]}'):
                min((r['energy_per_atom_eV'] for r in a.get('qualified_observations', [])
                     if r['paid_requests_through_outer'] <= horizon), default=None) for a in pair}))
    totals = {kind: sum(a['raw_costs'][kind]['requests'] for a in arms)
              for kind in ('search', 'fresh', 'reference')}
    confirmed = [a['slot'] for a in arms if ref_passed and
                 a.get('result', {}).get('first_target_cold_confirmed')]
    summary = dict(scope='same-model paper-geometry target; existing whole VC pipelines; no DFT/rate or isolated-component claim',
                   planned_arms=4, missing_slots=[a['slot'] for a in arms if not a['result_present']],
                   raw_costs=totals, reused_qualification_requests=31,
                   reference_cold_qualified=ref_passed, confirmed_slots=confirmed,
                   arms=arms, paired_common_cost=pairs, real_pes_requests_from_analysis=0)
    lines = ['# TiO2 VC anatase target', '',
             '| Input | Walker | Paid search | Valid observations | Cold-confirmed target | First target cost | Execution |',
             '|---|---|---:|---:|---|---:|---|']
    for a in arms:
        r = a.get('result', {})
        execution = r.get('status', 'missing result')
        if a['raw_costs']['search']['denials']:
            execution += ' / budget censored'
        lines.append(f'| {r.get("case")} | {r.get("arm")} | {a["raw_costs"]["search"]["requests"]} | '
                     f'{len(a.get("qualified_observations", []))} | {a["slot"] in confirmed} | '
                     f'{r.get("first_target_paid_requests")} | {execution} |')
    lines += ['', f'Current search/cold/reference requests: {totals}; prerequisite qualification31 separately.',
              'Valid observations can repeat basins; no paper success-rate estimate or optimizer ranking.',
              'Initial, best and selected first-target/last geometries alone receive independent cold checks.']
    (args.out / 'analysis.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    (args.out / 'README.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
