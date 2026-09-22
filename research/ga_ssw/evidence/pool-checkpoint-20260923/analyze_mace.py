#!/usr/bin/env python3
"""Offline ledger and continuation audit. No new PES calls; no efficiency claim."""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np

LEGS = ('continuous2', 'first1', 'resume1')


def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [plain(v) for v in value]
    if hasattr(value, '__dict__'): return plain(vars(value))
    return value


def read_pickle(path):
    if not path.exists(): return None
    with path.open('rb') as stream: return pickle.load(stream)


def ledger(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    attempted = [r for r in rows if r.get('status') != 'denied']
    return dict(present=path.exists(), attempted=len(attempted),
        successful=sum('energy_eV' in r for r in attempted),
        errors=sum(r.get('status') == 'error' for r in attempted),
        denied=len(rows)-len(attempted),
        by_leg={name: sum(r.get('leg') == name for r in attempted) for name in LEGS})


def case_analysis(root, name, budgets):
    folder = root/name
    search = ledger(folder/'requests.jsonl')
    fresh_ledgers = {leg: ledger(folder/leg/'fresh.jsonl') for leg in ('continuous2', 'resume1')}
    fresh_count = sum(x['attempted'] for x in fresh_ledgers.values())
    summary = json.loads((folder/'summary.json').read_text()) if (folder/'summary.json').exists() else {}
    fresh = [dict(leg=row.get('leg'), **check) for row in summary.get('rows', []) for check in row.get('fresh_checks', [])]
    results = {leg: read_pickle(folder/leg/'result.pkl') for leg in LEGS}
    report = dict(case=name, search_ledger=search, fresh_ledger=fresh_ledgers,
        fresh_checks=fresh, search_requests=search['attempted'], fresh_requests=fresh_count,
        total_ef=search['attempted']+fresh_count, runner_status=summary.get('status'),
        runner_error=summary.get('error'),
        budget_satisfied=search['attempted'] <= budgets['search_requests_per_case_all_legs'] and fresh_count <= budgets['fresh_terminal_checks_per_case_all_legs'],
        summary_ledger_match=summary.get('search_requests') == search['attempted'] and summary.get('fresh_requests') == fresh_count,
        fresh_qualified=len(fresh) == 4 and all(all(c.get(k, False) for k in ('force_qualified','composition_unchanged','cell_unchanged','pbc_unchanged')) for c in fresh),
        legs={})
    for leg, result in results.items():
        if result is None:
            report['legs'][leg] = dict(status='missing', actual_requests=search['by_leg'][leg])
            continue
        report['legs'][leg] = dict(status=result.status, records=len(result.records),
            statuses=[r.status for r in result.records],
            qualified_landings=sum(r.landing is not None and r.landing.converged for r in result.records),
            cumulative_requests=result.evaluation_requests, actual_requests=search['by_leg'][leg],
            cost_record_consistent=result.initial.evaluation_requests+sum(r.evaluation_requests for r in result.records)==result.evaluation_requests,
            selections=[r.starter_selection for r in result.records],
            ls_steps=[None if r.ls_update is None else r.ls_update.get('step') for r in result.records])
    if any(r is None for r in results.values()):
        report['execution_status'] = 'incomplete'
        return report
    continuous, first, resumed = (results[leg] for leg in LEGS)
    report['execution_status'] = 'completed' if all(r.status == 'completed' for r in results.values()) else 'failed_or_terminal'
    report['expected_record_counts'] = [len(r.records) for r in results.values()] == [2, 1, 2]
    actual = search['by_leg']
    report['request_accounting'] = dict(
        continuous_matches_ledger=continuous.evaluation_requests==actual['continuous2'],
        first_matches_ledger=first.evaluation_requests==actual['first1'],
        resumed_matches_ledger=resumed.evaluation_requests==actual['first1']+actual['resume1'],
        continuous_equals_resumed=continuous.evaluation_requests==resumed.evaluation_requests)
    comparisons = dict(current_position_max_abs_difference_A=float(np.max(np.abs(continuous.current.positions-resumed.current.positions))),
        cell_equal=np.array_equal(continuous.current.cell.array, resumed.current.cell.array),
        pbc_equal=np.array_equal(continuous.current.pbc,resumed.current.pbc),
        composition_equal=np.array_equal(continuous.current.numbers,resumed.current.numbers),
        full_selection_telemetry_equal=plain([r.starter_selection for r in continuous.records])==plain([r.starter_selection for r in resumed.records]),
        ls_steps_equal=report['legs']['continuous2']['ls_steps']==report['legs']['resume1']['ls_steps'],
        ls_updates_equal=plain([r.ls_update for r in continuous.records])==plain([r.ls_update for r in resumed.records]),
        native_mc_state_equal=plain(continuous.checkpoint.native_mc_state)==plain(resumed.checkpoint.native_mc_state))
    keys=('chosen_index','mc_current_index','last_landing_index','step','restarted','ls_reinitialized','direction_reinitialized')
    def choices(result):
        return [None if r.starter_selection is None else {k:r.starter_selection.get(k) for k in keys} for r in result.records]
    comparisons['choices_equal'] = choices(continuous)==choices(resumed)
    for filename, label in [('main-rng-state.pkl','main_rng'),('selector-rng-state.pkl','selector_rng'),('pool-export.pkl','pool_export')]:
        a,b=(read_pickle(folder/leg/filename) for leg in ('continuous2','resume1'))
        comparisons[label+'_present'] = a is not None and b is not None
        comparisons[label+'_equal'] = a is not None and b is not None and plain(a)==plain(b)
        if label=='pool_export' and a is not None and b is not None:
            comparisons['pool_mapping_equal'] = a['mapping']==b['mapping']
            ea,eb=a['archive']['entries'],b['archive']['entries']
            comparisons['pool_entry_counts']=[len(ea),len(eb)]
            comparisons['pool_energy_max_abs_difference_eV'] = max((abs(x['energy']-y['energy']) for x,y in zip(ea,eb)),default=0.) if len(ea)==len(eb) else None
    report['comparisons']=comparisons
    return report


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--input',type=Path,required=True)
    args=parser.parse_args(); root=args.input
    protocol=json.loads((root/'protocol.json').read_text())
    cases=[case_analysis(root,name,protocol['budgets']) for name in protocol['cases']]
    result=dict(purpose='checkpoint interface qualification, not search efficiency', cases=cases,
        total_ef=sum(c['total_ef'] for c in cases),
        total_budget=protocol['budgets']['total_ef_all_cases'])
    (root/'analysis.json').write_text(json.dumps(plain(result),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(cases=[dict(case=c['case'],execution_status=c['execution_status'],ef=c['total_ef']) for c in cases],total_ef=result['total_ef']),indent=2))

if __name__=='__main__': main()
