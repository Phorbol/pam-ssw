"""Zero-PES audit of all four preregistered held-out arms; never resubmit."""
import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
from ase.io import read
import reused_readout as prior

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent / 'c60-recovered-rotation-ls-prospective-20260922'
REFERENCE = Path('/home/gengjianrui/bin/pam-ssw-worktrees/ga-ssw-behavior-parity/research/ga_ssw/evidence/c60-reference-qualification-v2-20260917/final.extxyz')


def analyze(root):
    plan = prior.load(root / 'plan.json')
    graph = prior.graph_module()
    reference = read(REFERENCE)
    distance = np.linalg.norm(reference.positions[:, None] - reference.positions[None, :], axis=2)
    reference_graph = nx.Graph()
    reference_graph.add_nodes_from(range(60))
    reference_graph.add_edges_from(zip(*np.where(np.triu((distance < 1.8) & (distance > 0), 1))))
    assert graph.graph_row(reference.numbers, reference.positions, 1.8)['graph_cage_candidate']
    rows, errors = [], []
    for case in plan['cases']:
        for name in ('recovered_rotation_native_ls',):
            folder = root / f"{case}-seed{plan['seeds'][case]}"
            if not (folder / 'summary.json').exists():
                rows.append(dict(case=case, arm=name, status='missing', folder=str(folder)))
                errors.append(f'{case}/{name}: missing summary')
                continue
            summary = prior.load(folder / 'summary.json')
            try:
                row = prior.arm(root, case, plan['seeds'][case], True, graph, reference_graph)
                row['arm'] = name
                row['execution_class'] = summary.get('execution_class')
                row['all_fresh'] = summary.get('fresh', [])
                ledger = row['ledger']
                integrity = (ledger.get('present') and ledger.get('ids_contiguous') and
                             ledger.get('ids_unique') and ledger.get('summary_request_match') and
                             ledger.get('summary_denial_match') and not ledger.get('errors'))
                row['cost_closed'] = bool(integrity)
                result_path = folder / 'result.json'
                if result_path.exists():
                    result = prior.load(result_path)
                    total = result['initial']['evaluation_requests'] + sum(
                        r['evaluation_requests'] for r in result['records'])
                    row['record_request_sum'] = total
                    row['cost_closed'] &= total == result['evaluation_requests'] == summary['search_requests']
                    updates = []
                    for record in result['records']:
                        update = record.get('ls_update')
                        if update is not None:
                            response = record.get('energy_response')
                            observed = update.get('observed_response_mev_per_atom')
                            if response is None or observed is None or not np.isclose(observed, 1000 * response, rtol=1e-12, atol=1e-12):
                                errors.append(f'{case}/{name}: LS response units mismatch')
                            updates.append(update)
                        if record.get('status') == 'ls_prequench_failed' and update is not None:
                            errors.append(f'{case}/{name}: failed prequench updated LS')
                    row['ls_updates'] = updates
                    row['ls_preparations'] = sum(r.get('ls_preparation') is not None for r in result['records'])
                    candidates = [m for m in row['result']['minima']
                                  if m['converged'] and m['graph_cage_candidate']]
                    first = candidates[0]['index'] if candidates else None
                    best = min(range(len(result['minima'])),
                               key=lambda i: result['minima'][i]['energy'])
                    expected_extra = first is not None and first not in (0, best)
                    expected_index = first if expected_extra else None
                    expected_labels = ['initial', 'best'] + (['first_cage'] if expected_extra else [])
                    if (summary.get('fresh_initial_best_requests') != 2 or
                            summary.get('fresh_conditional_requests') != int(expected_extra) or
                            summary.get('conditional_cage_index') != expected_index or
                            [q.get('label') for q in row['all_fresh']] != expected_labels):
                        errors.append(f'{case}/{name}: conditional fresh selection/count mismatch')
                    label = ('initial' if first == 0 else 'best' if first == best else 'first_cage')
                    check = next((c for c in row['all_fresh'] if c['label'] == label), None)
                    row['first_cage_index'] = first
                    row['cage_independently_qualified'] = False
                    if first is not None:
                        traj = folder / f'{label}.traj'
                        if not check or not traj.exists():
                            errors.append(f'{case}/{name}: first cage lacks planned independent check')
                        else:
                            atoms = read(traj)
                            saved = result['minima'][first]['atoms']
                            exact = (np.array_equal(atoms.positions, saved['positions']) and
                                     np.array_equal(atoms.numbers, saved['numbers']) and
                                     np.array_equal(atoms.cell.array, saved['cell']) and
                                     np.array_equal(atoms.pbc, saved['pbc']))
                            row['cage_independently_qualified'] = bool(exact and check.get('numerical_qualified'))
                    best_fresh = next((c for c in row['all_fresh'] if c['label'] == 'best'), {})
                    row['energy_target_independently_qualified'] = bool(
                        best_fresh.get('numerical_qualified') and
                        best_fresh['energy_eV'] <= plan['acceptance']['energy_reference_eV'] +
                        plan['acceptance']['energy_target_margin_eV'])
                else:
                    row['failed_initial_present'] = (folder / 'failed-initial.json').is_file()
                    row['physical_qualification'] = 'no completed minimum result'
                    if summary.get('failed_initial_saved') or 'InitialQuenchError' in summary.get('error', ''):
                        if not row['failed_initial_present'] or not (folder / 'failed-initial.traj').is_file():
                            errors.append(f'{case}/{name}: missing failed initial certificate/structure')
                        else:
                            failed = prior.load(folder / 'failed-initial.json')
                            required = {'converged', 'evaluation_requests', 'optimizer_steps', 'surface'}
                            if not required.issubset(failed) or failed['converged']:
                                errors.append(f'{case}/{name}: invalid failed initial certificate')
                    if (summary.get('fresh_initial_best_requests') != 0 or
                            summary.get('fresh_conditional_requests') != 0 or row['all_fresh']):
                        errors.append(f'{case}/{name}: unexpected fresh calls without minima')
                if not row['cost_closed']:
                    errors.append(f'{case}/{name}: cost not closed')
                if summary['search_requests'] > plan['search_cap']:
                    errors.append(f'{case}/{name}: search budget exceeded')
                if summary['fresh_requests'] > plan['fresh_per_case'] + plan['conditional_first_cage_extra_per_case']:
                    errors.append(f'{case}/{name}: fresh budget exceeded')
                if summary['fresh_requests'] != len(row['all_fresh']):
                    errors.append(f'{case}/{name}: fresh accounting mismatch')
                if summary['fresh_requests'] != (summary.get('fresh_initial_best_requests', -1) +
                                                  summary.get('fresh_conditional_requests', -1)):
                    errors.append(f'{case}/{name}: fresh subcounts not closed')
                row['fresh_all_qualified'] = bool(row['all_fresh']) and all(
                    q.get('numerical_qualified') for q in row['all_fresh'])
                rows.append(row)
            except Exception as exc:
                rows.append(dict(case=case, arm=name, status='analysis_error', error=repr(exc)))
                errors.append(f'{case}/{name}: {exc!r}')
    totals = {
        'search_requests': sum(r.get('search_requests', 0) for r in rows),
        'actual_calculate': sum(r.get('actual_calculate', 0) or 0 for r in rows),
        'fresh_requests': sum(r.get('fresh_requests', 0) for r in rows),
        'fresh_actual_calculate': sum(r.get('fresh_actual_calculate', 0) or 0 for r in rows),
    }
    reused = []
    for case in plan['cases']:
        old = root.parent / 'c60-recovered-rotation-long-20260921' / f"{case}-seed{plan['seeds'][case]}" / 'summary.json'
        summary = prior.load(old)
        reused.append(dict(case=case, source=str(old), search_requests=summary['search_requests'],
                           actual_calculate=summary['search_calculator_calls'], fresh=summary['fresh']))
    return dict(reused_no_ls_baselines=reused, new_cost_only=totals, protocol='old development inputs; only NativeLS enabled; not independent validation',
                source=str(root), reference=str(REFERENCE), arms=rows, totals=totals, errors=errors,
                qualification_note='Saved force certificates qualify minima numerically; only recorded fresh frames were independently recomputed. Graph candidates alone are not independently validated cages.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze(args.root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(totals=result['totals'], errors=result['errors'])))
    raise SystemExit(2 if result['errors'] else 0)
