"""Offline readout of same-population research runs; no calculator calls."""
import argparse
import json
from pathlib import Path

import numpy as np
from ase.io import read
from research.ga_ssw.analyze_c60_random_development import graph_row


def analyze_run(folder):
    plan = json.loads((folder / 'plan.json').read_text())
    summary = json.loads((folder / 'summary.json').read_text())
    paid, refused, indices = 0, 0, []
    with (folder / 'evaluations.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['kind'] in ('request', 'failed_request'):
                paid += 1
                indices.append(row['request_index'])
            elif row['kind'] == 'refused':
                refused += 1
    errors = []
    if paid != summary['search_requests']:
        errors.append('paid ledger count != search_requests')
    if indices != list(range(1, paid+1)):
        errors.append('nonconsecutive paid request indices')
    if paid > plan['search_cap']:
        errors.append('search cap exceeded')
    frames = read(folder / 'observed_minima.extxyz', ':') if (folder / 'observed_minima.extxyz').exists() else []
    observations = []
    for index, atoms in enumerate(frames):
        # extxyz exposes the stored energy through its SinglePointCalculator.
        energy = float(atoms.get_potential_energy())
        info = atoms.info
        # ASE extxyz parses the literal surface=true as boolean True.
        stored_surface = info.get('surface')
        true_surface = stored_surface == 'true' or stored_surface is True or isinstance(stored_surface, np.bool_) and bool(stored_surface)
        numerical = (bool(info.get('converged', False)) and true_surface
                     and np.isfinite(energy) and float(info.get('max_force', float('inf'))) <= plan['ga']['quench_fmax'])
        row = dict(index=index, energy=energy, numerical=numerical,
                   phase=info.get('phase'), archive_eligible=info.get('eligible_for_archive'))
        if len(atoms)==60 and np.all(atoms.numbers==6):
            row['geometry'] = {str(c): graph_row(atoms.numbers, atoms.positions, c) for c in (1.64, 1.7, 1.8)}
            row['relative_energy_eV'] = energy - plan['reference_energy']
            row['cage_candidate'] = numerical and row['geometry']['1.8']['graph_cage_candidate']
            row['energy_candidate'] = numerical and energy <= plan['reference_energy'] + plan['acceptance']['energy_tolerance_eV']
        observations.append(row)
    phase_costs = {}
    for phase in summary.get('phases', []):
        if 'evaluation_requests' in phase:
            cost = phase['evaluation_requests']
        elif 'used' in phase:
            cost = phase['used']
        else:
            cost = max(0, phase.get('request_end', 0) - phase.get('request_start', 1) + 1)
        name = phase['phase']
        phase_costs[name] = phase_costs.get(name, 0) + cost
    if summary.get('phases') and sum(phase_costs.values()) != paid:
        errors.append('phase costs do not close to paid ledger')
    eligible = [r for r in observations if r['numerical']]
    best = min(eligible, key=lambda r:r['energy']) if eligible else None
    phase_best = {}
    for row in eligible:
        phase = row['phase']
        if phase not in phase_best or row['energy'] < phase_best[phase]['energy']:
            phase_best[phase] = {'index': row['index'], 'energy': row['energy']}
    fresh_path=folder/'fresh-checks.json'
    fresh = json.loads(fresh_path.read_text()) if fresh_path.exists() else None
    checked_indices = {row['geometry_index'] for row in (fresh or {}).get('checks', [])
                       if row.get('force_qualified') and row.get('composition_match')
                       and row.get('pbc_match') and np.isfinite(row.get('fresh_energy', np.nan))}
    confirmed_cage = any(row.get('cage_candidate', False) and row['index'] in checked_indices
                         for row in observations)
    confirmed_energy = any(row.get('energy_candidate', False) and row['index'] in checked_indices
                           and check.get('fresh_energy', float('inf')) <= plan['reference_energy'] + plan['acceptance']['energy_tolerance_eV']
                           for row in observations for check in (fresh or {}).get('checks', [])
                           if check.get('geometry_index') == row['index']) if 'reference_energy' in plan else None
    result=dict(folder=str(folder),arm=summary['arm'],seed=plan['seed'],status=summary['status'],
        requests=paid,refused=refused,wall_seconds=summary['wall_seconds'],
        errors=errors,observations=observations,best=best,fresh=fresh,
        phase_requests=phase_costs, phase_best=phase_best,
        calculator_calls=summary.get('search_calculator_calculate_calls'),
        confirmed_cage=confirmed_cage if 'reference_energy' in plan else None,
        confirmed_reference_energy=confirmed_energy,
        interpretation='Numerical candidates and independently checked endpoints are distinct. No unique-basin count, DFT validation, or reliable success probability is inferred.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-runs',type=int)
    args=parser.parse_args()
    folders=sorted(p.parent for p in args.root.glob('**/summary.json') if (p.parent/'evaluations.jsonl').exists())
    results=[analyze_run(folder) for folder in folders]
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([dict(folder=r['folder'],status=r['status'],requests=r['requests'],errors=r['errors'],best_energy=None if r['best'] is None else r['best']['energy']) for r in results],indent=2))
    assert results, 'no completed run summaries'
    if args.expected_runs is not None:
        assert len(results) == args.expected_runs, f'expected {args.expected_runs} completed summaries, found {len(results)}; missing arms are not successes'
    assert not any(r['errors'] for r in results), 'ledger audit failed; preserve output'

if __name__=='__main__':
    main()
