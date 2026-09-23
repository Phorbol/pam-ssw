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
        numerical = (bool(info.get('converged', False)) and info.get('surface') == 'true'
                     and np.isfinite(energy) and float(info.get('max_force', float('inf'))) <= plan['ga']['quench_fmax'])
        row = dict(index=index, energy=energy, numerical=numerical,
                   phase=info.get('phase'), archive_eligible=info.get('eligible_for_archive'))
        if len(atoms)==60 and np.all(atoms.numbers==6):
            row['geometry'] = {str(c): graph_row(atoms.numbers, atoms.positions, c) for c in (1.64, 1.7, 1.8)}
            row['relative_energy_eV'] = energy - plan['reference_energy']
            row['cage_candidate'] = numerical and row['geometry']['1.8']['graph_cage_candidate']
            row['energy_candidate'] = numerical and energy <= plan['reference_energy'] + plan['acceptance']['energy_tolerance_eV']
        observations.append(row)
    eligible = [r for r in observations if r['numerical']]
    best = min(eligible, key=lambda r:r['energy']) if eligible else None
    fresh_path=folder/'fresh-checks.json'
    fresh = json.loads(fresh_path.read_text()) if fresh_path.exists() else None
    result=dict(folder=str(folder),arm=summary['arm'],seed=plan['seed'],status=summary['status'],
        requests=paid,refused=refused,wall_seconds=summary['wall_seconds'],
        errors=errors,observations=observations,best=best,fresh=fresh,
        interpretation='Numerical candidates and independently checked endpoints are distinct. No unique-basin count, DFT validation, or reliable success probability is inferred.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    folders=sorted(p.parent for p in args.root.glob('**/summary.json') if (p.parent/'evaluations.jsonl').exists())
    results=[analyze_run(folder) for folder in folders]
    args.output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([dict(folder=r['folder'],status=r['status'],requests=r['requests'],errors=r['errors'],best_energy=None if r['best'] is None else r['best']['energy']) for r in results],indent=2))
    assert results, 'no completed run summaries'
    assert not any(r['errors'] for r in results), 'ledger audit failed; preserve output'

if __name__=='__main__':
    main()
