"""Check capture replay identity and locate fragmentation; no PES requests."""
import argparse
import json
import math
from pathlib import Path

import networkx as nx
import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent


def connectivity(atoms, cutoff):
    distances = atoms.get_all_distances()
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from(zip(*np.where(np.triu((distances > 0) & (distances < cutoff), 1))))
    components = sorted((sorted(c) for c in nx.connected_components(graph)), key=len, reverse=True)
    return dict(sizes=[len(c) for c in components], isolated=[c[0] for c in components if len(c) == 1])


def same(a, b):
    if a is None or b is None:
        return a is b
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
    return a == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=HERE/"stage-probe-runs")
    args = parser.parse_args()
    results = []
    for arm in ('global', 'paper'):
        for seed in (25092501, 25092502):
            folder = args.runs/f'lj38-{arm}-seed{seed}'
            row = dict(arm=arm, seed=seed, prefix_qualified=False)
            try:
                run = json.loads((folder/'summary.json').read_text())
                old_root = 'runs' if arm == 'global' else 'paper-direction-runs'
                old = [json.loads(line) for line in (HERE/old_root/f'lj38-seed{seed}'/'outer-steps.jsonl').read_text().splitlines()]
                old = {entry['step']: entry for entry in old if isinstance(entry.get('step'), int)}
                mismatch = []
                if run['initial_requests'] != old[-1]['requests']:
                    mismatch.append('initial_requests')
                steps = run['outer_steps']
                fields = {'status': 'status', 'accepted': 'accepted', 'outer_requests': 'step_requests',
                          'landing_energy_eV': 'landing_energy_eV', 'current_energy_after_eV': 'current_energy_eV',
                          'best_energy_after_eV': 'best_energy_eV'}
                for step in steps:
                    previous = old[step['outer_index']]
                    mismatch.extend(f"{step['outer_index']}:{key}" for key, old_key in fields.items()
                                    if not same(step[key], previous[old_key]))
                if run['search_requests'] != run['initial_requests'] + sum(s['outer_requests'] for s in steps):
                    mismatch.append('cost_accounting')
                row.update(status=run['status'], requests=run['search_requests'], mismatches=mismatch,
                           prefix_qualified=not mismatch and len(steps) == 3 and run['status'] == 'completed')
                stage_rows = []
                for step in steps:
                    outer = step['outer_index']
                    landing = folder/f'outer-{outer:02d}-landing.extxyz'
                    for stage in step['stages']:
                        index = stage['gaussian_index']
                        path = folder/'biased-endpoints'/f'outer-{outer:02d}-gaussian-{index:02d}.extxyz'
                        value = dict(outer=outer, gaussian=index, status=stage['status'],
                                     requests=stage['stage_requests'], rotation_requests=stage['rotation_force_cost'],
                                     quench_requests=stage.get('quench_requests'))
                        if path.exists():
                            atoms = read(path)
                            value['endpoints'] = {str(scale): connectivity(atoms, scale*2.7) for scale in (1.3, 1.5)}
                        if stage['mode_direction'] is not None:
                            direction = np.asarray(stage['mode_direction'])
                            weights = np.sum(direction**2, axis=1)
                            value.update(participation_ratio=float(weights.sum()**2/(len(weights)*np.sum(weights**2))),
                                         largest_atom_weight=float(weights.max()/weights.sum()),
                                         largest_atom_index=int(weights.argmax()))
                        stage_rows.append(value)
                    if landing.exists():
                        atoms = read(landing)
                        stage_rows.append(dict(outer=outer, gaussian='true_landing',
                            endpoints={str(scale): connectivity(atoms, scale*2.7) for scale in (1.3,1.5)}))
                row['stages'] = stage_rows
            except Exception as error:
                row['error'] = f'{type(error).__name__}: {error}'
            results.append(row)
    output = dict(scope='Outcome-selected diagnostic; connectivity and localization are not a causal mechanism or basin identity.',
                  potential_requests=0, all_prefixes_qualified=all(r['prefix_qualified'] for r in results), runs=results)
    (args.runs/'analysis.json').write_text(json.dumps(output, indent=2)+'\n')
    print('prefixes qualified:', output['all_prefixes_qualified'])
    for row in results:
        print(row['arm'],row['seed'],row.get('requests'),row.get('mismatches'),row.get('error'))
    if not output['all_prefixes_qualified']:
        raise SystemExit('Capture replay not qualified: do not interpret stage differences.')


if __name__ == '__main__':
    main()
