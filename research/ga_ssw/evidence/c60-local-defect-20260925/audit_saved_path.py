"""Zero-PES readout of saved biased-path geometries, not minima qualification."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))


def main():
    import numpy as np
    import networkx as nx
    from ase.io import read

    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = importlib.util.spec_from_file_location('graph_helper', ROOT / 'research/ga_ssw/analyze_c60_random_development.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    reference = read(HERE / 'qualification/isomer-1/final.extxyz')
    def graph(x, cutoff):
        d = np.linalg.norm(x[:, None] - x[None, :], axis=2)
        g = nx.Graph()
        g.add_nodes_from(range(60))
        g.add_edges_from(zip(*np.where(np.triu((d < cutoff) & (d > 0), 1))))
        return g
    cutoffs = (1.64, 1.7, 1.8)
    refs = {c: graph(reference.positions, c) for c in cutoffs}
    rows = []
    for folder in sorted((HERE / 'runs').glob('*-*')):
        source = folder / 'result.json'
        if not source.exists():
            continue
        result = json.loads(source.read_text())
        current_energy = result['initial']['energy']
        for record in result['records']:
            climb = record['climb']
            stages = []
            for j, stage in enumerate(climb):
                # SSW records the next Gaussian center at the previous biased
                # optimizer endpoint; final last_atoms precedes true quenching.
                positions = (climb[j+1]['center'] if j+1 < len(climb)
                             else record['last_atoms']['positions'])
                x = np.asarray(positions, float)
                stages.append(dict(index=stage['index'], height_eV=stage['weight'],
                    true_delta_from_current_eV=stage['true_energy']-current_energy,
                    cumulative_climb_requests=sum(g['requests'] for g in climb[:j+1]),
                    graphs={str(c): helper.graph_row([6]*60, x, c, refs[c]) for c in cutoffs}))
            rows.append(dict(arm=folder.name, outer_index=record['index'],
                accepted=record['accepted'], current_energy_eV=current_energy,
                stages=stages, landing_delta_from_current_eV=record['landing']['energy']-current_energy))
            if record['accepted']:
                current_energy = record['landing']['energy']
    output = dict(scope='Biased geometry and real energy at that geometry only; no quench or PES requests. Graph changes under strain do not establish irreversible bond breaking.',
                  endpoint_provenance='next climb.center; final record.last_atoms, per paper_reference.py run_ssw', rows=rows)
    args.output.write_text(json.dumps(output, indent=2)+'\n')


if __name__ == '__main__':
    main()
