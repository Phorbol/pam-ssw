"""Independent saved-geometry check of first Ih hits; zero PES requests."""
import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=HERE / "direction-probe/runs")
    parser.add_argument("--output", type=Path, default=HERE / "direction-probe/hit-geometry.json")
    args = parser.parse_args()
    import numpy as np
    from ase.io import read
    spec = importlib.util.spec_from_file_location('geometry_analysis', ROOT / 'research/ga_ssw/evidence/climb-depth-ablation-20260925/analyze.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    reference = read(HERE / 'qualification/isomer-1/final.extxyz')
    ref = dict(numbers=reference.numbers.tolist(), positions=reference.positions.tolist())
    ref_graph = helper.build_graph(ref, 'C60', cutoff=1.8)
    rows = []
    for checks_path in sorted(args.runs.glob('*-*/checks.json')):
        folder = checks_path.parent
        checks = json.loads((folder / 'checks.json').read_text())
        hits = [c for c in checks if c['role'].startswith('landing-') and all(c['graphs'][str(t)]['ih_graph_match'] for t in (1.64,1.7,1.8))]
        if not hits:
            continue
        hit = min(hits, key=lambda c:c['search_cost'])
        index = int(hit['role'].split('-')[-1])
        record = json.loads((folder / 'result.json').read_text())['records'][index]
        atoms = record['landing']['atoms']
        positions = np.asarray(atoms['positions'])
        distances = np.linalg.norm(positions[:,None]-positions[None,:], axis=2)
        rows.append(dict(arm=folder.name, first_hit=hit, accepted=record['accepted'],
            minimum_pair_distance_A=float(distances[np.triu_indices(60,1)].min()),
            independent_geometry_vs_reference=helper.proper_kabsch_rms(atoms,ref,helper.build_graph(atoms,'C60',cutoff=1.8),ref_graph)))
    output = args.output
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(dict(scope='Geometric identity check alongside existing fresh force/energy checks; not a TS/barrier or new Hessian certificate',rows=rows),indent=2)+'\n')
    print(output)


if __name__ == '__main__':
    main()
