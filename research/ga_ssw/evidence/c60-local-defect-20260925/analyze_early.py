"""Readout of saved-stage quenches with existing ledger and graph checks."""
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'research/ga_ssw')]


def main():
    import networkx as nx
    import numpy as np
    from ase.io import read
    from analyze_c4h6_ls_reaction_coverage import atoms_from_dict
    from analyze_c60_random_development import graph_row
    from escape_probe import cutoff_graph

    spec = importlib.util.spec_from_file_location('depth_analysis', ROOT / 'research/ga_ssw/evidence/climb-depth-ablation-20260925/analyze.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    out = HERE / 'early-quench'
    inputs = json.loads((out / 'inputs.json').read_text())
    ih = read(HERE / 'qualification/isomer-1/final.extxyz')
    defect = read(HERE / 'qualification/isomer-2/final.extxyz')
    e_ih = json.loads((HERE / 'escape-plan.json').read_text())['references']['ih_energy_eV']
    rows = []
    for item in inputs['rows']:
        folder = out / 'runs' / item['case_id']
        row = helper.analyze_row(item, folder)
        if (folder / 'result.json').exists():
            raw = json.loads((folder / 'result.json').read_text())
            q = raw.get('quench')
            if q:
                atoms = atoms_from_dict(q['atoms'])
                fresh = next(c for c in raw['fresh'] if c['label'] == 'truncated')
                row['target_checks'] = dict(graphs={str(c): dict(
                    **graph_row(atoms.numbers, atoms.positions, c, cutoff_graph(ih,c,nx,np)),
                    defect_graph_match=nx.is_isomorphic(cutoff_graph(atoms,c,nx,np),cutoff_graph(defect,c,nx,np)))
                    for c in (1.64,1.7,1.8)},
                    fresh_delta_Ih_eV=(fresh['energy']-e_ih if fresh.get('status')=='completed' else None))
        rows.append(row)
    result = dict(scope='Post-hoc saved-stage local-defect diagnosis, no new search or default recommendation', rows=rows)
    (out / 'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for r in rows:
        print(r['case_id'], r.get('quench_convergence'), r.get('target_checks'))


if __name__ == '__main__':
    main()
