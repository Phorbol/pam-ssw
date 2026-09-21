"""Offline periodic geometry diagnostics for the fixed-cell AlOH comparison.

Coordination cutoffs are diagnostic sensitivity bands, not search parameters
or proof of bonding, phase identity, or a positive Hessian. No PES calls.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.neighborlist import neighbor_list


def describe(a):
    ii, jj, dd, shifts = neighbor_list('ijdS', a, 6., self_interaction=False)
    z = a.numbers
    result = dict(formula=a.get_chemical_formula(), volume_A3=a.get_volume(),
                  minimum_distance_within_6A=float(dd.min()),
                  species_minimum_distances_within_6A={}, coordination={}, hydrogen_oxygen=[])
    for za in sorted(set(z)):
        for zb in sorted(set(z)):
            if za > zb:
                continue
            mask = (z[ii] == za) & (z[jj] == zb)
            result['species_minimum_distances_within_6A'][f'{za}-{zb}'] = float(dd[mask].min()) if mask.any() else None
    for center, neighbor, cutoffs in [(13, 8, [2.1, 2.3, 2.5]), (1, 8, [1.1, 1.2, 1.3, 1.5])]:
        ids = np.flatnonzero(z == center)
        result['coordination'][f'{center}-{neighbor}'] = dict(
            center_indices=ids.tolist(), counts={str(c): [int(np.sum((ii == i) & (z[jj] == neighbor) & (dd < c))) for i in ids] for c in cutoffs})
    for i in np.flatnonzero(z == 1):
        indices = np.flatnonzero((ii == i) & (z[jj] == 8))
        if len(indices):
            k = indices[np.argmin(dd[indices])]
            result['hydrogen_oxygen'].append(dict(h=int(i), o=int(jj[k]), image=shifts[k].tolist(), distance_A=float(dd[k])))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.directory.glob('frame*/result.json')):
        result = json.loads(path.read_text())
        if 'result' in result:
            result = result['result']
        initial = result['initial']['atoms']
        observations = []
        for index, m in enumerate(result['minima']):
            data = m['atoms']
            a = Atoms(numbers=data['numbers'], positions=data['positions'], cell=data['cell'], pbc=data['pbc'])
            observations.append(dict(index=index, energy=m['energy'],
                delta_initial_eV=m['energy']-result['initial']['energy'],
                numbers_preserved=data['numbers']==initial['numbers'],
                pbc_preserved=data['pbc']==initial['pbc'],
                cell_preserved=bool(np.array_equal(data['cell'], initial['cell'])),
                **describe(a)))
        rows.append(dict(arm=path.parent.name, observations=observations))
    output = args.directory/'root-periodic-geometry-audit.json'
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(dict(rows=rows, scope=__doc__), indent=2)+'\n')
    print(json.dumps(dict(arms=len(rows), observations=sum(len(r['observations']) for r in rows))))


if __name__ == '__main__':
    main()
