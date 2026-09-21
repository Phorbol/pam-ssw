"""Offline structural/MC audit for constrained direction diagnostic results."""
import json
from pathlib import Path
import numpy as np
from ase import Atoms
from scipy.optimize import linear_sum_assignment
from ase.geometry import find_mic


def atoms(data):
    return Atoms(numbers=data['numbers'], positions=data['positions'],
                 cell=data['cell'], pbc=data['pbc'])


def active_assignment(a, b, active):
    cost = np.full((len(active), len(active)), np.inf)
    for i, ai in enumerate(active):
        for j, bj in enumerate(active):
            if a.numbers[ai] != b.numbers[bj]:
                continue
            delta, _ = find_mic(a.positions[ai] - b.positions[bj], a.cell, a.pbc)
            cost[i, j] = float(np.dot(delta, delta))
    rows, cols = linear_sum_assignment(cost)
    if not np.isfinite(cost[rows, cols]).all():
        raise ValueError('active species composition differs')
    return float(np.sqrt(np.mean(cost[rows, cols]))), [(int(active[i]), int(active[j])) for i, j in zip(rows, cols)]


def main():
    root = Path('research/ga_ssw/evidence/constrained-direction-diagnostic-20260912')
    output = []
    for path in sorted(root.glob('*-result.json')):
        data = json.loads(path.read_text())
        result = data['result']; initial = atoms(result['initial']['atoms'])
        fixed = np.asarray(data['fixed_indices'], dtype=int)
        active = np.asarray(data['active_indices'], dtype=int)
        fixed_top = float(initial.positions[fixed, 2].max())
        tag1_support = np.setdiff1d(active, [len(initial) - 1])
        tag1_top = float(initial.positions[tag1_support, 2].max())
        first_record = next(record for record in result['records'][1:]
                            if record.get('landing') is not None)
        first = atoms(first_record['landing']['atoms'])
        accepted = [r.get('accepted') for r in result['records']]
        rows = []
        for index, minimum in enumerate(result['minima']):
            current = atoms(minimum['atoms'])
            rms_initial, assignment_initial = active_assignment(current, initial, active)
            rms_first, assignment_first = active_assignment(current, first, active)
            z = current.positions[:, 2]
            current_tag1_top = float(current.positions[tag1_support, 2].max())
            rows.append(dict(minimum=index, rms_active_to_initial=rms_initial,
                assignment_to_initial=assignment_initial, rms_active_to_first_landing=rms_first,
                assignment_to_first_landing=assignment_first,
                active_below_fixed_top=bool(np.any(z[active] < fixed_top)),
                highest_atom_index=int(np.argmax(z)),
                labeled_adatom_height_from_fixed_top=float(z[-1] - fixed_top),
                initial_tag1_top_z=tag1_top, current_tag1_top_z=current_tag1_top,
                labeled_adatom_height_from_initial_tag1_top=float(z[-1] - tag1_top),
                labeled_adatom_height_from_current_tag1_top=float(z[-1] - current_tag1_top)))
        output.append(dict(arm=path.stem, requests=data['requests'], record_count=len(result['records']),
            records_accepted=accepted, mc_rejections=sum(x is False for x in accepted),
            rotation_failed=sum(r.get('status') == 'rotation_failed' for r in result['records']),
            minima_count=len(result['minima']), minima=rows))
    (root / 'physical-audit.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()
